"""政治经济学反馈环 (v2.2 新增).

核心机制：
- 定义"政权支持度"状态变量，受就业、工资、不平等、福利影响
- 支持度下降 → 策略参数受约束（提取率上限下降、投资偏向福利）
- 产生"先跳后崩"模式：B策略初期成功 → 社会压力积累 → 政策被迫转向
- 抗议风险累积 → 当超过阈值时可能触发 social_collapse 或政策剧变

理论基础：
- Acemoglu & Robinson (2006) 的政治经济学框架
- Rodrik (2016) 的去工业化政治后果
- 本模型特有的"数字-工业"二元结构下的政治动力学
"""

import numpy as np
from ..core.state import SystemState, ALL_SECTORS
from ..core.state import SystemState, PoliticalSupport


class PoliticalEconomyModule:
    """政治经济学反馈模块.

    在每个时间步计算政治支持度的变化，并据此调整策略参数约束。
    不直接决定策略——而是限制策略空间。
    """

    def __init__(self, params: dict):
        self.p = params
        # 追踪历史状态用于检测趋势
        self._support_history: list = []
        self._last_update_t: float = 0.0
        # v2.2: 政策约束的惯性（避免政策频繁切换）
        self._constraint_momentum: float = 0.0

    def step(self, state: SystemState, dt: float):
        """一个时间步的政治经济学更新."""
        ps = state.political_support

        # 1. 计算各满意度分量
        self._compute_satisfaction_components(state)

        # 2. 综合支持度
        self._aggregate_support(state)

        # 3. 计算抗议风险
        self._compute_protest_risk(state)

        # 4. 计算政策约束（带惯性）
        self._compute_policy_constraint(state, dt)

        # 5. 计算改革动能
        self._compute_reform_momentum(state)

    def _compute_satisfaction_components(self, state: SystemState):
        """计算政治支持度的四个分量 (v3.2: 使用群组差异化政治权重)."""
        ps = state.political_support

        # 获取各群组的政治权重（归一化）
        hh = state.households
        groups = hh.all_groups()
        raw_powers = np.array([g.political_power for g in groups])
        pops = np.array([max(g.population, 0.01) for g in groups])
        # 政治影响力 = 政治权重 × 人口规模（人数越多影响力越大）
        weighted_power = raw_powers * pops
        total_power = weighted_power.sum()
        if total_power > 0:
            group_weights = weighted_power / total_power
        else:
            group_weights = np.ones(5) / 5
        # 存储用于后续计算
        self._group_political_weights = group_weights

        # (a) 就业满意度 — 基于总就业率和工业就业占比
        total_emp = sum(s.L for s in state.sectors.values())
        total_pop = total_emp + hh.unemployed.population
        emp_rate = total_emp / max(total_pop, 1.0)
        industrial_emp_share = state.sectors["i"].L / max(total_emp, 1.0)
        ps.employment_satisfaction = (
            0.4 * emp_rate + 0.4 * industrial_emp_share + 0.2 * (1 - state.unemployment_rate)
        )
        ps.employment_satisfaction = np.clip(ps.employment_satisfaction, 0.05, 1.0)

        # (b) v3.5: 工资满意度 — 6部门→5群组映射
        ref_wage = max(np.mean([state.sectors[k].W for k in ["d_infra", "d_app"]]), 0.01)
        # 数字群组(d_infra+d_app)、工业(i)、基础(b)、服务业(s_high+s_low)、失业
        d_L = state.sectors["d_infra"].L + state.sectors["d_app"].L
        w_d = ((state.sectors["d_infra"].W * state.sectors["d_infra"].L
               + state.sectors["d_app"].W * state.sectors["d_app"].L) / max(d_L, 0.01)
               if d_L > 0 else ref_wage)
        s_L = state.sectors["s_high"].L + state.sectors["s_low"].L
        w_s = ((state.sectors["s_high"].W * state.sectors["s_high"].L
               + state.sectors["s_low"].W * state.sectors["s_low"].L) / max(s_L, 0.01)
               if s_L > 0 else ref_wage * 0.3)
        wage_satisfactions = [
            min(1.0, w_d / ref_wage),                       # digital_high_skill
            min(1.0, state.sectors["i"].W / ref_wage),       # industrial_workers
            min(1.0, state.sectors["b"].W / ref_wage),       # base_workers
            min(1.0, w_s / ref_wage),                        # service_workers
        ]
        # 失业者
        hh_u = hh.unemployed
        if hh_u.population > 0 and hh_u.compensation_received > 0:
            avg_comp = hh_u.compensation_received / max(hh_u.population, 1.0)
            avg_wage_all = np.mean([state.sectors[k].W for k in ALL_SECTORS])
            u_sat = min(1.0, avg_comp / max(avg_wage_all * 0.3, 0.01))
        else:
            u_sat = 0.1
        wage_satisfactions.append(u_sat)

        # 按政治权重加权 (5群组)
        ps.wage_satisfaction = np.dot(group_weights, np.array(wage_satisfactions))
        ps.wage_satisfaction = np.clip(ps.wage_satisfaction, 0.05, 1.0)

        # (c) 不平等容忍度 — 基于代价基尼系数
        cost_gini = hh.cost_gini
        ps.inequality_tolerance = np.exp(-3.0 * cost_gini)
        ps.inequality_tolerance = np.clip(ps.inequality_tolerance, 0.05, 1.0)

        # (d) 福利满意度 — 基于总体福利指数
        ps.welfare_satisfaction = min(1.0, state.welfare_index / 2.0)
        ps.welfare_satisfaction = np.clip(ps.welfare_satisfaction, 0.05, 1.0)

    def _aggregate_support(self, state: SystemState):
        """加权聚合为综合支持度."""
        ps = state.political_support

        # 权重：就业最重要(0.35)，工资(0.25)，不平等(0.20)，福利(0.20)
        raw_support = (
            0.35 * ps.employment_satisfaction
            + 0.25 * ps.wage_satisfaction
            + 0.20 * ps.inequality_tolerance
            + 0.20 * ps.welfare_satisfaction
        )

        # 策略差异化调节
        regime = state.accumulation_regime
        if regime == "B":
            raw_support *= self.p.get("regime_adjust_B", 0.95)
        elif regime == "C":
            raw_support *= self.p.get("regime_adjust_C", 1.02)

        # 历史平滑（避免剧烈波动）
        if self._support_history:
            prev_support = self._support_history[-1]
            smooth = self.p.get("support_smoothing_factor", 0.85)
            ps.support_index = smooth * prev_support + (1 - smooth) * raw_support
        else:
            ps.support_index = raw_support

        ps.support_index = np.clip(ps.support_index, 0.0, 1.0)

    def _compute_protest_risk(self, state: SystemState):
        """计算抗议风险.

        高风险条件：
        - 支持度快速下降（趋势恶化）
        - 代价集中在特定群体（工业劳动者）
        - 失业率突然上升
        """
        ps = state.political_support

        # 趋势检测
        if len(self._support_history) >= 10:
            recent_trend = (
                self._support_history[-1] - self._support_history[-10]
            ) / 10.0
        else:
            recent_trend = 0.0

        # 代价集中度
        cost_ratio = state.households.cost_ratio_industrial_vs_digital
        concentration_risk = min(1.0, (cost_ratio - 1.0) / 3.0) if cost_ratio > 1.0 else 0.0

        # 失业风险
        u_risk = min(1.0, state.unemployment_rate / 0.25)

        # 综合抗议风险
        ps.protest_risk = (
            0.35 * max(0, -recent_trend * 5.0)  # 支持度下降速度
            + 0.30 * concentration_risk
            + 0.25 * u_risk
            + 0.10 * (1.0 - ps.support_index)   # 绝对支持度水平
        )
        ps.protest_risk = np.clip(ps.protest_risk, 0.0, 1.0)

    def _compute_policy_constraint(self, state: SystemState, dt: float):
        """计算政策约束乘数 (v2.2 增强).

        支持度高 (>0.6): 政策空间完整，constraint ≈ 1.0
        支持度中 (0.3-0.6): 提取率受限，投资需偏向福利
        支持度低 (<0.3): 严重受限，核心策略引擎可能瓦解

        带惯性效应：政策约束不会瞬时跳变。
        v2.2: 低支持度时约束的杀伤力显著增强 — 这是B可能崩溃的核心路径。
        """
        ps = state.political_support

        # 目标约束乘数 (v2.2: 低支持度时惩罚更严厉)
        if ps.support_index >= 0.6:
            target_constraint = 1.0
        elif ps.support_index >= 0.45:
            # 线性下降
            target_constraint = 0.75 + 0.25 * (ps.support_index - 0.45) / 0.15
        elif ps.support_index >= 0.30:
            # 加速下降
            target_constraint = 0.40 + 0.35 * (ps.support_index - 0.30) / 0.15
        elif ps.support_index >= 0.20:
            # 严重约束 — 核心策略可能瓦解
            target_constraint = 0.18 + 0.22 * (ps.support_index - 0.20) / 0.10
        else:
            # 几乎完全丧失政策空间 — 任何策略都难以维持
            target_constraint = 0.08 + 0.10 * ps.support_index / 0.20
            target_constraint = max(0.05, target_constraint)

        # 抗议风险的额外惩罚 (v2.2: 加强)
        if ps.protest_risk > 0.5:
            target_constraint *= 0.85
        if ps.protest_risk > 0.7:
            target_constraint *= 0.75
        if ps.protest_risk > 0.85:
            target_constraint *= 0.60  # 严重抗议 → 政策几乎瘫痪

        # 惯性平滑（约束松开较慢，收紧较快）(v2.2: 收紧更快)
        inertia_up = 0.93    # 约束放松的惯性（更难松绑）
        inertia_down = 0.70  # 约束收紧的惯性（加速收紧）

        if target_constraint >= ps.policy_constraint:
            ps.policy_constraint = (
                inertia_up * ps.policy_constraint
                + (1 - inertia_up) * target_constraint
            )
        else:
            ps.policy_constraint = (
                inertia_down * ps.policy_constraint
                + (1 - inertia_down) * target_constraint
            )

        # v2.2: 当约束<0.25时，策略的核心机制开始瓦解
        # 这模拟了"政策被迫放弃原策略"的现实
        if ps.policy_constraint < 0.25:
            regime = state.accumulation_regime
            if regime == "B":
                # B的核心引擎是高提取 → 数字投资
                # 约束太紧 → 无法维持DL_d → 反哺链条断裂
                # 数字部门失去投资 → 技术领先性退化
                dl_decay = (0.25 - ps.policy_constraint) * 0.05 * dt
                # Apply DL decay to both digital sectors (using their average)
                avg_dl = (state.sectors["d_infra"].DL + state.sectors["d_app"].DL) / 2.0
                new_avg_dl = max(0.05, avg_dl - dl_decay)
                decay_factor = new_avg_dl / max(avg_dl, 0.001)
                state.sectors["d_infra"].DL = max(0.05, state.sectors["d_infra"].DL * decay_factor)
                state.sectors["d_app"].DL = max(0.05, state.sectors["d_app"].DL * decay_factor)
            elif regime == "C":
                # C的核心依赖外部资源
                # 政治不稳定 → 外资加速撤离
                drain = (0.25 - ps.policy_constraint) * 0.08 * dt
                # Capital drain from both digital sectors
                drain_factor = max(0.0, 1.0 - drain)
                state.sectors["d_infra"].K = max(0.1, state.sectors["d_infra"].K * drain_factor)
                state.sectors["d_app"].K = max(0.1, state.sectors["d_app"].K * drain_factor)

        # 记录支持度历史
        self._support_history.append(ps.support_index)
        if len(self._support_history) > 100:
            self._support_history = self._support_history[-100:]

    def _compute_reform_momentum(self, state: SystemState):
        """计算改革动能 — 支持度回升的趋势强度.

        正的改革动能意味着社会矛盾在缓解，政策空间在扩大。
        这可能触发"政策窗口"——重新获得实施大胆改革的机会。
        """
        ps = state.political_support

        if len(self._support_history) >= 20:
            half = len(self._support_history) // 2
            earlier = np.mean(self._support_history[:half])
            later = np.mean(self._support_history[half:])
            momentum = (later - earlier) * 3.0  # 放大以便使用
        else:
            momentum = 0.0

        ps.reform_momentum = np.clip(momentum, -1.0, 1.0)

    def get_effective_extraction_ceiling(self, state: SystemState,
                                         nominal_ceiling: float) -> float:
        """获取受政治约束后的有效提取率上限."""
        ps = state.political_support
        return nominal_ceiling * ps.policy_constraint

    def get_effective_investment_bias(self, state: SystemState) -> float:
        """获取受政治约束的投资偏向调整.

        返回: welfare_bias ∈ [0, 1]
        - 0 = 完全按策略原始逻辑分配
        - 1 = 最大程度偏向福利/就业保障
        """
        ps = state.political_support
        if ps.support_index > 0.6:
            return 0.0
        elif ps.support_index > 0.3:
            return (0.6 - ps.support_index) / 0.3 * 0.5
        else:
            return 0.5 + (0.3 - ps.support_index) / 0.3 * 0.5

    def check_policy_regime_change(self, state: SystemState) -> str:
        """检查是否应该发生策略转变.

        返回:
            "none" — 维持现状
            "soften" — 需要软化当前策略（降低提取率、增加福利投资）
            "emergency" — 需要紧急政策转向
            "collapse" — 原策略完全无法维持，系统进入求生模式
        """
        ps = state.political_support

        # v2.2: 增加"collapse"级别 — 原策略核心引擎已死
        if ps.support_index < 0.15 and ps.protest_risk > 0.80:
            return "collapse"
        elif ps.support_index < 0.20 and ps.protest_risk > 0.65:
            return "emergency"
        elif ps.support_index < 0.25:
            return "soften"
        elif ps.protest_risk > 0.70 and ps.support_index < 0.35:
            return "soften"

        return "none"

    def summary(self, state: SystemState) -> dict:
        """返回政治经济学状态摘要."""
        ps = state.political_support
        return {
            "support_index": ps.support_index,
            "protest_risk": ps.protest_risk,
            "policy_constraint": ps.policy_constraint,
            "reform_momentum": ps.reform_momentum,
            "regime_change_needed": self.check_policy_regime_change(state),
            "employment_satisfaction": ps.employment_satisfaction,
            "wage_satisfaction": ps.wage_satisfaction,
            "inequality_tolerance": ps.inequality_tolerance,
        }

    # ─────────────────────────────────────────────
    # v3.6: 政策执行折扣 —— 官僚体系内部摩擦
    # ─────────────────────────────────────────────
    def compute_execution_discount(self, state: SystemState,
                                   strategy_complexity: float = 0.0) -> float:
        """计算官僚体系内部的策略执行折扣.

        即使是"最优策略"，在真实官僚体系中也会经历衰减：
        - 多部门协调成本（"九龙治水"）
        - 地方与中央的利益不一致
        - 信息传递失真
        - 制度能力的结构性约束

        Args:
            state: 系统状态
            strategy_complexity: 策略复杂度 [0,1]（0=纯策略, 1=高度混合）

        Returns:
            执行折扣 ∈ [0.5, 1.0]（1.0=完全执行, 0.5=仅执行50%）
        """
        inst_cap = self.p.get("institution_capacity", 0.6)
        base_efficiency = self.p.get("policy_execution_base_efficiency", 0.75)
        complexity_penalty = self.p.get("bureaucracy_complexity_penalty", 0.15)

        # 基础执行效率 = f(制度能力)
        execution = 0.5 + 0.5 * inst_cap  # [0.65, 1.0]

        # 策略复杂度惩罚
        # 混合策略需要更多部门协调 → 执行效率下降
        execution *= (1.0 - strategy_complexity * complexity_penalty)

        # 抗议风险额外折扣：社会不稳定时官僚体系更难运作
        ps = state.political_support
        if ps.protest_risk > 0.5:
            execution *= max(0.7, 1.0 - (ps.protest_risk - 0.5) * 0.4)

        # 政策约束的累积效应
        if ps.policy_constraint < 0.5:
            execution *= (0.7 + 0.3 * ps.policy_constraint / 0.5)

        execution = max(0.50, min(1.0, execution))

        # 记录到状态
        state.policy_execution_discount = execution

        return execution
