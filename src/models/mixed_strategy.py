"""v3.0 混合策略框架 —— λ向量驱动的策略混合.

核心思想：
- 现实世界中没有国家走纯粹的某一条路径
- 每种策略逻辑按 λ 权重混合，产生"主次分明"的策略组合
- 约束：主导策略≥50%，次要策略≤30%，第三策略≤20%

使用方式：
    mixed = MixedStrategy(params, lambda_vec=[0.7, 0.2, 0.1])
    tau = mixed.compute_extraction_rate(state)      # 混合后的提取率
    phi = mixed.compute_investment_allocation(state)  # 混合后的投资分配
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
from ..core.state import SystemState, ALL_SECTORS, DIGITAL_SECTORS, SERVICE_SECTORS
from .accumulation import (
    GradualAccumulation,
    LeapAccumulation,
    SubstitutionAccumulation,
)


# ─────────────────────────────────────────────
# 标准混合配比定义
# ─────────────────────────────────────────────
# 格式: (名称, (λ_A, λ_B, λ_C))
# 约束规则：主导策略≥50%，次要策略≤30%，第三策略≤20%
# 注：以下配比均满足约束；若需自定义违规配比，会被自动修正并发出警告
STANDARD_MIXTURES: Dict[str, Tuple[float, float, float]] = {
    # === 纯策略对照组 ===
    "pure_A":    (1.0, 0.0, 0.0),
    "pure_B":    (0.0, 1.0, 0.0),
    "pure_C":    (0.0, 0.0, 1.0),

    # === 主导混合（A为主） ===
    "A_dominant_B":   (0.7, 0.2, 0.1),   # A为主，B为辅，C微量
    "A_dominant_C":   (0.7, 0.1, 0.2),   # A为主，C为辅，B微量

    # === 主导混合（B为主） ===
    "B_dominant_A":   (0.2, 0.7, 0.1),   # B为主，A为辅，C微量
    "B_dominant_C":   (0.1, 0.7, 0.2),   # B为主，C为辅，A微量

    # === 主导混合（C为主） ===
    "C_dominant_A":   (0.2, 0.1, 0.7),   # C为主，A为辅，B微量
    "C_dominant_B":   (0.1, 0.2, 0.7),   # C为主，B为辅，A微量

    # === 双元混合（v3.2修正：满足约束） ===
    # 注：原定义(0.5,0.5,0.0)违反"次要≤30%"约束，已修正为主导70%
    "A_B_balanced":   (0.70, 0.30, 0.0),  # A主导+B辅助
    "A_C_balanced":   (0.70, 0.0, 0.30),  # A主导+C辅助
    "B_C_balanced":   (0.0, 0.70, 0.30),  # B主导+C辅助

    # === 三元均匀混合 ===
    # 注：均匀(1/3,1/3,1/3)违反"主导≥50%"约束，已修正为均衡配比
    "uniform":        (0.50, 0.30, 0.20),  # 三元均衡（满足约束下的最均匀配比）
}


@dataclass
class MixedStrategyConfig:
    """混合策略配置."""

    lambda_vec: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    # 约束参数
    enforce_constraints: bool = True
    min_dominant: float = 0.50       # 主导策略最低权重
    max_secondary: float = 0.30      # 次要策略最高权重
    max_tertiary: float = 0.20       # 第三策略最高权重

    @property
    def dominant_strategy(self) -> str:
        """返回主导策略."""
        idx = np.argmax(self.lambda_vec)
        return ["A", "B", "C"][idx]

    @property
    def is_pure(self) -> bool:
        """是否为纯策略."""
        return any(abs(l - 1.0) < 1e-6 for l in self.lambda_vec)

    def get_description(self) -> str:
        """返回人类可读的策略描述."""
        la, lb, lc = self.lambda_vec
        if self.is_pure:
            dom = self.dominant_strategy
            return f"纯策略{dom}"
        parts = []
        if la > 0.01:
            parts.append(f"A:{la:.0%}")
        if lb > 0.01:
            parts.append(f"B:{lb:.0%}")
        if lc > 0.01:
            parts.append(f"C:{lc:.0%}")
        return f"混合({'+'.join(parts)}, 主导={self.dominant_strategy})"


class MixedStrategy:
    """混合策略 —— 驱动三种纯策略逻辑按 λ 权重混合.

    不是在"选策略"（which），而是在"混合策略"（how much of each）。
    """

    def __init__(self, params: dict,
                 lambda_vec: Tuple[float, float, float] = None,
                 mixture_name: str = None,
                 config: MixedStrategyConfig = None):
        """
        Args:
            params: 全局参数字典
            lambda_vec: 策略配比 (λ_A, λ_B, λ_C), 默认纯A
            mixture_name: 使用预设混合配比的名称 (覆盖 lambda_vec)
            config: 混合策略配置
        """
        self.p = params

        # 从预设名称加载
        if mixture_name and mixture_name in STANDARD_MIXTURES:
            lambda_vec = STANDARD_MIXTURES[mixture_name]

        if lambda_vec is None:
            lambda_vec = (1.0, 0.0, 0.0)

        self.config = config or MixedStrategyConfig(lambda_vec=lambda_vec)
        self.config.lambda_vec = lambda_vec

        # 约束检查
        if self.config.enforce_constraints:
            self._validate_and_correct()

        # 实例化三种纯策略逻辑（需要时惰性创建）
        self._strategy_A: Optional[GradualAccumulation] = None
        self._strategy_B: Optional[LeapAccumulation] = None
        self._strategy_C: Optional[SubstitutionAccumulation] = None

        # v3.6: 策略不兼容矩阵 — 当混合存在内在冲突的策略时施加惩罚
        # 每个策略对包含三个维度：资源逻辑冲突、制度逻辑冲突、外部逻辑冲突
        self._incompatibility_matrix = {
            "AB": {
                "total":       self.p.get("M_AB", 0.25),
                "resource":    self.p.get("M_AB_resource", 0.35),
                "institution": self.p.get("M_AB_institution", 0.25),
                "external":    self.p.get("M_AB_external", 0.10),
            },
            "BC": {
                "total":       self.p.get("M_BC", 0.25),
                "resource":    self.p.get("M_BC_resource", 0.10),
                "institution": self.p.get("M_BC_institution", 0.30),
                "external":    self.p.get("M_BC_external", 0.35),
            },
            "AC": {
                "total":       self.p.get("M_AC", 0.15),
                "resource":    self.p.get("M_AC_resource", 0.15),
                "institution": self.p.get("M_AC_institution", 0.20),
                "external":    self.p.get("M_AC_external", 0.15),
            },
        }

    @property
    def lambda_vec(self) -> Tuple[float, float, float]:
        return self.config.lambda_vec

    @property
    def dominant(self) -> str:
        return self.config.dominant_strategy

    # ─────────────────────────────────────────────
    # 约束检查
    # ─────────────────────────────────────────────
    def _validate_and_correct(self):
        """验证并修正混合配比，确保满足约束.

        如果配比被修正，会发出 RuntimeWarning 提示用户。
        """
        la, lb, lc = self.config.lambda_vec
        original = (la, lb, lc)
        corrected = False

        # 归一化
        total = la + lb + lc
        if abs(total - 1.0) > 1e-6:
            la, lb, lc = la / total, lb / total, lc / total
            corrected = True

        # 检查并修正约束
        if self.config.enforce_constraints:
            # 1. 主导策略必须 ≥ 50%
            values = [la, lb, lc]
            dom_idx = np.argmax(values)
            if values[dom_idx] < self.config.min_dominant:
                corrected = True
                # 提升主导策略权重
                deficit = self.config.min_dominant - values[dom_idx]
                values[dom_idx] = self.config.min_dominant
                # 从其他策略按比例削减
                other_indices = [i for i in range(3) if i != dom_idx]
                other_total = sum(values[i] for i in other_indices)
                if other_total > 0:
                    for i in other_indices:
                        values[i] -= deficit * (values[i] / other_total)
                values = [max(0.0, v) for v in values]

            # 2. 第三策略 ≤ 20%
            nondom_indices = sorted(
                [i for i in range(3) if i != dom_idx],
                key=lambda i: -values[i]
            )
            if len(nondom_indices) >= 2:
                sec_idx, ter_idx = nondom_indices[0], nondom_indices[1]
                if values[ter_idx] > self.config.max_tertiary:
                    corrected = True
                    excess = values[ter_idx] - self.config.max_tertiary
                    values[ter_idx] = self.config.max_tertiary
                    values[sec_idx] += excess

            # 3. 次要策略 ≤ 30%
            if len(nondom_indices) >= 2:
                sec_idx = nondom_indices[0]
                if values[sec_idx] > self.config.max_secondary:
                    corrected = True
                    excess = values[sec_idx] - self.config.max_secondary
                    values[sec_idx] = self.config.max_secondary
                    values[dom_idx] += excess

            # 重新归一化
            total = sum(values)
            la, lb, lc = values[0] / total, values[1] / total, values[2] / total

        self.config.lambda_vec = (la, lb, lc)

        # v3.2: 如果配比被修正，发出警告
        if corrected:
            warnings.warn(
                f"混合策略配比从 {original} 被自动修正为 {self.config.lambda_vec}。"
                f"约束：主导≥{self.config.min_dominant:.0%}、"
                f"次要≤{self.config.max_secondary:.0%}、"
                f"第三≤{self.config.max_tertiary:.0%}。"
                f"请使用满足约束的配比以消除此警告。",
                RuntimeWarning, stacklevel=2
            )

    # ─────────────────────────────────────────────
    # 惰性策略实例化
    # ─────────────────────────────────────────────
    def _get_A(self) -> GradualAccumulation:
        if self._strategy_A is None:
            self._strategy_A = GradualAccumulation(self.p)
        return self._strategy_A

    def _get_B(self) -> LeapAccumulation:
        if self._strategy_B is None:
            self._strategy_B = LeapAccumulation(self.p)
        return self._strategy_B

    def _get_C(self) -> SubstitutionAccumulation:
        if self._strategy_C is None:
            self._strategy_C = SubstitutionAccumulation(self.p)
        return self._strategy_C

    # ─────────────────────────────────────────────
    # 核心混合逻辑
    # ─────────────────────────────────────────────
    def _compute_incompatibility_penalty(self, state: SystemState) -> float:
        """计算策略混合的不兼容惩罚.

        penalty = λ_A·λ_B·M_AB + λ_B·λ_C·M_BC + λ_A·λ_C·M_AC

        调节因子:
        - 制度能力越高，不兼容性越低（强制度可以调和策略冲突）
        - 技术差距 > 5 时，外部压力放大不兼容性

        Returns:
            不兼容惩罚值，范围 [0, 1]
        """
        la, lb, lc = self.lambda_vec
        M = self._incompatibility_matrix

        # 基础不兼容：各策略对的权重乘积 × 不兼容系数
        penalty = (
            la * lb * M["AB"]["total"]
            + lb * lc * M["BC"]["total"]
            + la * lc * M["AC"]["total"]
        )

        # 制度能力调节：更高制度能力 → 更低的实际不兼容成本
        # 调节范围 [0.4, 1.0]：强制度可将不兼容降低60%
        inst_cap = self.p.get("institution_capacity", 0.6)
        institution_moderator = max(0.4, 1.0 - inst_cap * 0.8)
        penalty *= institution_moderator

        # 外部压力放大：tech_gap > 5 时放大不兼容性
        # 技术差距越大，策略冲突的代价越高
        if state.tech_gap > 5.0:
            amplification = 1.0 + (state.tech_gap - 5.0) * 0.08
            penalty *= min(2.0, amplification)

        return min(1.0, max(0.0, penalty))

    def compute_extraction_rate(self, state: SystemState) -> float:
        """混合提取率 = λ_A·τ_A + λ_B·τ_B + λ_C·τ_C.

        v3.6: 策略不兼容惩罚降低有效提取率。
        """
        la, lb, lc = self.lambda_vec
        tau = 0.0
        if la > 0:
            tau += la * self._get_A().extraction_rate(state)
        if lb > 0:
            tau += lb * self._get_B().extraction_rate(state)
        if lc > 0:
            tau += lc * self._get_C().extraction_rate(state)

        # v3.6: 策略冲突降低有效提取效率（最多降低30%）
        penalty = self._compute_incompatibility_penalty(state)
        tau *= (1.0 - penalty * 0.3)

        return tau

    def compute_investment_allocation(self, state: SystemState,
                                      tau: float) -> dict:
        """混合投资分配 — 按 λ 加权各纯策略的分配逻辑，然后归一化.

        v3.6: 使用六部门结构 (ALL_SECTORS)，策略不兼容导致额外投资浪费。
        """
        la, lb, lc = self.lambda_vec

        # v3.6: 六部门结构
        phi = {k: 0.0 for k in ALL_SECTORS}

        if la > 0:
            phi_A = self._get_A().investment_allocation(state, tau)
            for k in ALL_SECTORS:
                phi[k] += la * phi_A.get(k, 0.0)

        if lb > 0:
            phi_B = self._get_B().investment_allocation(state, tau)
            for k in ALL_SECTORS:
                phi[k] += lb * phi_B.get(k, 0.0)

        if lc > 0:
            # 策略C的investment_allocation需要dt参数
            try:
                phi_C = self._get_C().investment_allocation(state, tau, dt=0.25)
            except TypeError:
                phi_C = self._get_C().investment_allocation(state, tau)
            for k in ALL_SECTORS:
                phi[k] += lc * phi_C.get(k, 0.0)

        # v3.6: 策略不兼容惩罚 — 部分投资因策略冲突而浪费（最多浪费40%）
        penalty = self._compute_incompatibility_penalty(state)
        waste_factor = 1.0 - penalty * 0.4
        for k in ALL_SECTORS:
            phi[k] *= waste_factor

        # 归一化
        total = sum(phi.values())
        if total > 0:
            return {k: v / total for k, v in phi.items()}
        # 六部门默认均分
        n = len(ALL_SECTORS)
        return {k: 1.0 / n for k in ALL_SECTORS}

    def compute_surplus(self, state: SystemState):
        """计算总剩余（委托给基类逻辑 — 使用 mixed regime 的 AccumulationRegime）."""
        # 混合策略使用与纯策略相同的剩余计算
        # 直接使用策略A的基类方法（所有策略的_compute_surplus是相同的）
        self._get_A()._compute_surplus(state)

    def apply_investment(self, state: SystemState, phi: dict, tau: float, dt: float):
        """应用投资（委托给基类）."""
        self._get_A()._apply_investment(state, phi, tau, dt)

    def step(self, state: SystemState, dt: float):
        """混合策略的完整步进逻辑.

        这是混合策略的"主入口"——模仿各纯策略的step方法，
        但在关键计算点使用混合逻辑。
        """
        # 1. 计算总剩余
        self._get_A()._compute_surplus(state)

        # 2. 混合提取率
        tau = self.compute_extraction_rate(state)

        # 3. 混合投资分配
        phi = self.compute_investment_allocation(state, tau)

        # 4. 应用投资
        self._get_A()._apply_investment(state, phi, tau, dt)

        # 5. v3.0: 混合内生成本 — 按各策略的 λ 比例应用其内生成本
        self._apply_mixed_endogenous_costs(state, tau, dt)

    def _apply_mixed_endogenous_costs(self, state: SystemState,
                                       tau: float, dt: float):
        """按 λ 权重混合各策略的内生社会经济成本.

        每种纯策略都有其特有的成本模型。混合策略继承了各纯策略成本
        的加权平均——不会因为混合比例小就完全消除某种成本，
        但可以显著稀释它。
        """
        la, lb, lc = self.lambda_vec

        # v3.0: 策略A内生成本
        if la > 0:
            self._apply_A_costs(state, dt, weight=la)

        # B的内生成本（v2.2已有）
        if lb > 0:
            self._apply_B_costs(state, tau, dt, weight=lb)

        # C的内生成本通过外部冲击机制运作（v2.2已有）
        if lc > 0:
            self._apply_C_costs(state, dt, weight=lc)

        # v3.6: 策略对之间的互斥不兼容成本
        self._apply_mutual_incompatibility_costs(state, dt)

    def _apply_mutual_incompatibility_costs(self, state: SystemState,
                                            dt: float):
        """v3.6: 应用策略对之间的互斥不兼容成本.

        当两个策略都具有显著权重（>0.15）时，其内在逻辑冲突
        会产生额外的结构性成本，这种成本无法通过 λ 加权简单稀释。
        """
        la, lb, lc = self.lambda_vec

        # ── A+B 冲突："保护 vs 提取" — 资源逻辑冲突 ──
        # 保护性工业政策需要稳定投入，但高提取率抽走资源，
        # 导致"既要保护又要榨取"的矛盾 → 工业投资效率下降
        if la > 0.15 and lb > 0.15:
            conflict_intensity = min(la, lb)
            efficiency_drop = conflict_intensity * 0.06 * dt
            state.sectors["i"].Y *= max(0.88, 1.0 - efficiency_drop)
            # 制度摩擦额外消耗积累基金
            state.accumulation_fund *= max(
                0.85, 1.0 - conflict_intensity * 0.08 * dt
            )

        # ── B+C 冲突："主权 vs FDI" — 制度逻辑冲突 ──
        # B追求自主可控，C依赖FDI/外部资源，
        # 同时推行 → 政策信号混乱 → 制度有效性下降
        if lb > 0.15 and lc > 0.15:
            conflict_intensity = min(lb, lc)
            inst_drag = conflict_intensity * 0.05 * dt
            # 外部依赖加剧，但同时试图保持自主 → 两头落空
            if hasattr(state, 'external_dependency'):
                state.external_dependency = min(
                    0.95,
                    state.external_dependency + conflict_intensity * 0.03 * dt
                )
            # 数字基础设施投资效率降低（FDI期待 vs 自主标准冲突）
            for d_key in DIGITAL_SECTORS:
                if d_key in state.sectors:
                    state.sectors[d_key].I *= max(
                        0.82, 1.0 - inst_drag * 1.5
                    )

        # ── A+C 冲突："渐进主义 vs 快速外部替代" — 政策不一致 ──
        # A强调渐进积累，C依赖外部快速替代，
        # 两者并存 → 政策摇摆不定 → 两个方向都受拖累
        if la > 0.15 and lc > 0.15:
            conflict_intensity = min(la, lc)
            policy_incoherence = conflict_intensity * 0.04 * dt
            # 技术追赶效率受损：方向不明确
            state.tech_gap *= (1.0 + policy_incoherence * 0.5)
            # 服务业受政策不确定影响
            for s_key in SERVICE_SECTORS:
                if s_key in state.sectors:
                    state.sectors[s_key].Y *= max(
                        0.90, 1.0 - policy_incoherence * 0.8
                    )

    # ─────────────────────────────────────────────
    # 策略A内生成本 (v3.0 新增)
    # ─────────────────────────────────────────────
    def _apply_A_costs(self, state: SystemState, dt: float, weight: float = 1.0):
        """策略A的内生成本：保护僵化、时间税、既得利益刚性、中间收入陷阱."""
        if not hasattr(self, '_A_cost_state'):
            self._A_cost_state = {
                "protection_duration": 0.0,
                "cumulative_lag": 0.0,
                "vested_interest_intensity": 0.0,
                "middle_income_trap_years": 0.0,
            }
        cs = self._A_cost_state

        # 1. 保护僵化成本
        safety_margin = self.p.get("A_safety_margin", 0.7)
        K_i_ratio = state.sectors["i"].K / max(state.initial_K_i, 1.0)
        # 保护过剩度：实际资本存量远超有效资本需求
        protection_excess = max(0.0, K_i_ratio / max(safety_margin, 0.01) - 1.0)
        efficiency_loss = protection_excess * 0.08 * weight
        state.sectors["i"].Y *= max(0.85, 1.0 - efficiency_loss * dt)

        # 2. 渐进主义时间税 — 走得太慢，世界在前进
        cs["protection_duration"] += dt
        g_world = self.p.get("g_world", 0.03)
        # v3.6: 六部门结构 — 使用 d_app（应用层）的技术水平作为数字部门追赶指标
        prev_A_d = getattr(self, '_prev_A_d', state.sectors["d_app"].A)
        g_d = (state.sectors["d_app"].A / max(prev_A_d, 0.01) - 1.0)
        # A逻辑下数字部门增长慢 → 追赶赤字累积
        lag_rate = max(0.0, g_world - g_d - 0.01)  # 0.01是自然追赶的缓冲
        cs["cumulative_lag"] += lag_rate * dt * weight
        # 累积滞后的后果：技术差距小幅额外扩大
        if cs["cumulative_lag"] > 0.1:  # 累积滞后显著时
            lag_penalty = cs["cumulative_lag"] * 0.03 * dt
            state.tech_gap *= (1.0 + lag_penalty)

        self._prev_A_d = state.sectors["d_app"].A

        # 3. 既得利益刚性
        # 保护持续时间越长 → 利益集团越强 → 改革阻力越大
        if cs["protection_duration"] > 15.0:  # 保护超过15年
            industry_emp_share = state.sectors["i"].L / max(state.households.total_population, 1.0)
            cs["vested_interest_intensity"] = min(
                0.5,
                (cs["protection_duration"] - 15.0) * 0.01 * industry_emp_share
            )
            # 改革阻力：计划中的数字投资被打了折扣
            reform_resistance = cs["vested_interest_intensity"] * weight
            # v3.6: 六部门结构 — 改革阻力影响所有数字子部门
            for d_key in DIGITAL_SECTORS:
                if d_key in state.sectors:
                    state.sectors[d_key].I *= max(0.5, 1.0 - reform_resistance * dt)

        # 4. 中间收入陷阱
        dd_i = state.sectors["i"].digital_density
        G = state.tech_gap
        if dd_i < 0.3 and G > 4.0:
            cs["middle_income_trap_years"] += dt
            if cs["middle_income_trap_years"] > 10.0:
                # 持续10年以上 → 结构性问题
                trap_penalty = 0.01 * (cs["middle_income_trap_years"] - 10.0) * weight
                state.sectors["i"].K *= max(0.90, 1.0 - trap_penalty * dt)
                state.tech_gap *= (1.0 + trap_penalty * 0.5 * dt)
        else:
            cs["middle_income_trap_years"] = max(0.0, cs["middle_income_trap_years"] - dt * 2)

    # ─────────────────────────────────────────────
    # 策略B内生成本（委托给现有实现）
    # ─────────────────────────────────────────────
    def _apply_B_costs(self, state: SystemState, tau: float, dt: float,
                        weight: float = 1.0):
        """策略B的内生成本，按混合权重缩放."""
        strategy_B = self._get_B()

        # 1. 就业破坏（按权重缩放）
        tau_baseline = 0.30
        if tau > tau_baseline:
            excess_tau = (tau - tau_baseline) * weight  # 缩放后的超额提取率
            extra_unemployment = (
                state.sectors["i"].L * excess_tau * 0.05 * dt
            )
            # v3.6: 六部门结构 — 数字部门吸收能力汇总 d_infra + d_app
            total_d_L = sum(state.sectors[k].L for k in DIGITAL_SECTORS
                           if k in state.sectors)
            avg_d_DL = (sum(state.sectors[k].DL for k in DIGITAL_SECTORS
                           if k in state.sectors)
                        / max(len(DIGITAL_SECTORS), 1))
            absorptive_d = total_d_L * avg_d_DL * 0.02 * dt
            net_unemployment = max(0, extra_unemployment - absorptive_d)
            state.households.unemployed.population += net_unemployment
            state.sectors["i"].L = max(0.5, state.sectors["i"].L - net_unemployment * 0.5)

        # 2. 消费萎缩
        disposable_share = max(0.3, 0.7 - tau * 0.5 * weight)
        consumption_drag = max(0.0, (0.70 - disposable_share) * 0.15)
        if consumption_drag > 0:
            state.sectors["i"].Y *= max(0.85, 1.0 - consumption_drag * dt)

        # 3. 泡沫压力（按权重积累）
        # v3.6: 六部门结构 — 使用 d_app 利润率（应用层回报最能反映泡沫风险）
        r_d = state.sectors["d_app"].r
        if r_d > 0.25:
            bubble_pressure = (r_d - 0.25) * 0.3 * dt * weight
            strategy_B._regime_state["bubble_pressure"] = (
                strategy_B._regime_state.get("bubble_pressure", 0.0) + bubble_pressure
            )

        # 4. 泡沫破裂检查
        strategy_B._check_bubble_burst(state, dt)

    # ─────────────────────────────────────────────
    # 策略C内生成本（委托给现有实现）
    # ─────────────────────────────────────────────
    def _apply_C_costs(self, state: SystemState, dt: float, weight: float = 1.0):
        """策略C的内生成本，按混合权重缩放."""
        strategy_C = self._get_C()

        # C的外部冲击检查（概率随依赖度调整，权重影响概率）
        ext_dep = state.external_dependency

        # FDI冲击 — 按权重降低概率（权重越小，越不可能受纯C的典型冲击）
        fdi_avail = self.p.get("fdi_availability", 1.0)
        if not strategy_C._regime_state.get("fdi_shock_active", False):
            base_prob = (ext_dep * 0.20 + (1 - fdi_avail) * 0.30) * 0.03 * weight
            if np.random.random() < base_prob:
                strategy_C._regime_state["fdi_shock_active"] = True
                strategy_C._regime_state["shock_timer"] = 0.0

        # 资源冲击 — 按权重降低概率
        if not strategy_C._regime_state.get("resource_shock_active", False):
            export_intensity = self.p.get("C_resource_export_intensity", 0.6)
            sigma_p = self.p.get("sigma_p", 0.15)
            base_prob = export_intensity * sigma_p * 0.10 * weight
            if np.random.random() < base_prob:
                strategy_C._regime_state["resource_shock_active"] = True
                strategy_C._regime_state["shock_timer"] = 0.0

        # 完美风暴检查
        if (strategy_C._regime_state.get("fdi_shock_active", False)
                and strategy_C._regime_state.get("resource_shock_active", False)):
            strategy_C._regime_state.setdefault("perfect_storm_count", 0)
            strategy_C._regime_state["perfect_storm_count"] += 1
            state.external_dependency = min(0.95, state.external_dependency + 0.03 * dt * weight)
            indigenous_rd = self.p.get("C_indigenous_rd_ratio", 0.03)
            rd_buffer = max(0.2, 1.0 - indigenous_rd * 10)
            extra_gap_growth = 0.06 * rd_buffer * dt * weight
            state.tech_gap *= (1.0 + extra_gap_growth)

    # ─────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────
    def get_cost_state(self) -> dict:
        """返回各策略的内生成本状态."""
        state = {
            "lambda": list(self.lambda_vec),
            "dominant": self.dominant,
            "is_pure": self.config.is_pure,
        }
        if hasattr(self, '_A_cost_state'):
            state["A_costs"] = self._A_cost_state.copy()
        if self._strategy_B is not None:
            state["B_costs"] = {
                "bubble_pressure": self._strategy_B._regime_state.get("bubble_pressure", 0.0),
                "feedback_failed": self._strategy_B._regime_state.get("feedback_failed_count", 0),
            }
        if self._strategy_C is not None:
            state["C_costs"] = {
                "fdi_shock": self._strategy_C._regime_state.get("fdi_shock_active", False),
                "resource_shock": self._strategy_C._regime_state.get("resource_shock_active", False),
                "perfect_storms": self._strategy_C._regime_state.get("perfect_storm_count", 0),
            }
        return state

    @staticmethod
    def list_standard_mixtures() -> Dict[str, Tuple[float, float, float]]:
        """列出所有标准混合配比."""
        return STANDARD_MIXTURES.copy()

    @staticmethod
    def is_valid_mixture(lambda_vec: Tuple[float, float, float],
                         enforce: bool = True) -> Tuple[bool, str]:
        """检查一个混合配比是否满足约束.

        Returns:
            (是否有效, 问题描述)
        """
        la, lb, lc = lambda_vec
        total = la + lb + lc

        if abs(total - 1.0) > 0.01:
            return False, f"权重之和={total:.3f}，应≈1.0"

        if not enforce:
            return True, "OK"

        dom = max(la, lb, lc)
        if dom < 0.50:
            return False, f"主导策略权重={dom:.2f}，应≥0.50"

        nondom = sorted([la, lb, lc], reverse=True)
        if nondom[1] > 0.30:
            return False, f"次要策略权重={nondom[1]:.2f}，应≤0.30"

        if nondom[2] > 0.20:
            return False, f"第三策略权重={nondom[2]:.2f}，应≤0.20"

        return True, "OK"
