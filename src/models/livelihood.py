"""v3.0 民生子模块 —— 健康·教育·社会保障·安全感.

从聚合福利指数拆分为四个可独立追踪的民生维度，
与五维评价体系的"民生福祉"维度对接。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from ..core.state import SystemState, ALL_SECTORS


@dataclass
class LivelihoodState:
    """民生状态 —— 四个子维度的快照."""

    # 健康
    health_index: float = 1.0         # [0,1]
    food_security: float = 1.0        # [0,1]
    medical_access: float = 0.5       # [0,1]
    pollution_level: float = 0.0      # 0=清洁, 1=严重

    # 教育
    education_index: float = 1.0      # [0,1]
    edu_spending_ratio: float = 0.03  # 教育支出/GDP
    edu_equity: float = 1.0           # 教育公平度
    skill_mismatch: float = 0.0       # 技能不匹配度

    # 社会保障
    social_security_index: float = 1.0  # [0,1]
    pension_adequacy: float = 0.5       # [0,1]
    housing_affordability: float = 1.0  # [0,1]
    unemployment_coverage: float = 0.5  # [0,1]

    # 安全感
    security_perception: float = 1.0    # [0,1]
    job_stability: float = 1.0          # [0,1]
    income_growth_expectation: float = 0.5  # [0,1]
    institutional_trust: float = 0.7    # [0,1]

    # 民生综合指数 (聚合到 welfare 模块使用的格式)
    composite: float = 1.0


class LivelihoodModule:
    """民生子模块 —— v3.0 核心升级.

    在 v2.2 的 WelfareModule 基础上深化，将聚合福利指数
    拆分为健康、教育、社会保障、安全感四个可独立追踪的维度。

    这些维度：
    1. 参与五维评价中的"民生福祉"维度计算
    2. 与人道主义红线检测联动
    3. 通过消费、人力资本、政治支持等渠道反馈到经济系统
    """

    def __init__(self, params: dict):
        self.p = params
        self._state = LivelihoodState()
        # 追踪器
        self._health_tracker: Dict[str, float] = {}
        self._edu_tracker: Dict[str, float] = {}
        self._security_tracker: Dict[str, float] = {}
        # v3.4: 教育投资队列管道 — 每笔投资在15-20年后才影响人力资本
        self._education_cohorts: list = []       # [(投资年份, 投资金额), ...]
        self._education_stock: float = 0.0       # 已成熟的累积教育资本存量
        self._edu_maturation_years: float = 18.0 # 教育投资→人力资本的平均成熟期

    def step(self, state: SystemState, dt: float):
        """更新所有民生子维度."""
        self._update_health(state, dt)
        self._update_education(state, dt)
        self._update_social_security(state, dt)
        self._update_security_perception(state, dt)
        self._compute_composite(state)
        # v3.4: 应用教育队列延迟效应到人力资本
        self._apply_education_cohort_effect(state, dt)

    # ─────────────────────────────────────────────
    # 健康模块
    # ─────────────────────────────────────────────
    def _update_health(self, state: SystemState, dt: float):
        """更新健康指数.

        影响因素：
        - 粮食安全：基础部门产出/人口
        - 医疗可及性：公共医疗支出/GDP
        - 环境污染：与工业资本和环保投资相关
        - 失业 → 精神健康的间接效应
        """
        ls = self._state

        # 1. 粮食安全
        food_per_cap = state.sectors["b"].Y / max(state.households.total_population, 1.0)
        # 基本需求线（模型单位，约0.5）
        subsistence = 0.30
        ls.food_security = np.clip(food_per_cap / (subsistence * 2), 0.0, 1.0)

        # 2. 医疗可及性
        medical_spending = state.gdp * 0.03 * self.p.get("social_coverage", 0.5)
        medical_per_cap = medical_spending / max(state.households.total_population, 1.0)
        ls.medical_access = np.clip(medical_per_cap / 0.15, 0.0, 1.0)

        # 3. 环境污染（工业活动副产品）
        industrial_intensity = state.sectors["i"].Y / max(state.gdp, 0.01)
        green_invest = state.accumulation_fund * 0.02  # 假设2%用于环保
        base_pollution = industrial_intensity * 0.5
        # 数字化程度高 → 污染相对少（智能制造更清洁）
        dd_i = state.sectors["i"].digital_density
        pollution_raw = base_pollution * (1.0 - dd_i * 0.4) - green_invest * 0.1
        ls.pollution_level = np.clip(pollution_raw, 0.0, 1.0)

        # 4. 健康综合指数
        u_rate = state.unemployment_rate
        mental_health_penalty = u_rate * 0.3  # 失业对精神健康的负面影响

        ls.health_index = np.clip(
            0.35 * ls.food_security
            + 0.35 * ls.medical_access
            + 0.20 * (1.0 - ls.pollution_level)
            + 0.10 * (1.0 - mental_health_penalty),
            0.0, 1.0
        )

    # ─────────────────────────────────────────────
    # 教育模块
    # ─────────────────────────────────────────────
    def _update_education(self, state: SystemState, dt: float):
        """更新教育指数.

        刻画人力资本的积累过程、教育的公平性和有效性。
        """
        ls = self._state

        # 1. 教育支出
        # 基础值 + 培训投资 + 再分配中的教育成分
        training_invest = self.p.get("training_invest", 0.015)
        edu_spending = state.gdp * (0.03 + training_invest)
        # 社保覆盖率影响公共教育支出的实际效果
        edu_spending *= (0.5 + 0.5 * self.p.get("social_coverage", 0.5))
        ls.edu_spending_ratio = edu_spending / max(state.gdp, 0.01)

        # 2. 教育公平度（反向：群组间人力资本差距）
        groups = state.households.all_groups()
        skill_levels = []
        for h in groups:
            if h.population > 0:
                skill_levels.append(h.skill_level)
        if len(skill_levels) >= 2:
            skill_arr = np.array(skill_levels)
            # 群组间技能差距的标准差
            skill_std = np.std(skill_arr) / max(np.mean(skill_arr), 0.01)
            ls.edu_equity = np.clip(1.0 - skill_std * 0.5, 0.0, 1.0)
        else:
            ls.edu_equity = 1.0

        # 3. 技能不匹配度
        # 数字部门技能需求 vs 工业/基础劳动者现有技能
        skill_demand = 0.5 * (state.sectors["d_infra"].H + state.sectors["d_app"].H)
        skill_supply_industrial = state.households.industrial_workers.skill_level
        if skill_demand > 0 and skill_supply_industrial > 0:
            mismatch = max(0.0, skill_demand / max(skill_supply_industrial, 0.01) - 1.0)
            ls.skill_mismatch = np.clip(mismatch / 3.0, 0.0, 1.0)
        else:
            ls.skill_mismatch = 0.0

        # 4. 教育综合指数
        ls.education_index = np.clip(
            0.30 * np.clip(ls.edu_spending_ratio / 0.06, 0.0, 1.0)  # 支出充足度
            + 0.30 * ls.edu_equity
            + 0.25 * (1.0 - ls.skill_mismatch)
            + 0.15 * np.clip(state.sectors["i"].H / 2.0, 0.0, 1.0),  # 工业人力资本
            0.0, 1.0
        )

        # v3.4: 将当期教育投资推入队列管道（约每年一次）
        if not hasattr(self, '_last_cohort_push'):
            self._last_cohort_push = 0.0
        self._last_cohort_push += dt
        if self._last_cohort_push >= 1.0:  # 每年记录一次
            self._last_cohort_push -= 1.0
            edu_investment = state.gdp * ls.edu_spending_ratio
            self._education_cohorts.append((state.t, edu_investment))
            # 限制队列长度
            if len(self._education_cohorts) > 40:
                self._education_cohorts = self._education_cohorts[-30:]

    def _apply_education_cohort_effect(self, state: SystemState, dt: float):
        """v3.4: 教育队列延迟效应 — 15-20年前的教育投资现在才影响人力资本.

        队列管道模拟了"教育投资→人力资本形成"的真实延迟：
        - 今天增加教育预算 → 约18年后劳动力技能才开始提升
        - 已成熟的教育存量随时间缓慢释放，提升各部门H
        """
        # 1. 处理成熟的队列（超过成熟期的投资移入教育存量）
        matured_amount = 0.0
        still_immature = []
        for cohort_time, amount in self._education_cohorts:
            age = state.t - cohort_time
            if age >= self._edu_maturation_years:
                matured_amount += amount
            else:
                still_immature.append((cohort_time, amount))
        self._education_cohorts = still_immature

        if matured_amount > 0:
            # 成熟教育投资进入存量（考虑制度效率折扣）
            inst_cap = self.p.get("institution_capacity", 0.6)
            self._education_stock += matured_amount * inst_cap

        # 2. 教育存量缓慢释放，提升各部门人力资本
        if self._education_stock > 0:
            # 年化释放率：教育存量的5%/年转化为实际H提升
            release_rate = 0.05 * dt
            released = self._education_stock * release_rate
            self._education_stock = max(0.0, self._education_stock - released)

            # 分配到各部门，优先提升低H部门
            total_H_gap = 0.0
            H_gaps = {}
            for key in ALL_SECTORS:
                gap = max(0.0, 2.0 - state.sectors[key].H)
                H_gaps[key] = gap
                total_H_gap += gap

            if total_H_gap > 0:
                for key in ALL_SECTORS:
                    # H提升量与当前H缺口成正比（缺得多的部门受益多）
                    H_boost = released * (H_gaps[key] / total_H_gap) / max(
                        state.sectors[key].L, 0.01)
                    # H提升有上限（不能一蹴而就）
                    state.sectors[key].H = min(3.0,
                        state.sectors[key].H + H_boost * 0.1)

    # ─────────────────────────────────────────────
    # 社会保障模块
    # ─────────────────────────────────────────────
    def _update_social_security(self, state: SystemState, dt: float):
        """更新社会保障指数.

        包括养老金、失业保险、住房可负担性。
        """
        ls = self._state
        social_cov = self.p.get("social_coverage", 0.5)

        # 1. 失业保险覆盖率
        u_pop = state.households.unemployed.population
        total_comp = state.households.unemployed.compensation_received
        # 失业保险替代率 = 人均补偿 / 平均工资
        avg_wage = np.mean([state.sectors[k].W for k in ALL_SECTORS])
        if u_pop > 0 and avg_wage > 0:
            replacement_rate = (total_comp / u_pop) / max(avg_wage, 0.01)
            ls.unemployment_coverage = np.clip(replacement_rate / 0.40, 0.0, 1.0)
        else:
            ls.unemployment_coverage = social_cov

        # 2. 养老金充足度（从积累和再分配推断）
        pension_fund = state.accumulation_fund * 0.05 * social_cov
        elderly_ratio = 0.15  # 假设15%老龄人口
        elderly_pop = state.households.total_population * elderly_ratio
        if elderly_pop > 0:
            ls.pension_adequacy = np.clip(
                pension_fund / max(elderly_pop * avg_wage * 0.4, 0.01),
                0.0, 1.0
            )
        else:
            ls.pension_adequacy = social_cov

        # 3. 住房可负担性
        # 用工资/住房成本比近似
        housing_cost_ratio = 0.3  # 默认住房支出占收入30%
        disposable_income_ratio = (1.0 - self.p.get("tau_w", 0.2))
        ls.housing_affordability = np.clip(
            disposable_income_ratio / max(housing_cost_ratio, 0.01) * 0.5,
            0.0, 1.0
        )

        # 4. 社保综合指数
        ls.social_security_index = np.clip(
            0.35 * ls.unemployment_coverage
            + 0.30 * ls.pension_adequacy
            + 0.20 * ls.housing_affordability
            + 0.15 * social_cov,
            0.0, 1.0
        )

    # ─────────────────────────────────────────────
    # 安全感模块
    # ─────────────────────────────────────────────
    def _update_security_perception(self, state: SystemState, dt: float):
        """更新民生安全感.

        这是"软"指标但对消费行为和政治支持度极为重要：
        - 对未来悲观的预期 → 预防性储蓄增加 → 消费减少 → 总需求不足
        """
        ls = self._state

        # 1. 就业稳定性预期
        u_rate = state.unemployment_rate
        # 失业率的变动趋势
        if not hasattr(self, '_prev_u_rate'):
            self._prev_u_rate = u_rate
        u_trend = u_rate - self._prev_u_rate
        # 失业率低且稳定 → 高安全感
        ls.job_stability = np.clip(
            1.0 - u_rate * 2.0 - max(0, u_trend) * 3.0,
            0.0, 1.0
        )
        self._prev_u_rate = u_rate

        # 2. 收入增长预期
        # GDP增长 + 工资增长 - 通货膨胀预期
        gdp_growth = 0.02  # 默认值，可从时间序列推断
        wage_growth = np.mean([
            state.sectors[k].W / max(getattr(self, f'_prev_W_{k}', state.sectors[k].W), 0.01) - 1.0
            for k in ALL_SECTORS
        ])
        ls.income_growth_expectation = np.clip(
            0.5 + (gdp_growth * 5.0) + (wage_growth * 3.0),
            0.0, 1.0
        )
        for k in ALL_SECTORS:
            setattr(self, f'_prev_W_{k}', state.sectors[k].W)

        # 3. 制度信任度（与政治支持度相关）
        ps = state.political_support
        ls.institutional_trust = np.clip(
            0.4 * ps.support_index
            + 0.3 * (1.0 - ps.protest_risk)
            + 0.3 * ps.welfare_satisfaction,
            0.0, 1.0
        )

        # 4. 安全感综合指数
        ls.security_perception = np.clip(
            0.35 * ls.job_stability
            + 0.25 * ls.income_growth_expectation
            + 0.25 * ls.institutional_trust
            + 0.15 * (1.0 - ls.skill_mismatch * 0.5),
            0.0, 1.0
        )

        # 安全感 → 消费行为的反馈
        # 低安全感 → 预防性储蓄增加 → 消费减少
        precautionary_saving = max(0.0, (0.7 - ls.security_perception) * 0.15)
        if precautionary_saving > 0:
            for h in state.households.all_groups():
                if h.population > 0:
                    drag = precautionary_saving * 0.1  # 额外储蓄率
                    h.consumption *= max(0.85, 1.0 - drag * dt)

    # ─────────────────────────────────────────────
    # 综合指数
    # ─────────────────────────────────────────────
    def _compute_composite(self, state: SystemState):
        """计算民生综合指数 —— 聚合四个子维度.

        权重设置：
        - 健康: 0.30 — 生存基础
        - 教育: 0.25 — 发展基础
        - 社保: 0.25 — 安全网
        - 安全感: 0.20 — 心理福祉
        """
        ls = self._state
        ls.composite = np.clip(
            0.30 * ls.health_index
            + 0.25 * ls.education_index
            + 0.25 * ls.social_security_index
            + 0.20 * ls.security_perception,
            0.0, 1.0
        )

    # ─────────────────────────────────────────────
    # 公共接口
    # ─────────────────────────────────────────────
    def get_state(self) -> LivelihoodState:
        """获取当前民生状态."""
        return self._state

    def get_summary(self) -> dict:
        """获取民生状态摘要."""
        ls = self._state
        return {
            "health_index": ls.health_index,
            "education_index": ls.education_index,
            "social_security_index": ls.social_security_index,
            "security_perception": ls.security_perception,
            "composite": ls.composite,
            # 关键子指标
            "food_security": ls.food_security,
            "skill_mismatch": ls.skill_mismatch,
            "unemployment_coverage": ls.unemployment_coverage,
            "job_stability": ls.job_stability,
            "institutional_trust": ls.institutional_trust,
        }

    def check_red_line_indicators(self) -> dict:
        """检查与人道主义红线相关的民生指标."""
        ls = self._state
        warnings = {}

        if ls.food_security < 0.30:
            warnings["food_crisis"] = {
                "level": "WARNING",
                "value": ls.food_security,
                "threshold": 0.30,
            }

        if ls.health_index < 0.25:
            warnings["health_crisis"] = {
                "level": "WARNING",
                "value": ls.health_index,
                "threshold": 0.25,
            }

        if ls.edu_spending_ratio < 0.02 and ls.skill_mismatch > 0.5:
            warnings["education_deficit"] = {
                "level": "WARNING",
                "value": ls.edu_spending_ratio,
                "threshold": 0.02,
            }

        if ls.security_perception < 0.20:
            warnings["social_despair"] = {
                "level": "WARNING",
                "value": ls.security_perception,
                "threshold": 0.20,
            }

        return warnings
