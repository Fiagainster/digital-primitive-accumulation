"""v3.6 内生人口动态模块 —— 队列分量人口模型.

核心机制：
- 简化队列分量法（cohort-component method）：5年年龄组
- 生育率内生调节（经济发展 → 生育率下降）
- 死亡率内生调节（健康水平 → 预期寿命延长）
- 城镇化驱动的劳动力部门间再分配

理论基础：
- UN Population Division 队列分量法
- 人口转型理论（Demographic Transition）
- Lewis 二元经济模型（劳动力无限供给 → 刘易斯拐点）
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from ..core.state import SystemState, ALL_SECTORS


# 默认年龄结构（5岁一组：0-4, 5-9, ..., 75-79, 80+），共17组
# 来自典型发展中国家的年龄金字塔
DEFAULT_AGE_STRUCTURE = np.array([
    0.12, 0.11, 0.10, 0.09, 0.09,  # 0-24
    0.08, 0.07, 0.07, 0.06,         # 25-44
    0.05, 0.05, 0.04, 0.03,         # 45-64
    0.02, 0.01, 0.008, 0.002,       # 65+
])
DEFAULT_AGE_STRUCTURE = DEFAULT_AGE_STRUCTURE / DEFAULT_AGE_STRUCTURE.sum()


@dataclass
class DemographyState:
    """人口状态快照."""

    # 年龄结构（5岁组份额）
    age_structure: np.ndarray = field(default_factory=lambda: DEFAULT_AGE_STRUCTURE.copy())

    # 宏观指标
    total_population: float = 7000.0       # 总人口（万人）
    working_age_pop: float = 4500.0        # 劳动年龄人口（15-64岁）
    dependency_ratio: float = 0.55         # 抚养比
    old_age_dependency: float = 0.10       # 老年抚养比（65+/15-64）

    # 生育与死亡
    fertility_rate: float = 2.5            # 总和生育率（TFR）
    crude_birth_rate: float = 20.0         # 粗出生率（‰）
    life_expectancy: float = 65.0          # 预期寿命（岁）
    crude_death_rate: float = 8.0          # 粗死亡率（‰）
    infant_mortality: float = 30.0         # 婴儿死亡率（‰）

    # 城镇化
    urbanization_rate: float = 0.35        # 城镇化率 [0,1]
    rural_to_urban_flow: float = 0.0       # 年农村→城市净迁移率

    # 劳动力
    labor_force_participation: float = 0.60  # 劳动参与率
    potential_workers: float = 3500.0        # 潜在劳动力供给

    # 人口红利指标
    demographic_dividend: float = 0.0       # 人口红利强度（正值=红利期）
    working_age_growth_rate: float = 0.0    # 劳动年龄人口增长率


class DemographyModule:
    """内生人口动态模块.

    在每个模拟年度更新人口结构、劳动力供给和城镇化进程。
    使用简化的队列分量法，避免完整的多队列模型的计算成本。
    """

    # 年龄组定义（岁）
    AGE_GROUP_BOUNDS = [
        (0, 4), (5, 9), (10, 14), (15, 19), (20, 24),
        (25, 29), (30, 34), (35, 39), (40, 44),
        (45, 49), (50, 54), (55, 59), (60, 64),
        (65, 69), (70, 74), (75, 79), (80, 999),
    ]
    N_AGE_GROUPS = len(AGE_GROUP_BOUNDS)
    WORKING_AGE_START = 3   # 15-19 组索引
    WORKING_AGE_END = 12     # 60-64 组索引（不含）

    def __init__(self, params: dict):
        self.p = params
        self._state = DemographyState()

        # 从参数读取初始值
        self._state.fertility_rate = params.get("IC_fertility_rate", 2.5)
        self._state.life_expectancy = params.get("IC_life_expectancy", 65.0)
        self._state.urbanization_rate = params.get("IC_urbanization_rate", 0.35)
        self._state.total_population = params.get("IC_total_population", 7000.0)

        # 初始化年龄结构
        init_age = params.get("IC_age_structure", None)
        if init_age is not None:
            self._state.age_structure = np.array(init_age) / np.sum(init_age)

        # 追踪器
        self._year_accumulator: float = 0.0
        self._prev_gdp_pc: float = 0.0
        self._fertility_history: list = []

    def step(self, state: SystemState, dt: float):
        """人口动态更新（每年执行一次以降低计算成本）."""
        self._year_accumulator += dt
        if self._year_accumulator < 1.0:
            # 季度更新时仅更新城镇化（连续过程）
            self._update_urbanization_continuous(state, dt)
            self._sync_to_state(state)
            return
        self._year_accumulator -= 1.0

        # 年度更新
        # 1. 生育率内生调节
        self._adjust_fertility(state)

        # 2. 死亡率内生调节
        self._adjust_mortality(state)

        # 3. 队列分量推进（年龄结构老化一年）
        self._cohort_component_step(state)

        # 4. 城镇化推进
        self._update_urbanization(state)

        # 5. 劳动力供给更新
        self._update_labor_supply(state)

        # 6. 同步到全局状态
        self._sync_to_state(state)

    # ─────────────────────────────────────────────
    # 生育率内生调节
    # ─────────────────────────────────────────────
    def _adjust_fertility(self, state: SystemState):
        """生育率随经济发展和教育水平下降 —— 标准人口转型路径.

        影响因素：
        1. 人均GDP上升 → 生育率下降（机会成本效应）
        2. 女性教育水平上升 → 生育率下降
        3. 城镇化率上升 → 生育率下降（城市养育成本高）
        4. 社会保障覆盖 → 生育率下降（"养儿防老"需求减弱）
        """
        ds = self._state
        base_fertility = self.p.get("IC_fertility_rate", 2.5)

        # 1. 人均GDP效应（对数形式，边际递减）
        gdp_pc = state.gdp / max(state.total_population, 1.0)
        # GDP per capita 的对数效应：翻倍约降低0.3
        gdp_effect = 0.3 * np.log1p(max(0, gdp_pc - 0.5)) / np.log(2)

        # 2. 教育效应（女性教育 = 高技能服务业 + 数字部门女性H的代理）
        female_edu_proxy = 0.5 * (
            state.sectors["s_high"].H + state.sectors["d_app"].H
        )
        education_effect = 0.4 * (female_edu_proxy - 1.0)

        # 3. 城镇化效应
        urbanization_effect = 0.3 * ds.urbanization_rate

        # 4. 社保效应
        social_cov = self.p.get("social_coverage", 0.5)
        social_security_effect = 0.2 * social_cov

        # 合成
        effective_fertility = base_fertility - (
            gdp_effect + education_effect + urbanization_effect + social_security_effect
        )

        # 生育率有下限（约1.1，低于此值难以回升）和上限（生物极限）
        ds.fertility_rate = max(1.1, min(6.5, effective_fertility))

        # 粗出生率 ≈ 生育率 × 育龄妇女占比 × 系数
        # 简化：生育率/2.1（替代水平）× 基准出生率
        ds.crude_birth_rate = (ds.fertility_rate / 2.1) * 18.0

        self._fertility_history.append(ds.fertility_rate)
        if len(self._fertility_history) > 50:
            self._fertility_history = self._fertility_history[-50:]

    # ─────────────────────────────────────────────
    # 死亡率内生调节
    # ─────────────────────────────────────────────
    def _adjust_mortality(self, state: SystemState):
        """死亡率随健康水平和医疗条件下降.

        影响因素：
        1. 人均GDP上升 → 营养改善 → 死亡率下降
        2. 医疗可及性 → 死亡率下降
        3. 环境污染 → 死亡率上升
        """
        ds = self._state
        base_life_exp = self.p.get("IC_life_expectancy", 65.0)

        # 1. GDP效应
        gdp_pc = state.gdp / max(state.total_population, 1.0)
        gdp_effect = 5.0 * np.log1p(max(0, gdp_pc - 0.5)) / np.log(2)

        # 2. 医疗效应
        medical_spending = state.gdp * 0.03 * self.p.get("social_coverage", 0.5)
        medical_per_cap = medical_spending / max(state.total_population, 1.0)
        medical_effect = 3.0 * medical_per_cap / 0.5

        # 3. 污染惩罚
        industrial_intensity = state.sectors["i"].Y / max(state.gdp, 0.01)
        pollution_penalty = -1.5 * industrial_intensity * (1.0 - state.sectors["i"].digital_density * 0.5)

        # 合成
        ds.life_expectancy = max(40.0, min(85.0,
            base_life_exp + gdp_effect + medical_effect + pollution_penalty
        ))

        # 粗死亡率 = f(预期寿命, 年龄结构)
        # 简化：长寿→低死亡率
        ds.crude_death_rate = max(3.0, 50.0 - ds.life_expectancy * 0.55)

        # 婴儿死亡率（与预期寿命高度负相关）
        ds.infant_mortality = max(2.0, 120.0 - ds.life_expectancy * 1.6)

    # ─────────────────────────────────────────────
    # 队列分量推进
    # ─────────────────────────────────────────────
    def _cohort_component_step(self, state: SystemState):
        """简化队列分量法：年龄结构老化 + 出生 + 死亡.

        每一年：
        1. 各年龄组存活者上移至下一组（5年实际步进1/5组）
        2. 新生儿进入0-4组
        3. 死亡从各组按死亡率移除
        """
        ds = self._state
        age = ds.age_structure

        # 1. 各年龄组每年的存活率
        # 使用 Coale-Demeny 模型生命表的简化版
        # 存活率 = f(预期寿命, 年龄组)
        survival_rates = self._compute_survival_rates(ds.life_expectancy)

        # 2. 老化：每组的1/5上移至下一组（5岁组每年步进1岁）
        new_age = np.zeros_like(age)
        # 所有组存活后上移1/5
        for i in range(self.N_AGE_GROUPS - 1):
            outflow = age[i] / 5.0  # 每年1/5流出
            survivors = outflow * survival_rates[i]
            new_age[i + 1] += survivors
            new_age[i] += age[i] - outflow  # 留在原组的4/5

        # 最后一组（80+）留存
        new_age[-1] += age[-1] * (1.0 - 1.0 / 5.0) * survival_rates[-1]
        # 以及从倒数第二组流入
        new_age[-1] += (age[-2] / 5.0) * survival_rates[-2]

        # 3. 新生儿
        # 出生人数 = 育龄妇女(15-49, 组3-9) × 年龄别生育率 × 女性比例
        women_15_49 = sum(age[3:10]) * ds.total_population * 0.5
        births = women_15_49 * (ds.fertility_rate / 30.0)  # TFR→年出生率
        births = max(0, births)
        new_age[0] += births / max(ds.total_population, 1.0)

        # 4. 总人口变化
        natural_increase = births - (
            ds.total_population * ds.crude_death_rate / 1000.0
        )
        ds.total_population = max(10.0, ds.total_population + natural_increase)

        # 5. 归一化年龄结构
        ds.age_structure = new_age / max(new_age.sum(), 1e-10)

    def _compute_survival_rates(self, life_expectancy: float) -> np.ndarray:
        """基于预期寿命计算各年龄组的年存活率."""
        # 基准存活率（对应预期寿命65岁）
        base_rates = np.array([
            0.995, 0.998, 0.997, 0.996, 0.995,   # 0-24
            0.994, 0.993, 0.992, 0.990,            # 25-44
            0.987, 0.983, 0.977, 0.968,            # 45-64
            0.955, 0.935, 0.900, 0.800,            # 65+
        ])

        # 调整因子：预期寿命每增加1岁，存活率微量上升
        adjustment = (life_expectancy - 65.0) * 0.0005
        return np.clip(base_rates + adjustment, 0.7, 0.999)

    # ─────────────────────────────────────────────
    # 城镇化
    # ─────────────────────────────────────────────
    def _update_urbanization(self, state: SystemState):
        """年度城镇化推进.

        城镇化速率取决于：
        1. 工业和服务业就业扩张速度（拉力）
        2. 农业劳动生产率提升（推力）
        3. 当前城镇化水平（饱和效应）
        """
        ds = self._state

        # 城市拉力：非农就业增长率
        non_ag_L = sum(
            state.sectors[k].L
            for k in ALL_SECTORS if k != "b"
        )
        urban_pull = non_ag_L / max(state.total_population, 1.0)

        # 农村推力：农业劳动生产率
        ag_productivity = state.sectors["b"].Y / max(state.sectors["b"].L, 0.01)
        rural_push = min(0.1, ag_productivity / 5.0)

        # 饱和效应：城镇化率越高，进一步城镇化越难
        saturation = 1.0 - ds.urbanization_rate / 0.85

        # 城镇化增量
        d_urban = (urban_pull * 0.03 + rural_push * 0.02) * saturation
        ds.urbanization_rate = max(0.10, min(0.90,
            ds.urbanization_rate + d_urban
        ))

        ds.rural_to_urban_flow = d_urban

    def _update_urbanization_continuous(self, state: SystemState, dt: float):
        """城镇化在季度步长下的连续更新（仅追踪，不改变年龄结构）."""
        ds = self._state
        non_ag_L = sum(
            state.sectors[k].L
            for k in ALL_SECTORS if k != "b"
        )
        urban_pull = non_ag_L / max(state.total_population, 1.0)
        saturation = 1.0 - ds.urbanization_rate / 0.85
        d_urban = urban_pull * 0.03 * saturation * dt
        ds.urbanization_rate = max(0.10, min(0.90,
            ds.urbanization_rate + d_urban
        ))

    # ─────────────────────────────────────────────
    # 劳动力供给
    # ─────────────────────────────────────────────
    def _update_labor_supply(self, state: SystemState):
        """更新劳动年龄人口和劳动力供给."""
        ds = self._state

        # 劳动年龄人口（15-64岁，组3-12）
        working_age_share = ds.age_structure[self.WORKING_AGE_START:self.WORKING_AGE_END].sum()
        prev_working = ds.working_age_pop
        ds.working_age_pop = ds.total_population * working_age_share

        # 劳动年龄人口增长率（年度）
        if prev_working > 0:
            ds.working_age_growth_rate = (
                ds.working_age_pop / prev_working - 1.0
            )

        # 抚养比
        non_working = ds.total_population - ds.working_age_pop
        ds.dependency_ratio = non_working / max(ds.working_age_pop, 1.0)

        # 老年抚养比（65+ / 15-64）
        old_age_share = ds.age_structure[13:].sum()
        ds.old_age_dependency = (
            ds.total_population * old_age_share
        ) / max(ds.working_age_pop, 1.0)

        # 劳动参与率（受城镇化、教育、社保影响）
        ds.labor_force_participation = np.clip(
            0.55
            + ds.urbanization_rate * 0.10
            - ds.old_age_dependency * 0.15
            + self.p.get("social_coverage", 0.5) * 0.05,
            0.40, 0.75
        )

        # 潜在劳动力
        ds.potential_workers = ds.working_age_pop * ds.labor_force_participation

        # 人口红利强度
        # 正值 = 劳动年龄人口增长快于总人口（"人口红利"）
        # 负值 = 老龄化加速（"人口负债"）
        ds.demographic_dividend = (
            ds.working_age_growth_rate
            - (ds.total_population / max(prev_working / working_age_share, 1.0) - 1.0)
        )

    # ─────────────────────────────────────────────
    # 同步到全局状态
    # ─────────────────────────────────────────────
    def _sync_to_state(self, state: SystemState):
        """将人口状态同步到全局 SystemState."""
        ds = self._state
        state.total_population = ds.total_population
        state.working_age_pop = ds.working_age_pop
        state.dependency_ratio = ds.dependency_ratio
        state.fertility_rate = ds.fertility_rate
        state.life_expectancy = ds.life_expectancy
        state.urbanization_rate = ds.urbanization_rate

        # 劳动力总供给随人口动态调整
        # 劳动年龄人口变化 → 调整各家庭群组人口边界
        # 注意：这里只更新"潜在"劳动力，实际就业由 labor_market.py 决定

    # ─────────────────────────────────────────────
    # 公共接口
    # ─────────────────────────────────────────────
    def get_state(self) -> DemographyState:
        """获取当前人口状态."""
        return self._state

    def get_summary(self) -> dict:
        """获取人口状态摘要."""
        ds = self._state
        return {
            "total_population": ds.total_population,
            "working_age_pop": ds.working_age_pop,
            "dependency_ratio": ds.dependency_ratio,
            "old_age_dependency": ds.old_age_dependency,
            "fertility_rate": ds.fertility_rate,
            "life_expectancy": ds.life_expectancy,
            "urbanization_rate": ds.urbanization_rate,
            "labor_force_participation": ds.labor_force_participation,
            "demographic_dividend": ds.demographic_dividend,
            "working_age_growth_rate": ds.working_age_growth_rate,
            "crude_birth_rate": ds.crude_birth_rate,
            "crude_death_rate": ds.crude_death_rate,
        }

    def is_demographic_dividend(self) -> bool:
        """是否处于人口红利期."""
        return self._state.demographic_dividend > 0.005

    def is_aging_society(self) -> bool:
        """是否进入老龄化社会（老年抚养比 > 0.14）."""
        return self._state.old_age_dependency > 0.14
