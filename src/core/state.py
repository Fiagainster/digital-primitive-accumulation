"""系统状态容器 —— 所有状态变量的统一数据结构 (v3.5: 七部门 + 就业质量)."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SectorState:
    """单个部门的状态快照."""

    K: float = 0.0          # 固定资本存量
    H: float = 1.0          # 人力资本指数
    L: float = 0.0          # 就业人数（万人）
    DL: float = 0.0         # 数字化水平 [0,1]
    Y: float = 0.0          # 当期产出
    W: float = 0.0          # 平均工资
    r: float = 0.0          # 利润率
    A: float = 1.0          # 技术水平
    I: float = 0.0          # 当期实际投资
    DS: float = 0.0         # 数字服务投入量
    M_used: float = 0.0     # 原材料使用量
    profit: float = 0.0     # 部门利润
    absorption_cap: float = 0.0  # 投资吸收能力
    digital_density: float = 0.0  # v2.2: 资本中嵌入数字技术的比例 [0,1]
    jobs_created: float = 0.0     # v2.2: 当期新增就业
    jobs_destroyed: float = 0.0   # v2.2: 当期减少就业
    # v3.3: 部门类型元数据（服务业与制造业的区分）
    sector_type: str = "goods"     # "goods" | "services" | "digital_infra" | "digital_app"
    # v3.5: 进口依赖度（区分数字部门infra/app和服务业各子类的对外依赖）
    import_dependency: float = 0.0  # [0,1] 该部门对外部技术的依赖程度
    # v3.5: 就业质量（区分正式就业和零工经济）
    employment_quality: float = 1.0  # [0,1] 1=正式稳定就业, 0.4=零工, 0.1=非正式


@dataclass
class DataAssets:
    """异质性数据资产."""

    producer_data: float = 0.0
    consumer_data: float = 0.0
    public_data: float = 0.0

    @property
    def effective_stock(self) -> float:
        cross = self.producer_data * (1.0 + 0.3 * self.consumer_data)
        public = self.public_data * 1.5
        return cross + public


@dataclass
class DigitalDependencyProfile:
    """v3.5: 六维数字依附评估（原三维扩展）."""

    # 基础设施层依附
    chip_dependency: float = 0.0          # 芯片进口依赖
    cloud_dependency: float = 0.0         # 云服务依赖
    os_dependency: float = 0.0            # 操作系统依赖
    # 数据层依附
    data_dependency: float = 0.0          # 数据存储与处理能力依赖
    # 标准层依附
    standards_autonomy: float = 0.5       # 标准参与权(反向)
    # v3.5: 新增维度
    capital_dependency: float = 0.0       # 资本依附（外资在数字部门持股比例）
    talent_dependency: float = 0.0        # 人才依附（高技能人才海外依赖）
    ecosystem_dependency: float = 0.0     # 生态依附（应用生态/开发者社区外国主导）

    @property
    def infrastructure_dependency(self) -> float:
        """基础设施综合依赖 = 芯片/云/OS加权."""
        return (0.45 * self.chip_dependency
                + 0.30 * self.cloud_dependency
                + 0.25 * self.os_dependency)

    @property
    def composite(self) -> float:
        """六维综合依附指数."""
        return (0.25 * self.infrastructure_dependency
                + 0.20 * self.data_dependency
                + 0.15 * (1.0 - self.standards_autonomy)
                + 0.20 * self.capital_dependency
                + 0.10 * self.talent_dependency
                + 0.10 * self.ecosystem_dependency)


@dataclass
class HouseholdGroup:
    """单个家庭群组."""

    population: float = 0.0
    employed: float = 0.0
    avg_wage: float = 0.0
    skill_level: float = 1.0
    welfare_index: float = 1.0
    cumulative_cost: float = 0.0
    compensation_received: float = 0.0
    political_power: float = 0.20  # v3.2: 政治影响力乘数 [0,1]
    income: float = 0.0
    consumption: float = 0.0
    # v3.2: 家庭群组储蓄与债务
    savings: float = 0.0           # 储蓄存量
    debt: float = 0.0              # 债务存量
    # v3.4: 财富存量
    equity: float = 0.0            # 股权持有（数字+工业资本所有权）


@dataclass
class HouseholdGroups:
    """v3.3: 五个异质性家庭群组（新增服务业劳动者）."""

    digital_high_skill: HouseholdGroup = field(default_factory=HouseholdGroup)
    industrial_workers: HouseholdGroup = field(default_factory=HouseholdGroup)
    base_workers: HouseholdGroup = field(default_factory=HouseholdGroup)
    service_workers: HouseholdGroup = field(default_factory=HouseholdGroup)  # v3.3 新增
    unemployed: HouseholdGroup = field(default_factory=HouseholdGroup)

    @property
    def total_population(self) -> float:
        return (self.digital_high_skill.population
                + self.industrial_workers.population
                + self.base_workers.population
                + self.service_workers.population  # v3.3
                + self.unemployed.population)

    @property
    def cost_gini(self) -> float:
        costs = np.array([
            self.digital_high_skill.cumulative_cost,
            self.industrial_workers.cumulative_cost,
            self.base_workers.cumulative_cost,
            self.service_workers.cumulative_cost,  # v3.3
            self.unemployed.cumulative_cost,
        ])
        pops = np.array([
            self.digital_high_skill.population,
            self.industrial_workers.population,
            self.base_workers.population,
            self.service_workers.population,  # v3.3
            self.unemployed.population,
        ])
        if costs.sum() == 0 or pops.sum() == 0:
            return 0.0
        per_capita = costs / np.maximum(pops, 1.0)
        return _gini(per_capita)

    @property
    def cost_ratio_industrial_vs_digital(self) -> float:
        d_cost = max(self.digital_high_skill.cumulative_cost, 1e-10)
        i_cost = self.industrial_workers.cumulative_cost
        return i_cost / d_cost

    @property
    def wealth_gini(self) -> float:
        """v3.4: 财富基尼系数 — 基于净值(savings + equity - debt)."""
        net_worths = np.array([
            max(0.0, self.digital_high_skill.savings + self.digital_high_skill.equity - self.digital_high_skill.debt),
            max(0.0, self.industrial_workers.savings + self.industrial_workers.equity - self.industrial_workers.debt),
            max(0.0, self.base_workers.savings + self.base_workers.equity - self.base_workers.debt),
            max(0.0, self.service_workers.savings + self.service_workers.equity - self.service_workers.debt),
            max(0.0, self.unemployed.savings + self.unemployed.equity - self.unemployed.debt),
        ])
        pops = np.array([
            self.digital_high_skill.population,
            self.industrial_workers.population,
            self.base_workers.population,
            self.service_workers.population,
            self.unemployed.population,
        ])
        if net_worths.sum() == 0 or pops.sum() == 0:
            return 0.0
        per_capita = net_worths / np.maximum(pops, 1.0)
        return _gini(per_capita)

    def all_groups(self) -> List[HouseholdGroup]:
        return [self.digital_high_skill, self.industrial_workers,
                self.base_workers, self.service_workers, self.unemployed]


@dataclass
class BalanceSheet:
    """部门资产负债表."""

    physical_capital: float = 0.0
    financial_assets: float = 0.0
    data_assets_value: float = 0.0
    debt_domestic: float = 0.0
    debt_foreign: float = 0.0
    equity_foreign: float = 0.0

    @property
    def total_assets(self) -> float:
        return self.physical_capital + self.financial_assets + self.data_assets_value

    @property
    def total_liabilities(self) -> float:
        return self.debt_domestic + self.debt_foreign + self.equity_foreign

    @property
    def net_worth(self) -> float:
        return self.total_assets - self.total_liabilities


@dataclass
class CostMetrics:
    """代价指标集合."""

    unemployment_stock: float = 0.0
    skill_depreciation: float = 0.0
    sectoral_disparity: float = 1.0
    regional_imbalance: float = 0.0
    intergenerational_transfer: float = 0.0
    cost_concentration: float = 0.0
    irreversibility_index: float = 0.0
    cost_by_group: Dict[str, float] = field(default_factory=dict)
    cost_ratio_industrial_digital: float = 1.0
    compensation_gap: float = 0.0
    deind_type: str = ""
    deind_quality: float = 1.0


@dataclass
class FractureStatus:
    """断裂状态."""

    active_fractures: List[str] = field(default_factory=list)
    warning_signals: Dict[str, float] = field(default_factory=dict)
    first_warning_time: Dict[str, float] = field(default_factory=dict)
    fracture_time: Dict[str, float] = field(default_factory=dict)
    is_terminal: bool = False
    terminal_type: str = ""


@dataclass
class EWSResult:
    """早期预警信号结果."""

    variable: str = ""
    variance_trend: float = 0.0
    ar1_trend: float = 0.0
    skewness: float = 0.0
    flickering_pvalue: float = 1.0
    es_proximity: float = 0.0
    composite_score: float = 0.0
    status: str = "normal"
    # v2.2: 韧性维度
    resilience_index: float = 0.0     # 综合韧性指数 [0,1], 1=高韧性
    stress_duration: float = 0.0      # EWS连续处于高位的时间步数
    recovery_speed: float = 0.0       # 从扰动恢复的速度


@dataclass
class PoliticalSupport:
    """v2.2: 政治支持度 — 政治经济学反馈环的核心状态变量."""

    support_index: float = 0.7        # 综合支持度 [0,1], <0.3触发政策约束
    employment_satisfaction: float = 0.7  # 就业满意度分量
    wage_satisfaction: float = 0.7    # 工资满意度分量
    inequality_tolerance: float = 0.7 # 不平等容忍度分量
    welfare_satisfaction: float = 0.7 # 福利满意度分量
    policy_constraint: float = 1.0    # 政策约束乘数 [0.3, 1.0], 低支持度时收紧
    protest_risk: float = 0.0         # 抗议风险 [0,1]
    reform_momentum: float = 0.0      # 改革动能（支持度回升趋势）


@dataclass
class SystemState:
    """全局系统状态 —— 所有变量在单一时点的完整快照."""

    t: float = 0.0
    step: int = 0
    sectors: Dict[str, SectorState] = field(default_factory=dict)
    data_assets: DataAssets = field(default_factory=DataAssets)
    dependency: DigitalDependencyProfile = field(default_factory=DigitalDependencyProfile)
    households: HouseholdGroups = field(default_factory=HouseholdGroups)
    balance_sheets: Dict[str, BalanceSheet] = field(default_factory=dict)

    total_surplus: float = 0.0
    accumulation_fund: float = 0.0
    tech_gap: float = 1.0
    competitiveness: float = 1.0
    external_dependency: float = 0.0
    welfare_index: float = 1.0
    gdp: float = 0.0
    gini: float = 0.35
    wealth_gini: float = 0.40  # v3.4: 财富基尼系数（净值分布）
    unemployment_rate: float = 0.0
    fiscal_debt_ratio: float = 0.0
    transaction_residual: float = 0.0

    cost_metrics: CostMetrics = field(default_factory=CostMetrics)
    ews_signals: Dict[str, EWSResult] = field(default_factory=dict)
    fracture_status: FractureStatus = field(default_factory=FractureStatus)
    accumulation_regime: str = "A"
    political_support: PoliticalSupport = field(default_factory=PoliticalSupport)  # v2.2 新增

    initial_K_i: float = 1.0
    initial_K_b: float = 1.0
    initial_K_d: float = 1.0
    initial_K_s: float = 1.0  # v3.3
    initial_gdp_pc: float = 1.0  # v3.1: 初始人均GDP（用于增长率计算）
    # v3.2: 劳动力技能侵蚀（由引擎在主循环中填充）
    skill_erosion_i: float = 1.0
    skill_erosion_b: float = 1.0
    # v3.3: 低端服务业（非正式就业缓冲带）
    informal_service_L: float = 0.0    # 低端服务业/零工经济劳动力
    informal_service_W: float = 0.0    # 低端服务业/零工经济平均工资
    # v3.5: 就业质量综合指数（加权平均）
    employment_quality_index: float = 1.0  # [0,1] 整体就业质量
    # v3.5: 数字部门infra/app分层技术差距
    tech_gap_infra: float = 3.0        # 基础设施层技术差距
    tech_gap_app: float = 3.0          # 应用层技术差距
    # v3.3: 金融部门
    credit_rationing_ratio: float = 1.0  # 信贷可得性 [0,1]
    real_exchange_rate: float = 1.0      # 实际汇率（1=基准）
    # v3.3: 内生国际
    effective_block_max: float = 0.5
    effective_fdi_availability: float = 1.0
    initial_tech_gap: float = 3.0        # 初始技术差距（用于内生封锁计算）
    # v3.3: 气候/环境
    carbon_emissions: float = 0.0        # 当期碳排放
    cumulative_carbon: float = 0.0       # 累计碳排放
    carbon_damage_multiplier: float = 1.0  # 气候损害乘数 (>1=损害)
    green_transition_pressure: float = 0.0  # 绿色转型压力 [0,1]

    # v3.6: 平台垄断
    digital_hhi: float = 0.0              # 数字部门HHI市场集中度 [0,1]
    platform_monopoly_rent: float = 0.0   # 平台垄断租金占数字利润比例 [0,1]
    data_network_moat: float = 0.0        # 数据网络护城河强度 [0,1]

    # v3.6: 内生人口
    total_population: float = 0.0         # 总人口（万人）——由demography模块内生更新
    working_age_pop: float = 0.0          # 劳动年龄人口（15-64岁）
    dependency_ratio: float = 0.5         # 抚养比（非劳动年龄/劳动年龄）
    fertility_rate: float = 2.5           # 总和生育率
    life_expectancy: float = 65.0         # 预期寿命（年）
    urbanization_rate: float = 0.35       # 城镇化率 [0,1]

    # v3.6: 空间维度
    coastal_industry_share: float = 0.60       # 沿海/城市区域工业份额
    coastal_digital_infra_share: float = 0.75  # 沿海数字infra份额
    coastal_digital_app_share: float = 0.80    # 沿海数字app份额
    coastal_service_high_share: float = 0.70   # 沿海高技能服务业份额
    coastal_population_share: float = 0.45     # 沿海/城市人口份额
    urban_rural_productivity_ratio: float = 2.5  # 城乡生产率比
    spatial_gini: float = 0.0                   # 区域不平等（基尼系数）
    agglomeration_bonus: float = 0.0            # 集聚经济效应
    congestion_penalty: float = 0.0             # 拥挤成本

    # v3.6: 政策执行
    policy_execution_discount: float = 1.0  # 官僚体系策略执行效率 [0.5, 1.0]


# v3.5: 七部门标识（数字拆分为infra+app, 服务业拆分为high+low）
ALL_SECTORS = ["b", "i", "d_infra", "d_app", "s_high", "s_low"]

# v3.5: 向后兼容的部门组
DIGITAL_SECTORS = ["d_infra", "d_app"]
SERVICE_SECTORS = ["s_high", "s_low"]
PRODUCTION_SECTORS = ["b", "i"]  # 物质生产部门
# 旧代码兼容映射（逐步迁移）
LEGACY_d = DIGITAL_SECTORS  # "d" → ["d_infra", "d_app"]
LEGACY_s = SERVICE_SECTORS  # "s" → ["s_high", "s_low"]

# v3.5: 就业质量分类
EMPLOYMENT_FORMAL = 1.0       # 正式稳定就业
EMPLOYMENT_GIG = 0.4          # 零工经济（有收入但不稳定）
EMPLOYMENT_INFORMAL = 0.1     # 非正式就业/隐藏失业


def _gini(x: np.ndarray) -> float:
    """计算基尼系数."""
    if len(x) < 2 or x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))
