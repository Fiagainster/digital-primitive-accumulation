"""国情预设体系 —— 六种后发国家的初始条件参数化.

设计原则：
- 国情、策略、场景三维正交 —— 选什么国家 × 选什么策略 × 处在什么国际环境
- 预设覆盖典型结构性差异，不是穷举国家
- 所有值均可被用户覆盖

六种预设：
1. 制造业大国 (manufacturing_giant)    — 中国c.2000, 韩国c.1980
2. 资源富国 (resource_rich)            — 沙特, 尼日利亚, 委内瑞拉
3. 数字跳跃者 (digital_leapfrogger)    — 印度c.2015, 肯尼亚, 爱沙尼亚
4. 小型开放经济体 (small_open)         — 越南c.2010, 孟加拉, 柬埔寨
5. 后苏联工业国 (post_soviet)          — 俄罗斯, 乌克兰
6. 人口大国弱工业 (populous_weak_industry) — 印度c.1990, 巴基斯坦, 埃塞俄比亚
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CountryProfile:
    """一个后发国家的初始条件完整刻画."""

    # 元数据
    name: str                          # 内部ID
    label: str                         # 中文名称
    description: str                   # 一句话描述
    typical_references: str            # 典型参照国家

    # 部门初始资本存量
    K_b: float = 200.0
    K_i: float = 300.0
    K_d: float = 20.0

    # 部门初始劳动力（万人）
    L_b: float = 25.0
    L_i: float = 30.0
    L_d: float = 5.0

    # 部门初始人力资本
    H_b: float = 0.8
    H_i: float = 1.0
    H_d: float = 1.5

    # 部门初始数字化水平 [0,1]
    DL_b: float = 0.05
    DL_i: float = 0.10
    DL_d: float = 0.15

    # 部门初始技术水平
    A_b: float = 1.0
    A_i: float = 1.2
    A_d: float = 1.5
    A_s: float = 0.8    # v3.3: 服务业初始技术水平

    # 初始工资
    W_b: float = 2.0
    W_i: float = 3.0
    W_d: float = 5.0
    W_s: float = 1.5    # v3.3: 服务业初始工资（低）

    # v2.2: 部门初始数字密度
    DD_b: float = 0.01
    DD_i: float = 0.03
    DD_d: float = 0.05
    DD_s: float = 0.02   # v3.3: 服务业数字密度（较低）

    # v3.3: 服务业初始条件
    K_s: float = 50.0    # 服务业资本存量（店面/设备等）
    L_s: float = 15.0    # 服务业劳动力（零售/餐饮/家政等）
    H_s: float = 0.6     # 服务业人力资本（较低）
    DL_s: float = 0.05   # 服务业数字化水平

    # 初始资源禀赋 [0,1]
    R_initial: float = 0.60

    # 初始制度能力 [0.3, 1.0]
    institution_capacity: float = 0.60

    # 初始基尼系数
    gini_initial: float = 0.38

    # 初始技术差距
    tech_gap_initial: float = 3.0

    # 初始数据资产
    data_producer_initial: float = 1.0
    data_consumer_initial: float = 0.5
    data_public_initial: float = 0.3

    # 初始数字服务流
    DS_b_initial: float = 0.5
    DS_i_initial: float = 1.0
    DS_d_initial: float = 2.0

    # 政策/策略默认倾向（仅文档参考，不影响模拟）
    natural_strategy: str = "A"        # 该国情"最自然"的策略
    strategy_risks: str = ""           # 该国情选择某些策略的额外风险

    def to_params_dict(self) -> Dict[str, float]:
        """转换为可直接合并到模拟参数字典的格式.

        所有键使用 IC_ 前缀，以区别于其他参数层。
        """
        return {
            # 资本
            "IC_K_b": self.K_b,
            "IC_K_i": self.K_i,
            "IC_K_d": self.K_d,
            # 劳动力
            "IC_L_b": self.L_b,
            "IC_L_i": self.L_i,
            "IC_L_d": self.L_d,
            # 人力资本
            "IC_H_b": self.H_b,
            "IC_H_i": self.H_i,
            "IC_H_d": self.H_d,
            # 数字化
            "IC_DL_b": self.DL_b,
            "IC_DL_i": self.DL_i,
            "IC_DL_d": self.DL_d,
            # 技术水平
            "IC_A_b": self.A_b,
            "IC_A_i": self.A_i,
            "IC_A_d": self.A_d,
            # 工资
            "IC_W_b": self.W_b,
            "IC_W_i": self.W_i,
            "IC_W_d": self.W_d,
            # 数字密度
            "IC_DD_b": self.DD_b,
            "IC_DD_i": self.DD_i,
            "IC_DD_d": self.DD_d,
            "IC_DD_s": self.DD_s,
            # v3.3: 服务业
            "IC_K_s": self.K_s,
            "IC_L_s": self.L_s,
            "IC_H_s": self.H_s,
            "IC_DL_s": self.DL_s,
            "IC_A_s": self.A_s,
            "IC_W_s": self.W_s,
            "IC_DS_s": 0.5,  # 服务业初始数字服务流
            # 资源
            "IC_R_initial": self.R_initial,
            "R_initial_ratio": self.R_initial,  # 兼容旧参数名
            # 制度
            "IC_institution_capacity": self.institution_capacity,
            "institution_capacity": self.institution_capacity,  # 兼容旧参数名
            # 不平等
            "IC_gini_initial": self.gini_initial,
            # 技术差距
            "IC_tech_gap_initial": self.tech_gap_initial,
            # 数据资产
            "IC_data_producer": self.data_producer_initial,
            "IC_data_consumer": self.data_consumer_initial,
            "IC_data_public": self.data_public_initial,
            # 数字服务
            "IC_DS_b": self.DS_b_initial,
            "IC_DS_i": self.DS_i_initial,
            "IC_DS_d": self.DS_d_initial,
        }

    def summary(self) -> str:
        """人类可读的单行摘要."""
        total_L = self.L_b + self.L_i + self.L_d
        return (f"[{self.name}] {self.label}: "
                f"K_i={self.K_i:.0f} K_d={self.K_d:.0f} "
                f"R={self.R_initial:.2f} H_i={self.H_i:.1f} "
                f"L={total_L:.0f} inst={self.institution_capacity:.2f} "
                f"gini={self.gini_initial:.2f}")


# ─────────────────────────────────────────────
# 六种国情预设
# ─────────────────────────────────────────────

def profile_manufacturing_giant() -> CountryProfile:
    """制造业大国 —— 工业雄厚，劳动力充足，数字起步."""
    return CountryProfile(
        name="manufacturing_giant",
        label="制造业大国",
        description="工业资本雄厚，劳动力充足，数字部门刚起步，人力资本中等，制度能力中等偏高",
        typical_references="中国c.2000, 韩国c.1980",
        K_b=200.0, K_i=400.0, K_d=15.0, K_s=80.0,
        L_b=25.0, L_i=45.0, L_d=3.0, L_s=20.0,
        H_b=0.7, H_i=1.0, H_d=1.2, H_s=0.7,
        DL_b=0.03, DL_i=0.08, DL_d=0.10, DL_s=0.08,
        A_b=1.0, A_i=1.3, A_d=1.5, A_s=0.9,
        W_b=2.0, W_i=3.5, W_d=6.0, W_s=2.0,
        DD_b=0.01, DD_i=0.03, DD_d=0.05, DD_s=0.02,
        R_initial=0.60,
        institution_capacity=0.65,
        gini_initial=0.38,
        tech_gap_initial=3.0,
        data_producer_initial=1.0,
        data_consumer_initial=0.5,
        data_public_initial=0.3,
        natural_strategy="A",
        strategy_risks="B需克服工业惯性；C浪费自身积累能力且外资可能控制关键产业",
    )


def profile_resource_rich() -> CountryProfile:
    """资源富国 —— 资源充足但工业薄弱，数字基础差，不平等高."""
    return CountryProfile(
        name="resource_rich",
        label="资源富国",
        description="资源充足但工业薄弱，数字基础差，人力资本偏低，制度能力弱，不平等高",
        typical_references="沙特, 尼日利亚, 委内瑞拉",
        K_b=150.0, K_i=100.0, K_d=5.0,
        L_b=20.0, L_i=15.0, L_d=1.0,
        H_b=0.5, H_i=0.7, H_d=1.0,
        DL_b=0.01, DL_i=0.03, DL_d=0.05,
        A_b=1.2, A_i=0.8, A_d=1.0,
        W_b=1.5, W_i=2.5, W_d=4.0,
        DD_b=0.00, DD_i=0.01, DD_d=0.02,
        R_initial=0.95,
        institution_capacity=0.35,
        gini_initial=0.50,
        tech_gap_initial=5.0,
        data_producer_initial=0.3,
        data_consumer_initial=0.2,
        data_public_initial=0.1,
        natural_strategy="C",
        strategy_risks="C的荷兰病+政策俘获是该国情的最大陷阱；A需从无到有建设工业(非常困难)；B缺乏数字人才基础",
    )


def profile_digital_leapfrogger() -> CountryProfile:
    """数字跳跃者 —— 工业中等偏弱但数字活跃，人力资本两极分化."""
    return CountryProfile(
        name="digital_leapfrogger",
        label="数字跳跃者",
        description="工业中等偏弱但数字部门活跃(IT外包/金融科技)，人力资本两极分化，制度能力中等",
        typical_references="印度c.2015, 肯尼亚(M-Pesa), 爱沙尼亚",
        K_b=120.0, K_i=150.0, K_d=30.0,
        L_b=30.0, L_i=25.0, L_d=8.0,
        H_b=0.6, H_i=0.8, H_d=1.6,
        DL_b=0.05, DL_i=0.20, DL_d=0.30,
        A_b=0.9, A_i=1.0, A_d=1.8,
        W_b=1.5, W_i=3.0, W_d=8.0,
        DD_b=0.02, DD_i=0.08, DD_d=0.12,
        R_initial=0.40,
        institution_capacity=0.55,
        gini_initial=0.42,
        tech_gap_initial=4.0,
        data_producer_initial=2.0,
        data_consumer_initial=1.0,
        data_public_initial=0.4,
        natural_strategy="B",
        strategy_risks="B的撕裂陷阱(数字精英vs工业劳动者)是该国情的最大风险；A可补齐工业短板但浪费数字人才优势",
    )


def profile_small_open() -> CountryProfile:
    """小型开放经济体 —— 工业靠FDI驱动，数字极弱，谈判力弱."""
    return CountryProfile(
        name="small_open",
        label="小型开放经济体",
        description="人口适中偏小，工业靠FDI驱动(低端制造)，数字极弱，人力资本低，制度能力中等偏低",
        typical_references="越南c.2010, 孟加拉, 柬埔寨",
        K_b=100.0, K_i=120.0, K_d=3.0,
        L_b=15.0, L_i=20.0, L_d=1.0,
        H_b=0.4, H_i=0.6, H_d=0.8,
        DL_b=0.02, DL_i=0.05, DL_d=0.03,
        A_b=0.8, A_i=1.0, A_d=0.8,
        W_b=1.0, W_i=2.0, W_d=3.0,
        DD_b=0.00, DD_i=0.01, DD_d=0.01,
        R_initial=0.30,
        institution_capacity=0.40,
        gini_initial=0.35,
        tech_gap_initial=4.5,
        data_producer_initial=0.2,
        data_consumer_initial=0.1,
        data_public_initial=0.1,
        natural_strategy="A+C混合",
        strategy_risks="纯A缺乏国内市场规模；纯C的FDI依赖可能被外资控制；纯B缺乏数字人才和资本基础",
    )


def profile_post_soviet() -> CountryProfile:
    """后苏联工业国 —— 重工业遗产(资本陈旧)，高人力资本，制度转型不彻底."""
    return CountryProfile(
        name="post_soviet",
        label="后苏联工业国",
        description="继承了苏联重工业遗产(资本陈旧)+高人力资本+制度转型不彻底+资源丰富+不平等极高",
        typical_references="俄罗斯, 乌克兰, 白俄罗斯",
        K_b=180.0, K_i=350.0, K_d=10.0,
        L_b=20.0, L_i=35.0, L_d=4.0,
        H_b=0.8, H_i=1.2, H_d=1.4,
        DL_b=0.03, DL_i=0.05, DL_d=0.08,
        A_b=1.0, A_i=1.1, A_d=1.3,
        W_b=2.0, W_i=3.0, W_d=5.0,
        DD_b=0.01, DD_i=0.02, DD_d=0.04,
        R_initial=0.85,
        institution_capacity=0.40,
        gini_initial=0.55,
        tech_gap_initial=3.5,
        data_producer_initial=0.8,
        data_consumer_initial=0.3,
        data_public_initial=0.3,
        natural_strategy="A",
        strategy_risks="A的挑战在于旧资本的数字密度极低(DD_i=0.02)——升级而非新建；B可能引发严重社会撕裂(基尼已0.55)；C的资源依赖是诱惑也是陷阱",
    )


def profile_populous_weak_industry() -> CountryProfile:
    """人口大国弱工业 —— 最艰难的起点，无论选什么策略都面临巨大结构约束."""
    return CountryProfile(
        name="populous_weak_industry",
        label="人口大国弱工业",
        description="人口极多但工业极弱，农业为主，数字几乎为零，人力资本很低，制度能力弱",
        typical_references="印度c.1990, 巴基斯坦, 埃塞俄比亚",
        K_b=100.0, K_i=80.0, K_d=2.0,
        L_b=55.0, L_i=12.0, L_d=0.5,
        H_b=0.4, H_i=0.5, H_d=0.7,
        DL_b=0.01, DL_i=0.01, DL_d=0.01,
        A_b=0.7, A_i=0.8, A_d=0.6,
        W_b=0.8, W_i=1.5, W_d=3.0,
        DD_b=0.00, DD_i=0.00, DD_d=0.00,
        R_initial=0.35,
        institution_capacity=0.30,
        gini_initial=0.45,
        tech_gap_initial=7.0,
        data_producer_initial=0.1,
        data_consumer_initial=0.1,
        data_public_initial=0.1,
        natural_strategy="A",
        strategy_risks="无论选什么策略都极其艰难——A的'保护'可能变成'保护落后'；B完全缺乏数字基础；C缺乏有吸引力的资源和制度环境吸引FDI",
    )


# ─────────────────────────────────────────────
# 注册表与工具函数
# ─────────────────────────────────────────────

COUNTRY_PROFILES: Dict[str, CountryProfile] = {
    "manufacturing_giant": profile_manufacturing_giant(),
    "resource_rich": profile_resource_rich(),
    "digital_leapfrogger": profile_digital_leapfrogger(),
    "small_open": profile_small_open(),
    "post_soviet": profile_post_soviet(),
    "populous_weak_industry": profile_populous_weak_industry(),
}


def get_country_profile(name: str) -> CountryProfile:
    """获取指定国情预设.

    Args:
        name: 预设名称，如 "manufacturing_giant"

    Returns:
        CountryProfile 对象

    Raises:
        ValueError: 如果预设名称未知
    """
    if name not in COUNTRY_PROFILES:
        available = ", ".join(COUNTRY_PROFILES.keys())
        raise ValueError(
            f"未知的国情预设: '{name}'。可用预设: {available}"
        )
    return COUNTRY_PROFILES[name]


def list_country_profiles() -> List[Dict]:
    """列出所有可用的国情预设及其元数据."""
    result = []
    for name, profile in COUNTRY_PROFILES.items():
        result.append({
            "name": profile.name,
            "label": profile.label,
            "description": profile.description,
            "typical_references": profile.typical_references,
            "natural_strategy": profile.natural_strategy,
            "strategy_risks": profile.strategy_risks,
            "summary": profile.summary(),
        })
    return result


def get_profile_names() -> List[str]:
    """返回所有可用的预设名称列表."""
    return list(COUNTRY_PROFILES.keys())


def build_params_with_profile(
    profile_name: str,
    regime: str = "A",
    scenario: str = None,
    user_overrides: Dict[str, float] = None,
) -> Dict[str, float]:
    """一站式构建完整参数集：策略默认 + 国情预设 + 场景 + 用户覆盖.

    参数加载优先级（从低到高）：
    1. schema.build_defaults(regime)        — 策略/结构默认参数
    2. country_profile.to_params_dict()     — 国情初始条件覆盖
    3. get_scenario(scenario)              — 国际场景参数覆盖
    4. user_overrides                      — 用户自定义覆盖（最高优先级）

    Args:
        profile_name: 国情预设名称
        regime: 积累策略 A/B/C/M
        scenario: 国际场景名称（可选）
        user_overrides: 用户自定义参数覆盖（可选）

    Returns:
        完整的参数字典
    """
    from .schema import get_schema
    from .presets import get_scenario

    schema = get_schema()

    # 1. 策略默认参数（最低优先级）
    params = schema.build_defaults(regime)

    # 2. 国情预设覆盖
    profile = get_country_profile(profile_name)
    params.update(profile.to_params_dict())

    # 3. 国际场景覆盖
    if scenario:
        scene_params = get_scenario(scenario)
        params.update(scene_params)

    # 4. 用户自定义覆盖（最高优先级）
    if user_overrides:
        params.update(user_overrides)

    return params
