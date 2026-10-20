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
    K_d: float = 20.0          # 兼容旧参数
    # v3.5: 数字部门infra/app分层
    K_d_infra: float = 8.0     # 基础设施层资本（芯片厂/数据中心）
    K_d_app: float = 12.0      # 应用层资本（平台/软件）
    # v3.5: 服务业high/low分层
    K_s_high: float = 30.0     # 高技能服务业资本
    K_s_low: float = 20.0      # 低端服务业资本

    # 部门初始劳动力（万人）
    L_b: float = 25.0
    L_i: float = 30.0
    L_d: float = 5.0           # 兼容
    L_d_infra: float = 1.5     # infra劳动力少而精
    L_d_app: float = 3.5       # app劳动力较多
    L_s_high: float = 8.0      # 高技能服务劳动力
    L_s_low: float = 7.0       # 低端服务劳动力

    # 部门初始人力资本
    H_b: float = 0.8
    H_i: float = 1.0
    H_d: float = 1.5           # 兼容
    H_d_infra: float = 1.6     # infra需要极高技能
    H_d_app: float = 1.4       # app技能偏高
    H_s_high: float = 1.1      # 高技能服务人力资本
    H_s_low: float = 0.5       # 低端服务人力资本低

    # 部门初始数字化水平 [0,1]
    DL_b: float = 0.05
    DL_i: float = 0.10
    DL_d: float = 0.15         # 兼容
    DL_d_infra: float = 0.08
    DL_d_app: float = 0.20
    DL_s_high: float = 0.08
    DL_s_low: float = 0.03

    # 部门初始技术水平
    A_b: float = 1.0
    A_i: float = 1.2
    A_d: float = 1.5           # 兼容
    A_d_infra: float = 1.2
    A_d_app: float = 1.8
    A_s: float = 0.8           # 兼容
    A_s_high: float = 1.0
    A_s_low: float = 0.7

    # 初始工资
    W_b: float = 2.0
    W_i: float = 3.0
    W_d: float = 5.0           # 兼容
    W_d_infra: float = 6.0
    W_d_app: float = 5.0
    W_s: float = 1.5           # 兼容
    W_s_high: float = 3.5
    W_s_low: float = 1.2

    # v2.2: 部门初始数字密度
    DD_b: float = 0.01
    DD_i: float = 0.03
    DD_d: float = 0.05         # 兼容
    DD_d_infra: float = 0.03
    DD_d_app: float = 0.06
    DD_s: float = 0.02         # 兼容
    DD_s_high: float = 0.04
    DD_s_low: float = 0.01

    # v3.5: 进口依赖度初始值
    import_dep_d_infra: float = 0.75   # infra高度依赖进口
    import_dep_d_app: float = 0.30     # app国产替代更多

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

    # 政策/策略默认倾向
    natural_strategy: str = "A"         # 该国情"最自然"的策略 (人类可读)
    recommended_regime: str = "A"       # 🆕 v3.7.1: 机器可读的推荐策略 (A/B/C/M_*)
    viable_regimes: list = None         # 🆕 v3.7.1: 可行的策略列表
    contraindicated: list = None        # 🆕 v3.7.1: 应避免的策略
    strategy_risks: str = ""            # 该国情选择某些策略的额外风险

    def __post_init__(self):
        if self.viable_regimes is None:
            self.viable_regimes = ["A", "B", "C"]
        if self.contraindicated is None:
            self.contraindicated = []

    def to_params_dict(self) -> Dict[str, float]:
        """转换为可直接合并到模拟参数字典的格式 (v3.5: 六部门)."""
        result = {
            # 资本 (兼容旧参数)
            "IC_K_b": self.K_b, "IC_K_i": self.K_i, "IC_K_d": self.K_d,
            # v3.5: 数字infra/app
            "IC_K_d_infra": self.K_d_infra, "IC_K_d_app": self.K_d_app,
            # v3.5: 服务业high/low
            "IC_K_s_high": self.K_s_high, "IC_K_s_low": self.K_s_low,
            # 劳动力
            "IC_L_b": self.L_b, "IC_L_i": self.L_i, "IC_L_d": self.L_d,
            "IC_L_d_infra": self.L_d_infra, "IC_L_d_app": self.L_d_app,
            "IC_L_s_high": self.L_s_high, "IC_L_s_low": self.L_s_low,
            # 人力资本
            "IC_H_b": self.H_b, "IC_H_i": self.H_i, "IC_H_d": self.H_d,
            "IC_H_d_infra": self.H_d_infra, "IC_H_d_app": self.H_d_app,
            "IC_H_s_high": self.H_s_high, "IC_H_s_low": self.H_s_low,
            # 数字化
            "IC_DL_b": self.DL_b, "IC_DL_i": self.DL_i, "IC_DL_d": self.DL_d,
            "IC_DL_d_infra": self.DL_d_infra, "IC_DL_d_app": self.DL_d_app,
            "IC_DL_s_high": self.DL_s_high, "IC_DL_s_low": self.DL_s_low,
            # 技术水平
            "IC_A_b": self.A_b, "IC_A_i": self.A_i, "IC_A_d": self.A_d,
            "IC_A_d_infra": self.A_d_infra, "IC_A_d_app": self.A_d_app,
            "IC_A_s_high": self.A_s_high, "IC_A_s_low": self.A_s_low,
            # 工资
            "IC_W_b": self.W_b, "IC_W_i": self.W_i, "IC_W_d": self.W_d,
            "IC_W_d_infra": self.W_d_infra, "IC_W_d_app": self.W_d_app,
            "IC_W_s_high": self.W_s_high, "IC_W_s_low": self.W_s_low,
            # 数字密度
            "IC_DD_b": self.DD_b, "IC_DD_i": self.DD_i, "IC_DD_d": self.DD_d,
            "IC_DD_d_infra": self.DD_d_infra, "IC_DD_d_app": self.DD_d_app,
            "IC_DD_s_high": self.DD_s_high, "IC_DD_s_low": self.DD_s_low,
            # 兼容旧服务业参数
            "IC_K_s": self.K_s_low, "IC_L_s": self.L_s_low,
            "IC_H_s": self.H_s_low, "IC_DL_s": self.DL_s_low,
            "IC_A_s": self.A_s_low, "IC_W_s": self.W_s_low, "IC_DD_s": self.DD_s_low,
            # v3.5: 进口依赖度
            "IC_import_dep_d_infra": self.import_dep_d_infra,
            "IC_import_dep_d_app": self.import_dep_d_app,
            # 资源/制度/不平等
            "IC_R_initial": self.R_initial,
            "R_initial_ratio": self.R_initial,
            "IC_institution_capacity": self.institution_capacity,
            "institution_capacity": self.institution_capacity,
            "IC_gini_initial": self.gini_initial,
            "IC_tech_gap_initial": self.tech_gap_initial,
            "IC_tech_gap_infra_initial": self.tech_gap_initial * 1.3,
            "IC_tech_gap_app_initial": self.tech_gap_initial * 0.8,
            # 数据资产
            "IC_data_producer": self.data_producer_initial,
            "IC_data_consumer": self.data_consumer_initial,
            "IC_data_public": self.data_public_initial,
            # 数字服务
            "IC_DS_b": self.DS_b_initial,
            "IC_DS_i": self.DS_i_initial,
            "IC_DS_d": self.DS_d_initial,
            "IC_DS_d_infra": self.DS_d_initial * 0.6,
            "IC_DS_d_app": self.DS_d_initial * 0.8,
            "IC_DS_s_high": 0.8, "IC_DS_s_low": 0.3,
        }
        return result

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
        K_b=200.0, K_i=400.0, K_d=15.0,
        K_d_infra=5.0, K_d_app=10.0,
        K_s_high=40.0, K_s_low=30.0,
        L_b=25.0, L_i=45.0, L_d=3.0,
        L_d_infra=1.0, L_d_app=2.0,
        L_s_high=10.0, L_s_low=10.0,
        H_b=0.7, H_i=1.0, H_d=1.2,
        H_d_infra=1.3, H_d_app=1.1,
        H_s_high=1.0, H_s_low=0.6,
        DL_b=0.03, DL_i=0.08, DL_d=0.10,
        DL_d_infra=0.05, DL_d_app=0.12,
        DL_s_high=0.08, DL_s_low=0.05,
        A_b=1.0, A_i=1.3, A_d=1.5,
        A_d_infra=1.1, A_d_app=1.6,
        A_s_high=1.1, A_s_low=0.8,
        W_b=2.0, W_i=3.5, W_d=6.0,
        W_d_infra=5.5, W_d_app=6.5,
        W_s_high=4.0, W_s_low=1.8,
        DD_b=0.01, DD_i=0.03, DD_d=0.05,
        DD_d_infra=0.02, DD_d_app=0.06,
        DD_s_high=0.04, DD_s_low=0.01,
        import_dep_d_infra=0.75, import_dep_d_app=0.25,
        R_initial=0.60,
        institution_capacity=0.65,
        gini_initial=0.38,
        tech_gap_initial=3.0,
        data_producer_initial=1.0,
        data_consumer_initial=0.5,
        data_public_initial=0.3,
        natural_strategy="A",
        recommended_regime="A",
        viable_regimes=["A", "M_A_dominant_B", "M_A_dominant_C"],
        contraindicated=["C"],
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
        K_d_infra=1.8, K_d_app=3.2,
        K_s_high=18.0, K_s_low=12.0,
        L_b=20.0, L_i=15.0, L_d=1.0,
        L_d_infra=0.3, L_d_app=0.7,
        L_s_high=8.0, L_s_low=7.0,
        H_b=0.5, H_i=0.7, H_d=1.0,
        H_d_infra=1.1, H_d_app=0.9,
        H_s_high=0.9, H_s_low=0.4,
        DL_b=0.01, DL_i=0.03, DL_d=0.05,
        DL_d_infra=0.03, DL_d_app=0.07,
        DL_s_high=0.08, DL_s_low=0.03,
        A_b=1.2, A_i=0.8, A_d=1.0,
        A_d_infra=0.8, A_d_app=1.2,
        A_s_high=1.0, A_s_low=0.6,
        W_b=1.5, W_i=2.5, W_d=4.0,
        W_d_infra=3.5, W_d_app=4.5,
        W_s_high=3.0, W_s_low=1.0,
        DD_b=0.00, DD_i=0.01, DD_d=0.02,
        DD_d_infra=0.01, DD_d_app=0.02,
        DD_s_high=0.04, DD_s_low=0.01,
        import_dep_d_infra=0.80, import_dep_d_app=0.40,
        R_initial=0.95,
        institution_capacity=0.35,
        gini_initial=0.50,
        tech_gap_initial=5.0,
        data_producer_initial=0.3,
        data_consumer_initial=0.2,
        data_public_initial=0.1,
        natural_strategy="C",
        recommended_regime="C",
        viable_regimes=["C", "M_C_dominant_A"],
        contraindicated=["B"],
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
        K_d_infra=10.0, K_d_app=20.0,
        K_s_high=30.0, K_s_low=20.0,
        L_b=30.0, L_i=25.0, L_d=8.0,
        L_d_infra=2.4, L_d_app=5.6,
        L_s_high=6.5, L_s_low=5.5,
        H_b=0.6, H_i=0.8, H_d=1.6,
        H_d_infra=1.7, H_d_app=1.5,
        H_s_high=1.0, H_s_low=0.4,
        DL_b=0.05, DL_i=0.20, DL_d=0.30,
        DL_d_infra=0.18, DL_d_app=0.40,
        DL_s_high=0.15, DL_s_low=0.06,
        A_b=0.9, A_i=1.0, A_d=1.8,
        A_d_infra=1.4, A_d_app=2.2,
        A_s_high=1.1, A_s_low=0.7,
        W_b=1.5, W_i=3.0, W_d=8.0,
        W_d_infra=7.0, W_d_app=9.0,
        W_s_high=4.0, W_s_low=1.4,
        DD_b=0.02, DD_i=0.08, DD_d=0.12,
        DD_d_infra=0.07, DD_d_app=0.14,
        DD_s_high=0.06, DD_s_low=0.02,
        import_dep_d_infra=0.70, import_dep_d_app=0.35,
        R_initial=0.40,
        institution_capacity=0.55,
        gini_initial=0.42,
        tech_gap_initial=4.0,
        data_producer_initial=2.0,
        data_consumer_initial=1.0,
        data_public_initial=0.4,
        natural_strategy="B",
        recommended_regime="B",
        viable_regimes=["B", "M_B_dominant_A", "A"],
        contraindicated=["C"],
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
        K_d_infra=1.0, K_d_app=2.0,
        K_s_high=15.0, K_s_low=10.0,
        L_b=15.0, L_i=20.0, L_d=1.0,
        L_d_infra=0.3, L_d_app=0.7,
        L_s_high=5.5, L_s_low=4.5,
        H_b=0.4, H_i=0.6, H_d=0.8,
        H_d_infra=0.9, H_d_app=0.7,
        H_s_high=0.8, H_s_low=0.3,
        DL_b=0.02, DL_i=0.05, DL_d=0.03,
        DL_d_infra=0.02, DL_d_app=0.04,
        DL_s_high=0.05, DL_s_low=0.02,
        A_b=0.8, A_i=1.0, A_d=0.8,
        A_d_infra=0.6, A_d_app=1.0,
        A_s_high=0.8, A_s_low=0.6,
        W_b=1.0, W_i=2.0, W_d=3.0,
        W_d_infra=2.5, W_d_app=3.5,
        W_s_high=2.0, W_s_low=0.7,
        DD_b=0.00, DD_i=0.01, DD_d=0.01,
        DD_d_infra=0.01, DD_d_app=0.01,
        DD_s_high=0.02, DD_s_low=0.01,
        import_dep_d_infra=0.85, import_dep_d_app=0.50,
        R_initial=0.30,
        institution_capacity=0.40,
        gini_initial=0.35,
        tech_gap_initial=4.5,
        data_producer_initial=0.2,
        data_consumer_initial=0.1,
        data_public_initial=0.1,
        natural_strategy="A+C混合",
        recommended_regime="M_A_dominant_C",
        viable_regimes=["A", "C", "M_A_dominant_C", "M_C_dominant_A"],
        contraindicated=["B"],
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
        K_d_infra=3.5, K_d_app=6.5,
        K_s_high=21.0, K_s_low=14.0,
        L_b=20.0, L_i=35.0, L_d=4.0,
        L_d_infra=1.2, L_d_app=2.8,
        L_s_high=10.0, L_s_low=8.0,
        H_b=0.8, H_i=1.2, H_d=1.4,
        H_d_infra=1.5, H_d_app=1.3,
        H_s_high=1.2, H_s_low=0.5,
        DL_b=0.03, DL_i=0.05, DL_d=0.08,
        DL_d_infra=0.05, DL_d_app=0.10,
        DL_s_high=0.08, DL_s_low=0.03,
        A_b=1.0, A_i=1.1, A_d=1.3,
        A_d_infra=1.0, A_d_app=1.6,
        A_s_high=1.1, A_s_low=0.7,
        W_b=2.0, W_i=3.0, W_d=5.0,
        W_d_infra=4.5, W_d_app=5.5,
        W_s_high=4.0, W_s_low=1.4,
        DD_b=0.01, DD_i=0.02, DD_d=0.04,
        DD_d_infra=0.02, DD_d_app=0.05,
        DD_s_high=0.04, DD_s_low=0.01,
        import_dep_d_infra=0.80, import_dep_d_app=0.40,
        R_initial=0.85,
        institution_capacity=0.40,
        gini_initial=0.55,
        tech_gap_initial=3.5,
        data_producer_initial=0.8,
        data_consumer_initial=0.3,
        data_public_initial=0.3,
        natural_strategy="A",
        recommended_regime="A",
        viable_regimes=["A", "M_A_dominant_B"],
        contraindicated=["C"],
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
        K_d_infra=0.7, K_d_app=1.3,
        K_s_high=12.0, K_s_low=8.0,
        L_b=55.0, L_i=12.0, L_d=0.5,
        L_d_infra=0.15, L_d_app=0.35,
        L_s_high=4.5, L_s_low=3.5,
        H_b=0.4, H_i=0.5, H_d=0.7,
        H_d_infra=0.8, H_d_app=0.6,
        H_s_high=0.6, H_s_low=0.2,
        DL_b=0.01, DL_i=0.01, DL_d=0.01,
        DL_d_infra=0.01, DL_d_app=0.01,
        DL_s_high=0.03, DL_s_low=0.01,
        A_b=0.7, A_i=0.8, A_d=0.6,
        A_d_infra=0.5, A_d_app=0.7,
        A_s_high=0.7, A_s_low=0.5,
        W_b=0.8, W_i=1.5, W_d=3.0,
        W_d_infra=2.5, W_d_app=3.5,
        W_s_high=1.6, W_s_low=0.6,
        DD_b=0.00, DD_i=0.00, DD_d=0.00,
        DD_d_infra=0.00, DD_d_app=0.00,
        DD_s_high=0.00, DD_s_low=0.00,
        import_dep_d_infra=0.90, import_dep_d_app=0.50,
        R_initial=0.35,
        institution_capacity=0.30,
        gini_initial=0.45,
        tech_gap_initial=7.0,
        data_producer_initial=0.1,
        data_consumer_initial=0.1,
        data_public_initial=0.1,
        natural_strategy="A",
        recommended_regime="A",
        viable_regimes=["A", "M_A_dominant_C"],
        contraindicated=["B", "C"],
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
            "recommended_regime": profile.recommended_regime,
            "viable_regimes": profile.viable_regimes,
            "contraindicated": profile.contraindicated,
            "strategy_risks": profile.strategy_risks,
            "summary": profile.summary(),
        })
    return result


def get_recommended_regime(profile_name: str) -> str:
    """根据国情预设获取推荐的积累策略.

    可用于脚本自动选择默认策略，例如:
        regime = get_recommended_regime("digital_leapfrogger")  # → "B"

    Args:
        profile_name: 国情预设名称

    Returns:
        推荐的策略代码 (A/B/C/M_*)，若国情不存在则返回 "A"
    """
    profile = COUNTRY_PROFILES.get(profile_name)
    if profile is None:
        return "A"
    return profile.recommended_regime


def get_viable_regimes(profile_name: str) -> List[str]:
    """获取某国情下可行的策略列表."""
    profile = COUNTRY_PROFILES.get(profile_name)
    if profile is None:
        return ["A", "B", "C"]
    return profile.viable_regimes


def get_contraindicated_regimes(profile_name: str) -> List[str]:
    """获取某国情下应避免的策略列表."""
    profile = COUNTRY_PROFILES.get(profile_name)
    if profile is None:
        return []
    return profile.contraindicated


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
