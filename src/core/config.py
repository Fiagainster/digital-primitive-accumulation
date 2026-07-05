"""模拟配置."""

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """单次模拟的运行配置."""

    T_max: float = 100.0          # 模拟总年数
    dt_base: float = 0.25         # 基础步长（年）
    dt_min: float = 0.02          # 最细步长（约1周）
    adaptive_stepping: bool = True
    ews_window_size: int = 40     # EWS滑动窗口步数
    ews_warning_threshold: float = 0.4
    ews_critical_threshold: float = 0.7
    # v2.2: EWS韧性参数
    ews_resilience_window: int = 60    # 韧性评估窗口（需足够长以观察恢复）
    ews_resilience_weight: float = 0.35  # 韧性指数在综合EWS评分中的权重
    ews_stress_duration_threshold: int = 30  # 连续高压步数阈值（超过此值才视为"结构性高压"）
    # v2.2: 政治经济学反馈参数
    political_feedback_enabled: bool = True
    political_update_interval: float = 1.0  # 政治支持度更新间隔（年）
    sfc_tolerance: float = 1e-4
    random_seed: int = 42
    verbose: bool = False
    # v3.0: 多维评价参数
    evaluation_enabled: bool = True           # 是否启用五维评价
    evaluation_interval: float = 1.0          # 评价更新间隔（年）
    cdi_weight_preset: str = "default"        # CDI权重预设: default/growth_first/livelihood_first/security_first/balanced
    red_line_check_enabled: bool = True       # 是否启用人道主义红线检测

    # v3.6: 新增模块开关
    platform_enabled: bool = True             # 是否启用平台垄断模块
    demography_enabled: bool = True           # 是否启用内生人口动态
    demography_update_interval: float = 1.0   # 人口更新间隔（年）
    spatial_enabled: bool = True              # 是否启用双区域空间模块
    policy_execution_discount_enabled: bool = True  # 是否启用政策执行折扣
