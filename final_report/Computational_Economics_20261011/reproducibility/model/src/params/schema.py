"""参数模式定义 —— 所有参数的定义域、默认值、分类.

v3.7: 初始化时自动加载 config/default.yaml。
      硬编码默认值仅作为YAML缺失时的安全网。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import os
import warnings
import numpy as np


# v3.7: 自动定位项目根目录和默认配置文件
def _find_default_config() -> Optional[str]:
    """查找默认YAML配置文件.

    从当前文件位置向上查找，支持多种安装方式。
    """
    # 方法1: 相对于本文件 (src/params/schema.py → ../../config/default.yaml)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(current_dir, "..", "..", "config", "default.yaml"))
    if os.path.exists(candidate):
        return candidate

    # 方法2: 从当前工作目录查找
    cwd_candidate = os.path.join(os.getcwd(), "config", "default.yaml")
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    return None


@dataclass
class ParamDef:
    """单个参数定义."""

    name: str
    symbol: str
    description: str
    domain: Tuple[float, float]
    default: float
    layer: str  # "structural" | "strategy" | "scenario" | "meta"
    log_scale: bool = False


class ParameterSchema:
    """完整的参数模式.

    v3.7: 自动从 config/default.yaml 加载默认值。
          硬编码值仅在YAML文件不存在或参数缺失时作为安全网。
    """

    def __init__(self):
        self._params: Dict[str, ParamDef] = {}
        self._register_all()
        # v3.7: 自动加载YAML配置覆盖硬编码默认值
        self._load_yaml_defaults()

    def _reg(self, pid: str, symbol: str, desc: str, lo: float, hi: float,
             default: float, layer: str, log: bool = False):
        self._params[pid] = ParamDef(pid, symbol, desc, (lo, hi), default, layer, log)

    def _load_yaml_defaults(self):
        """v3.7: 从 config/default.yaml 加载参数值，覆盖硬编码默认值.

        只从规范默认区域提取参数（structural, strategy, scenario_defaults, meta,
        simulation, cdi_weights），跳过场景预设和国情预设中的极端/特定值。

        如果YAML文件不存在，静默跳过（向后兼容）。
        """
        config_path = _find_default_config()
        if config_path is None:
            return

        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)

            # v3.7: 只从规范默认区域提取，避免场景/国情预设污染默认值
            default_sections = [
                "structural",
                "strategy",
                "scenario_defaults",
                "meta",
                "simulation",
                "cdi_weights",
            ]

            yaml_flat = {}
            def _flatten(d, prefix=""):
                for k, v in d.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        _flatten(v, key)
                    elif isinstance(v, (int, float)):
                        leaf = key.split(".")[-1]
                        yaml_flat[leaf] = float(v)

            for section in default_sections:
                if section in raw:
                    _flatten(raw[section])

            # 覆盖已注册参数的默认值
            updated_count = 0
            for pid in self._params:
                if pid in yaml_flat:
                    old = self._params[pid].default
                    new = yaml_flat[pid]
                    # 检查定义域
                    lo, hi = self._params[pid].domain
                    if lo <= new <= hi:
                        if abs(old - new) > 1e-10:
                            self._params[pid].default = new
                            updated_count += 1

            # 静默加载

        except Exception:
            # YAML加载失败时静默回退到硬编码默认值
            pass

    def _register_all(self):
        # ============ L1: 结构参数 ============
        # 生产函数
        self._reg("alpha_b", "α_b", "基础部门资本产出弹性", 0.2, 0.5, 0.35, "structural")
        self._reg("alpha_i", "α_i", "工业部门资本产出弹性", 0.3, 0.6, 0.45, "structural")
        self._reg("alpha_d", "α_d", "数字部门资本产出弹性(兼容旧参数)", 0.15, 0.4, 0.25, "structural")
        # v3.5: 数字部门infra/app分层
        self._reg("alpha_d_infra", "α_d_infra", "数字基础设施层资本产出弹性", 0.35, 0.65, 0.55, "structural")
        self._reg("alpha_d_app", "α_d_app", "数字应用层资本产出弹性", 0.15, 0.40, 0.30, "structural")
        self._reg("delta_d_infra_hw", "δ_d_infra", "基础硬件对infra产出弹性", 0.05, 0.30, 0.20, "structural")
        self._reg("delta_d_app_data", "δ_d_app", "数据对app产出弹性", 0.10, 0.40, 0.25, "structural")
        self._reg("eta_network_app", "η_net_app", "应用层网络效应强度", 0.1, 0.6, 0.35, "structural")
        self._reg("xi_infra_to_app", "ξ_infra_app", "infra→app技术约束弹性", 0.1, 0.5, 0.30, "structural")
        self._reg("sigma_b", "σ_b", "基础部门替代弹性", 0.3, 1.5, 0.8, "structural")
        self._reg("theta_b", "θ_b", "基础部门资本份额", 0.2, 0.5, 0.35, "structural")
        self._reg("lambda_b", "λ_b", "资源约束弹性", 0.3, 1.5, 0.8, "structural")
        self._reg("gamma_i", "γ_i", "数字服务对工业产出弹性", 0.05, 0.35, 0.15, "structural")
        self._reg("delta_d_hw", "δ_d", "硬件对数字部门产出弹性(兼容)", 0.05, 0.3, 0.15, "structural")
        self._reg("epsilon_d", "ε_d", "数据存量弹性", 0.05, 0.3, 0.15, "structural")
        self._reg("mu_i", "μ_i", "原材料约束弹性", 0.3, 1.0, 0.7, "structural")
        # v3.3: 服务业部门 (v3.5: 保留兼容，新增分层参数)
        self._reg("alpha_s", "α_s", "服务业资本产出弹性(兼容)", 0.15, 0.35, 0.25, "structural")
        self._reg("alpha_s_high", "α_s_high", "高技能服务业资本产出弹性", 0.20, 0.45, 0.35, "structural")
        self._reg("alpha_s_low", "α_s_low", "低端服务业资本产出弹性", 0.05, 0.25, 0.15, "structural")
        self._reg("alpha_s_dig", "α_s_dig", "服务业数字化最大增益(兼容)", 0.2, 0.8, 0.5, "structural")
        self._reg("beta_s_dig", "β_s_dig", "服务业数字化弹性(兼容)", 0.3, 0.8, 0.5, "structural")
        self._reg("xi_s_digital_link", "ξ_s_dig_link", "高技能服务业与数字部门关联弹性", 0.1, 0.5, 0.30, "structural")
        self._reg("platform_empowerment", "plat_emp", "数字平台对低端服务业赋能强度", 0.1, 0.7, 0.40, "structural")

        # 数字化
        self._reg("alpha_b_dig", "α_b_dig", "基础部门数字化最大增益", 0.1, 0.6, 0.3, "structural")
        self._reg("alpha_i_dig", "α_i_dig", "工业部门数字化最大增益", 0.3, 1.5, 0.8, "structural")
        self._reg("beta_b_dig", "β_b_dig", "基础部门数字化弹性", 0.3, 0.8, 0.5, "structural")
        self._reg("beta_i_dig", "β_i_dig", "工业部门数字化弹性", 0.3, 0.8, 0.5, "structural")
        self._reg("eta_network", "η_net", "网络效应强度", 0.0, 0.5, 0.2, "structural")

        # 资本积累
        self._reg("delta_b", "δ_b", "基础部门折旧率", 0.03, 0.10, 0.05, "structural")
        self._reg("delta_i", "δ_i", "工业部门折旧率", 0.05, 0.15, 0.08, "structural")
        self._reg("delta_d", "δ_d", "数字部门折旧率(兼容)", 0.10, 0.30, 0.20, "structural")
        self._reg("delta_d_infra", "δ_d_infra", "数字基础设施层折旧率", 0.03, 0.10, 0.05, "structural")
        self._reg("delta_d_app", "δ_d_app", "数字应用层折旧率", 0.10, 0.30, 0.20, "structural")
        self._reg("delta_s", "δ_s", "服务业折旧率(兼容)", 0.03, 0.08, 0.05, "structural")
        self._reg("delta_s_high", "δ_s_high", "高技能服务业折旧率", 0.03, 0.08, 0.05, "structural")
        self._reg("delta_s_low", "δ_s_low", "低端服务业折旧率", 0.05, 0.12, 0.08, "structural")
        self._reg("nu_b", "ν_b", "基础部门投资吸收率", 0.1, 0.5, 0.3, "structural")
        self._reg("nu_i", "ν_i", "工业部门投资吸收率", 0.2, 0.6, 0.4, "structural")
        self._reg("nu_d", "ν_d", "数字部门投资吸收率(兼容)", 0.1, 0.4, 0.25, "structural")
        self._reg("nu_d_infra", "ν_d_infra", "数字infra投资吸收率", 0.08, 0.35, 0.20, "structural")
        self._reg("nu_d_app", "ν_d_app", "数字app投资吸收率", 0.15, 0.45, 0.30, "structural")
        self._reg("nu_s", "ν_s", "服务业投资吸收率(兼容)", 0.1, 0.5, 0.35, "structural")
        self._reg("nu_s_high", "ν_s_high", "高技能服务业投资吸收率", 0.15, 0.55, 0.40, "structural")
        self._reg("nu_s_low", "ν_s_low", "低端服务业投资吸收率", 0.1, 0.45, 0.30, "structural")
        self._reg("chi_absorb", "χ_abs", "人力资本对吸收能力弹性", 0.3, 1.5, 0.8, "structural")
        self._reg("psi_obsolescence", "ψ", "技术淘汰系数", 0.0, 0.3, 0.1, "structural")

        # 劳动力市场
        self._reg("lambda_flow", "λ_flow", "基础流动概率", 0.01, 0.15, 0.05, "structural")
        self._reg("sigma_skill", "σ_skill", "技能匹配敏感度", 0.1, 0.5, 0.25, "structural")
        self._reg("lambda_cost", "λ_cost", "转移成本敏感度", 0.5, 3.0, 1.0, "structural")
        self._reg("bp_base", "bp_base", "基础谈判力", 0.3, 0.8, 0.5, "structural")

        # 技术追赶
        self._reg("zeta_d", "ζ_d", "自主创新系数", 0.01, 0.10, 0.03, "structural")
        self._reg("phi_rd", "φ_rd", "研发投入弹性", 0.3, 0.8, 0.5, "structural")
        self._reg("chi_absorb_tech", "χ_tech", "技术吸收系数", 0.02, 0.20, 0.08, "structural")
        self._reg("xi_gap", "ξ", "技术差距弹性", 0.3, 1.5, 0.8, "structural")
        self._reg("G_critical", "G_crit", "追赶陷阱临界差距", 3.0, 20.0, 8.0, "structural")
        self._reg("lambda_trap", "λ_trap", "陷阱衰减速度", 0.1, 1.0, 0.3, "structural")
        self._reg("phi_trap", "φ_trap", "依赖陷阱强度", 0.0, 0.5, 0.2, "structural")

        # 国际
        self._reg("theta_p", "θ_p", "资源价格均值回复速度", 0.1, 0.5, 0.3, "structural")
        self._reg("sigma_p", "σ_p", "资源价格波动率", 0.05, 0.40, 0.15, "structural")
        self._reg("block_max", "block_max", "最大技术封锁溢价", 0.0, 0.9, 0.5, "structural")
        self._reg("fdi_sensitivity", "FDI_sens", "外资对回报率敏感度", 0.5, 3.0, 1.5, "structural")

        # 福利
        # v3.7: 默认值从3.0降至1.5, 处于Atkinson(1970)不平等厌恶参数标准范围[0.5,2.0]内,
        # 避免eps>2时c^(1-eps)=c^(-k)对接近零的福利值数学上过度敏感
        self._reg("epsilon_welfare", "ε_welfare", "福利分量替代弹性(Atkinson不平等厌恶)", 0.5, 2.5, 1.5, "structural")
        self._reg("data_depreciation", "δ_data", "数据折旧率", 0.05, 0.30, 0.15, "structural")

        # v2.0 异质性数据
        self._reg("phi_prod_data", "φ_prod", "生产者数据生成率", 0.01, 0.10, 0.04, "structural")
        self._reg("phi_cons_data", "φ_cons", "消费者数据生成率", 0.02, 0.15, 0.06, "structural")
        self._reg("gamma_cross_data", "γ_cross", "消费者→生产者交叉弹性", 0.0, 0.5, 0.2, "structural")

        # v2.0 知识扩散
        self._reg("chi_spont", "χ_spont", "自发扩散系数", 0.01, 0.10, 0.04, "structural")
        self._reg("eta_diff", "η_diff", "人力资本对扩散敏感度", 0.5, 3.0, 1.5, "structural")

        # v2.0 数字依附
        self._reg("lambda_dep", "λ_dep", "依附对自主创新抑制强度", 0.0, 1.0, 0.5, "structural")

        # ============ L2: 策略参数 ============
        # 通用
        self._reg("tau_w", "τ_w", "再分配力度", 0.0, 0.5, 0.15, "strategy")
        self._reg("kappa", "κ", "规划强度", 0.0, 1.0, 0.3, "strategy")
        self._reg("pub_infra_invest", "pub_infra", "公共数字基础设施投资占比", 0.0, 0.3, 0.1, "strategy")
        self._reg("social_coverage", "social_cov", "社会保障覆盖率", 0.0, 1.0, 0.5, "strategy")
        self._reg("training_invest", "train_inv", "培训投入占GDP比", 0.0, 0.05, 0.01, "strategy")
        self._reg("kappa_diff", "κ_diff", "[⚠已废弃] 知识扩散政策力度——请改用diff_policy_intensity", 0.0, 0.10, 0.03, "strategy")

        # 策略A：渐进积累
        self._reg("A_tau_min", "τ_min", "最低提取率", 0.02, 0.10, 0.05, "strategy")
        self._reg("A_tau_max", "τ_max", "最高提取率", 0.10, 0.30, 0.20, "strategy")
        self._reg("A_theta_elasticity", "θ_elas", "提取弹性", 0.1, 1.0, 0.5, "strategy")
        self._reg("A_safety_margin", "safety", "工业资本安全边际", 0.5, 0.9, 0.7, "strategy")
        self._reg("A_phi_d_premium", "φ_d_prem", "数字投资额外权重", 0.0, 0.2, 0.1, "strategy")

        # 策略B：跳跃积累
        self._reg("B_g_target", "g_target", "数字部门目标增长率", 0.10, 0.40, 0.25, "strategy")
        self._reg("B_tau_ceiling", "τ_ceil", "名义提取上限", 0.20, 0.60, 0.40, "strategy")
        self._reg("B_tau_actual_max", "τ_act_max", "实际可能提取上限", 0.30, 0.80, 0.60, "strategy")
        self._reg("B_industry_tolerance", "ind_tol", "工业萎缩容忍度", 0.3, 0.7, 0.5, "strategy")
        self._reg("B_feedback_trigger", "fb_trig", "反哺触发阈值(DL_d)", 0.3, 0.7, 0.5, "strategy")
        self._reg("B_feedback_ratio", "fb_ratio", "反哺比例", 0.05, 0.30, 0.15, "strategy")
        self._reg("B_phi_d_jump", "φ_d_jump", "跳跃期数字投资份额", 0.4, 0.8, 0.6, "strategy")

        # 策略C：替代积累
        self._reg("C_resource_export_intensity", "res_exp", "资源出口强度", 0.3, 0.9, 0.6, "strategy")
        self._reg("C_fdi_openness", "fdi_open", "外资开放度", 0.1, 0.8, 0.4, "strategy")
        self._reg("C_tech_import_preference", "tech_imp", "技术进口偏好", 0.3, 0.9, 0.6, "strategy")
        self._reg("C_indigenous_rd_ratio", "indig_rd", "自主创新投入", 0.01, 0.10, 0.03, "strategy")
        self._reg("C_profit_repat_limit", "profit_rep", "利润汇出限制", 0.1, 0.7, 0.3, "strategy")
        self._reg("C_foreign_debt_ceiling", "debt_ceil", "外债上限(占GDP)", 0.2, 1.0, 0.5, "strategy")

        # ============ L3: 场景参数 ============
        self._reg("g_world", "g_world", "国际技术前沿增长率", 0.01, 0.12, 0.03, "scenario")
        self._reg("mu_p_resource", "μ_p", "长期资源均价水平", 0.3, 1.5, 1.0, "scenario")
        self._reg("world_digital_demand", "world_dem", "国际数字需求水平", 0.3, 2.0, 1.0, "scenario")
        self._reg("fdi_availability", "fdi_avail", "外资可得性", 0.1, 2.0, 1.0, "scenario")
        self._reg("R_initial_ratio", "R0_ratio", "初始资源存量比", 0.2, 1.0, 0.8, "scenario")
        self._reg("institution_capacity", "inst_cap", "制度执行能力", 0.3, 1.0, 0.6, "scenario")

        # v2.1 新增策略参数
        self._reg("diff_policy_intensity", "diff_pol", "知识扩散政策强度(替代默认kappa_diff)", 0.0, 0.15, 0.05, "strategy")

        # ============ L4: 元参数 ============
        self._reg("fracture_threshold_mult", "frac_mult", "断裂阈值全局乘数", 0.7, 1.3, 1.0, "meta")
        self._reg("cascade_strength", "cascade", "断裂级联强度", 0.05, 0.30, 0.15, "meta")

        # v3.5: 混合策略互斥矩阵参数
        self._reg("incompat_AB_resource", "M_AB_res", "A↔B资源争夺互斥强度", 0.0, 0.6, 0.35, "meta")
        self._reg("incompat_AB_institution", "M_AB_inst", "A↔B制度冲突互斥强度", 0.0, 0.5, 0.25, "meta")
        self._reg("incompat_AB_external", "M_AB_ext", "A↔B外部矛盾互斥强度", 0.0, 0.4, 0.10, "meta")
        self._reg("incompat_BC_resource", "M_BC_res", "B↔C资源争夺互斥强度", 0.0, 0.3, 0.10, "meta")
        self._reg("incompat_BC_institution", "M_BC_inst", "B↔C制度冲突互斥强度", 0.0, 0.5, 0.30, "meta")
        self._reg("incompat_BC_external", "M_BC_ext", "B↔C外部矛盾互斥强度", 0.0, 0.6, 0.35, "meta")
        self._reg("incompat_AC_resource", "M_AC_res", "A↔C资源争夺互斥强度", 0.0, 0.4, 0.15, "meta")
        self._reg("incompat_AC_institution", "M_AC_inst", "A↔C制度冲突互斥强度", 0.0, 0.4, 0.20, "meta")
        self._reg("incompat_AC_external", "M_AC_ext", "A↔C外部矛盾互斥强度", 0.0, 0.4, 0.15, "meta")
        self._reg("incompat_moderation_inst", "M_mod_inst", "制度能力对互斥的缓和系数", 0.0, 0.5, 0.30, "meta")
        self._reg("incompat_amplify_pressure", "M_amp_press", "外部压力对互斥的放大系数", 0.0, 0.4, 0.20, "meta")

        # v3.5: 金融维度参数
        self._reg("capital_flight_sensitivity", "cap_flight", "资本外逃敏感度", 0.0, 0.5, 0.15, "structural")
        self._reg("asset_price_volatility", "asset_vol", "数字资产价格波动率", 0.05, 0.50, 0.25, "structural")
        self._reg("credit_bias_digital", "cred_bias_d", "信贷数字部门偏好乘数", 0.8, 1.8, 1.3, "structural")
        self._reg("credit_bias_industry", "cred_bias_i", "信贷工业部门偏好乘数", 0.5, 1.2, 0.8, "structural")
        self._reg("housing_wealth_share", "house_wealth", "房产在家庭财富中的基准占比", 0.2, 0.7, 0.45, "structural")

        # v3.6: 平台垄断参数
        self._reg("anti_monopoly_policy", "anti_mono", "反垄断执法力度", 0.0, 1.0, 0.2, "strategy")
        self._reg("data_portability_mandate", "data_port", "数据可携带权强度", 0.0, 1.0, 0.1, "strategy")
        self._reg("platform_interoperability", "plat_interop", "平台互操作性要求", 0.0, 1.0, 0.1, "strategy")

        # v3.6: 初始条件参数（用于校准和国家设定）
        self._reg("IC_K_i", "IC_K_i", "初始工业资本存量", 50.0, 800.0, 300.0, "structural")
        self._reg("IC_tech_gap_initial", "IC_G0", "初始技术差距", 1.5, 10.0, 3.0, "structural")
        self._reg("IC_gini_initial", "IC_gini0", "初始基尼系数", 0.15, 0.55, 0.35, "structural")

        # v3.6: 内生人口参数
        self._reg("IC_fertility_rate", "IC_fert", "初始总和生育率", 1.1, 6.0, 2.5, "structural")
        self._reg("IC_life_expectancy", "IC_life_exp", "初始预期寿命(年)", 40.0, 85.0, 65.0, "structural")
        self._reg("IC_urbanization_rate", "IC_urban", "初始城镇化率", 0.10, 0.90, 0.35, "structural")
        self._reg("IC_total_population", "IC_pop", "初始总人口(万人)", 100.0, 150000.0, 7000.0, "structural")
        self._reg("migration_cost_index", "mig_cost", "劳动力空间流动摩擦成本", 0.0, 0.8, 0.30, "structural")

        # v3.6: 空间维度参数
        self._reg("coastal_productivity_premium", "coast_prem", "沿海/城市生产率溢价", 0.05, 0.50, 0.20, "structural")
        self._reg("agglomeration_strength", "agglom_str", "集聚经济强度", 0.0, 0.30, 0.15, "structural")
        self._reg("congestion_threshold", "cong_thresh", "拥挤成本触发阈值(沿海人口占比)", 0.5, 0.8, 0.60, "structural")
        self._reg("IC_coastal_industry_share", "IC_coast_i", "初始沿海工业份额", 0.3, 0.9, 0.60, "structural")
        self._reg("IC_coastal_digital_infra_share", "IC_coast_di", "初始沿海数字infra份额", 0.4, 1.0, 0.75, "structural")
        self._reg("IC_coastal_digital_app_share", "IC_coast_da", "初始沿海数字app份额", 0.4, 1.0, 0.80, "structural")
        self._reg("IC_coastal_population_share", "IC_coast_pop", "初始沿海人口份额", 0.15, 0.80, 0.45, "structural")
        self._reg("IC_urban_rural_productivity_ratio", "IC_urb_rur", "初始城乡生产率比", 1.2, 6.0, 2.5, "structural")

        # v3.6: 政策执行折扣参数
        self._reg("bureaucracy_complexity_penalty", "bur_comp", "官僚协调复杂度惩罚系数", 0.0, 0.3, 0.15, "structural")
        self._reg("policy_execution_base_efficiency", "pol_exec", "政策执行基础效率", 0.5, 1.0, 0.75, "structural")

        # ── v3.7.1: 可配置的硬编码系数 ──

        # 生产系数
        self._reg("M_needed_ratio", "M_need", "工业原材料需求/资本比", 0.02, 0.30, 0.1, "structural")
        self._reg("M_avail_ratio", "M_avail", "基础部门原材料可供应比例", 0.1, 0.6, 0.3, "structural")
        self._reg("HW_needed_ratio", "HW_need", "infra硬件需求/资本比", 0.02, 0.20, 0.08, "structural")
        self._reg("HW_avail_ratio_i", "HW_avail_i", "工业硬件供应比例", 0.05, 0.50, 0.2, "structural")
        self._reg("HW_avail_floor", "HW_floor", "硬件可用性下限", 0.01, 0.5, 0.1, "structural")
        self._reg("import_penalty_factor", "imp_pen", "进口依赖惩罚系数", 0.1, 0.8, 0.4, "structural")
        self._reg("d_app_users_per_L", "dapp_uL", "app层每单位劳动力对应消费者", 5.0, 50.0, 15.0, "structural")
        self._reg("d_app_users_min", "dapp_umin", "app层网络效应最小用户", 10.0, 200.0, 50.0, "structural")
        self._reg("d_app_net_effect_cap", "dapp_netcap", "app层网络效应增益上限", 0.5, 5.0, 2.5, "structural")
        self._reg("d_app_min_A_effective", "dapp_Amineff", "app层infra约束下最低效率", 0.3, 0.9, 0.6, "structural")
        self._reg("s_low_users_per_L", "slow_uL", "低端服务每单位劳动力对应消费者", 3.0, 30.0, 10.0, "structural")
        self._reg("s_low_users_min", "slow_umin", "低端服务网络效应最小用户", 10.0, 100.0, 30.0, "structural")
        self._reg("s_low_net_effect_cap", "slow_netcap", "低端服务网络效应增益上限", 0.2, 3.0, 1.0, "structural")
        self._reg("s_low_net_effect_weight", "slow_netw", "低端服务网络效应权重", 0.1, 0.6, 0.3, "structural")
        self._reg("s_high_density_gain_factor", "shigh_dgf", "高技能服务数字密度增益系数", 0.1, 0.8, 0.40, "structural")
        self._reg("s_low_platform_sat_start", "plat_sat_start", "平台赋能饱和起始DL", 0.5, 0.95, 0.8, "structural")
        self._reg("s_low_platform_sat_decay", "plat_sat_decay", "平台赋能饱和衰减率", 0.1, 0.6, 0.3, "structural")
        self._reg("s_low_platform_sat_range", "plat_sat_range", "平台赋能饱和过渡区间", 0.05, 0.5, 0.2, "structural")
        self._reg("s_low_platform_sat_floor", "plat_sat_floor", "平台赋能饱和下限", 0.2, 0.8, 0.5, "structural")
        self._reg("i_density_gain_factor", "i_dgf", "工业数字密度增益系数", 0.1, 1.0, 0.5, "structural")

        # 积累系数
        self._reg("gov_consumption_share", "gov_share", "政府消费/GDP", 0.03, 0.25, 0.1, "structural")
        self._reg("accumulation_fund_share", "acc_fund_share", "积累基金/总剩余", 0.5, 0.95, 0.8, "structural")
        self._reg("credit_ext_dep_factor", "cred_extdep", "外部依赖对信贷可得性惩罚", 0.1, 0.8, 0.4, "structural")
        self._reg("credit_availability_min", "cred_min", "信贷可得性下限", 0.1, 0.5, 0.25, "structural")
        self._reg("absorb_DL_bonus_factor", "abs_DL_bonus", "吸收能力DL加成系数", 0.1, 1.0, 0.5, "structural")
        self._reg("absorb_H_reference", "abs_H_ref", "吸收能力H参考值", 0.2, 1.0, 0.5, "structural")
        self._reg("digital_share_i", "dig_share_i", "工业投资中数字密度占比", 0.1, 0.5, 0.3, "structural")
        self._reg("digital_share_b", "dig_share_b", "基础部门投资中数字密度占比", 0.05, 0.3, 0.15, "structural")
        self._reg("digital_share_other", "dig_share_other", "其他部门投资中数字密度占比", 0.2, 0.7, 0.5, "structural")
        self._reg("obsolescence_delta_threshold", "obs_delta_thresh", "技术淘汰触发折旧阈值", 0.02, 0.10, 0.05, "structural")
        self._reg("K_floor", "K_floor", "资本存量下限", 0.001, 0.1, 0.01, "structural")
        self._reg("L_floor", "L_floor", "劳动力下限", 0.1, 2.0, 0.5, "structural")

        # 劳动力市场(可配置系数)
        self._reg("informal_wage_ratio", "inf_wage_ratio", "非正式工资/基础部门工资", 0.3, 0.9, 0.6, "structural")
        self._reg("informal_ratio_saturation", "inf_sat_ratio", "非正式就业饱和阈值", 0.1, 0.5, 0.25, "structural")
        self._reg("saturated_back_to_base_ratio", "sat_back_base", "饱和时回流基础部门比例", 0.1, 0.6, 0.4, "structural")
        self._reg("labor_surplus_wage_penalty_max", "lab_surp_wage_max", "劳动力过剩对工资最大惩罚", 0.05, 0.4, 0.2, "structural")
        self._reg("labor_surplus_wage_coefficient", "lab_surp_wage_coef", "劳动力过剩工资惩罚系数", 0.05, 0.3, 0.15, "structural")
        self._reg("informal_reflow_unsaturated", "inf_reflow_low", "未饱和时非正式回流率", 0.01, 0.15, 0.05, "structural")
        self._reg("informal_reflow_near_sat", "inf_reflow_mid", "接近饱和时非正式回流率", 0.01, 0.10, 0.03, "structural")
        self._reg("informal_reflow_saturated", "inf_reflow_high", "饱和时非正式回流率", 0.005, 0.05, 0.015, "structural")
        self._reg("informal_reflow_formal_success", "inf_reflow_succ", "非正式回流正式成功率", 0.1, 0.6, 0.3, "structural")
        self._reg("digital_skill_obsolescence_rate", "dig_skill_obs", "数字技能年化贬值率", 0.02, 0.15, 0.06, "structural")
        self._reg("industrial_skill_erosion_rate", "ind_skill_ero", "工业技能年化侵蚀率", 0.01, 0.08, 0.025, "structural")
        self._reg("industrial_skill_erosion_max", "ind_skill_ero_max", "工业技能最大侵蚀比例", 0.1, 0.5, 0.30, "structural")
        self._reg("skill_selection_base", "skill_sel_base", "技能选择基础强度", 0.1, 0.5, 0.3, "structural")
        self._reg("skill_selection_match_factor", "skill_sel_match", "技能选择匹配因子", 0.3, 0.9, 0.7, "structural")
        self._reg("H_bounds_min", "H_min", "人力资本下限", 0.1, 0.5, 0.2, "structural")
        self._reg("H_bounds_max", "H_max", "人力资本上限", 2.0, 5.0, 3.0, "structural")
        self._reg("H_flow_damping", "H_damp", "劳动力流动H变化阻尼系数", 0.2, 0.8, 0.5, "structural")

        # 国际系数
        self._reg("rer_resource_share_coefficient", "rer_coef", "汇率-资源出口份额系数", 0.3, 2.0, 0.8, "structural")
        self._reg("digital_export_share", "dig_exp_share", "数字出口占数字产出比例", 0.05, 0.5, 0.2, "structural")
        self._reg("fdi_inflow_gdp_share", "fdi_gdp_share", "FDI流入基准占GDP比例", 0.01, 0.15, 0.05, "structural")
        self._reg("fdi_outflow_rate", "fdi_out_rate", "FDI利润汇出率", 0.02, 0.20, 0.08, "structural")
        self._reg("tech_import_gdp_share", "tech_imp_gdp", "技术进口基准占GDP比例", 0.02, 0.25, 0.1, "structural")
        self._reg("blockade_sigmoid_steep", "block_sig_steep", "封锁Sigmoid函数陡度", 2.0, 10.0, 5.0, "structural")
        self._reg("blockade_sigmoid_mid", "block_sig_mid", "封锁Sigmoid函数中点", 1.0, 6.0, 3.0, "structural")
        self._reg("resource_export_price_factor", "res_exp_price", "资源出口价格因子", 0.2, 0.8, 0.5, "structural")

        # 福利系数
        self._reg("welfare_consumption_exponent", "welf_cons_exp", "福利消费指数幂", 0.3, 0.7, 0.5, "structural")
        self._reg("welfare_employment_exponent", "welf_emp_exp", "福利就业指数幂", 0.1, 0.5, 0.3, "structural")
        self._reg("welfare_skill_exponent", "welf_skill_exp", "福利技能指数幂", 0.1, 0.4, 0.2, "structural")
        self._reg("consumption_propensity", "cons_prop", "家庭消费倾向", 0.7, 0.95, 0.85, "structural")
        self._reg("unemployment_cost_rate", "u_cost_rate", "失业者人均年代价系数", 1.0, 8.0, 3.0, "structural")

        # 策略A 内生成本(补充)
        self._reg("protection_excess_coefficient", "prot_excess_coef", "保护过剩效率损失系数", 0.02, 0.20, 0.08, "strategy")
        self._reg("time_tax_accumulation_threshold", "time_tax_thresh", "累积滞后触发阈值", 0.03, 0.30, 0.1, "strategy")
        self._reg("time_tax_penalty_rate", "time_tax_rate", "累积滞后惩罚率", 0.01, 0.08, 0.03, "strategy")
        self._reg("vested_interest_duration", "vested_dur", "既得利益触发年限", 5.0, 30.0, 15.0, "strategy")
        self._reg("vested_interest_coefficient", "vested_coef", "既得利益强度系数", 0.005, 0.03, 0.01, "strategy")
        self._reg("middle_income_DD_threshold", "mid_inc_DD", "中间收入陷阱数字密度阈值", 0.1, 0.5, 0.3, "strategy")
        self._reg("middle_income_G_threshold", "mid_inc_G", "中间收入陷阱技术差距阈值", 2.0, 8.0, 4.0, "strategy")
        self._reg("middle_income_duration", "mid_inc_dur", "中间收入陷阱持续年限", 5.0, 20.0, 10.0, "strategy")
        self._reg("A_g_i_max", "A_gimax", "策略A工业GDP份额上限", 0.3, 0.7, 0.5, "strategy")
        self._reg("A_protection_excess_floor", "A_prot_floor", "策略A保护过剩产出下限", 0.7, 0.95, 0.85, "strategy")
        self._reg("A_time_tax_natural_buffer", "A_time_buf", "策略A自然追赶缓冲", 0.0, 0.05, 0.01, "strategy")
        self._reg("A_vested_interest_max_intensity", "A_vest_max", "策略A既得利益最大强度", 0.2, 0.8, 0.5, "strategy")
        self._reg("A_vested_interest_invest_floor", "A_vest_invfloor", "策略A既得利益投资下限", 0.3, 0.8, 0.5, "strategy")
        self._reg("A_trap_penalty_rate", "A_trap_pen", "策略A中间收入陷阱惩罚率", 0.005, 0.03, 0.01, "strategy")
        self._reg("A_trap_K_floor", "A_trap_Kfloor", "策略A陷阱资本下限", 0.8, 0.95, 0.90, "strategy")
        self._reg("A_trap_recovery_rate", "A_trap_recov", "策略A陷阱恢复速率乘数", 1.0, 4.0, 2.0, "strategy")
        self._reg("A_extraction_floor_ratio", "A_ext_floor", "策略A安全边际下提取率下限比", 0.1, 0.5, 0.3, "strategy")

        # 策略B 内生成本(补充)
        self._reg("B_tau_baseline", "B_tau_base", "策略B就业破坏触发提取率阈值", 0.15, 0.45, 0.30, "strategy")
        self._reg("B_unemployment_coefficient", "B_unemp_coef", "策略B超额提取→失业转化率", 0.02, 0.12, 0.05, "strategy")
        self._reg("B_disposable_income_base", "B_disp_base", "策略B基准可支配收入占比", 0.55, 0.85, 0.70, "strategy")
        self._reg("B_disposable_income_tau_sens", "B_disp_sens", "策略B可支配收入对提取率敏感度", 0.2, 0.8, 0.5, "strategy")
        self._reg("B_consumption_drag_coefficient", "B_cons_drag", "策略B消费萎缩→产出拖累系数", 0.05, 0.30, 0.15, "strategy")
        self._reg("B_consumption_output_floor", "B_cons_floor", "策略B消费萎缩产出下限", 0.7, 0.95, 0.85, "strategy")
        self._reg("B_bubble_trigger_r", "B_bub_trig_r", "策略B泡沫触发利润率阈值", 0.10, 0.40, 0.25, "strategy")
        self._reg("B_bubble_pressure_sensitivity", "B_bub_sens", "策略B利润率→泡沫压力敏感度", 0.1, 0.6, 0.3, "strategy")
        self._reg("B_bubble_burst_base_prob", "B_bub_prob", "策略B泡沫破裂基础概率", 0.001, 0.01, 0.003, "strategy")
        self._reg("B_feedback_base_success", "B_fb_base", "策略B反哺基础成功概率", 0.6, 0.95, 0.85, "strategy")
        self._reg("B_feedback_profit_penalty_high", "B_fb_pen_hi", "策略B高利润率惩罚(r>0.30)", 0.05, 0.40, 0.20, "strategy")
        self._reg("B_feedback_profit_penalty_mid", "B_fb_pen_mid", "策略B中等利润率惩罚(r>0.20)", 0.02, 0.20, 0.10, "strategy")
        self._reg("B_bubble_decay_rate", "B_bub_decay", "策略B泡沫自然衰减率", 0.005, 0.03, 0.01, "strategy")
        self._reg("B_feedback_min_success_prob", "B_fb_min_p", "策略B反哺最低成功概率", 0.05, 0.3, 0.15, "strategy")
        self._reg("B_feedback_max_success_prob", "B_fb_max_p", "策略B反哺最高成功概率", 0.8, 1.0, 0.95, "strategy")
        self._reg("B_feedback_execution_floor", "B_fb_exec_floor", "策略B反哺执行下限", 0.1, 0.5, 0.3, "strategy")
        self._reg("B_feedback_resistance_coefficient", "B_fb_resist_coef", "策略B反哺抵制系数", 0.05, 0.3, 0.15, "strategy")
        self._reg("B_panic_extraction_G_threshold", "B_panic_G", "策略B恐慌提取G阈值", 4.0, 15.0, 8.0, "strategy")
        self._reg("B_panic_extraction_K_ratio", "B_panic_Kr", "策略B恐慌提取K比阈值", 0.3, 0.8, 0.6, "strategy")
        self._reg("B_panic_extraction_max_bonus", "B_panic_bonus", "策略B恐慌提取最大额外比例", 0.05, 0.3, 0.15, "strategy")
        self._reg("B_industry_tau_floor", "B_tau_floor", "策略B工业容忍提取下限", 0.1, 0.5, 0.3, "strategy")
        self._reg("B_soft_landing_inst_cap", "B_soft_inst", "策略B软着陆制度能力阈值", 0.5, 0.9, 0.7, "strategy")
        self._reg("B_soft_landing_bp_threshold", "B_soft_bp", "策略B软着陆泡沫压力阈值", 2.0, 5.0, 3.0, "strategy")
        self._reg("B_soft_landing_bp_decay", "B_soft_decay", "策略B软着陆泡沫衰减率", 0.01, 0.05, 0.02, "strategy")
        self._reg("B_d_infra_severity_scale", "B_sev_infra", "策略B破裂infra冲击倍数", 0.3, 1.0, 0.6, "strategy")
        self._reg("B_d_app_severity_scale", "B_sev_app", "策略B破裂app冲击倍数", 0.5, 2.0, 1.2, "strategy")

        # 策略C 内生成本(补充)
        self._reg("C_resource_shock_tau_mult", "C_rs_tau", "策略C资源冲击提取率乘数", 0.3, 0.9, 0.6, "strategy")
        self._reg("C_resource_shock_phi_mult", "C_rs_phi", "策略C资源冲击投资乘数", 0.5, 0.9, 0.7, "strategy")
        self._reg("C_fdi_shock_effective_mult", "C_fdi_eff", "策略C FDI冲击有效乘数", 0.1, 0.5, 0.3, "strategy")
        self._reg("C_d_app_max_share", "C_dapp_max", "策略C数字app最大投资份额", 0.3, 0.7, 0.55, "strategy")
        self._reg("C_phi_b_floor", "C_phib_floor", "策略C基础部门最小投资份额", 0.01, 0.10, 0.03, "strategy")

        # 政治经济学
        self._reg("support_smoothing_factor", "supp_smooth", "支持度指数平滑因子", 0.8, 0.95, 0.85, "structural")
        self._reg("regime_adjust_B", "reg_adj_B", "策略B支持度调整乘数", 0.85, 1.0, 0.95, "structural")
        self._reg("regime_adjust_C", "reg_adj_C", "策略C支持度调整乘数", 1.0, 1.1, 1.02, "structural")
        self._reg("protest_trend_window", "prot_trend_win", "抗议风险趋势检测窗口", 5, 20, 10, "structural")

        # 技术加权
        self._reg("A_world_infra_weight", "A_w_infra", "综合技术差距infra权重", 0.2, 0.6, 0.4, "structural")
        self._reg("A_world_app_weight", "A_w_app", "综合技术差距app权重", 0.4, 0.8, 0.6, "structural")

    def get(self, pid: str) -> ParamDef:
        return self._params[pid]

    def get_value(self, pid: str, overrides: Optional[Dict[str, float]] = None) -> float:
        if overrides and pid in overrides:
            return overrides[pid]
        return self._params[pid].default

    def structural_ids(self) -> List[str]:
        return [k for k, v in self._params.items() if v.layer == "structural"]

    def strategy_ids(self) -> List[str]:
        return [k for k, v in self._params.items() if v.layer == "strategy"]

    def scenario_ids(self) -> List[str]:
        return [k for k, v in self._params.items() if v.layer == "scenario"]

    def meta_ids(self) -> List[str]:
        return [k for k, v in self._params.items() if v.layer == "meta"]

    def all_ids(self) -> List[str]:
        return list(self._params.keys())

    def build_defaults(self, regime: str = "A") -> Dict[str, float]:
        """构建默认参数集."""
        params = {pid: v.default for pid, v in self._params.items()}
        # 不修改非活跃策略参数——保持默认值在定义域内
        return params

    def build_from_profile(self, profile_name: str, regime: str = "A",
                           scenario: str = None,
                           user_overrides: Dict[str, float] = None) -> Dict[str, float]:
        """v3.1: 从国情预设一站式构建完整参数集.

        参数加载优先级（从低到高）：
        1. build_defaults(regime)        — 策略/结构默认参数
        2. 国情预设覆盖                  — 初始条件
        3. 国际场景覆盖                  — 外部环境
        4. user_overrides               — 用户自定义（最高优先级）

        Args:
            profile_name: 国情预设名称
            regime: 积累策略 A/B/C/M
            scenario: 国际场景名称（可选）
            user_overrides: 用户自定义参数覆盖（可选）

        Returns:
            完整的参数字典
        """
        from .country_profiles import build_params_with_profile
        return build_params_with_profile(
            profile_name=profile_name,
            regime=regime,
            scenario=scenario,
            user_overrides=user_overrides,
        )

    def build_from_yaml(self, config_path: str,
                        profile_name: str = None,
                        regime: str = "A",
                        scenario: str = None,
                        user_overrides: Dict[str, float] = None) -> Dict[str, float]:
        """v3.7: 从YAML配置文件构建参数集.

        YAML中的参数会覆盖schema中的硬编码默认值。
        如果YAML中缺少某参数，回退到schema默认值。

        参数加载优先级（从低到高）：
        1. schema硬编码默认值                — 作为安全网
        2. YAML配置文件中的参数              — 覆盖默认值
        3. 国情预设覆盖 (从YAML或代码)       — 初始条件
        4. 国际场景覆盖 (从YAML或代码)       — 外部环境
        5. user_overrides                   — 用户自定义（最高优先级）

        Args:
            config_path: YAML配置文件路径
            profile_name: 国情预设名称
            regime: 积累策略 A/B/C/M
            scenario: 国际场景名称（可选）
            user_overrides: 用户自定义参数覆盖（可选）

        Returns:
            完整的参数字典
        """
        from .yaml_loader import build_params_from_yaml

        # 1. 从YAML加载参数
        yaml_params = build_params_from_yaml(
            config_path=config_path,
            profile_name=profile_name,
            regime=regime,
            scenario=scenario,
            user_overrides=user_overrides,
        )

        # 2. 用schema默认值填充YAML中缺失的参数（安全网）
        defaults = self.build_defaults(regime)
        for pid, default_val in defaults.items():
            if pid not in yaml_params:
                yaml_params[pid] = default_val

        return yaml_params

    def build_from_overrides(self, overrides: Dict[str, float],
                              regime: str = "A") -> Dict[str, float]:
        """构建参数集并应用覆盖."""
        params = self.build_defaults(regime)
        params.update(overrides)
        return params

    def domain_for(self, pid: str) -> Tuple[float, float]:
        return self._params[pid].domain

    def is_log(self, pid: str) -> bool:
        return self._params[pid].log_scale

    def validate(self, params: Dict[str, float]) -> List[str]:
        """验证参数是否在定义域内，返回违规列表."""
        violations = []
        for pid, pdef in self._params.items():
            if pid in params:
                lo, hi = pdef.domain
                val = params[pid]
                if val < lo or val > hi:
                    violations.append(f"{pid}={val:.4f} ∉ [{lo},{hi}]")
        return violations

    def sample_lhs(self, n: int, param_ids: List[str],
                    rng: np.random.Generator) -> np.ndarray:
        """拉丁超立方采样.

        修复（2026-10-08）：先前 `qmc.LatinHypercube(d=d)` 未接收 `rng`，
        导致同一 seed 两次调用得到不同参数矩阵、抽样点无法复现
        （见《新增代码核对与待确认清单》第一项）。现把 `rng` 传给采样器。
        注意：v3.8 归档的 cost_lhs 批次是用旧行为生成的，其逐运行参数
        未落盘，故那 450 条记录无法从种子恢复；此后新批次可复现。
        """
        from scipy.stats import qmc
        d = len(param_ids)
        sampler = qmc.LatinHypercube(d=d, seed=rng)
        sample = sampler.random(n=n)

        for i, pid in enumerate(param_ids):
            lo, hi = self.domain_for(pid)
            if self.is_log(pid):
                sample[:, i] = np.exp(np.log(lo) + sample[:, i] * (np.log(hi) - np.log(lo)))
            else:
                sample[:, i] = lo + sample[:, i] * (hi - lo)

        return sample

    def __len__(self) -> int:
        return len(self._params)


# 全局单例
_schema_instance: Optional[ParameterSchema] = None


def get_schema() -> ParameterSchema:
    global _schema_instance
    if _schema_instance is None:
        _schema_instance = ParameterSchema()
    return _schema_instance
