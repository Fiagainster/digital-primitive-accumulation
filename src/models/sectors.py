"""v3.3 四部门模型 —— 基础、工业、数字、服务业."""

import numpy as np
from ..core.state import SystemState, SectorState, ALL_SECTORS
from ..params.schema import get_schema


class SectorModel:
    """四部门模型聚合器."""

    def __init__(self, params: dict):
        self.p = params
        self.schema = get_schema()

    def compute_all_production(self, state: SystemState):
        """并行计算四部门产出."""
        self._base_production(state)
        self._industry_production(state)
        self._digital_production(state)
        self._service_production(state)  # v3.3

    def _base_production(self, state: SystemState):
        s = state.sectors["b"]
        p = self.p

        # CES聚合
        sigma = p["sigma_b"]
        theta = p["theta_b"]
        eps = (sigma - 1) / sigma if sigma != 0 else 0.0
        if abs(eps) < 1e-10:
            F = s.K ** theta * (s.H * s.L) ** (1 - theta)
        else:
            F = (theta * s.K ** eps + (1 - theta) * (s.H * s.L) ** eps) ** (1 / eps)

        # 数字化增强
        dig_gain = 1 + p["alpha_b_dig"] * s.DL ** p["beta_b_dig"]

        # 资源约束
        R_ratio = max(0.01, state.sectors["b"].A)  # 用A_b代理资源存量
        resource_factor = R_ratio ** p["lambda_b"]

        s.Y = s.A * F * dig_gain * resource_factor
        s.profit = s.Y - s.W * s.L

    def _industry_production(self, state: SystemState):
        s = state.sectors["i"]
        p = self.p

        # Cobb-Douglas + 数字化 + 数字服务密度 + 原材料约束
        base = s.A * s.K ** p["alpha_i"] * (s.H * s.L) ** (1 - p["alpha_i"])
        dig_gain = 1 + p["alpha_i_dig"] * s.DL ** p["beta_i_dig"]

        # v2.2: 数字密度增强 — 嵌入数字技术的资本比例越高，生产效率越高
        density_gain = 1.0 + s.digital_density * 0.5  # 最高50%额外增益

        # 数字服务密度（防零）
        ds_density = max(s.DS, 0.01) / max(s.K, 0.01)
        ds_gain = ds_density ** p["gamma_i"]

        # 原材料约束
        M_needed = 0.1 * s.K  # 每单位资本需要0.1单位原材料
        M_avail = state.sectors["b"].Y * 0.3  # 30%基础部门产出可用作原材料
        material_factor = min(1.0, M_avail / max(M_needed, 0.01)) ** p["mu_i"]
        s.M_used = min(M_needed, M_avail)

        s.Y = base * dig_gain * density_gain * ds_gain * material_factor
        s.profit = s.Y - s.W * s.L

    def _digital_production(self, state: SystemState):
        s = state.sectors["d"]
        p = self.p

        # 硬件密度（防零基线）
        HW_needed = 0.05 * s.K
        HW_avail = max(state.sectors["i"].Y * 0.2, 0.1)  # 至少0.1单位硬件
        hw_density = max(min(HW_avail, HW_needed), 0.01) / max(s.K, 0.01)

        # 有效数据存量（防零基线）
        data_eff = max(state.data_assets.effective_stock, 1.0)
        data_factor = (data_eff / 1.0) ** p["epsilon_d"]

        # 网络效应
        users = max(0, s.L * 10)  # 每万就业 ≈ 10万用户
        users_min = 100.0
        net_effect = max(0, (users / users_min - 1)) if users >= users_min else 0.0
        net_gain = 1 + p["eta_network"] * min(net_effect, 2.0)

        base = s.A * s.K ** p["alpha_d"] * (s.H * s.L) ** (1 - p["alpha_d"])
        s.Y = base * (hw_density / 0.05) ** p["delta_d_hw"] * data_factor * net_gain
        s.profit = s.Y - s.W * s.L

    def _service_production(self, state: SystemState):
        """v3.3: 服务业生产函数.

        服务业与工业的关键差异：
        - 资本密集度低（α_s < α_i）——服务业更多依赖劳动力
        - 数字化增益中等——数字平台可提升服务效率（外卖/打车/电商）
        - 不受原材料约束——服务业不需要矿产资源
        - 人力资本敏感——教育水平直接影响服务质量
        - 网络效应——平台型服务的用户规模效应
        """
        s = state.sectors["s"]
        p = self.p

        # Cobb-Douglas：资本份额低（~25%），劳动份额高（~75%）
        base = s.A * s.K ** p["alpha_s"] * (s.H * s.L) ** (1 - p["alpha_s"])

        # 数字化增益：数字平台提升服务效率
        dig_gain = 1 + p["alpha_s_dig"] * s.DL ** p["beta_s_dig"]

        # 数字密度增强：数字支付/在线预订/智能调度等
        density_gain = 1.0 + s.digital_density * 0.35

        # 数字服务投入：服务业也使用数字工具
        ds_density = max(s.DS, 0.01) / max(s.K, 0.01)
        ds_gain = ds_density ** 0.10  # 弹性较低

        # 网络效应（平台型服务）
        users = max(0, s.L * 10)
        users_min = 50.0
        net_effect = max(0, (users / users_min - 1)) if users >= users_min else 0.0
        net_gain = 1 + p["eta_network"] * 0.5 * min(net_effect, 1.5)

        s.Y = base * dig_gain * density_gain * ds_gain * net_gain
        s.profit = s.Y - s.W * s.L

    def compute_wages(self, state: SystemState):
        """计算各部门工资（边际劳动产出 × 谈判力）."""
        for key in ALL_SECTORS:  # v3.3
            s = state.sectors[key]
            # MPL近似 = (1-alpha) * Y / L
            if key == "b":
                alpha_eff = self.p["theta_b"]
            elif key == "i":
                alpha_eff = self.p["alpha_i"]
            elif key == "d":
                alpha_eff = self.p["alpha_d"]
            else:  # "s"
                alpha_eff = self.p["alpha_s"]

            mpl = (1 - alpha_eff) * s.Y / max(s.L, 0.01)
            # 谈判力受失业率影响
            u_rate = state.unemployment_rate
            tightness = max(0.3, 1.0 - u_rate)
            bp = self.p["bp_base"] * tightness
            s.W = mpl * bp

    def compute_profit_rates(self, state: SystemState):
        """更新各部门利润率."""
        for key in ALL_SECTORS:  # v3.3
            s = state.sectors[key]
            s.r = s.profit / max(s.K, 0.01)

    def initial_state(self, state: SystemState):
        """设置各部门初始条件 —— 从参数读取，无参数时使用默认值(向后兼容).

        优先从 params 中读取 IC_* 系列参数（由国情预设设置），
        如果不存在则回退到硬编码默认值（制造业大国基准）。
        """
        p = self.p

        # ── 基础部门 ──
        state.sectors["b"] = SectorState(
            K=p.get("IC_K_b", 200.0),
            H=p.get("IC_H_b", 0.8),
            L=p.get("IC_L_b", 40.0),
            DL=p.get("IC_DL_b", 0.05),
            A=p.get("IC_A_b", 1.0),
            W=p.get("IC_W_b", 2.0),
            digital_density=p.get("IC_DD_b", 0.01),
        )
        # ── 工业部门 ──
        state.sectors["i"] = SectorState(
            K=p.get("IC_K_i", 300.0),
            H=p.get("IC_H_i", 1.0),
            L=p.get("IC_L_i", 30.0),
            DL=p.get("IC_DL_i", 0.10),
            A=p.get("IC_A_i", 1.2),
            W=p.get("IC_W_i", 3.0),
            digital_density=p.get("IC_DD_i", 0.03),
        )
        # ── 数字部门 ──
        state.sectors["d"] = SectorState(
            K=p.get("IC_K_d", 20.0),
            H=p.get("IC_H_d", 1.5),
            L=p.get("IC_L_d", 5.0),
            DL=p.get("IC_DL_d", 0.15),
            A=p.get("IC_A_d", 1.5),
            W=p.get("IC_W_d", 5.0),
            digital_density=p.get("IC_DD_d", 0.05),
        )
        # ── v3.3: 服务业部门 ──
        state.sectors["s"] = SectorState(
            K=p.get("IC_K_s", 50.0),
            H=p.get("IC_H_s", 0.6),
            L=p.get("IC_L_s", 15.0),
            DL=p.get("IC_DL_s", 0.05),
            A=p.get("IC_A_s", 0.8),
            W=p.get("IC_W_s", 1.5),
            digital_density=p.get("IC_DD_s", 0.02),
            sector_type="services",
        )

        # 记录初始资本（后续断裂检测和策略评估的基准）
        state.initial_K_i = state.sectors["i"].K
        state.initial_K_b = state.sectors["b"].K
        state.initial_K_d = state.sectors["d"].K
        state.initial_K_s = state.sectors["s"].K  # v3.3

        # 预置初始数据资产
        state.data_assets.producer_data = p.get("IC_data_producer", 1.0)
        state.data_assets.consumer_data = p.get("IC_data_consumer", 0.5)
        state.data_assets.public_data = p.get("IC_data_public", 0.3)

        # 初始数字服务流
        state.sectors["b"].DS = p.get("IC_DS_b", 0.5)
        state.sectors["i"].DS = p.get("IC_DS_i", 1.0)
        state.sectors["d"].DS = p.get("IC_DS_d", 2.0)
        state.sectors["s"].DS = p.get("IC_DS_s", 0.5)  # v3.3

        # 初始基尼系数
        state.gini = p.get("IC_gini_initial", 0.38)

        # 初始产出
        self.compute_all_production(state)
        self.compute_wages(state)
        self.compute_profit_rates(state)

        # 初始GDP
        state.gdp = sum(s.Y for s in state.sectors.values())
