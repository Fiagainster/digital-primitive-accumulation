"""v3.5 七部门模型 —— 基础、工业、数字infra、数字app、高技能服务、低端服务."""

import numpy as np
from ..core.state import SystemState, SectorState, ALL_SECTORS, DIGITAL_SECTORS, SERVICE_SECTORS


class SectorModel:
    """七部门模型聚合器 (v3.5: 数字拆分为infra+app, 服务业拆分为high+low)."""

    def __init__(self, params: dict):
        self.p = params

    def compute_all_production(self, state: SystemState):
        """并行计算六部门产出.

        v3.7 修复: 调整计算顺序——s_high依赖gdp_share_digital(需要其他部门产出已知),
        故将其放在最后; s_low依赖平均数字DL, 放到s_high之前.
        """
        self._base_production(state)
        self._industry_production(state)
        self._digital_infra_production(state)
        self._digital_app_production(state)
        self._service_low_production(state)
        self._service_high_production(state)

    def _base_production(self, state: SystemState):
        s = state.sectors["b"]
        p = self.p

        sigma = p["sigma_b"]
        theta = p["theta_b"]
        eps = (sigma - 1) / sigma if sigma != 0 else 0.0
        if abs(eps) < 1e-10:
            F = s.K ** theta * (s.H * s.L) ** (1 - theta)
        else:
            F = (theta * s.K ** eps + (1 - theta) * (s.H * s.L) ** eps) ** (1 / eps)

        dig_gain = 1 + p["alpha_b_dig"] * s.DL ** p["beta_b_dig"]

        R_ratio = max(0.01, state.sectors["b"].A)
        resource_factor = R_ratio ** p["lambda_b"]

        s.Y = s.A * F * dig_gain * resource_factor
        s.profit = s.Y - s.W * s.L

    def _industry_production(self, state: SystemState):
        s = state.sectors["i"]
        p = self.p

        base = s.A * s.K ** p["alpha_i"] * (s.H * s.L) ** (1 - p["alpha_i"])
        dig_gain = 1 + p["alpha_i_dig"] * s.DL ** p["beta_i_dig"]

        density_gain = 1.0 + s.digital_density * 0.5

        # 数字服务密度 — 来自两个数字子部门
        ds_total = state.sectors["d_infra"].DS + state.sectors["d_app"].DS
        ds_density = max(ds_total, 0.01) / max(s.K, 0.01)
        ds_gain = ds_density ** p["gamma_i"]

        M_needed = 0.1 * s.K
        M_avail = state.sectors["b"].Y * 0.3
        material_factor = min(1.0, M_avail / max(M_needed, 0.01)) ** p["mu_i"]
        s.M_used = min(M_needed, M_avail)

        s.Y = base * dig_gain * density_gain * ds_gain * material_factor
        s.profit = s.Y - s.W * s.L

    def _digital_infra_production(self, state: SystemState):
        """v3.5: 数字基础设施层 — 芯片、云、OS.

        特征：高资本密集(α≈0.55)、高硬件依赖、长回报周期、
        受制于外国技术的程度影响生产效率。
        """
        s = state.sectors["d_infra"]
        p = self.p

        # 硬件密度
        HW_needed = 0.08 * s.K
        HW_avail = max(state.sectors["i"].Y * 0.2, 0.1)
        hw_density = max(min(HW_avail, HW_needed), 0.01) / max(s.K, 0.01)

        # 进口依赖折扣：infra高度依赖外国技术 → import_dep越高效率越低
        import_penalty = 1.0 - 0.4 * s.import_dependency

        # Cobb-Douglas: 高资本份额
        base = s.A * s.K ** p["alpha_d_infra"] * (s.H * s.L) ** (1 - p["alpha_d_infra"])
        s.Y = base * (hw_density / 0.08) ** p["delta_d_infra_hw"] * import_penalty
        s.profit = s.Y - s.W * s.L

    def _digital_app_production(self, state: SystemState):
        """v3.5: 数字应用层 — 平台、AI应用.

        特征：中低资本密集(α≈0.30)、高人力资本依赖、
        强网络效应、受infra层技术水平约束。
        """
        s = state.sectors["d_app"]
        p = self.p

        # 有效数据存量
        data_eff = max(state.data_assets.effective_stock, 1.0)
        data_factor = (data_eff / 1.0) ** p["delta_d_app_data"]

        # 网络效应（应用层显著强于基础设施层）
        users = max(0, s.L * 15)
        users_min = 50.0
        net_effect = max(0, (users / users_min - 1)) if users >= users_min else 0.0
        net_gain = 1 + p["eta_network_app"] * min(net_effect, 2.5)

        # v3.5: infra→app技术约束
        # 应用层效率受基础设施层技术水平约束
        A_infra = state.sectors["d_infra"].A
        A_app_potential = s.A
        infra_constraint = 1.0 - p["xi_infra_to_app"] * max(0.0, 1.0 - A_infra / max(A_app_potential, 0.01))
        A_effective = A_app_potential * max(0.6, infra_constraint)

        base = A_effective * s.K ** p["alpha_d_app"] * (s.H * s.L) ** (1 - p["alpha_d_app"])
        s.Y = base * data_factor * net_gain
        s.profit = s.Y - s.W * s.L

    def _service_high_production(self, state: SystemState):
        """v3.5: 高技能服务业 — 金融、研发、咨询.

        特征：与数字部门紧密关联、高人力资本、中等资本密集。
        数字部门繁荣 → 高技能服务业同步增长。
        """
        s = state.sectors["s_high"]
        p = self.p

        base = s.A * s.K ** p["alpha_s_high"] * (s.H * s.L) ** (1 - p["alpha_s_high"])

        # 数字化增益
        dig_gain = 1 + p["alpha_s_dig"] * s.DL ** p["beta_s_dig"]

        # v3.5: 数字部门联动增益
        # 数字infra+app的产出增长 → 拉动高技能服务需求
        # v3.7 修复: 使用所有其他部门产出之和作为GDP代理, 避免初始化时state.gdp=0
        # 导致除以0.01分母而使得digital_link_gain膨胀250x的bug
        d_total_output = state.sectors["d_infra"].Y + state.sectors["d_app"].Y
        other_sectors_Y = sum(
            state.sectors[k].Y for k in ["b", "i", "d_infra", "d_app", "s_low"]
        )
        gdp_proxy = max(state.gdp, other_sectors_Y, 0.01)
        gdp_share_digital = d_total_output / gdp_proxy
        digital_link_gain = 1.0 + p["xi_s_digital_link"] * gdp_share_digital

        # 数字密度增强
        density_gain = 1.0 + s.digital_density * 0.40

        s.Y = base * dig_gain * digital_link_gain * density_gain
        s.profit = s.Y - s.W * s.L

    def _service_low_production(self, state: SystemState):
        """v3.5: 低端服务业 — 餐饮、零售、家政.

        特征：极低资本密集(α≈0.15)、低技能劳动力密集、
        受数字平台赋能（外卖/打车/电商提升匹配效率）。
        """
        s = state.sectors["s_low"]
        p = self.p

        base = s.A * s.K ** p["alpha_s_low"] * (s.H * s.L) ** (1 - p["alpha_s_low"])

        # 数字化增益
        dig_gain = 1 + p["alpha_s_dig"] * s.DL ** p["beta_s_dig"]

        # v3.5: 数字平台赋能 — 随数字部门DL增长而增强
        avg_digital_DL = 0.5 * (state.sectors["d_infra"].DL + state.sectors["d_app"].DL)
        platform_boost = 1.0 + p["platform_empowerment"] * avg_digital_DL
        # 饱和效应: DL超过0.8后边际递减
        if avg_digital_DL > 0.8:
            saturation = 1.0 - 0.3 * (avg_digital_DL - 0.8) / 0.2
            platform_boost = 1.0 + p["platform_empowerment"] * 0.8 * max(0.5, saturation)

        # 网络效应（平台型：外卖/打车用户规模→匹配效率）
        users = max(0, s.L * 10)
        users_min = 30.0
        net_effect = max(0, (users / users_min - 1)) if users >= users_min else 0.0
        net_gain = 1 + p.get("eta_network", 0.2) * 0.3 * min(net_effect, 1.0)

        s.Y = base * dig_gain * platform_boost * net_gain
        s.profit = s.Y - s.W * s.L

    def compute_wages(self, state: SystemState):
        """计算各部门工资（边际劳动产出 × 谈判力）."""
        alpha_map = {
            "b": self.p.get("theta_b", 0.35),
            "i": self.p["alpha_i"],
            "d_infra": self.p["alpha_d_infra"],
            "d_app": self.p["alpha_d_app"],
            "s_high": self.p["alpha_s_high"],
            "s_low": self.p["alpha_s_low"],
        }

        for key in ALL_SECTORS:
            s = state.sectors[key]
            alpha_eff = alpha_map.get(key, 0.25)
            mpl = (1 - alpha_eff) * s.Y / max(s.L, 0.01)
            u_rate = state.unemployment_rate
            tightness = max(0.3, 1.0 - u_rate)
            bp = self.p["bp_base"] * tightness
            s.W = mpl * bp

    def compute_profit_rates(self, state: SystemState):
        """更新各部门利润率."""
        for key in ALL_SECTORS:
            s = state.sectors[key]
            s.r = s.profit / max(s.K, 0.01)

    def initial_state(self, state: SystemState):
        """设置各部门初始条件 (v3.5: 六部门)."""
        p = self.p

        # ── 基础部门 ──
        state.sectors["b"] = SectorState(
            K=p.get("IC_K_b", 200.0), H=p.get("IC_H_b", 0.8),
            L=p.get("IC_L_b", 40.0), DL=p.get("IC_DL_b", 0.05),
            A=p.get("IC_A_b", 1.0), W=p.get("IC_W_b", 2.0),
            digital_density=p.get("IC_DD_b", 0.01),
            sector_type="goods",
        )
        # ── 工业部门 ──
        state.sectors["i"] = SectorState(
            K=p.get("IC_K_i", 300.0), H=p.get("IC_H_i", 1.0),
            L=p.get("IC_L_i", 30.0), DL=p.get("IC_DL_i", 0.10),
            A=p.get("IC_A_i", 1.2), W=p.get("IC_W_i", 3.0),
            digital_density=p.get("IC_DD_i", 0.03),
            sector_type="goods",
        )
        # ── v3.5: 数字基础设施层 ──
        state.sectors["d_infra"] = SectorState(
            K=p.get("IC_K_d_infra", 8.0), H=p.get("IC_H_d_infra", 1.6),
            L=p.get("IC_L_d_infra", 1.5), DL=p.get("IC_DL_d_infra", 0.08),
            A=p.get("IC_A_d_infra", 1.2), W=p.get("IC_W_d_infra", 6.0),
            digital_density=p.get("IC_DD_d_infra", 0.03),
            import_dependency=p.get("IC_import_dep_d_infra", 0.75),
            sector_type="digital_infra",
        )
        # ── v3.5: 数字应用层 ──
        state.sectors["d_app"] = SectorState(
            K=p.get("IC_K_d_app", 12.0), H=p.get("IC_H_d_app", 1.4),
            L=p.get("IC_L_d_app", 3.5), DL=p.get("IC_DL_d_app", 0.20),
            A=p.get("IC_A_d_app", 1.8), W=p.get("IC_W_d_app", 5.0),
            digital_density=p.get("IC_DD_d_app", 0.06),
            import_dependency=p.get("IC_import_dep_d_app", 0.30),
            sector_type="digital_app",
        )
        # ── v3.5: 高技能服务业 ──
        state.sectors["s_high"] = SectorState(
            K=p.get("IC_K_s_high", 30.0), H=p.get("IC_H_s_high", 1.1),
            L=p.get("IC_L_s_high", 8.0), DL=p.get("IC_DL_s_high", 0.08),
            A=p.get("IC_A_s_high", 1.0), W=p.get("IC_W_s_high", 3.5),
            digital_density=p.get("IC_DD_s_high", 0.04),
            sector_type="services",
            employment_quality=1.0,
        )
        # ── v3.5: 低端服务业 ──
        state.sectors["s_low"] = SectorState(
            K=p.get("IC_K_s_low", 20.0), H=p.get("IC_H_s_low", 0.5),
            L=p.get("IC_L_s_low", 7.0), DL=p.get("IC_DL_s_low", 0.03),
            A=p.get("IC_A_s_low", 0.7), W=p.get("IC_W_s_low", 1.2),
            digital_density=p.get("IC_DD_s_low", 0.01),
            sector_type="services",
            employment_quality=0.6,  # 低端服务业就业质量偏低
        )

        # 记录初始资本
        for key in ALL_SECTORS:
            setattr(state, f"initial_K_{key}", state.sectors[key].K)

        # 预置初始数据资产
        state.data_assets.producer_data = p.get("IC_data_producer", 1.0)
        state.data_assets.consumer_data = p.get("IC_data_consumer", 0.5)
        state.data_assets.public_data = p.get("IC_data_public", 0.3)

        # 初始数字服务流
        state.sectors["b"].DS = p.get("IC_DS_b", 0.5)
        state.sectors["i"].DS = p.get("IC_DS_i", 1.0)
        state.sectors["d_infra"].DS = p.get("IC_DS_d_infra", 1.0)
        state.sectors["d_app"].DS = p.get("IC_DS_d_app", 1.5)
        state.sectors["s_high"].DS = p.get("IC_DS_s_high", 0.8)
        state.sectors["s_low"].DS = p.get("IC_DS_s_low", 0.3)

        # 初始基尼系数
        state.gini = p.get("IC_gini_initial", 0.38)

        # 初始产出
        self.compute_all_production(state)
        self.compute_wages(state)
        self.compute_profit_rates(state)

        # 初始GDP
        state.gdp = sum(s.Y for s in state.sectors.values())

        # v3.5: 初始技术差距（infra/app分层）
        state.tech_gap = p.get("IC_tech_gap_initial", 3.0)
        state.tech_gap_infra = p.get("IC_tech_gap_infra_initial", state.tech_gap * 1.3)
        state.tech_gap_app = p.get("IC_tech_gap_app_initial", state.tech_gap * 0.8)
