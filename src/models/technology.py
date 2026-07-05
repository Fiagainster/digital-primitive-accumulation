"""技术追赶 —— 自主创新 + 技术吸收 + 追赶陷阱 + 知识扩散.

v2.1 升级：策略差异化技术追赶
- 不同策略产生显著不同的技术轨迹
- 策略A：保守投资 → 吸收能力弱 → 技术锁死风险高
- 策略B：激进投资 → 吸收能力强 → 追赶速度快，但有过热风险
- 策略C：依赖进口 → 初期吸收快但自主创新退化加速
"""

import numpy as np
from ..core.state import SystemState


class TechnologyDynamics:
    """技术动态模型（v2.1 策略差异化升级）."""

    def __init__(self, params: dict):
        self.p = params

    def step(self, state: SystemState, dt: float):
        """一个时间步的技术更新 (v3.5: infra/app分层)."""
        # 1. 更新国际前沿
        self._update_frontier(state, dt)
        # 2. 更新数字基础设施层技术
        self._update_digital_infra_tech(state, dt)
        # 3. 更新数字应用层技术
        self._update_digital_app_tech(state, dt)
        # 4. 技术扩散到工业、基础和高技能服务部门
        self._diffuse_technology(state, dt)
        # 5. 更新数字依附评估（六维）
        self._update_dependency(state)

    def _update_frontier(self, state: SystemState, dt: float):
        """国际技术前沿增长 (v3.5: infra/app分层)."""
        g = self.p["g_world"]
        # 综合技术差距：加权infra和app
        A_d_infra = state.sectors["d_infra"].A
        A_d_app = state.sectors["d_app"].A
        G_infra = state.tech_gap_infra
        G_app = state.tech_gap_app

        A_world_infra = A_d_infra * G_infra if G_infra > 0 else A_d_infra * 1.0
        A_world_app = A_d_app * G_app if G_app > 0 else A_d_app * 1.0

        A_world_infra *= (1 + g * dt)
        A_world_app *= (1 + g * dt)

        state.tech_gap_infra = A_world_infra / max(A_d_infra, 0.01)
        state.tech_gap_app = A_world_app / max(A_d_app, 0.01)
        # 综合技术差距（用于向后兼容）
        w_infra = self.p.get("A_world_infra_weight", 0.4)
        w_app = self.p.get("A_world_app_weight", 0.6)
        A_world_avg = w_infra * A_world_infra + w_app * A_world_app
        A_d_avg = w_infra * A_d_infra + w_app * A_d_app
        state.tech_gap = A_world_avg / max(A_d_avg, 0.01)

    def _regime_multipliers(self, state: SystemState) -> dict:
        """根据策略返回技术追赶的差异化乘数."""
        regime = state.accumulation_regime
        p = self.p

        # 基础乘数
        multipliers = {
            "innovation": 1.0,
            "absorption": 1.0,
            "trap_sensitivity": 1.0,
            "dependency_decay": 1.0,
            "diffusion_spont": 1.0,
        }

        if regime == "A":
            # 策略A：保守投资 → 数字部门研发投入不足
            # v2.2: 但如果工业数字密度高，说明"保护+升级"路径成功
            # 数字密度可以大幅改善A的吸收和扩散能力
            phi_d = p.get("A_phi_d_premium", 0.1)
            dd_i = state.sectors["i"].digital_density  # 工业数字密度
            # 基础乘数仍然偏低（数字投资确实少），但数字密度显著弥补
            multipliers["innovation"] = 0.6 + phi_d + dd_i * 0.4   # 0.7~1.1
            multipliers["absorption"] = 0.5 + phi_d * 1.5 + dd_i * 0.6  # 0.65~1.25
            multipliers["trap_sensitivity"] = 1.3 - dd_i * 0.6  # 1.3~0.7 (高密度=低敏感)
            multipliers["diffusion_spont"] = 0.6 + dd_i * 0.8  # 0.6~1.4 (高密度=扩散更快)

        elif regime == "B":
            # 策略B：激进投资 → 数字部门获得大量资源
            phi_d = p.get("B_phi_d_jump", 0.6)
            multipliers["innovation"] = 1.0 + phi_d * 0.5  # ~1.3
            multipliers["absorption"] = 1.0 + phi_d * 0.8  # ~1.48
            multipliers["trap_sensitivity"] = 0.7  # 对追赶陷阱更有韧性
            multipliers["diffusion_spont"] = 1.2  # 扩散更快

        elif regime == "C":
            # 策略C：依赖外部 → 初期吸收强但自主能力显著退化
            tech_imp = p.get("C_tech_import_preference", 0.6)
            fdi_open = p.get("C_fdi_openness", 0.4)
            multipliers["innovation"] = max(0.3, 0.8 - tech_imp * 0.6)  # 进口越多自主越弱(~0.44)
            multipliers["absorption"] = 0.7 + fdi_open * 1.0  # FDI带来吸收能力(~1.1)
            multipliers["trap_sensitivity"] = 1.2 + tech_imp * 0.5  # 过度进口增加陷阱风险
            multipliers["dependency_decay"] = 1.5 + tech_imp * 1.0  # 依附退化显著加速(~2.1)
            multipliers["diffusion_spont"] = 0.7  # 扩散偏弱

        return multipliers

    def _update_digital_infra_tech(self, state: SystemState, dt: float):
        """v3.5: 数字基础设施层技术进步 — 芯片/云/OS.

        infra技术追赶更依赖自主创新(无法进口核心基础设施技术)，
        受封锁影响最大，追赶速度慢但一旦突破则稳定性高。
        """
        p = self.p
        s = state.sectors["d_infra"]
        G = state.tech_gap_infra
        mult = self._regime_multipliers(state)

        # 自主创新（infra更依赖自主研发）
        rd_ratio = p.get("C_indigenous_rd_ratio", 0.03)
        innovation = (
            p["zeta_d"] * mult["innovation"] * 0.7  # infra创新效率偏低(硬科技)
            * (rd_ratio) ** p["phi_rd"] * s.H ** 0.5
        )

        # 技术吸收（infra的技术吸收受封锁严重限制）
        block = getattr(state, "effective_block_max", p.get("block_max", 0.5))
        absorption_cap = max(0.05, 1.0 - block * 1.2)  # 封锁对infra的吸收影响更大
        absorption = (
            p["chi_absorb_tech"] * mult["absorption"] * absorption_cap
            * (1.0 / max(G, 0.1)) ** p["xi_gap"]
        )

        # 追赶陷阱
        if G > p["G_critical"]:
            trap_strength = p["lambda_trap"] * mult["trap_sensitivity"]
            absorption *= np.exp(-trap_strength * (G - p["G_critical"]))

        # 依附对创新的抑制（infra更敏感）
        dep = state.dependency.composite
        innovation *= (1 - p["lambda_dep"] * dep * 1.3)

        # v3.5: 自主化突破 — 当自主创新投资足够高时可能逆转依赖
        if rd_ratio > 0.06 and p.get("institution_capacity", 0.6) > 0.65:
            innovation *= 1.3  # 制度+研发投入→创新加速
            s.import_dependency = max(0.2, s.import_dependency - 0.01 * dt)

        dA = (innovation + absorption) * s.A
        s.A += dA * dt
        s.A = max(0.1, s.A)

    def _update_digital_app_tech(self, state: SystemState, dt: float):
        """v3.5: 数字应用层技术进步 — 平台/AI应用.

        app技术追赶相对灵活，网络效应和数据资产是关键驱动力，
        但受infra层技术水平的约束。
        """
        p = self.p
        s = state.sectors["d_app"]
        G = state.tech_gap_app
        mult = self._regime_multipliers(state)

        # 自主创新（app创新效率高，受益于网络效应和数据资产）
        rd_ratio = p.get("C_indigenous_rd_ratio", 0.03)
        data_bonus = 1.0 + 0.3 * (state.data_assets.effective_stock / 10.0)
        innovation = (
            p["zeta_d"] * mult["innovation"] * 1.3 * data_bonus
            * (rd_ratio) ** p["phi_rd"] * s.H ** 0.5
        )

        # 技术吸收（app更容易吸收外国技术，但受封锁影响）
        block = getattr(state, "effective_block_max", p.get("block_max", 0.5))
        absorption_cap = max(0.1, 1.0 - block * 0.8)
        absorption = (
            p["chi_absorb_tech"] * mult["absorption"] * absorption_cap
            * (1.0 / max(G, 0.1)) ** p["xi_gap"]
        )

        # 网络效应加速学习
        net_bonus = 1.0 + 0.2 * s.DL
        absorption *= net_bonus

        # 追赶陷阱
        if G > p["G_critical"]:
            trap_strength = p["lambda_trap"] * mult["trap_sensitivity"] * 0.8
            absorption *= np.exp(-trap_strength * (G - p["G_critical"]))

        # v3.5: infra→app技术约束
        A_infra = state.sectors["d_infra"].A
        A_app_potential = s.A
        if A_infra < A_app_potential * 0.7:
            # infra严重落后 → app技术天花板
            innovation *= max(0.5, A_infra / max(A_app_potential * 0.7, 0.01))

        # 依附对创新的抑制（app敏感度较低——应用层更容易绕过封锁）
        dep = state.dependency.composite
        innovation *= (1 - p["lambda_dep"] * dep * 0.7)

        dA = (innovation + absorption) * s.A
        s.A += dA * dt
        s.A = max(0.1, s.A)

    def _diffuse_technology(self, state: SystemState, dt: float):
        """技术从数字部门扩散到工业、基础和高技能服务部门 (v3.5)."""
        p = self.p
        # 使用infra+app的加权平均技术水平作为扩散源
        A_d_weighted = (0.4 * state.sectors["d_infra"].A
                        + 0.6 * state.sectors["d_app"].A)
        mult = self._regime_multipliers(state)

        # v3.5: 扩散到工业、基础部门和高技能服务业
        for key in ["i", "b", "s_high"]:
            s = state.sectors[key]
            gap = A_d_weighted / max(s.A, 1.0)

            spont = (
                p["chi_spont"]
                * mult["diffusion_spont"]
                * (gap - 1)
                * (1 - np.exp(-p["eta_diff"] * s.H))
            )

            density_bonus = 1.0 + s.digital_density * 0.8

            policy = (
                p.get("diff_policy_intensity", p.get("kappa_diff", 0.03))
                * p["institution_capacity"]
                * s.DL
                * (1 + p["kappa"] * 0.5)
            )

            # v3.5: 扩散质量受数字依附约束
            dep_quality = 1.0 - 0.4 * state.dependency.composite

            dA_diff = (spont + policy) * s.A * density_bonus * dep_quality
            s.A += dA_diff * dt
            s.A = max(0.1, s.A)

        # v3.5: 工业→数字反向扩散（制造经验→IoT数据→AI训练）
        self._reverse_diffusion_industry_to_digital(state, dt)

    def _reverse_diffusion_industry_to_digital(self, state: SystemState, dt: float):
        """v3.5: 工业→数字反向扩散 — 制造经验丰富生产者数据.

        工业的制造经验和工艺知识 → 工业IoT数据 → 丰富生产者数据资产 →
        增强数字部门的AI训练数据。扩散效率随工业数字密度增加而上升。
        """
        p = self.p
        dd_i = state.sectors["i"].digital_density
        # 反向扩散强度：工业数字密度越高，产生的数据越多
        reverse_strength = dd_i * 0.05 * dt
        if reverse_strength > 0:
            # 丰富生产者数据
            state.data_assets.producer_data += (
                state.sectors["i"].Y * reverse_strength * 0.1
            )
            # 对app层技术水平有微弱提升（更好的训练数据）
            state.sectors["d_app"].A *= (1.0 + reverse_strength * 0.02)
            state.sectors["d_app"].A = min(5.0, state.sectors["d_app"].A)

    def _update_dependency(self, state: SystemState):
        """v3.5: 六维数字依附评估."""
        p = self.p

        # 基础设施依附（芯片+云+OS）
        tech_imp = p.get("C_tech_import_preference", 0.4)
        fdi = p.get("fdi_availability", 1.0)
        block = getattr(state, "effective_block_max", p.get("block_max", 0.5))

        state.dependency.chip_dependency = (
            0.6 * tech_imp + 0.4 * block
        )
        state.dependency.cloud_dependency = (
            0.5 * fdi * 0.6 + 0.5 * tech_imp * 0.5
        )
        state.dependency.os_dependency = (
            0.7 * tech_imp + 0.3 * block
        )

        # 数据依赖
        total_data = state.data_assets.producer_data + state.data_assets.consumer_data
        if total_data > 0:
            state.dependency.data_dependency = (
                0.6 * state.data_assets.consumer_data / total_data
            )
        else:
            state.dependency.data_dependency = 0.5

        # v3.5: 资本依附 — 外资在数字部门的持股比例
        bs_d = state.balance_sheets.get("d_infra", None)
        bs_app = state.balance_sheets.get("d_app", None)
        foreign_equity = 0.0
        total_equity = 1.0
        for bs in [bs_d, bs_app]:
            if bs:
                foreign_equity += bs.equity_foreign
                total_equity += bs.net_worth
        state.dependency.capital_dependency = min(1.0, foreign_equity / max(total_equity, 1.0))

        # v3.5: 人才依附 — 高技能人才海外就业/教育依赖
        digital_emp_share = (state.sectors["d_infra"].L + state.sectors["d_app"].L) / max(
            state.households.total_population, 1.0)
        # 数字就业岗位不足+技术进口高→人才外流风险
        state.dependency.talent_dependency = max(0.0,
            0.6 * tech_imp - 0.3 * digital_emp_share)

        # v3.5: 生态依附 — 应用生态/开发者社区的外国主导程度
        state.dependency.ecosystem_dependency = (
            0.5 * state.dependency.os_dependency
            + 0.3 * state.dependency.cloud_dependency
            + 0.2 * (1.0 - state.sectors["d_app"].DL)
        )

        # 标准自主权（锁定效应——v3.2保留）
        if not hasattr(self, '_tech_import_duration'):
            self._tech_import_duration = 0.0
            self._lock_in_multiplier = 1.0

        if tech_imp > 0.4:
            self._tech_import_duration += 0.25
        else:
            self._tech_import_duration = max(0.0, self._tech_import_duration - 0.25 * 0.5)

        lock_in_years = self._tech_import_duration
        if lock_in_years > 10.0:
            self._lock_in_multiplier = min(1.8, 1.0 + (lock_in_years - 10.0) * 0.04)

        state.dependency.standards_autonomy = max(
            0.1, 1.0 - state.external_dependency * 0.5 * self._lock_in_multiplier
        )

        # 外部依赖度更新
        raw_dependency = tech_imp * 0.7 + fdi * 0.3 / 2.0
        if raw_dependency < state.external_dependency and self._lock_in_multiplier > 1.0:
            state.external_dependency += (raw_dependency - state.external_dependency) * 0.5
        else:
            state.external_dependency = raw_dependency

        # v3.5: 依赖自我强化 — 依附越深→自主能力越弱→更加依赖
        if state.dependency.composite > 0.6:
            reinforce = (state.dependency.composite - 0.6) * 0.02 * 0.25  # 季度化
            state.dependency.chip_dependency = min(0.98,
                state.dependency.chip_dependency + reinforce)
            state.dependency.ecosystem_dependency = min(0.98,
                state.dependency.ecosystem_dependency + reinforce * 1.5)
