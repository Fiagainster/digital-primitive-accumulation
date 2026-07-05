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
        """一个时间步的技术更新."""
        # 1. 更新国际前沿
        self._update_frontier(state, dt)
        # 2. 更新国内数字部门技术（策略差异化）
        self._update_digital_tech(state, dt)
        # 3. 技术扩散到工业和基础部门
        self._diffuse_technology(state, dt)
        # 4. 更新数字依附评估
        self._update_dependency(state)

    def _update_frontier(self, state: SystemState, dt: float):
        """国际技术前沿增长."""
        g = self.p["g_world"]
        A_d = state.sectors["d"].A
        G = state.tech_gap
        A_world = A_d * G if G > 0 else A_d * 1.0

        A_world *= (1 + g * dt)
        state.tech_gap = A_world / max(A_d, 0.01)

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

    def _update_digital_tech(self, state: SystemState, dt: float):
        """数字部门技术进步（v2.1 策略差异化）."""
        p = self.p
        s = state.sectors["d"]
        G = state.tech_gap
        mult = self._regime_multipliers(state)

        # === 自主创新 ===
        rd_ratio = p.get("C_indigenous_rd_ratio", 0.03)
        innovation = (
            p["zeta_d"]
            * mult["innovation"]
            * (rd_ratio) ** p["phi_rd"]
            * s.H ** 0.5
        )

        # === 技术吸收 ===
        tech_import_share = p.get("C_tech_import_preference", 0.4)
        # 策略差异化：A的吸收弱，B的吸收强
        absorption = (
            p["chi_absorb_tech"]
            * mult["absorption"]
            * (1.0 / max(G, 0.1)) ** p["xi_gap"]
        )
        # 技术进口对吸收的贡献（策略C从中获益更多）
        tech_import_bonus = tech_import_share * (1 + p.get("fdi_availability", 1.0) * 0.1)
        absorption *= (1 + tech_import_bonus * 0.3)

        # === 追赶陷阱（策略差异化敏感度） ===
        if G > p["G_critical"]:
            trap_strength = p["lambda_trap"] * mult["trap_sensitivity"]
            absorption *= np.exp(-trap_strength * (G - p["G_critical"]))

        # === 依赖陷阱（策略C加速退化） ===
        if tech_import_share > 0.5:
            decay = p["phi_trap"] * mult.get("dependency_decay", 1.0) * (tech_import_share - 0.5)
            innovation *= (1 - min(decay, 0.85))

        # === 数字依附对自主创新的抑制 ===
        dep = state.dependency.composite
        innovation *= (1 - p["lambda_dep"] * dep)

        dA = (innovation + absorption) * s.A
        s.A += dA * dt
        s.A = max(0.1, s.A)

        # 更新技术差距
        A_world = s.A * G
        state.tech_gap = A_world / max(s.A, 0.01)

    def _diffuse_technology(self, state: SystemState, dt: float):
        """技术从数字部门扩散到工业和基础部门（v2.2 数字密度增强 + v2.1 策略差异化）."""
        p = self.p
        A_d = state.sectors["d"].A
        mult = self._regime_multipliers(state)

        for key in ["i", "b"]:
            s = state.sectors[key]
            gap = A_d / max(s.A, 1.0)

            # 自发扩散（策略差异化）
            spont = (
                p["chi_spont"]
                * mult["diffusion_spont"]
                * (gap - 1)
                * (1 - np.exp(-p["eta_diff"] * s.H))
            )

            # v2.2: 数字密度加速扩散 — 资本中嵌入的数字技术越多，扩散越快
            density_bonus = 1.0 + s.digital_density * 0.8

            # 政策驱动扩散（双轨制：规划力度 + 知识扩散政策）
            policy = (
                p["kappa_diff"]
                * p["institution_capacity"]
                * s.DL
                * (1 + p["kappa"] * 0.5)  # 规划强度增强扩散
            )

            dA_diff = (spont + policy) * s.A * density_bonus
            s.A += dA_diff * dt
            s.A = max(0.1, s.A)

    def _update_dependency(self, state: SystemState):
        """更新数字依附评估."""
        p = self.p

        # 基础设施依赖：基于技术进口和外资
        tech_imp = p.get("C_tech_import_preference", 0.4)
        fdi = p.get("fdi_availability", 1.0)
        state.dependency.infrastructure_dependency = (
            0.5 * tech_imp + 0.3 * max(0, 1 - 1 / max(fdi, 1.0))
        )

        # 数据依赖：基于消费者数据占比
        total_data = state.data_assets.producer_data + state.data_assets.consumer_data
        if total_data > 0:
            state.dependency.data_dependency = (
                0.6 * state.data_assets.consumer_data / total_data
            )
        else:
            state.dependency.data_dependency = 0.5

        # 标准自主权 (v3.2: 增加路径依赖锁定效应)
        # 一旦采用外国标准，即使技术差距缩小也难以逆转——锁定效应
        if not hasattr(self, '_tech_import_duration'):
            self._tech_import_duration = 0.0
            self._lock_in_multiplier = 1.0

        # 追踪技术进口持续时间
        if tech_imp > 0.4:  # 显著依赖技术进口
            self._tech_import_duration += 0.25  # 每季度累积
        else:
            self._tech_import_duration = max(0.0, self._tech_import_duration - 0.25 * 0.5)

        # 锁定乘数：长期技术进口(>10年)→即使gap缩小也难以降低依赖
        lock_in_years = self._tech_import_duration
        if lock_in_years > 10.0:
            # 锁定效应在10-30年间从1.0增长到1.8
            self._lock_in_multiplier = min(1.8, 1.0 + (lock_in_years - 10.0) * 0.04)

        state.dependency.standards_autonomy = max(
            0.1, 1.0 - state.external_dependency * 0.5 * self._lock_in_multiplier
        )

        # 外部依赖度更新 (v3.2: 锁定使依赖更难降低)
        raw_dependency = tech_imp * 0.7 + fdi * 0.3 / 2.0
        # 当锁定乘数>1时，外部依赖度有"粘性"——下降更慢
        if raw_dependency < state.external_dependency and self._lock_in_multiplier > 1.0:
            # 有锁定时依赖度下降速度减半
            state.external_dependency += (raw_dependency - state.external_dependency) * 0.5
        else:
            state.external_dependency = raw_dependency
