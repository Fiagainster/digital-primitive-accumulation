"""知识扩散政策模块 (v2.1 新增).

实现 Delli Gatti et al. (2024) 的双轨制产业政策：
- 轨道1：R&D补贴 — 支持创新前沿（数字部门）
- 轨道2：知识扩散政策 — 帮助落后部门（工业、基础）采用已有技术

核心发现：R&D补贴单独使用无效。只有当与知识扩散政策结合时，产业政策才有效。

本模块功能：
1. 知识扩散效率计算 — 从数字部门到其他部门的技术传播
2. 吸收能力评估 — 目标部门的准备度
3. 扩散政策力度与制度能力的交互
4. 部门内领先-落后差距的演化
"""

import numpy as np
from dataclasses import dataclass
from ..core.state import SystemState


@dataclass
class DiffusionMetrics:
    """知识扩散指标."""
    diffusion_rate_d_to_i: float = 0.0   # D→I扩散率
    diffusion_rate_d_to_b: float = 0.0   # D→B扩散率
    absorption_readiness_i: float = 0.0  # 工业吸收准备度
    absorption_readiness_b: float = 0.0  # 基础部门吸收准备度
    policy_effectiveness: float = 0.0    # 政策效果
    frontier_laggard_gap_i: float = 0.0  # 部门内差距（近似）
    frontier_laggard_gap_b: float = 0.0  # 部门内差距（近似）


class TechnologyDiffusion:
    """知识扩散模型 — 双轨制产业政策.

    技术扩散不是自动的——它需要制度和政策支持。
    特别是在"跳跃策略"下，工业部门可能既缺乏数字技术，
    又缺乏采用数字技术的能力。
    """

    def __init__(self, params: dict):
        self.p = params
        self.metrics = DiffusionMetrics()
        # 追踪扩散的累计效果
        self._cumulative_diffusion = {"i": 0.0, "b": 0.0}
        # 部门内领先-落后差距（简化二分）
        self._sector_gaps = {
            "i": {"frontier": 1.2, "laggard": 1.0},
            "b": {"frontier": 1.1, "laggard": 1.0},
            "d": {"frontier": 1.5, "laggard": 1.0},
        }

    def step(self, state: SystemState, dt: float):
        """一个时间步的知识扩散更新."""
        # 1. 计算扩散率
        self._compute_diffusion_rates(state)
        # 2. 应用知识扩散（增强现有技术扩散）
        self._apply_diffusion_boost(state, dt)
        # 3. 更新部门内差距
        self._update_sector_gaps(state, dt)
        # 4. 记录指标
        self._update_metrics(state)

    def _compute_diffusion_rates(self, state: SystemState):
        """计算从数字部门到其他部门的知识扩散率."""
        p = self.p
        A_d = state.sectors["d"].A

        for key in ["i", "b"]:
            s = state.sectors[key]
            gap = A_d / max(s.A, 1.0)

            # === 自发扩散（市场机制） ===
            # 人员流动、供应链关系、竞争模仿
            spontaneous = (
                p["chi_spont"]
                * min(gap - 1, 5.0)  # 差距过大反而无法学习
                * s.DL  # 数字化水平影响接收能力
            )

            # === 政策驱动扩散（制度机制） ===
            # 政府组织的培训、示范项目、技术推广站
            kappa_diff = p.get("diff_policy_intensity", p["kappa_diff"])
            inst_cap = p["institution_capacity"]
            kappa_plan = p["kappa"]

            # 政策效果 = 扩散力度 × 制度能力 × 规划强度 × 部门准备度
            policy_driven = (
                kappa_diff
                * inst_cap
                * (1 + kappa_plan * 0.5)
                * self._absorption_readiness(s, key)
            )

            # === 数字部门成熟度的溢出 ===
            # DL_d越高，知识溢出越强（数字生态的外部性）
            dl_spillover = state.sectors["d"].DL * 0.05 * gap

            # 总扩散率
            total = spontaneous + policy_driven + dl_spillover

            if key == "i":
                self.metrics.diffusion_rate_d_to_i = total
                self.metrics.absorption_readiness_i = self._absorption_readiness(s, "i")
            else:
                self.metrics.diffusion_rate_d_to_b = total
                self.metrics.absorption_readiness_b = self._absorption_readiness(s, "b")

    def _absorption_readiness(self, s, key: str) -> float:
        """评估部门的吸收准备度.

        考虑因素：
        - 人力资本水平
        - 数字化水平
        - v2.2: 数字密度（嵌入数字技术的资本比例）
        - 现有技术水平
        """
        h_factor = min(1.0, s.H / 1.5)  # 人力资本相对水平
        dl_factor = s.DL  # 数字化就绪度
        # v2.2: 数字密度因子 — 已经嵌入数字技术的资本越多，吸收新技术越容易
        dd_factor = 0.5 + 0.5 * s.digital_density
        # 综合准备度
        readiness = 0.3 * h_factor + 0.35 * dl_factor + 0.15 * dd_factor + 0.2 * 0.5
        return max(0.05, readiness)

    def _apply_diffusion_boost(self, state: SystemState, dt: float):
        """将知识扩散增强应用到技术追赶中.

        扩散效果通过两个渠道起作用：
        1. 增强自主吸收（帮助理解数字技术）
        2. 降低技能贬值（培训使工人适应新技术）
        """
        # 对工业部门的人力资本增强
        for key in ["i", "b"]:
            s = state.sectors[key]
            rate = (self.metrics.diffusion_rate_d_to_i if key == "i"
                    else self.metrics.diffusion_rate_d_to_b)

            # 扩散 → 人力资本缓慢提升（学习效果）
            h_boost = rate * 0.02 * s.H
            s.H += h_boost * dt
            s.H = min(3.0, s.H)  # 人力资本有上限

            # 扩散 → 加速数字化吸收（知道怎么用数字技术了）
            dl_boost = rate * 0.01 * (1 - s.DL)
            s.DL += dl_boost * dt
            s.DL = np.clip(s.DL, 0.0, 0.99)

            # 记录累计扩散
            self._cumulative_diffusion[key] += rate * dt

    def _update_sector_gaps(self, state: SystemState, dt: float):
        """更新部门内的前沿-落后差距.

        简化的二分模型：
        - 前沿企业：能从数字部门直接学习，技术进步快
        - 落后企业：需要政策帮助才能跟上
        - 知识扩散政策帮助缩小这个差距
        """
        p = self.p
        kappa_diff = p.get("diff_policy_intensity", p["kappa_diff"])

        for key in ["i", "b"]:
            gap = self._sector_gaps[key]
            # 前沿自然进步（来自数字溢出）
            dl_factor = state.sectors["d"].DL
            gap["frontier"] += dl_factor * 0.03 * dt

            # 落后企业追赶
            gap_size = gap["frontier"] / max(gap["laggard"], 0.1) - 1.0
            # 追赶速度 = 自发追赶 + 政策加速
            catch_up = (
                0.02  # 自发追赶
                + kappa_diff * p["institution_capacity"] * 0.05  # 政策加速
            )
            gap["laggard"] += catch_up * gap_size * gap["laggard"] * dt

            # 限制差距范围
            gap["frontier"] = min(5.0, max(0.5, gap["frontier"]))
            gap["laggard"] = min(5.0, max(0.5, gap["laggard"]))

    def _update_metrics(self, state: SystemState):
        """更新指标."""
        for key in ["i", "b"]:
            f = self._sector_gaps[key]["frontier"]
            l = self._sector_gaps[key]["laggard"]
            gap = (f / max(l, 0.1)) - 1.0
            if key == "i":
                self.metrics.frontier_laggard_gap_i = gap
            else:
                self.metrics.frontier_laggard_gap_b = gap

        # 政策效果评估
        kappa_diff = self.p.get("diff_policy_intensity", self.p["kappa_diff"])
        inst_cap = self.p["institution_capacity"]
        self.metrics.policy_effectiveness = kappa_diff * inst_cap * 10.0

    def get_sector_gap(self, sector: str) -> float:
        """获取部门内前沿-落后技术差距."""
        if sector in self._sector_gaps:
            f = self._sector_gaps[sector]["frontier"]
            l = self._sector_gaps[sector]["laggard"]
            return f / max(l, 0.1) - 1.0
        return 0.0

    def get_cumulative_diffusion(self, sector: str) -> float:
        """获取累计知识扩散量."""
        return self._cumulative_diffusion.get(sector, 0.0)
