"""v3.6 平台垄断模块 —— 数字部门市场集中度内生演化.

核心机制：
- 市场集中度 HHI 内生演化（网络效应 + 数据飞轮的正反馈）
- 平台垄断租金随 HHI 非线性增长
- 数据网络护城河阻碍技术扩散
- 反垄断政策参数化

理论基础：
- Autor et al. (2020) — 超级明星企业的崛起
- Philippon (2019) — 市场集中度上升与竞争减弱
- Acemoglu & Restrepo (2021) — 自动化与劳动份额下降
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from ..core.state import SystemState, ALL_SECTORS, DIGITAL_SECTORS


@dataclass
class PlatformState:
    """平台垄断状态快照."""

    # 核心指标
    digital_hhi: float = 0.0              # 数字部门 HHI 市场集中度 [0,1]
    platform_monopoly_rent: float = 0.0    # 平台垄断租金占数字利润比例 [0,1]
    data_network_moat: float = 0.0         # 数据网络护城河强度 [0,1]

    # 反垄断政策效力
    anti_monopoly_policy: float = 0.0      # 反垄断执法力度 [0,1]
    data_portability_mandate: float = 0.0  # 数据可携带权 [0,1]
    platform_interoperability: float = 0.0 # 平台互操作性 [0,1]

    # 诊断指标
    concentration_drive_network: float = 0.0   # 网络效应驱动力
    concentration_drive_data: float = 0.0      # 数据飞轮驱动力
    concentration_decay_competition: float = 0.0  # 竞争衰减力
    monopoly_innovation_drag: float = 0.0      # 垄断对创新的抑制效应

    # 历史追踪
    hhi_growth_rate: float = 0.0           # HHI 年化增长率
    months_above_critical: float = 0.0     # HHI 超过临界值的累积月数


class PlatformModule:
    """数字平台市场结构模块.

    追踪数字部门的市场集中度如何内生演化，以及垄断结构
    如何影响创新扩散、利润分配和政治稳定性。
    """

    def __init__(self, params: dict):
        self.p = params
        self._state = PlatformState()

        # 从参数读取反垄断政策
        self._state.anti_monopoly_policy = params.get("anti_monopoly_policy", 0.0)
        self._state.data_portability_mandate = params.get("data_portability_mandate", 0.0)
        self._state.platform_interoperability = params.get("platform_interoperability", 0.0)

        # 追踪器
        self._prev_hhi: float = 0.0
        self._hhi_history: list = []

    def step(self, state: SystemState, dt: float):
        """一个时间步的平台结构更新."""
        # 1. 更新市场集中度
        self._update_market_concentration(state, dt)

        # 2. 更新垄断租金
        self._update_monopoly_rent(state)

        # 3. 更新数据护城河
        self._update_data_moat(state, dt)

        # 4. 计算垄断对创新的抑制
        self._update_innovation_drag(state)

        # 5. 应用垄断效应到经济系统
        self._apply_monopoly_effects(state, dt)

    # ─────────────────────────────────────────────
    # 市场集中度内生演化
    # ─────────────────────────────────────────────
    def _update_market_concentration(self, state: SystemState, dt: float):
        """市场集中度内生演化 —— 基于网络效应 + 数据飞轮的正反馈.

        核心微分方程：
        d(HHI)/dt = (network_drive + data_drive) * (1 - HHI)
                   - competition_decay
                   - anti_monopoly_effect

        其中各项在 [0,1] 范围内，确保 HHI ∈ [0, 0.95].
        """
        ps = self._state

        # 1. 网络效应驱动力：用户越多 → 平台价值越大 → 集中度上升
        total_d_users = sum(state.sectors[k].L for k in DIGITAL_SECTORS) * 15
        # 对数饱和：前100万用户影响力最大，之后再增加边际效应递减
        network_drive = 0.025 * np.log1p(total_d_users / 100.0)

        # 2. 数据飞轮：消费者数据越多 → 算法优势越大 → 护城河越深
        # 消费者数据的边际效应递减（对数形式）
        data_stock = max(state.data_assets.consumer_data, 0.01)
        data_drive = 0.02 * np.log1p(data_stock)

        # app 层的网络效应更强（直接面向消费者）
        if state.sectors["d_app"].DL > 0.3:
            # 高数字化水平 + 消费数据 → 强化数据飞轮
            app_amplifier = state.sectors["d_app"].DL * 0.015
            data_drive += app_amplifier

        # 3. 自然竞争衰减（新进入者、技术变革）
        # 当 HHI 较高时，高利润吸引新进入者 → 衰减力增强
        competition_decay = 0.008 * (1.0 + ps.digital_hhi)

        # 4. 反垄断政策效应
        anti_monopoly_effect = (
            ps.anti_monopoly_policy * 0.020          # 反垄断执法
            + ps.data_portability_mandate * 0.012     # 数据可携带权（削弱数据飞轮）
            + ps.platform_interoperability * 0.008    # 平台互操作性（降低网络效应）
        )

        # 制度能力调节：强制度 → 反垄断更有效
        inst_cap = self.p.get("institution_capacity", 0.6)
        anti_monopoly_effect *= (0.5 + inst_cap * 0.5)

        # 5. 合成 HHI 变化率
        # 逻辑斯蒂形式：(drive - decay) * (1 - HHI) * HHI —— 在极端值处减慢
        d_hhi = (
            (network_drive + data_drive) * (1.0 - ps.digital_hhi)
            - competition_decay * ps.digital_hhi
            - anti_monopoly_effect
        )
        # 确保 HHI 在 [0, 0.95] 范围内
        d_hhi = d_hhi * (1.0 - ps.digital_hhi / 0.95) if d_hhi > 0 else d_hhi * (ps.digital_hhi / 0.95)

        self._prev_hhi = ps.digital_hhi
        ps.digital_hhi = max(0.0, min(0.95, ps.digital_hhi + d_hhi * dt))

        # 追踪驱动力分量
        ps.concentration_drive_network = network_drive
        ps.concentration_drive_data = data_drive
        ps.concentration_decay_competition = competition_decay

        # 6. 历史追踪
        self._hhi_history.append(ps.digital_hhi)
        if len(self._hhi_history) > 200:
            self._hhi_history = self._hhi_history[-100:]

        if len(self._hhi_history) >= 20:
            # 年化增长率估算
            ps.hhi_growth_rate = (
                (ps.digital_hhi - self._hhi_history[-20]) / 5.0
            )

        # HHI 超过临界值（0.7）的累积时间
        if ps.digital_hhi > 0.7:
            ps.months_above_critical += dt

    # ─────────────────────────────────────────────
    # 垄断租金计算
    # ─────────────────────────────────────────────
    def _update_monopoly_rent(self, state: SystemState):
        """垄断租金随 HHI 非线性增长.

        当 HHI > 0.5 时，市场开始表现出垄断特征：
        - 数字部门利润率中的垄断成分 = f(HHI)
        - 租金随 HHI 加速增长（HHI 从 0.5 → 0.8 时租金翻三倍）
        """
        ps = self._state

        if ps.digital_hhi <= 0.4:
            # 竞争性市场，无显著垄断租金
            ps.platform_monopoly_rent = 0.0
        else:
            # 垄断租金 = (HHI - 0.4)^2 的非线性函数
            excess_hhi = ps.digital_hhi - 0.4
            ps.platform_monopoly_rent = min(0.8, excess_hhi ** 1.5 * 1.8)

    # ─────────────────────────────────────────────
    # 数据护城河
    # ─────────────────────────────────────────────
    def _update_data_moat(self, state: SystemState, dt: float):
        """数据网络护城河 —— 消费者数据的锁定效应.

        数据护城河衡量：已有平台通过数据积累形成的进入壁垒。
        - 消费者数据越多 → 个性化推荐越精准 → 用户粘性越强
        - 高护城河 → 技术扩散率下降（算法和数据不共享）
        """
        ps = self._state

        # 护城河 = f(消费者数据, 数字部门HHI, app层DL)
        data_factor = np.log1p(state.data_assets.consumer_data) / np.log1p(100.0)
        ps.data_network_moat = np.clip(
            data_factor * 0.5
            + ps.digital_hhi * 0.3
            + state.sectors["d_app"].DL * 0.2,
            0.0, 1.0
        )

        # 数据可携带权削弱护城河
        if ps.data_portability_mandate > 0:
            moat_decay = ps.data_portability_mandate * 0.15
            ps.data_network_moat = max(0.0, ps.data_network_moat - moat_decay * dt)

    # ─────────────────────────────────────────────
    # 垄断对创新的抑制
    # ─────────────────────────────────────────────
    def _update_innovation_drag(self, state: SystemState):
        """垄断对创新的抑制效应.

        渠道：
        1. 垄断平台减少了对激进创新的激励（"杀手并购"）
        2. 数据护城河阻碍知识扩散
        3. 高集中度 → 新进入者创新被压制
        """
        ps = self._state

        # 创新抑制 = HHI 的非线性函数
        # 适度集中(0.3-0.5)可能有助创新（熊彼特效应），但过度集中抑制创新
        if ps.digital_hhi < 0.5:
            drag = 0.0
        else:
            drag = (ps.digital_hhi - 0.5) ** 2 * 1.5
        # 数据护城河叠加效应
        drag += ps.data_network_moat * 0.3

        ps.monopoly_innovation_drag = min(0.5, drag)

    # ─────────────────────────────────────────────
    # 垄断效应对经济系统的反馈
    # ─────────────────────────────────────────────
    def _apply_monopoly_effects(self, state: SystemState, dt: float):
        """将垄断结构的变化反馈到经济系统的其他部分."""
        ps = self._state

        # 1. 垄断租金 → 数字部门利润率上升
        # 高 HHI 允许平台制定更高的加成率（markup）
        if ps.platform_monopoly_rent > 0:
            markup_boost = 1.0 + ps.platform_monopoly_rent * 0.4
            for d_key in DIGITAL_SECTORS:
                # 调整利润率（向下的工资压力 + 向上的价格加成）
                old_profit = state.sectors[d_key].profit
                state.sectors[d_key].profit = old_profit * markup_boost
                # 重新计算利润率
                state.sectors[d_key].r = (
                    state.sectors[d_key].profit
                    / max(state.sectors[d_key].K, 0.01)
                )

        # 2. 垄断 + 护城河 → 技术扩散减速
        # 数据和技术被封锁在平台内部，不向工业和基础部门溢出
        if ps.data_network_moat > 0.3:
            diffusion_penalty = 1.0 - ps.data_network_moat * 0.25
            # 影响技术扩散模块中的扩散速率
            for key in ["i", "b", "s_low"]:
                if key in state.sectors:
                    state.sectors[key].DL *= max(
                        0.95,
                        1.0 - ps.data_network_moat * 0.02 * dt
                    )

        # 3. 垄断 → 就业吸收能力下降
        # "明星公司"高利润但低就业：数字部门雇人随 HHI 上升而减缓
        if ps.digital_hhi > 0.6 and state.unemployment_rate > 0.05:
            # 高集中度下，同样产出增长创造的就业更少
            employment_multiplier = 1.0 - (ps.digital_hhi - 0.6) * 0.5
            for d_key in DIGITAL_SECTORS:
                # 减少新增就业吸收
                state.sectors[d_key].jobs_created *= max(0.5, employment_multiplier)

        # 4. 垄断 → 财富集中加速
        # 垄断利润不成比例地流向数字精英
        if ps.platform_monopoly_rent > 0.3:
            wealth_boost = ps.platform_monopoly_rent * 0.02 * dt
            digital_hh = state.households.digital_high_skill
            # 数字精英的股权价值上升快于其他群体
            digital_hh.equity *= (1.0 + wealth_boost)
            # 其他群体不分享这些垄断收益
            industrial_share = state.households.industrial_workers.equity
            state.households.industrial_workers.equity = max(
                0.01, industrial_share * (1.0 - wealth_boost * 0.3)
            )

        # 5. 记录到状态变量（供断裂检测和五维评价使用）
        state.digital_hhi = ps.digital_hhi
        state.platform_monopoly_rent = ps.platform_monopoly_rent
        state.data_network_moat = ps.data_network_moat

    # ─────────────────────────────────────────────
    # 公共接口
    # ─────────────────────────────────────────────
    def get_state(self) -> PlatformState:
        """获取当前平台垄断状态."""
        return self._state

    def get_summary(self) -> dict:
        """获取平台垄断状态摘要."""
        ps = self._state
        return {
            "digital_hhi": ps.digital_hhi,
            "platform_monopoly_rent": ps.platform_monopoly_rent,
            "data_network_moat": ps.data_network_moat,
            "anti_monopoly_policy": ps.anti_monopoly_policy,
            "monopoly_innovation_drag": ps.monopoly_innovation_drag,
            "hhi_growth_rate": ps.hhi_growth_rate,
            "months_above_critical": ps.months_above_critical,
            "concentration_drive_network": ps.concentration_drive_network,
            "concentration_drive_data": ps.concentration_drive_data,
        }

    def check_platform_related_risks(self, state: SystemState) -> list:
        """检查与平台垄断相关的系统风险.

        Returns:
            风险列表，每个风险包含 type/severity/description
        """
        ps = self._state
        risks = []

        # 风险1：数字封建化
        if (ps.digital_hhi > 0.8
                and ps.platform_monopoly_rent > 0.6
                and state.wealth_gini > 0.6):
            risks.append({
                "type": "digital_feudalization",
                "severity": "CRITICAL",
                "description": (
                    f"数字部门退化为租金抽取机器：HHI={ps.digital_hhi:.2f}, "
                    f"垄断租金={ps.platform_monopoly_rent:.2f}, "
                    f"财富基尼={state.wealth_gini:.2f}"
                ),
            })

        # 风险2：创新停滞
        if ps.monopoly_innovation_drag > 0.3:
            risks.append({
                "type": "innovation_stagnation",
                "severity": "WARNING" if ps.monopoly_innovation_drag < 0.4 else "HIGH",
                "description": (
                    f"垄断抑制创新：创新拖累={ps.monopoly_innovation_drag:.2f}, "
                    f"HHI={ps.digital_hhi:.2f}"
                ),
            })

        # 风险3：数据殖民
        if (ps.data_network_moat > 0.7
                and state.data_assets.consumer_data > 50):
            risks.append({
                "type": "data_colonialism",
                "severity": "WARNING",
                "description": (
                    f"数据护城河过高：护城河={ps.data_network_moat:.2f}, "
                    f"消费者数据={state.data_assets.consumer_data:.1f}"
                ),
            })

        return risks
