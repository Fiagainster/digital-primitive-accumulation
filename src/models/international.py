"""国际交互 —— 资源出口、技术进口、外资、数字出口."""

import numpy as np
from ..core.state import SystemState


class InternationalModule:
    """国际交互模型 (v3.3: 汇率渠道 + 内生封锁/FDI)."""

    def __init__(self, params: dict):
        self.p = params
        self._resource_price = 1.0
        # v3.3: 追踪历史tech_gap用于内生封锁
        self._tech_gap_history = []

    def step(self, state: SystemState, dt: float):
        """一个时间步的国际结算."""
        # 1. 资源价格动态
        self._update_resource_price(dt)
        # 2. 资源出口收入
        export_income = self._resource_export(state)
        # 3. v3.3: 更新汇率
        self._update_exchange_rate(state)
        # 4. v3.3: 更新内生封锁和FDI
        self._update_endogenous_conditions(state, dt)
        # 5. 技术进口
        tech_import = self._technology_import(state)
        # 6. 外资流动
        fdi_net = self._fdi_flows(state)
        # 7. 数字出口
        digital_export = self._digital_export(state)
        # 8. 更新国际竞争力
        self._update_competitiveness(state)

        # 净外部收入注入积累基金
        net_external = export_income + fdi_net + digital_export - tech_import
        state.accumulation_fund = max(0.0, state.accumulation_fund + net_external)

        # v3.3: 记录tech_gap历史
        self._tech_gap_history.append(state.tech_gap)
        if len(self._tech_gap_history) > 100:
            self._tech_gap_history = self._tech_gap_history[-100:]

    def _update_exchange_rate(self, state: SystemState):
        """v3.3: 实际汇率——资源出口繁荣→汇率升值(荷兰病渠道)."""
        resource_export = state.sectors["b"].Y * self.p.get(
            "C_resource_export_intensity", 0.3
        ) * self._resource_price * 0.5
        resource_share = resource_export / max(state.gdp, 0.01)
        # 资源出口每占GDP 10%，汇率升值约8%
        state.real_exchange_rate = 1.0 + resource_share * 0.8
        state.real_exchange_rate = max(0.5, min(state.real_exchange_rate, 3.0))

    def _update_endogenous_conditions(self, state: SystemState, dt: float):
        """v3.3: 内生封锁程度和FDI可得性.

        - 封锁：你追得越快→封锁越紧
        - FDI：政治稳定性越好→外资越愿意来
        """
        base_block = self.p.get("block_max", 0.5)
        base_fdi = self.p.get("fdi_availability", 1.0)

        # 封锁响应：如果tech_gap在过去5年(20步)缩小了超过15%
        if len(self._tech_gap_history) >= 20:
            gap_5yr_ago = np.mean(self._tech_gap_history[-20:-15])
            gap_now = np.mean(self._tech_gap_history[-5:])
            if gap_5yr_ago > 0:
                gap_reduction = max(0.0, (gap_5yr_ago - gap_now) / gap_5yr_ago)
                if gap_reduction > 0.15:
                    # 你追得越快→封锁越紧（最多比基础值高50%）
                    escalation = 1.0 + gap_reduction * 2.0
                    state.effective_block_max = min(0.95, base_block * escalation)
                else:
                    state.effective_block_max = base_block
            else:
                state.effective_block_max = base_block
        else:
            state.effective_block_max = base_block

        # FDI响应：政治支持度下降→外资撤离
        ps = state.political_support.support_index
        state.effective_fdi_availability = base_fdi * (0.4 + 0.6 * ps)
        state.effective_fdi_availability = max(0.05, state.effective_fdi_availability)

    def _update_resource_price(self, dt: float):
        """均值回复 + 随机冲击."""
        theta = self.p["theta_p"]
        mu = self.p["mu_p_resource"]
        sigma = self.p["sigma_p"]
        dP = theta * (mu - self._resource_price) * dt + sigma * np.random.randn() * np.sqrt(dt)
        self._resource_price += dP
        self._resource_price = max(0.1, self._resource_price)

    def _resource_export(self, state: SystemState) -> float:
        """资源出口收入."""
        intensity = self.p.get("C_resource_export_intensity", 0.3)
        base_output = state.sectors["b"].Y * intensity
        return base_output * self._resource_price * 0.5

    def _technology_import(self, state: SystemState) -> float:
        """技术进口成本."""
        tech_pref = self.p.get("C_tech_import_preference", 0.4)
        G = state.tech_gap

        # 封锁溢价 (v3.3: 使用有效封锁值——可内生上升)
        block = getattr(state, "effective_block_max", self.p["block_max"])
        blockade = block / (1 + np.exp(-5 * (G - 3.0)))

        demand = tech_pref * state.gdp * 0.1
        cost = demand * (1 + blockade)
        return cost

    def _fdi_flows(self, state: SystemState) -> float:
        """外资净流入."""
        fdi_open = self.p.get("C_fdi_openness", 0.3)
        fdi_avail = getattr(state, "effective_fdi_availability",
                            self.p.get("fdi_availability", 1.0))

        # 国内回报率
        r_domestic = state.sectors["d"].r
        r_world = 0.05

        inflow = fdi_open * fdi_avail * state.gdp * 0.05
        inflow *= (1 + self.p["fdi_sensitivity"] * max(-0.5, r_domestic - r_world))

        # 利润汇出
        repat_limit = self.p.get("C_profit_repat_limit", 0.3)
        existing_fdi = state.balance_sheets.get("d", None)
        if existing_fdi:
            outflow = existing_fdi.equity_foreign * 0.08 * repat_limit
        else:
            outflow = 0.0

        # 更新资产负债表
        if "d" not in state.balance_sheets:
            from ..core.state import BalanceSheet
            state.balance_sheets["d"] = BalanceSheet()
        state.balance_sheets["d"].equity_foreign += (inflow - outflow)

        return inflow - outflow

    def _digital_export(self, state: SystemState) -> float:
        """数字服务出口."""
        world_demand = self.p.get("world_digital_demand", 1.0)
        comp = state.competitiveness
        export = state.sectors["d"].Y * 0.2 * world_demand * comp
        return export

    def _update_competitiveness(self, state: SystemState):
        """更新国际竞争力."""
        G = state.tech_gap
        state.competitiveness = 1.0 / max(G, 0.1)
