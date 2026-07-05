"""福利与代价 —— 按群组的福利计算与代价追踪."""

import numpy as np
from ..core.state import SystemState, ALL_SECTORS


class WelfareModule:
    """福利计算与代价追踪."""

    def __init__(self, params: dict):
        self.p = params

    def step(self, state: SystemState, dt: float):
        """一个时间步的福利更新 (v3.2: 增加储蓄/债务追踪)."""
        # 1. 计算各群组福利
        self._compute_group_welfare(state)
        # 2. 计算总体福利
        self._compute_aggregate_welfare(state)
        # 3. 追踪代价
        self._track_costs(state, dt)
        # 4. 去工业化诊断
        self._diagnose_deindustrialization(state, dt)
        # v3.2: 5. 更新家庭群组储蓄与债务
        self._update_household_balance_sheets(state, dt)

    def _compute_group_welfare(self, state: SystemState):
        """计算每个家庭群组的福利."""
        for h in state.households.all_groups():
            # 人均消费
            cons_per_cap = h.consumption / max(h.population, 0.01)
            # 就业率
            emp_rate = h.employed / max(h.population, 0.01)
            # 福利 = 几何平均
            h.welfare_index = (cons_per_cap ** 0.5) * (emp_rate ** 0.3) * (h.skill_level ** 0.2)

    def _compute_aggregate_welfare(self, state: SystemState):
        """计算总体福利（等弹性聚合）."""
        eps = self.p["epsilon_welfare"]
        components = []
        weights = []

        for h in state.households.all_groups():
            if h.population > 0:
                components.append(h.welfare_index)
                weights.append(h.population)

        if not components:
            state.welfare_index = 0.0
            return

        # 加权等弹性聚合
        total_w = sum(weights)
        if total_w == 0:
            state.welfare_index = 0.0
            return
        w = np.array(weights) / total_w
        c = np.maximum(np.array(components), 1e-6)
        try:
            if eps == 1.0:
                state.welfare_index = float(np.exp(np.sum(w * np.log(c))))
            else:
                val = float(np.sum(w * c ** (1 - eps)) ** (1 / (1 - eps)))
                state.welfare_index = max(0.0, min(val, 1e6))  # 防溢出
        except (OverflowError, FloatingPointError):
            state.welfare_index = 0.01

        # 基尼系数（基于群组收入简化）
        incomes = np.array([h.income / max(h.population, 0.01) for h in state.households.all_groups()])
        pops = np.array([h.population for h in state.households.all_groups()])
        if pops.sum() > 0 and incomes.sum() > 0:
            state.gini = self._approx_gini(incomes, pops)

    def _approx_gini(self, incomes: np.ndarray, pops: np.ndarray) -> float:
        """基于群组的近似基尼系数."""
        idx = np.argsort(incomes)
        incomes = incomes[idx]
        pops = pops[idx]
        total_pop = pops.sum()
        total_inc = (incomes * pops).sum()
        if total_inc == 0:
            return 0.0
        cum_pop = np.cumsum(pops) / total_pop
        cum_inc = np.cumsum(incomes * pops) / total_inc
        # 使用梯形法则近似积分（兼容新旧numpy版本）
        try:
            from scipy.integrate import trapezoid as trapz_fn
        except ImportError:
            try:
                trapz_fn = np.trapezoid
            except AttributeError:
                trapz_fn = np.trapz
        return 1.0 - 2.0 * trapz_fn(cum_inc, cum_pop)

    def _track_costs(self, state: SystemState, dt: float):
        """追踪并累积代价."""
        if not hasattr(self, '_peak_wages'):
            self._peak_wages = {k: state.sectors[k].W for k in ALL_SECTORS}

        cm = state.cost_metrics

        # 更新各部门工资峰值
        for k in ALL_SECTORS:
            self._peak_wages[k] = max(self._peak_wages[k], state.sectors[k].W)

            # 失业代价
        u_stock = state.households.unemployed.population
        if u_stock > 0:
            w_avg = np.mean([state.sectors[k].W for k in ["b", "i", "d_infra", "d_app"]])
            cm.unemployment_stock += u_stock * w_avg * dt

        # 技能贬值：各群组相对于自身历史峰值的工资损失
        for h, sector_keys in [
            (state.households.industrial_workers, ["i"]),
            (state.households.base_workers, ["b"]),
            (state.households.digital_high_skill, ["d_infra", "d_app"]),
            (state.households.service_workers, ["s_high", "s_low"]),
        ]:
            peak_w = np.mean([self._peak_wages[k] for k in sector_keys])
            w_loss = max(0, peak_w - h.avg_wage) * h.population
            h.cumulative_cost += w_loss * dt

        # 失业者成本
        state.households.unemployed.cumulative_cost += (
            state.households.unemployed.population * 3.0 * dt
        )

        # v3.3: 低端服务业"隐藏失业"成本
        informal_L = getattr(state, "informal_service_L", 0.0)
        informal_W = getattr(state, "informal_service_W", 0.0)
        if informal_L > 0:
            # 低端服务业工资极低 → 生活质量损失
            avg_formal_W = np.mean([state.sectors[k].W for k in ["b", "i", "d_infra", "d_app"]])
            wage_gap = max(0.0, avg_formal_W - informal_W) * informal_L
            state.households.unemployed.cumulative_cost += wage_gap * 0.5 * dt

        # 部门差距（生产率差距）
        prods = [s.Y / max(s.L, 0.01) for s in state.sectors.values()]
        if min(prods) > 1e-10:
            cm.sectoral_disparity = max(prods) / min(prods)

        # 不可逆指数
        crossed = len(state.fracture_status.active_fractures)
        cm.irreversibility_index = min(1.0, 0.15 * crossed)

        # 按群组的代价
        cm.cost_by_group = {
            "digital_high_skill": state.households.digital_high_skill.cumulative_cost,
            "industrial_workers": state.households.industrial_workers.cumulative_cost,
            "base_workers": state.households.base_workers.cumulative_cost,
            "service_workers": state.households.service_workers.cumulative_cost,  # v3.3
            "unemployed": state.households.unemployed.cumulative_cost,
        }

        # 代价集中度
        cm.cost_concentration = state.households.cost_gini
        cm.cost_ratio_industrial_digital = state.households.cost_ratio_industrial_vs_digital

    def _diagnose_deindustrialization(self, state: SystemState, dt: float):
        """去工业化质量诊断（周期性地更新）."""
        if not hasattr(self, '_prev_L_i'):
            self._prev_L_i = state.sectors["i"].L
            self._prev_W_i = state.sectors["i"].W
            self._deind_persistence = 0
            return

        dL_i = state.sectors["i"].L - self._prev_L_i
        dW_i = state.sectors["i"].W - self._prev_W_i

        if dL_i < -0.01:  # 工业劳动力在减少
            flow_to_d = max(0, (state.sectors["d_infra"].L + state.sectors["d_app"].L)
                            - getattr(self, '_prev_L_d', state.sectors["d_infra"].L + state.sectors["d_app"].L))
            flow_to_b = max(0, state.sectors["b"].L - getattr(self, '_prev_L_b', state.sectors["b"].L))
            flow_to_u = -dL_i - flow_to_d - flow_to_b

            if flow_to_d / abs(dL_i) > 0.5 and dW_i > 0:
                deind_type = "normal"
            elif flow_to_b / abs(dL_i) > 0.5 and dW_i < 0:
                deind_type = "premature"
            elif flow_to_u / abs(dL_i) > 0.3:
                deind_type = "collapse"
            elif state.sectors["i"].K / max(state.initial_K_i, 1.0) < 0.5:
                deind_type = "digital_leap_induced"
            else:
                deind_type = "mixed"

            state.cost_metrics.deind_type = deind_type
            state.cost_metrics.deind_quality = (
                1.0 if deind_type == "normal" else
                0.5 if deind_type == "mixed" else 0.2
            )

        # 保存上期值
        self._prev_L_i = state.sectors["i"].L
        self._prev_W_i = state.sectors["i"].W
        self._prev_L_d = state.sectors["d_infra"].L + state.sectors["d_app"].L
        self._prev_L_b = state.sectors["b"].L

    def update_reallocation(self, state: SystemState):
        """再分配：通过税收和转移支付."""
        tau_w = self.p["tau_w"]
        social_cov = self.p["social_coverage"]

        # 税收
        total_tax = state.gdp * tau_w

        # 转移支付：偏向失业者和基础部门
        u_pop = state.households.unemployed.population
        b_pop = state.households.base_workers.population
        total_needy = u_pop + b_pop * 0.3

        if total_needy > 0:
            state.households.unemployed.compensation_received = (
                total_tax * 0.6 * social_cov * u_pop / total_needy
            )
            state.households.base_workers.compensation_received = (
                total_tax * 0.3 * social_cov * (b_pop * 0.3) / total_needy
            )

        state.cost_metrics.compensation_gap = (
            state.households.unemployed.cumulative_cost
            - state.households.unemployed.compensation_received
        )

    def _update_household_balance_sheets(self, state: SystemState, dt: float):
        """v3.2: 更新五个家庭群组的储蓄和债务存量.

        追踪财富分配（存量）而不仅是收入流量。
        各群组的储蓄率不同——高收入群组储蓄率高，低收入群组可能负储蓄（借债）。
        """
        # 各群组储蓄率：数字>工业>服务业>基础>失业
        savings_rates = {
            "digital": 0.25,     # 高收入 → 高储蓄
            "industrial": 0.15,  # 中等收入
            "service": 0.08,     # 低收入 → 低储蓄
            "base": 0.10,        # 基础部门
            "unemployed": -0.05, # 负储蓄（消耗储蓄/借债）
        }

        for h, group_name in [
            (state.households.digital_high_skill, "digital"),
            (state.households.industrial_workers, "industrial"),
            (state.households.service_workers, "service"),
            (state.households.base_workers, "base"),
            (state.households.unemployed, "unemployed"),
        ]:
            if h.population > 0:
                sr = savings_rates[group_name]
                # 储蓄变化 = 收入 × 储蓄率
                saving_flow = h.income * sr * dt
                h.savings = max(0.0, h.savings + saving_flow)

                # 债务变化：当负储蓄（消费>收入）时债务增加
                if sr < 0:
                    # 入不敷出 → 债务累积
                    debt_increase = h.income * abs(sr) * dt
                    h.debt += debt_increase
                elif h.debt > 0 and h.savings > h.debt * 0.1:
                    # 有储蓄时逐步还债
                    debt_repayment = min(h.debt, h.savings * 0.3 * dt)
                    h.debt = max(0.0, h.debt - debt_repayment)
                    h.savings -= debt_repayment

                # 债务利息（年化5%）
                if h.debt > 0:
                    interest = h.debt * 0.05 * dt
                    h.debt += interest
                    # 利息支付减少可支配收入
                    h.income = max(0.01, h.income - interest)

                # v3.4: 股权价值变动 — 随部门资本增长而增值
                if group_name == "digital":
                    h.equity = (state.sectors["d_infra"].K + state.sectors["d_app"].K) * 0.40  # 数字高技能持有40%数字股权
                elif group_name == "industrial":
                    h.equity = state.sectors["i"].K * 0.25  # 工业劳动者持有25%工业股权
                elif group_name == "base":
                    h.equity = state.sectors["b"].K * 0.15
                elif group_name == "service":
                    h.equity = (state.sectors["s_high"].K + state.sectors["s_low"].K) * 0.10

        # v3.4: 更新系统级财富基尼系数
        state.wealth_gini = state.households.wealth_gini
