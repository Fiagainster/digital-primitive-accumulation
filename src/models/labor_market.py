"""劳动力市场 —— 部门间流动、工资、失业 (v3.5: 六部门 + 技能分化路由)."""

import numpy as np
from ..core.state import SystemState, ALL_SECTORS


class LaborMarket:
    """多部门劳动力市场模型."""

    def __init__(self, params: dict):
        self.p = params

    def step(self, state: SystemState, dt: float):
        """一个时间步内的劳动力市场更新."""
        # 1. 计算流动
        flows = self._compute_flows(state)
        # 2. 应用流动
        self._apply_flows(state, flows, dt)
        # 3. 更新家庭群组
        self._update_households(state)

    def _compute_flows(self, state: SystemState) -> dict:
        """计算部门间劳动力流动 (v3.5: 六部门 ALL_SECTORS)."""
        sectors = ALL_SECTORS
        flows = {}

        for src in sectors:
            for dst in sectors:
                if src == dst:
                    continue
                key = f"{src}->{dst}"
                L_src = state.sectors[src].L
                W_src = state.sectors[src].W
                W_dst = state.sectors[dst].W

                # 工资差距驱动
                wage_gap = max(0, (W_dst / max(W_src, 0.01) - 1))
                f_gap = min(1.0, wage_gap / 0.5)

                # 技能匹配
                H_src = state.sectors[src].H
                H_dst = state.sectors[dst].H
                match = min(H_src, H_dst) / max(H_src, H_dst, 0.01)
                f_skill = np.exp(-(1 - match) ** 2 / self.p["sigma_skill"] ** 2)

                # 转移成本
                f_cost = np.exp(-1.0 / max(W_src, 0.01))

                prob = self.p["lambda_flow"] * f_gap * f_skill * f_cost
                flows[key] = L_src * prob

        return flows

    def _apply_flows(self, state: SystemState, flows: dict, dt: float):
        """应用劳动力流动 + v3.2 技能选择 + v3.5 技能分化路由.

        v3.2 (Q5): 当工业劳动力流向数字部门时，技能匹配度高的劳动者
        优先流出。留在工业部门的劳动者平均技能下降。

        v3.5: 六部门全迭代 + 基于技能水平的差异化路由,
        高技能劳动者流向 d_app/s_high, 低技能流向 s_low/informal_service_L.
        """
        net_flow = {k: 0.0 for k in ALL_SECTORS}

        for key, flow in flows.items():
            src, dst = key.split("->")
            net_flow[src] -= flow
            net_flow[dst] += flow

        # v3.2: 技能选择效应 — 流出部门的H因"高技能先走"而下降
        H_changes = {k: 0.0 for k in ALL_SECTORS}
        for key, flow in flows.items():
            src, dst = key.split("->")
            if flow > 0 and state.sectors[src].L > 0.5:
                # 流出比例
                outflow_ratio = flow * dt / state.sectors[src].L
                # 技能匹配度：高匹配 → 高技能劳动者更容易流出
                H_src = state.sectors[src].H
                H_dst = state.sectors[dst].H
                skill_match = min(H_src, H_dst) / max(H_src, H_dst, 0.01)
                # 选择强度：技能匹配越高 → 流出的劳动者技能高于部门平均
                sel_base = self.p.get("skill_selection_base", 0.3)
                sel_match = self.p.get("skill_selection_match_factor", 0.7)
                selection_intensity = sel_base + sel_match * skill_match
                # 流出劳动者的平均技能 = 部门平均 × 选择强度
                leaver_H = H_src * selection_intensity
                # 流出后剩余劳动者的平均技能
                if outflow_ratio < 1.0:
                    remaining_H = (H_src - leaver_H * outflow_ratio) / (1.0 - outflow_ratio)
                else:
                    remaining_H = H_src * 0.8  # 大量流出时的底限
                H_changes[src] -= (H_src - remaining_H)
                # 流入部门H略微提升（高技能流入）
                H_changes[dst] += (leaver_H - H_dst) * outflow_ratio * 0.3  # 流入稀释效应

        # v3.5: 应用净流动到所有六个部门 (替换原来的 ["b", "i", "d"])
        for key in ALL_SECTORS:
            state.sectors[key].L += net_flow[key] * dt
            state.sectors[key].L = max(0.5, state.sectors[key].L)
            # v3.2: 应用技能选择效应
            state.sectors[key].H += H_changes[key] * self.p.get("H_flow_damping", 0.5)
            state.sectors[key].H = max(self.p.get("H_bounds_min", 0.2), min(state.sectors[key].H, self.p.get("H_bounds_max", 3.0)))

        # 失业变化 + v3.5 技能分化路由
        total_outflow = sum(max(0, -v) for v in net_flow.values())
        total_inflow = sum(max(0, v) for v in net_flow.values())
        net_unemployed = total_outflow - total_inflow

        # v3.5: 技能分化路由 — 对每个净流出部门，根据技能水平路由被替代劳动者
        # v3.5: 检查低端服务业饱和度
        total_formal_L = sum(state.sectors[k].L for k in ALL_SECTORS)
        informal_ratio = state.informal_service_L / max(total_formal_L, 1.0)
        service_saturated = informal_ratio > self.p.get("informal_ratio_saturation", 0.25)

        for key, outflow in net_flow.items():
            if outflow < 0:  # 该部门在流失劳动力
                abs_outflow = abs(outflow)
                # v3.5: 使用技能分化路由替代简单的 low_skill_share
                routed_to_formal, routed_to_informal = self._route_displaced_workers(
                    state, key, abs_outflow, dt)

                if service_saturated:
                    # 服务业饱和 → 部分回流到基础部门（压低基础工资）
                    back_ratio = self.p.get("saturated_back_to_base_ratio", 0.4)
                    back_to_base = routed_to_informal * back_ratio
                    state.sectors["b"].L += back_to_base
                    # 基础部门工资因劳动力过剩而下降
                    wage_penalty_max = self.p.get("labor_surplus_wage_penalty_max", 0.2)
                    wage_penalty_coef = self.p.get("labor_surplus_wage_coefficient", 0.15)
                    labor_surplus_penalty = min(wage_penalty_max, informal_ratio * wage_penalty_coef)
                    state.sectors["b"].W *= (1.0 - labor_surplus_penalty * dt)
                    # 剩余仍进入低端服务业（但工资更低）
                    state.informal_service_L += routed_to_informal * (1.0 - back_ratio)
                else:
                    state.informal_service_L += routed_to_informal

                # 扣减失业：已路由的劳动者不进入失业统计
                net_unemployed -= (routed_to_formal + routed_to_informal)

        # v3.2: 低端服务业回流的梯度路径
        if state.informal_service_L > 0:
            informal_ratio = state.informal_service_L / max(
                sum(state.sectors[k].L for k in ALL_SECTORS), 1.0)

            reflow_unsat = self.p.get("informal_reflow_unsaturated", 0.05)
            reflow_near = self.p.get("informal_reflow_near_sat", 0.03)
            reflow_sat = self.p.get("informal_reflow_saturated", 0.015)
            if informal_ratio < 0.15:
                back_to_formal = state.informal_service_L * reflow_unsat * dt
            elif informal_ratio < self.p.get("informal_ratio_saturation", 0.25):
                back_to_formal = state.informal_service_L * reflow_near * dt
            else:
                back_to_formal = state.informal_service_L * reflow_sat * dt

            state.informal_service_L -= back_to_formal
            reflow_success = self.p.get("informal_reflow_formal_success", 0.3)
            net_unemployed += back_to_formal * reflow_success

        state.informal_service_L = max(0.0, state.informal_service_L)
        # 低端服务业工资 = 基础部门的一定比例
        state.informal_service_W = state.sectors["b"].W * self.p.get("informal_wage_ratio", 0.6)

        # v3.7: 失业者再就业机制 — 当劳动力市场有净需求时, 失业者可以找到工作
        # 基于Pissarides(2000)搜寻-匹配理论: 就业率越高→失业者找到工作的概率越高
        unemployed_pool = state.households.unemployed.population
        if unemployed_pool > 0 and net_unemployed < 0:
            # 净就业需求 = -net_unemployed (正值表示需要更多劳动力)
            labor_demand = -net_unemployed * dt
            # 失业者找到工作的概率: 基于就业率(市场紧度)和技能匹配
            total_employed = sum(state.sectors[k].L for k in ALL_SECTORS)
            total_labor_force = total_employed + unemployed_pool + state.informal_service_L
            employment_rate = total_employed / max(total_labor_force, 1.0)
            # Beveridge曲线: 高就业率→更容易找到工作 (Pissarides匹配函数)
            job_finding_rate = 0.15 * employment_rate  # 基准15%/期, 按就业率调整
            reemployed = min(labor_demand, unemployed_pool * job_finding_rate * dt)
            state.households.unemployed.population -= reemployed
            state.households.unemployed.population = max(0.0, state.households.unemployed.population)
            # 再就业者进入部门(按部门规模分配)
            for key in ALL_SECTORS:
                if reemployed <= 0:
                    break
                share = state.sectors[key].L / max(total_employed, 1.0)
                added = reemployed * share
                state.sectors[key].L += added
            net_unemployed += reemployed / max(dt, 0.01)  # 调整净失业变化

        state.households.unemployed.population += net_unemployed * dt
        state.households.unemployed.population = max(0.0, state.households.unemployed.population)

    def _route_displaced_workers(self, state: SystemState, source_sector: str,
                                   displaced_count: float, dt: float):
        """v3.5: 根据技能水平将被替代劳动者路由到不同部门.

        高技能 (H > 1.0): 流向 d_app 和 s_high —— 技能可迁移到高技术服务业
        低技能 (H < 0.8): 流向 s_low, 溢出到 informal_service_L —— 只能进入低端岗位
        中等技能 (0.8 <= H <= 1.0): 按比例分配到各吸收性部门

        Args:
            state: 系统状态
            source_sector: 流出部门标识 (如 "i")
            displaced_count: 被替代劳动者数量 (流动率，尚未乘dt)
            dt: 时间步长

        Returns:
            (routed_to_formal, routed_to_informal): 路由到正式部门和非正式部门的
            劳动者数量（已乘dt的实际人数）
        """
        src_H = state.sectors[source_sector].H
        displaced = displaced_count * dt  # 实际人数

        if src_H > 1.0:
            # 高技能劳动者: 70% → d_app, 30% → s_high
            to_d_app = displaced * 0.7
            to_s_high = displaced * 0.3
            state.sectors["d_app"].L += to_d_app
            state.sectors["s_high"].L += to_s_high
            return displaced, 0.0  # 全部进入正式部门

        elif src_H < 0.8:
            # 低技能劳动者: 60% → s_low, 40% → informal_service_L
            to_s_low = displaced * 0.6
            to_informal = displaced * 0.4
            state.sectors["s_low"].L += to_s_low
            return to_s_low, to_informal

        else:
            # 中等技能: 均匀分配到各吸收性部门
            to_d_app = displaced * 0.25
            to_s_high = displaced * 0.25
            to_s_low = displaced * 0.25
            to_informal = displaced * 0.25
            state.sectors["d_app"].L += to_d_app
            state.sectors["s_high"].L += to_s_high
            state.sectors["s_low"].L += to_s_low
            return (to_d_app + to_s_high + to_s_low), to_informal

    def _update_households(self, state: SystemState):
        """v3.5: 将六部门就业映射到家庭群组.

        - d_infra + d_app → digital_high_skill (合并, 加权平均技能)
        - s_high + s_low → service_workers (合并, 加权平均工资/技能)
        - 就业质量综合指数: d_infra/d_app/i/s_high=1.0, b=0.8, s_low=0.6
        - 失业率: 六部门 + informal_service_L 隐藏失业
        """

        # ── v3.5: 数字部门 (d_infra + d_app) → 高技能数字家庭 ──
        d_infra_L = state.sectors["d_infra"].L
        d_app_L = state.sectors["d_app"].L
        total_digital_L = d_infra_L + d_app_L

        state.households.digital_high_skill.population = total_digital_L
        state.households.digital_high_skill.employed = total_digital_L

        if total_digital_L > 0:
            # 加权平均工资
            state.households.digital_high_skill.avg_wage = (
                d_infra_L * state.sectors["d_infra"].W
                + d_app_L * state.sectors["d_app"].W
            ) / total_digital_L
            # v3.5: 加权平均技能 (d_infra.H 与 d_app.H 的就业加权)
            combined_digital_H = (
                d_infra_L * state.sectors["d_infra"].H
                + d_app_L * state.sectors["d_app"].H
            ) / total_digital_L
        else:
            state.households.digital_high_skill.avg_wage = 0.0
            combined_digital_H = 1.0

        # v3.2: 数字技能半衰期 — 技术迭代导致已有技能贬值
        digital_obsolescence_rate = self.p.get("digital_skill_obsolescence_rate", 0.06)  # 年化6%
        prev_d_skill = state.households.digital_high_skill.skill_level
        state.households.digital_high_skill.skill_level = (
            combined_digital_H * (1.0 - digital_obsolescence_rate * 0.25)
            + prev_d_skill * digital_obsolescence_rate * 0.25  # 存量技能部分保留
        )

        # ── 工业部门 → 工业劳动者家庭 ──
        state.households.industrial_workers.population = state.sectors["i"].L
        state.households.industrial_workers.employed = state.sectors["i"].L
        state.households.industrial_workers.avg_wage = state.sectors["i"].W
        # v3.2: 长期在服务业/失业的工业劳动者技能折旧
        if hasattr(self, '_industrial_skill_erosion_duration'):
            erosion_years = self._industrial_skill_erosion_duration
            erosion_max = self.p.get("industrial_skill_erosion_max", 0.30)
            erosion_rate_coef = self.p.get("industrial_skill_erosion_rate", 0.025)
            erosion_rate = min(erosion_max, erosion_years * erosion_rate_coef)
            state.households.industrial_workers.skill_level = (
                state.sectors["i"].H * (1.0 - erosion_rate)
            )
        else:
            state.households.industrial_workers.skill_level = state.sectors["i"].H

        # ── 基础部门 → 基础部门劳动者家庭 ──
        state.households.base_workers.population = state.sectors["b"].L
        state.households.base_workers.employed = state.sectors["b"].L
        state.households.base_workers.avg_wage = state.sectors["b"].W
        state.households.base_workers.skill_level = state.sectors["b"].H

        # ── v3.5: 服务业部门 (s_high + s_low) → 服务业劳动者家庭 ──
        s_high_L = state.sectors["s_high"].L
        s_low_L = state.sectors["s_low"].L
        total_service_L = s_high_L + s_low_L

        state.households.service_workers.population = total_service_L
        state.households.service_workers.employed = total_service_L

        if total_service_L > 0:
            # 加权平均工资（s_high 工资高于 s_low）
            state.households.service_workers.avg_wage = (
                s_high_L * state.sectors["s_high"].W
                + s_low_L * state.sectors["s_low"].W
            ) / total_service_L
            # 加权平均技能
            state.households.service_workers.skill_level = (
                s_high_L * state.sectors["s_high"].H
                + s_low_L * state.sectors["s_low"].H
            ) / total_service_L
        else:
            state.households.service_workers.avg_wage = 0.0
            state.households.service_workers.skill_level = 1.0

        # ── 各群组收入 ──
        for h in state.households.all_groups():
            h.income = h.employed * h.avg_wage + h.compensation_received
            h.consumption = h.income * self.p.get("consumption_propensity", 0.85)  # 简化消费倾向

        # ── v3.2: 追踪工业技能侵蚀持续时间 ──
        if not hasattr(self, '_peak_industrial_L'):
            self._peak_industrial_L = state.sectors["i"].L
            self._industrial_skill_erosion_duration = 0.0
        self._peak_industrial_L = max(self._peak_industrial_L, state.sectors["i"].L)
        if state.sectors["i"].L < self._peak_industrial_L * 0.9:
            self._industrial_skill_erosion_duration += 0.25  # 每季度
        else:
            self._industrial_skill_erosion_duration = max(0.0,
                self._industrial_skill_erosion_duration - 0.25 * 2)  # 恢复更快

        # ── v3.5: 就业质量综合指数 ──
        # d_infra, d_app, i, s_high → quality=1.0 (正式稳定就业)
        # b → quality=0.8 (基础部门)
        # s_low → quality=0.6 (低端服务业/零工经济)
        total_formal_L = sum(state.sectors[k].L for k in ALL_SECTORS)
        if total_formal_L > 0:
            quality_map = {
                "d_infra": 1.0,
                "d_app": 1.0,
                "i": 1.0,
                "s_high": 1.0,
                "s_low": 0.6,
                "b": 0.8,
            }
            state.employment_quality_index = sum(
                state.sectors[k].L * quality_map[k] for k in ALL_SECTORS
            ) / total_formal_L
        else:
            state.employment_quality_index = 1.0

        # ── v3.5: 总失业率（六部门 + 低端服务业"隐藏失业"） ──
        total_L = sum(s.L for s in state.sectors.values())
        # 有效失业 = 统计失业 + 低端服务业(按0.5折算——他们毕竟有收入)
        effective_unemployed = (state.households.unemployed.population
                                + state.informal_service_L * 0.5)
        total_pop = total_L + effective_unemployed + state.informal_service_L
        state.unemployment_rate = effective_unemployed / max(total_pop, 1.0)
