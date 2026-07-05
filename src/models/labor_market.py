"""劳动力市场 —— 部门间流动、工资、失业."""

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
        """计算部门间劳动力流动."""
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
        """应用劳动力流动，更新各部门L + v3.2 技能选择效应.

        v3.2 (Q5): 当工业劳动力流向数字部门时，技能匹配度高的劳动者
        优先流出。留在工业部门的劳动者平均技能下降。
        这修正了总量流动模型对策略B工业人力资本破坏的低估。
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
                selection_intensity = 0.3 + 0.7 * skill_match  # [0.3, 1.0]
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

        for key in ["b", "i", "d"]:
            state.sectors[key].L += net_flow[key] * dt
            state.sectors[key].L = max(0.5, state.sectors[key].L)
            # v3.2: 应用技能选择效应
            state.sectors[key].H += H_changes[key] * 0.5  # 阻尼系数—避免过度震荡
            state.sectors[key].H = max(0.2, min(state.sectors[key].H, 3.0))

        # 失业变化 + v3.3 低端服务业分流
        total_outflow = sum(max(0, -v) for v in net_flow.values())
        total_inflow = sum(max(0, v) for v in net_flow.values())
        net_unemployed = total_outflow - total_inflow

        # v3.3: 低端服务业分流 — 从萎缩部门流出的低技能劳动力
        # v3.2: 增加梯度流动——服务业饱和后流向基础部门
        for key, outflow in net_flow.items():
            if outflow < 0:  # 该部门在流失劳动力
                abs_outflow = abs(outflow)
                # 流失劳动力中技能不足以进入其他正式部门的比例
                skill_level = state.sectors[key].H
                low_skill_share = max(0.0, 1.0 - skill_level / 1.5)

                # v3.2: 检查低端服务业是否饱和
                total_formal_L = sum(state.sectors[k].L for k in ALL_SECTORS)
                informal_ratio = state.informal_service_L / max(total_formal_L, 1.0)
                # 饱和阈值：非正式就业超过正式就业的25%
                service_saturated = informal_ratio > 0.25

                informal_inflow = abs_outflow * low_skill_share * dt

                if service_saturated:
                    # 服务业饱和 → 部分回流到基础部门（压低基础工资）
                    back_to_base = informal_inflow * 0.4  # 40%被迫回到基础部门
                    state.sectors["b"].L += back_to_base
                    # 基础部门工资因劳动力过剩而下降
                    labor_surplus_penalty = min(0.2, informal_ratio * 0.15)
                    state.sectors["b"].W *= (1.0 - labor_surplus_penalty * dt)
                    # 剩余仍进入低端服务业（但工资更低）
                    informal_inflow *= 0.6
                    state.informal_service_L += informal_inflow
                else:
                    state.informal_service_L += informal_inflow

                net_unemployed -= (abs_outflow * low_skill_share * dt)  # 原始全部扣除

        # v3.2: 低端服务业回流的梯度路径
        if state.informal_service_L > 0:
            informal_ratio = state.informal_service_L / max(
                sum(state.sectors[k].L for k in ALL_SECTORS), 1.0)

            if informal_ratio < 0.15:
                # 未饱和：5%/年回流正式部门
                back_to_formal = state.informal_service_L * 0.05 * dt
            elif informal_ratio < 0.25:
                # 接近饱和：3%/年回流，成功率下降
                back_to_formal = state.informal_service_L * 0.03 * dt
            else:
                # 已饱和：1.5%/年回流，大量劳动力困在非正式就业
                back_to_formal = state.informal_service_L * 0.015 * dt

            state.informal_service_L -= back_to_formal
            net_unemployed += back_to_formal * 0.3  # 30%成功回到正式就业

        state.informal_service_L = max(0.0, state.informal_service_L)
        # 低端服务业工资 = 基础部门的60%（极低）
        state.informal_service_W = state.sectors["b"].W * 0.6

        state.households.unemployed.population += net_unemployed * dt
        state.households.unemployed.population = max(0.0, state.households.unemployed.population)

    def _update_households(self, state: SystemState):
        """将部门就业映射到家庭群组 (v3.2: 增加技能折旧追踪)."""
        # 数字部门 → 高技能数字家庭
        state.households.digital_high_skill.population = state.sectors["d"].L
        state.households.digital_high_skill.employed = state.sectors["d"].L
        state.households.digital_high_skill.avg_wage = state.sectors["d"].W
        # v3.2: 数字技能半衰期 — 技术迭代导致已有技能贬值
        # 数字技术更新周期约3-5年 → 年化折旧率5-8%
        digital_obsolescence_rate = 0.06  # 年化6%
        prev_d_skill = state.households.digital_high_skill.skill_level
        state.households.digital_high_skill.skill_level = (
            state.sectors["d"].H * (1.0 - digital_obsolescence_rate * 0.25)
            + prev_d_skill * digital_obsolescence_rate * 0.25  # 存量技能部分保留
        )

        # 工业部门 → 工业劳动者家庭
        state.households.industrial_workers.population = state.sectors["i"].L
        state.households.industrial_workers.employed = state.sectors["i"].L
        state.households.industrial_workers.avg_wage = state.sectors["i"].W
        # v3.2: 长期在服务业/失业的工业劳动者技能折旧
        if hasattr(self, '_industrial_skill_erosion_duration'):
            erosion_years = self._industrial_skill_erosion_duration
            # 离开工业岗位每年技能贬值2-3%
            erosion_rate = min(0.30, erosion_years * 0.025)
            state.households.industrial_workers.skill_level = (
                state.sectors["i"].H * (1.0 - erosion_rate)
            )
        else:
            state.households.industrial_workers.skill_level = state.sectors["i"].H

        # 基础部门 → 基础部门劳动者家庭
        state.households.base_workers.population = state.sectors["b"].L
        state.households.base_workers.employed = state.sectors["b"].L
        state.households.base_workers.avg_wage = state.sectors["b"].W
        state.households.base_workers.skill_level = state.sectors["b"].H

        # v3.3: 服务业部门 → 服务业劳动者家庭
        state.households.service_workers.population = state.sectors["s"].L
        state.households.service_workers.employed = state.sectors["s"].L
        state.households.service_workers.avg_wage = state.sectors["s"].W
        state.households.service_workers.skill_level = state.sectors["s"].H

        # 各群组收入
        for h in state.households.all_groups():
            h.income = h.employed * h.avg_wage + h.compensation_received
            h.consumption = h.income * 0.85  # 简化消费倾向

        # v3.2: 追踪工业技能侵蚀持续时间
        # 当工业就业低于峰值时累积侵蚀时间
        if not hasattr(self, '_peak_industrial_L'):
            self._peak_industrial_L = state.sectors["i"].L
            self._industrial_skill_erosion_duration = 0.0
        self._peak_industrial_L = max(self._peak_industrial_L, state.sectors["i"].L)
        if state.sectors["i"].L < self._peak_industrial_L * 0.9:
            self._industrial_skill_erosion_duration += 0.25  # 每季度
        else:
            self._industrial_skill_erosion_duration = max(0.0,
                self._industrial_skill_erosion_duration - 0.25 * 2)  # 恢复更快

        # 总失业率（v3.3: 统计失业 + 低端服务业"隐藏失业"）
        total_L = sum(s.L for s in state.sectors.values())
        # 有效失业 = 统计失业 + 低端服务业(按0.5折算——他们毕竟有收入)
        effective_unemployed = (state.households.unemployed.population
                                + state.informal_service_L * 0.5)
        total_pop = total_L + effective_unemployed + state.informal_service_L
        state.unemployment_rate = effective_unemployed / max(total_pop, 1.0)
