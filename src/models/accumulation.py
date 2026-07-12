"""资本积累与投资分配 —— 三种积累策略 (v2.1 升级).

v2.1 升级：
- 策略B：增加反哺失败概率（数字部门可能截留利润而非反哺工业）
- 策略C：增加外部冲击事件（外资撤离、资源价格断崖）
- 策略A：增加工业保护的有效性约束
"""

import numpy as np
from ..core.state import SystemState, ALL_SECTORS


class AccumulationRegime:
    """积累策略基类 + 工厂方法."""

    def __init__(self, params: dict):
        self.p = params
        # v2.1: 策略状态追踪
        self._regime_state = {
            "feedback_active": False,
            "feedback_failed_count": 0,
            "fdi_shock_active": False,
            "resource_shock_active": False,
            "tech_import_break_active": False,  # v3.2: 独立技术进口断裂冲击
            "shock_timer": 0.0,
        }

    @staticmethod
    def create(regime: str, params: dict) -> "AccumulationRegime":
        if regime == "A":
            return GradualAccumulation(params)
        elif regime == "B":
            return LeapAccumulation(params)
        elif regime == "C":
            return SubstitutionAccumulation(params)
        elif regime.startswith("M"):  # v3.0: 混合策略
            from .mixed_strategy import MixedStrategy
            # 从参数中解析混合配比
            lambda_vec = params.get("mixed_lambda", (1/3, 1/3, 1/3))
            mixture_name = params.get("mixed_name", None)
            return MixedStrategy(params, lambda_vec=lambda_vec,
                                mixture_name=mixture_name)
        else:
            raise ValueError(f"Unknown regime: {regime}")

    def step(self, state: SystemState, dt: float):
        """一个时间步的积累逻辑."""
        # 1. 计算总剩余和积累基金
        self._compute_surplus(state)
        # 2. 计算提取率
        tau = self.extraction_rate(state)
        # 3. 投资分配
        phi = self.investment_allocation(state, tau)
        # 4. 应用投资（含吸收能力约束）
        self._apply_investment(state, phi, tau, dt)

    def _compute_surplus(self, state: SystemState):
        """计算总剩余."""
        Y_total = sum(s.Y for s in state.sectors.values())
        state.gdp = Y_total

        # 总消费
        C_total = sum(h.consumption for h in state.households.all_groups())
        gov_consumption = self.p.get("gov_consumption_share", 0.1) * Y_total
        depreciation = sum(
            self.p[f"delta_{k}"] * state.sectors[k].K
            for k in ALL_SECTORS
        )
        state.total_surplus = Y_total - C_total - gov_consumption - depreciation
        state.accumulation_fund = max(0.0, state.total_surplus * self.p.get("accumulation_fund_share", 0.8))

    def extraction_rate(self, state: SystemState) -> float:
        raise NotImplementedError

    def investment_allocation(self, state: SystemState, tau: float) -> dict:
        raise NotImplementedError

    def _apply_investment(self, state: SystemState, phi: dict, tau: float, dt: float):
        """应用投资 + 信贷配给 + 吸收能力约束 + 资本更新 + v2.2 数字密度."""
        fund = state.accumulation_fund

        # v3.3: 信贷配给 — 金融发展不足时，不是所有积累基金都能转化为投资
        inst_cap = self.p.get("institution_capacity", 0.6)
        ext_dep = state.external_dependency
        credit_ext_factor = self.p.get("credit_ext_dep_factor", 0.4)
        credit_availability = inst_cap * (1.0 - ext_dep * credit_ext_factor)
        credit_availability = max(
            self.p.get("credit_availability_min", 0.25),
            min(1.0, credit_availability)
        )
        state.credit_rationing_ratio = credit_availability
        fund = fund * credit_availability

        total_waste = 0.0

        for key in ALL_SECTORS:
            s = state.sectors[key]
            I_planned = phi.get(key, 0.0) * fund

            # 吸收能力
            nu = self.p[f"nu_{key}"]
            absorb_DL_bonus = self.p.get("absorb_DL_bonus_factor", 0.5)
            absorb_H_ref = self.p.get("absorb_H_reference", 0.5)
            absorb = nu * s.K * (1 + absorb_DL_bonus * s.DL) * (s.H / absorb_H_ref) ** self.p["chi_absorb"]
            I_actual = min(I_planned, absorb)
            total_waste += I_planned - I_actual

            # v2.2: 投资中的数字密度提升
            # 每单位投资中，数字技术服务/软件/智能设备的比例取决于部门数字化水平
            if I_actual > 0:
                # 数字投资比例 ≈ 数字化水平 × 部门数字化弹性
                ds_i = self.p.get("digital_share_i", 0.3)
                ds_b = self.p.get("digital_share_b", 0.15)
                ds_other = self.p.get("digital_share_other", 0.5)
                digital_share = s.DL * (ds_i if key == "i" else ds_b if key == "b" else ds_other)
                # 数字密度增量 = 数字投资 / 总资本存量
                dd_increment = digital_share * I_actual / max(s.K, 0.01) * dt
                # 折旧也会减少数字密度（旧数字设备淘汰）
                obsolescence_dd = s.digital_density * self.p[f"delta_{key}"] * dt
                s.digital_density += dd_increment - obsolescence_dd
                s.digital_density = np.clip(s.digital_density, 0.0, 0.95)

            # 资本动态
            delta = self.p[f"delta_{key}"]
            obs_threshold = self.p.get("obsolescence_delta_threshold", 0.05)
            obsolescence = self.p["psi_obsolescence"] * s.K * max(0, obs_threshold - delta)
            dK = I_actual - delta * s.K - obsolescence

            s.K += dK * dt
            s.K = max(self.p.get("K_floor", 0.01), s.K)
            s.I = I_actual
            s.absorption_cap = absorb

        # 投资浪费记录在SFC残差中
        state.transaction_residual += total_waste


class GradualAccumulation(AccumulationRegime):
    """策略A：渐进积累 (v3.2 升级).

    v3.2: 增加内生成本机制 — 保护僵化、渐进时间税、
          既得利益刚性、中间收入陷阱。
    v2.1: 增加"保护效率"约束 — 当技术差距过大时，
    保护工业反而使工业丧失数字化转型的紧迫性和资源。
    """

    def step(self, state: SystemState, dt: float):
        """v3.2: 覆写step以加入A的内生成本."""
        # 1. 计算总剩余和积累基金
        self._compute_surplus(state)
        # 2. 计算提取率
        tau = self.extraction_rate(state)
        # 3. 投资分配
        phi = self.investment_allocation(state, tau)
        # 4. 应用投资
        self._apply_investment(state, phi, tau, dt)
        # v3.2: 5. 应用策略A的内生成本
        self._apply_A_endogenous_costs(state, dt)

    def extraction_rate(self, state: SystemState) -> float:
        p = self.p
        g_i = (state.sectors["i"].Y / max(state.gdp, 0.01))
        g_i_max = p.get("A_g_i_max", 0.5)
        tau = (p["A_tau_min"]
               + (p["A_tau_max"] - p["A_tau_min"])
               * min(1.0, g_i / g_i_max) ** p["A_theta_elasticity"])

        # 安全边际
        K_ratio = state.sectors["i"].K / max(state.initial_K_i, 1.0)
        safety = p["A_safety_margin"]
        if K_ratio < safety:
            tau *= K_ratio / safety

        # v2.2: 保护过度惩罚 — 当技术差距大且工业数字密度低时
        # 过度保护会导致数字投资不足
        # 但如果工业已经高度数字化（数字密度高），说明"保护+升级"路径有效
        G = state.tech_gap
        dd_i = state.sectors["i"].digital_density
        overprotect_G = p.get("middle_income_G_threshold", 4.0) + 1.0  # 默认5.0
        overprotect_DD = p.get("middle_income_DD_threshold", 0.3)
        if G > overprotect_G and dd_i < overprotect_DD:
            # 差距大 + 工业数字化不足 → 需要更多数字投资
            tau = max(tau, p["A_tau_min"] * 1.5)
        elif G > overprotect_G and dd_i >= overprotect_DD:
            # 差距大 + 工业已数字化 → 保护是有意义的，不必强制提高提取
            pass  # 允许维持较低提取率

        return tau

    def investment_allocation(self, state: SystemState, tau: float) -> dict:
        total_K = sum(s.K for s in state.sectors.values())
        phi = {}
        for key in ALL_SECTORS:
            phi[key] = state.sectors[key].K / max(total_K, 1.0)

        # 略偏向数字（两个子部门分享溢价）
        premium = self.p["A_phi_d_premium"]
        phi["d_infra"] += premium * 0.4
        phi["d_app"] += premium * 0.6
        phi["i"] -= premium * 0.35
        phi["b"] -= premium * 0.15
        phi["s_high"] -= premium * 0.05
        phi["s_low"] -= premium * 0.05

        # 归一化
        total = sum(phi.values())
        return {k: v / total for k, v in phi.items()}

    def _apply_A_endogenous_costs(self, state: SystemState, dt: float):
        """v3.7.2: 策略A的内生成本 — 含自修正机制.

        纯策略A运行时，这四种成本也会生效（之前仅在混合策略框架中触发）。

        v3.7.2 改进：
        - 中间收入陷阱恢复速度从2.0→4.0（破坏:恢复比从4:1→2:1）
        - 新增保护持续时间衰减机制（条件改善时缓慢退出保护状态）
        - 既得利益刚性在工业升级成功后可逆转
        """
        if not hasattr(self, '_A_cost_state'):
            self._A_cost_state = {
                "protection_duration": 0.0,
                "cumulative_lag": 0.0,
                "vested_interest_intensity": 0.0,
                "middle_income_trap_years": 0.0,
            }
        cs = self._A_cost_state

        p = self.p
        dd_i = state.sectors["i"].digital_density
        G = state.tech_gap

        # 1. 保护僵化成本
        safety_margin = p.get("A_safety_margin", 0.7)
        K_i_ratio = state.sectors["i"].K / max(state.initial_K_i, 1.0)
        protection_excess = max(0.0, K_i_ratio / max(safety_margin, 0.01) - 1.0)
        protection_coef = p.get("protection_excess_coefficient", 0.08)
        protection_floor = p.get("A_protection_excess_floor", 0.85)
        efficiency_loss = protection_excess * protection_coef
        state.sectors["i"].Y *= max(protection_floor, 1.0 - efficiency_loss * dt)

        # v3.7.2: 保护持续时间跟踪——条件改善时衰减
        if G > 4.0 or dd_i < 0.2:
            # 技术差距大或数字密度低 → 保护期累积
            cs["protection_duration"] += dt
        elif dd_i > 0.4 and G < 3.0:
            # 工业数字化成功+技术差距缩小 → 保护可以逐步退出
            cs["protection_duration"] = max(0.0,
                cs["protection_duration"] - dt * 1.5)

        # 2. 渐进主义时间税 — 走得太慢，世界在前进
        g_world = p.get("g_world", 0.03)
        A_d_app = state.sectors["d_app"].A
        prev_A_d = getattr(self, '_prev_A_d', A_d_app)
        g_d = (A_d_app / max(prev_A_d, 0.01) - 1.0)
        self._prev_A_d = A_d_app
        natural_buffer = p.get("A_time_tax_natural_buffer", 0.01)
        lag_rate = max(0.0, g_world - g_d - natural_buffer)
        cs["cumulative_lag"] += lag_rate * dt
        # v3.7.2: 累积滞后也应有衰减——如果数字部门加速赶上
        if lag_rate <= 0.0:
            cs["cumulative_lag"] = max(0.0,
                cs["cumulative_lag"] + lag_rate * dt)  # lag_rate为负时减少
        accum_threshold = p.get("time_tax_accumulation_threshold", 0.1)
        penalty_rate = p.get("time_tax_penalty_rate", 0.03)
        if cs["cumulative_lag"] > accum_threshold:
            lag_penalty = cs["cumulative_lag"] * penalty_rate * dt
            state.tech_gap *= (1.0 + lag_penalty)

        # 3. 既得利益刚性
        vested_duration = p.get("vested_interest_duration", 15.0)
        if cs["protection_duration"] > vested_duration:
            industry_emp_share = state.sectors["i"].L / max(state.households.total_population, 1.0)
            vested_coef = p.get("vested_interest_coefficient", 0.01)
            vested_max = p.get("A_vested_interest_max_intensity", 0.5)
            cs["vested_interest_intensity"] = min(
                vested_max,
                (cs["protection_duration"] - vested_duration) * vested_coef * industry_emp_share
            )
            reform_resistance = cs["vested_interest_intensity"]
            invest_floor = p.get("A_vested_interest_invest_floor", 0.5)
            for d_key in ["d_infra", "d_app"]:
                state.sectors[d_key].I *= max(invest_floor, 1.0 - reform_resistance * dt)
        else:
            # v3.7.2: 保护期未满15年或保护已退出 → 既得利益强度衰减
            cs["vested_interest_intensity"] = max(0.0,
                cs["vested_interest_intensity"] - 0.015 * dt)

        # 4. 中间收入陷阱
        trap_DD = p.get("middle_income_DD_threshold", 0.3)
        trap_G = p.get("middle_income_G_threshold", 4.0)
        trap_duration = p.get("middle_income_duration", 10.0)
        trap_penalty_rate = p.get("A_trap_penalty_rate", 0.01)
        trap_K_floor = p.get("A_trap_K_floor", 0.90)
        # v3.7.2: 恢复速度从2.0→4.0，破坏:恢复比从4:1→2:1
        trap_recovery = p.get("A_trap_recovery_rate", 4.0)
        if dd_i < trap_DD and G > trap_G:
            cs["middle_income_trap_years"] += dt
            if cs["middle_income_trap_years"] > trap_duration:
                trap_penalty = trap_penalty_rate * (cs["middle_income_trap_years"] - trap_duration)
                state.sectors["i"].K *= max(trap_K_floor, 1.0 - trap_penalty * dt)
                state.tech_gap *= (1.0 + trap_penalty * 0.5 * dt)
        else:
            # v3.7.2: 恢复速度加倍（4.0/年 vs 原2.0/年）
            cs["middle_income_trap_years"] = max(0.0,
                cs["middle_income_trap_years"] - dt * trap_recovery)


class LeapAccumulation(AccumulationRegime):
    """策略B：跳跃积累 (v2.2 升级).

    v2.2 新增内生社会经济成本：
    - 高提取率 → 就业破坏与消费萎缩 → 国内需求下降
    - 数字经济高利润 → 资产泡沫/贫富差距 → 渐进式社会压力
    - 这些成本不会立刻导致崩溃，但会削弱B的长期优势

    v2.1 新增反哺失败机制：
    - 即使DL_d达到了反馈阈值，数字部门也可能截留利润
    - 失败概率取决于：数字部门利润率、外部竞争压力、制度能力
    - 反哺失败时，数字部门继续扩张但工业得不到补偿 → 工业枯萎风险
    """

    def _agg_digital_K(self, state: SystemState) -> float:
        """v3.5: 聚合数字部门资本."""
        return state.sectors["d_infra"].K + state.sectors["d_app"].K

    def _agg_digital_Y(self, state: SystemState) -> float:
        """v3.5: 聚合数字部门产出."""
        return state.sectors["d_infra"].Y + state.sectors["d_app"].Y

    def _agg_digital_L(self, state: SystemState) -> float:
        """v3.5: 聚合数字部门劳动力."""
        return state.sectors["d_infra"].L + state.sectors["d_app"].L

    def _agg_digital_DL(self, state: SystemState) -> float:
        """v3.5: 聚合数字部门数字化水平(加权)."""
        K_total = self._agg_digital_K(state)
        if K_total > 0:
            return (state.sectors["d_infra"].DL * state.sectors["d_infra"].K
                    + state.sectors["d_app"].DL * state.sectors["d_app"].K) / K_total
        return 0.0

    def _agg_digital_r(self, state: SystemState) -> float:
        """v3.5: 聚合数字部门利润率(加权)."""
        K_total = self._agg_digital_K(state)
        if K_total > 0:
            return (state.sectors["d_infra"].r * state.sectors["d_infra"].K
                    + state.sectors["d_app"].r * state.sectors["d_app"].K) / K_total
        return 0.0

    def extraction_rate(self, state: SystemState) -> float:
        """计算策略B的提取率.

        v3.7 修复: 原公式 tau = min(g_target * K_d_invest_needed / fund, ...) 中
        K_d_invest_needed = g_target * K_d_total, 导致 g_target 被乘了两次(g_target²).
        正确公式: tau = min(K_d_invest_needed / fund, B_tau_actual_max)
                  = min(g_target * K_d_total / fund, B_tau_actual_max)
        经济含义: 提取率 = 满足数字部门目标增长所需的投资 / 积累基金,
        即"我们需要多少投资"除以"我们有多少钱"。
        """
        p = self.p
        g_target = p["B_g_target"]
        K_d_total = self._agg_digital_K(state)
        K_d_invest_needed = g_target * K_d_total
        fund = max(state.accumulation_fund, 1.0)
        tau = min(K_d_invest_needed / fund, p["B_tau_actual_max"])

        # v2.2: 政治约束 — 提取率受政治支持度制约
        ps = state.political_support
        if ps.policy_constraint < 1.0:
            tau *= ps.policy_constraint

        # 工业容忍度检查
        K_i_ratio = state.sectors["i"].K / max(state.initial_K_i, 1.0)
        tolerance = p["B_industry_tolerance"]
        if K_i_ratio < tolerance:
            tau_floor = p.get("B_industry_tau_floor", 0.3)
            tau *= max(tau_floor, K_i_ratio / tolerance)

        # v2.1: 提取超调风险 — 数字部门增长目标驱使下，
        # 即使工业已经脆弱，提取率也可能"意外"偏高
        G = state.tech_gap
        panic_G = p.get("B_panic_extraction_G_threshold", 8.0)
        panic_Kr = p.get("B_panic_extraction_K_ratio", 0.6)
        panic_bonus = p.get("B_panic_extraction_max_bonus", 0.15)
        if G > panic_G and K_i_ratio < panic_Kr:
            # 技术差距大 + 工业脆弱 → 恐慌性提取（all-in数字化）
            tau = min(p["B_tau_actual_max"], tau * (1.0 + np.random.uniform(0, panic_bonus)))

        return tau

    def investment_allocation(self, state: SystemState, tau: float) -> dict:
        p = self.p
        DL_d = self._agg_digital_DL(state)
        # v3.5: infra/app投资分配比例 — infra需要更多投入但回报慢
        infra_share_of_digital = 0.35  # 数字投资中infra占35%

        if DL_d >= p["B_feedback_trigger"]:
            feedback_success = self._feedback_success_probability(state)
            if feedback_success:
                fb = p["B_feedback_ratio"]
                resistance = self._regime_state.get("feedback_resistance", 0.0)
                exec_floor = p.get("B_feedback_execution_floor", 0.3)
                resist_coef = p.get("B_feedback_resistance_coefficient", 0.15)
                execution_rate = max(exec_floor, 1.0 - resistance * resist_coef)
                fb_effective = fb * execution_rate
                self._regime_state["feedback_active"] = True
                self._regime_state["feedback_effective_rate"] = fb_effective
                phi_d_total = 0.55 - fb_effective * 0.5
                return {"b": 0.10, "i": 0.25 + fb_effective * 0.5,
                        "d_infra": phi_d_total * infra_share_of_digital,
                        "d_app": phi_d_total * (1 - infra_share_of_digital),
                        "s_high": 0.05, "s_low": 0.05}
            else:
                self._regime_state["feedback_failed_count"] += 1
                phi_d_total = min(0.85, p["B_phi_d_jump"] + 0.10)
                return {"b": 0.03, "i": 1.0 - phi_d_total - 0.04,
                        "d_infra": phi_d_total * infra_share_of_digital,
                        "d_app": phi_d_total * (1 - infra_share_of_digital),
                        "s_high": 0.02, "s_low": 0.02}
        else:
            phi_d_total = p["B_phi_d_jump"]
            return {"b": 0.03, "i": 1.0 - phi_d_total - 0.04,
                    "d_infra": phi_d_total * infra_share_of_digital,
                    "d_app": phi_d_total * (1 - infra_share_of_digital),
                    "s_high": 0.02, "s_low": 0.02}

    def _apply_socioeconomic_costs(self, state: SystemState, tau: float, dt: float):
        """v2.2: 计算并应用高提取策略的社会经济成本."""
        p = self.p
        # 1. 就业破坏效应
        tau_baseline = p.get("B_tau_baseline", 0.30)
        if tau > tau_baseline:
            excess_tau = tau - tau_baseline
            unemp_coef = p.get("B_unemployment_coefficient", 0.05)
            extra_unemployment = (
                state.sectors["i"].L * excess_tau * unemp_coef * dt
            )
            absorptive_d = self._agg_digital_L(state) * self._agg_digital_DL(state) * 0.02 * dt
            net_unemployment = max(0, extra_unemployment - absorptive_d)
            state.households.unemployed.population += net_unemployment
            state.households.unemployed.population = max(
                0, state.households.unemployed.population
            )
            state.sectors["i"].L -= net_unemployment * 0.5
            L_floor = p.get("L_floor", 0.5)
            state.sectors["i"].L = max(L_floor, state.sectors["i"].L)

        # 2. 消费萎缩效应
        disp_base = p.get("B_disposable_income_base", 0.70)
        disp_sens = p.get("B_disposable_income_tau_sens", 0.5)
        disposable_share = max(0.3, disp_base - tau * disp_sens)
        consumption_drag = (disp_base - disposable_share) * p.get("B_consumption_drag_coefficient", 0.15)
        if consumption_drag > 0:
            cons_floor = p.get("B_consumption_output_floor", 0.85)
            demand_constraint = 1.0 - consumption_drag * dt
            state.sectors["i"].Y *= max(cons_floor, demand_constraint)

        # 3. 资产泡沫/不平等风险
        r_d = self._agg_digital_r(state)
        bubble_trigger = p.get("B_bubble_trigger_r", 0.25)
        bubble_sens = p.get("B_bubble_pressure_sensitivity", 0.3)
        if r_d > bubble_trigger:
            bubble_pressure = (r_d - bubble_trigger) * bubble_sens * dt
            self._regime_state["bubble_pressure"] = (
                self._regime_state.get("bubble_pressure", 0.0) + bubble_pressure
            )
        else:
            bubble_decay = p.get("B_bubble_decay_rate", 0.01)
            self._regime_state["bubble_pressure"] = max(
                0, self._regime_state.get("bubble_pressure", 0.0) - bubble_decay * dt
            )

    def step(self, state: SystemState, dt: float):
        """v2.2: 覆写step以加入社会经济成本和泡沫破裂风险."""
        # 1. 计算总剩余和积累基金
        self._compute_surplus(state)
        # 2. 计算提取率
        tau = self.extraction_rate(state)
        # 3. 投资分配
        phi = self.investment_allocation(state, tau)
        # 4. 应用投资
        self._apply_investment(state, phi, tau, dt)
        # v2.2: 5. 应用社会经济成本
        self._apply_socioeconomic_costs(state, tau, dt)
        # v2.2: 6. 检查泡沫破裂
        self._check_bubble_burst(state, dt)

    def _feedback_success_probability(self, state: SystemState) -> bool:
        """计算反哺成功的概率 (v3.5: 使用聚合数字指标)."""
        p = self.p
        base_success = p.get("B_feedback_base_success", 0.85)

        r_d = self._agg_digital_r(state)
        profit_high = p.get("B_feedback_profit_penalty_high", 0.20)
        profit_mid = p.get("B_feedback_profit_penalty_mid", 0.10)
        if r_d > 0.30:
            base_success -= profit_high
        elif r_d > 0.20:
            base_success -= profit_mid

        inst_cap = p.get("institution_capacity", 0.6)
        base_success += (inst_cap - 0.5) * 0.30

        d_gdp_share = self._agg_digital_Y(state) / max(state.gdp, 0.01)
        feedback_resistance = d_gdp_share * (r_d / 0.20)
        base_success -= feedback_resistance * 0.08

        G = state.tech_gap
        if G > 8.0:
            base_success -= 0.15

        failed = self._regime_state["feedback_failed_count"]
        if failed > 0:
            base_success -= failed * 0.05

        min_p = p.get("B_feedback_min_success_prob", 0.15)
        max_p = p.get("B_feedback_max_success_prob", 0.95)
        base_success = max(min_p, min(max_p, base_success))
        self._regime_state["feedback_resistance"] = feedback_resistance
        return np.random.random() < base_success

    def _check_bubble_burst(self, state: SystemState, dt: float):
        """v2.2: 数字泡沫破裂风险."""
        p = self.p
        bp = self._regime_state.get("bubble_pressure", 0.0)
        if bp < 2.0:  # 需要显著积累才会破裂
            return

        # 制度能力决定是"硬破裂"还是"软着陆"
        inst_cap = p.get("institution_capacity", 0.6)

        # 硬破裂概率 = 泡沫压力 × 脆弱性 / 制度缓冲
        base_prob = p.get("B_bubble_burst_base_prob", 0.003)
        burst_prob = base_prob * (bp / 2.0) ** 3 * dt

        # 脆弱性因子（条件触发，不是随机骰子）
        fragility = 1.0
        u_rate = state.unemployment_rate
        if u_rate > 0.15:
            fragility *= 1.5
        if u_rate > 0.20:
            fragility *= 2.0

        ps = state.political_support
        if ps.support_index < 0.35:
            fragility *= 2.5  # 政治崩盘时泡沫破裂几乎是必然的

        # 制度缓冲：强制度可以有序去泡沫
        fragility *= (1.5 - inst_cap)

        # 泡沫破裂判定
        if np.random.random() < burst_prob * fragility:
            # === 硬破裂：同时影响infra和app ===
            severity = min(0.45, 0.15 + bp * 0.06)
            # app受到的冲击更大（估值泡沫主要在应用层）
            infra_scale = p.get("B_d_infra_severity_scale", 0.6)
            app_scale = p.get("B_d_app_severity_scale", 1.2)
            state.sectors["d_infra"].K *= (1.0 - severity * infra_scale)
            state.sectors["d_app"].K *= (1.0 - severity * app_scale)

            # 就业冲击
            layoff_rate = severity * 1.2
            L_floor = p.get("L_floor", 0.5)
            laid_off_infra = state.sectors["d_infra"].L * layoff_rate * 0.5
            laid_off_app = state.sectors["d_app"].L * layoff_rate * 1.5
            state.sectors["d_infra"].L = max(L_floor * 0.6, state.sectors["d_infra"].L - laid_off_infra)
            state.sectors["d_app"].L = max(L_floor, state.sectors["d_app"].L - laid_off_app)
            state.households.unemployed.population += (laid_off_infra + laid_off_app) * 0.6

            # 数据资产贬值
            state.data_assets.producer_data *= (1.0 - severity * 0.8)
            state.data_assets.consumer_data *= (1.0 - severity * 0.8)

            # 政治冲击
            state.political_support.support_index *= (1.0 - severity * 0.5)

            self._regime_state["bubble_pressure"] *= 0.35
            self._regime_state["feedback_failed_count"] += 1
            self._regime_state.setdefault("bubble_bursts", 0)
            self._regime_state["bubble_bursts"] += 1

            # v3.5: 泡沫破裂财富效应 — 应用层股权缩水更严重
            state.households.digital_high_skill.equity *= (1.0 - severity)
            state.households.industrial_workers.equity *= (1.0 - severity * 0.3)

        elif inst_cap > p.get("B_soft_landing_inst_cap", 0.7) and bp > p.get("B_soft_landing_bp_threshold", 3.0):
            # === 软着陆 ===
            soft_decay = p.get("B_soft_landing_bp_decay", 0.02)
            self._regime_state["bubble_pressure"] -= soft_decay * dt
            state.sectors["d_infra"].K *= (1.0 - 0.003 * dt)
            state.sectors["d_app"].K *= (1.0 - 0.005 * dt)
            state.political_support.support_index *= 0.995


class SubstitutionAccumulation(AccumulationRegime):
    """策略C：替代积累 (v3.7.2 升级).

    v3.7.2 新增：连续内生成本机制 —— 与策略A/B的机制丰富度对称。
    - 荷兰病支出效应：资源繁荣→汇率升值→工业竞争力持续退化
    - 政策俘获：外部依赖→外资利益绑架国内政策→投资分配扭曲
    - 技术退化：长期技术进口→自主创新持续萎缩
    - 消费主义陷阱：FDI偏向消费→国内储蓄率下降→积累能力减弱
    - 人才外流：依附加深→高技能人才持续外流→人力资本退化

    v2.2 新增：完美风暴机制 — FDI和资源同时断裂时，技术差距加速扩大
    v2.1 新增外部冲击机制：
    - 外资突然撤离：概率性事件，对数字部门投资造成冲击
    - 资源价格断崖：价格跌至极低水平，出口收入锐减
    - 冲击的严重程度取决于外部依赖度
    """

    def step(self, state: SystemState, dt: float):
        """v3.7.2: 覆写step以加入C的连续内生成本."""
        self._compute_surplus(state)
        tau = self.extraction_rate(state)
        phi = self.investment_allocation(state, tau, dt)
        self._apply_investment(state, phi, tau, dt)
        # v3.7.2: 应用策略C的连续内生成本
        self._apply_C_endogenous_costs(state, dt)

    def extraction_rate(self, state: SystemState) -> float:
        # v2.1: 外部冲击影响提取能力
        base_rate = 0.25

        # 如果处于资源冲击中，可提取的剩余减少
        if self._regime_state["resource_shock_active"]:
            base_rate *= 0.6

        return base_rate

    def investment_allocation(self, state: SystemState, tau: float, dt: float = 0.25) -> dict:
        p = self.p

        self._check_external_shocks(state, dt)

        export_share = p["C_resource_export_intensity"]
        fdi_share = p["C_fdi_openness"]
        tech_import = p["C_tech_import_preference"]

        fdi_effective = fdi_share
        if self._regime_state["fdi_shock_active"]:
            fdi_effective *= 0.3

        # v3.5: 数字部门投资分配 — FDI主要流向app层
        phi_d_total = 0.25 + 0.15 * fdi_effective + 0.10 * tech_import
        if self._regime_state["resource_shock_active"]:
            phi_d_total *= 0.7

        # infra/app分配: C策略下app获益更多（FDI偏好应用层）
        phi_d_infra = phi_d_total * 0.25
        phi_d_app = phi_d_total * 0.75

        phi_i = 0.30 + 0.10 * export_share
        phi_s_high = 0.08
        phi_s_low = 0.07
        phi_b = 1.0 - phi_d_infra - phi_d_app - phi_i - phi_s_high - phi_s_low

        return {"b": max(0.03, phi_b), "i": phi_i,
                "d_infra": max(0.02, phi_d_infra),
                "d_app": min(0.55, phi_d_app),
                "s_high": phi_s_high, "s_low": phi_s_low}

    def _check_external_shocks(self, state: SystemState, dt: float = 0.25):
        """检测外部冲击事件 (v3.2 增强 — 三重独立冲击).

        - FDI撤离：当外部依赖度高 + 全球经济环境恶化时概率触发
        - 资源价格断崖：资源出口强度高时，价格波动可触发断崖
        - v3.2: 技术进口断裂：技术进口偏好高+封锁升级时概率触发（独立于FDI/资源）
        - v2.2: 双重/三重冲击 — 多个冲击可能同时发生，互相放大
        """
        p = self.p

        # === FDI撤离冲击 ===
        ext_dep = state.external_dependency
        fdi_avail = p.get("fdi_availability", 1.0)

        if not self._regime_state["fdi_shock_active"]:
            shock_prob = (ext_dep * 0.20 + (1 - fdi_avail) * 0.30) * 0.03
            if np.random.random() < shock_prob:
                self._regime_state["fdi_shock_active"] = True
                self._regime_state["shock_timer"] = 0.0

        if self._regime_state["fdi_shock_active"]:
            self._regime_state["shock_timer"] += 0.25
            if self._regime_state["shock_timer"] > np.random.uniform(8.0, 15.0):
                self._regime_state["fdi_shock_active"] = False

        # === 资源价格断崖冲击 ===
        if not self._regime_state["resource_shock_active"]:
            export_intensity = p.get("C_resource_export_intensity", 0.6)
            sigma_p = p.get("sigma_p", 0.15)
            shock_prob = export_intensity * sigma_p * 0.10
            if np.random.random() < shock_prob:
                self._regime_state["resource_shock_active"] = True
                self._regime_state["shock_timer"] = 0.0

        if self._regime_state["resource_shock_active"]:
            self._regime_state["shock_timer"] += 0.25
            if self._regime_state["shock_timer"] > np.random.uniform(5.0, 12.0):
                self._regime_state["resource_shock_active"] = False

        # === v3.2: 技术进口断裂冲击（独立于FDI和资源） ===
        # 触发条件：技术进口偏好高 + 封锁升级 + 知识产权壁垒
        tech_import = p.get("C_tech_import_preference", 0.6)
        effective_block = getattr(state, "effective_block_max", p.get("block_max", 0.5))
        if not self._regime_state.get("tech_import_break_active", False):
            # 概率随技术进口依赖度和封锁强度上升
            tech_break_prob = (
                tech_import * 0.15  # 技术进口偏好 → 越依赖越脆弱
                + effective_block * 0.20  # 封锁强度 → 封锁越高越难获取技术
            ) * 0.02 * dt * 4  # 按季度概率转换
            if np.random.random() < tech_break_prob:
                self._regime_state["tech_import_break_active"] = True
                self._regime_state["shock_timer"] = 0.0

        if self._regime_state.get("tech_import_break_active", False):
            self._regime_state["shock_timer"] += 0.25
            # 技术进口断裂持续时间取决于封锁强度（封锁越高恢复越慢）
            recovery_years = np.random.uniform(6.0, 10.0) * (1.0 + effective_block)
            if self._regime_state["shock_timer"] > recovery_years:
                self._regime_state["tech_import_break_active"] = False

        # v3.2: 三重诅咒 — 统计活跃冲击数量并应用叠加效应
        active_shocks = sum([
            self._regime_state.get("fdi_shock_active", False),
            self._regime_state.get("resource_shock_active", False),
            self._regime_state.get("tech_import_break_active", False),
        ])

        if active_shocks >= 2:
            self._regime_state.setdefault("perfect_storm_count", 0)
            self._regime_state["perfect_storm_count"] += 1
            # 叠加乘数：2个冲击×1.0, 3个冲击×1.5
            storm_multiplier = 1.0 if active_shocks == 2 else 1.5
            state.external_dependency = min(
                0.95, state.external_dependency + 0.03 * dt * storm_multiplier
            )
            indigenous_rd = self.p.get("C_indigenous_rd_ratio", 0.03)
            rd_buffer = max(0.2, 1.0 - indigenous_rd * 10)
            extra_gap_growth = 0.06 * rd_buffer * dt * storm_multiplier
            state.tech_gap *= (1.0 + extra_gap_growth)
            for d_key in ["d_infra", "d_app"]:
                state.sectors[d_key].K = max(
                    0.5, state.sectors[d_key].K * (1.0 - 0.03 * dt * storm_multiplier)
                )
            # 技术进口断裂叠加 → 数字部门技术水平退化
            if self._regime_state.get("tech_import_break_active", False):
                for d_key in ["d_infra", "d_app"]:
                    state.sectors[d_key].A = max(0.1,
                        state.sectors[d_key].A * (1.0 - 0.02 * dt * storm_multiplier))

            if self._regime_state["perfect_storm_count"] > 20:
                state.political_support.support_index *= (1.0 - 0.01 * dt)

    def _apply_C_endogenous_costs(self, state: SystemState, dt: float):
        """v3.7.2: 策略C的连续内生成本机制.

        与策略A的_apply_A_endogenous_costs和策略B的_apply_socioeconomic_costs对称，
        实现README文档中描述的五种内生退化机制作为连续运行的成本。
        """
        if not hasattr(self, '_C_cost_state'):
            self._C_cost_state = {
                "dutch_disease_pressure": 0.0,      # 荷兰病累积压力
                "policy_capture_intensity": 0.0,     # 政策俘获强度
                "tech_degeneration_years": 0.0,      # 技术退化持续年数
                "consumerism_drag": 0.0,              # 消费主义拖累
                "brain_drain_cumulative": 0.0,        # 人才外流累积
            }
        cs = self._C_cost_state
        p = self.p

        # ================================================================
        # 1. 荷兰病支出效应 (Dutch Disease expenditure effect)
        # 资源出口繁荣 → 实际汇率升值 → 工业部门竞争力持续丧失
        # ================================================================
        resource_intensity = p.get("C_resource_export_intensity", 0.6)
        rer = getattr(state, "real_exchange_rate", 1.0)
        if rer > 1.05 and resource_intensity > 0.3:
            # 汇率升值幅度 × 资源依赖度 → 工业竞争力折扣
            dutch_severity = (rer - 1.0) * resource_intensity
            cs["dutch_disease_pressure"] += dutch_severity * dt
            # 工业出口竞争力持续下降
            industrial_penalty = min(0.15, cs["dutch_disease_pressure"] * 0.02)
            state.sectors["i"].Y *= max(0.70, 1.0 - industrial_penalty * dt)
            # 长期荷兰病 → 工业资本积累减速
            if cs["dutch_disease_pressure"] > 2.0:
                extra_decay = (cs["dutch_disease_pressure"] - 2.0) * 0.005
                state.sectors["i"].K = max(0.3 * state.initial_K_i,
                    state.sectors["i"].K * (1.0 - extra_decay * dt))
        else:
            # 汇率正常时缓慢恢复
            cs["dutch_disease_pressure"] = max(0.0,
                cs["dutch_disease_pressure"] - 0.03 * dt)

        # ================================================================
        # 2. 政策俘获 (Policy Capture)
        # 外部依赖升高 → 外资利益绑架国内政策 → 投资分配偏向外资偏好
        # ================================================================
        ext_dep = state.external_dependency
        if ext_dep > 0.4:
            cs["policy_capture_intensity"] += (ext_dep - 0.4) * 0.06 * dt
            cs["policy_capture_intensity"] = min(0.5, cs["policy_capture_intensity"])
            # 政策俘获 → FDI优惠 → 国内积累基金被外资利润汇出侵蚀
            capture_leakage = cs["policy_capture_intensity"] * 0.08
            state.accumulation_fund = max(0.0,
                state.accumulation_fund * (1.0 - capture_leakage * dt))
            # 严重俘获 → 工业保护被削弱（外资偏好服务业和资源）
            if cs["policy_capture_intensity"] > 0.25:
                protection_erosion = (cs["policy_capture_intensity"] - 0.25) * 0.05
                state.sectors["i"].I *= max(0.6, 1.0 - protection_erosion * dt)
        else:
            cs["policy_capture_intensity"] = max(0.0,
                cs["policy_capture_intensity"] - 0.02 * dt)

        # ================================================================
        # 3. 技术退化 (Technology Degeneration)
        # 长期依赖技术进口 → 自主创新能力持续下降
        # ================================================================
        tech_import = p.get("C_tech_import_preference", 0.6)
        indigenous_rd = p.get("C_indigenous_rd_ratio", 0.03)
        if tech_import > 0.4 and indigenous_rd < 0.05:
            cs["tech_degeneration_years"] += dt
            # 进口依赖 → "用进废退" → 自主创新效率持续下降
            degeneration_rate = (tech_import - 0.4) * 0.015
            # 对两个数字部门的技术水平施加退化压力
            for d_key in ["d_infra", "d_app"]:
                state.sectors[d_key].A *= max(0.7,
                    1.0 - degeneration_rate * dt)
            # 长期进口依赖(>10年) → 技术吸收能力也开始退化
            if cs["tech_degeneration_years"] > 10.0:
                absorption_decay = (cs["tech_degeneration_years"] - 10.0) * 0.003
                for d_key in ["d_infra", "d_app"]:
                    state.sectors[d_key].H = max(0.3,
                        state.sectors[d_key].H * (1.0 - absorption_decay * dt))
        else:
            cs["tech_degeneration_years"] = max(0.0,
                cs["tech_degeneration_years"] - dt * 1.5)  # 恢复慢于退化

        # ================================================================
        # 4. 消费主义陷阱 (Consumerism Trap)
        # FDI偏向消费领域 → 国内储蓄率下降 → 积累能力持续减弱
        # ================================================================
        fdi_open = p.get("C_fdi_openness", 0.4)
        if fdi_open > 0.3:
            consumerism_effect = (fdi_open - 0.3) * 0.03
            cs["consumerism_drag"] += consumerism_effect * dt
            cs["consumerism_drag"] = min(0.3, cs["consumerism_drag"])
            # 储蓄率下降 → 积累基金减少
            savings_penalty = cs["consumerism_drag"] * 0.1
            accumulation_fund_share = p.get("accumulation_fund_share", 0.8)
            effective_share = max(0.3, accumulation_fund_share - savings_penalty)
            # 通过降低积累基金比例来实现（间接机制）
            state.accumulation_fund *= (effective_share / max(accumulation_fund_share, 0.01))
            state.accumulation_fund = max(0.0, state.accumulation_fund)
        else:
            cs["consumerism_drag"] = max(0.0,
                cs["consumerism_drag"] - 0.01 * dt)

        # ================================================================
        # 5. 人才外流 (Brain Drain)
        # 依附加深 → 高技能人才海外流失 → 人力资本持续退化
        # ================================================================
        dep_composite = state.dependency.composite
        if dep_composite > 0.4:
            drain_rate = (dep_composite - 0.4) * 0.04
            cs["brain_drain_cumulative"] += drain_rate * dt
            # 人才外流主要影响数字部门和高技能服务业的人力资本
            for key in ["d_infra", "d_app", "s_high"]:
                s = state.sectors[key]
                # H的年度损失率 = 人才外流累积 × 敏感度
                h_loss = cs["brain_drain_cumulative"] * 0.03
                s.H = max(0.2, s.H * (1.0 - h_loss * dt))
            # 极端人才外流 → 数据资产质量下降
            if cs["brain_drain_cumulative"] > 2.0:
                state.data_assets.producer_data *= max(0.85,
                    1.0 - (cs["brain_drain_cumulative"] - 2.0) * 0.01 * dt)
        else:
            cs["brain_drain_cumulative"] = max(0.0,
                cs["brain_drain_cumulative"] - 0.02 * dt)

        # 记录成本状态用于监控
        self._regime_state["C_endogenous_costs"] = {
            "dutch_disease_pressure": cs["dutch_disease_pressure"],
            "policy_capture_intensity": cs["policy_capture_intensity"],
            "tech_degeneration_years": cs["tech_degeneration_years"],
            "consumerism_drag": cs["consumerism_drag"],
            "brain_drain_cumulative": cs["brain_drain_cumulative"],
        }
