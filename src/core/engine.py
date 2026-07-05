"""模拟引擎 —— 主循环，协调所有模块 (v3.0 升级)."""

import numpy as np
from .config import SimulationConfig
from .state import SystemState, ALL_SECTORS
from ..params.schema import get_schema
from ..models.sectors import SectorModel
from ..models.labor_market import LaborMarket
from ..models.accumulation import AccumulationRegime
from ..models.digitalization import DigitalizationModel
from ..models.technology import TechnologyDynamics
from ..models.technology_diffusion import TechnologyDiffusion
from ..models.international import InternationalModule
from ..models.welfare import WelfareModule
from ..models.political_economy import PoliticalEconomyModule  # v2.2 新增
from ..models.livelihood import LivelihoodModule  # v3.0: 民生子模块
from ..models.environment import EnvironmentModule  # v3.3: 气候/环境
from ..monitors.fracture import FractureMonitor
from ..monitors.ews import EarlyWarningMonitor
from ..monitors.academic_validator import AcademicValidator  # v3.2: 学术验证
from ..sfc.accounting import SFCClosure
from ..io.recorder import TimeSeriesRecorder
# v3.0: 多维评价体系
from ..evaluation import (
    FiveDimensionEvaluator, DimensionScores,
    RedLineDetector, RedLineResult,
    CDICalculator, CDIResult,
)


class SimulationEngine:
    """三部门政治经济学模拟引擎 (v2.2)."""

    def __init__(self, params: dict = None, config: SimulationConfig = None,
                 regime: str = "A", country_profile: str = None,
                 scenario: str = None):
        self.schema = get_schema()

        # v3.1: 链式参数加载
        if params is not None:
            self.p = params
        elif country_profile is not None:
            # 使用国情预设一站式构建
            self.p = self.schema.build_from_profile(
                profile_name=country_profile,
                regime=regime,
                scenario=scenario,
            )
        elif scenario is not None:
            # 无国情预设但有场景：先建默认再覆盖场景
            from ..params.presets import get_scenario
            self.p = self.schema.build_defaults(regime)
            self.p.update(get_scenario(scenario))
        else:
            self.p = self.schema.build_defaults(regime)

        self.config = config if config else SimulationConfig()
        self.regime = regime
        self._country_profile = country_profile  # 记录用于报告

        # v3.0: 解析混合策略参数
        self._is_mixed = regime.startswith("M")
        if self._is_mixed:
            from ..models.mixed_strategy import STANDARD_MIXTURES
            mixture_name = self.p.get("mixed_name", None)
            if mixture_name and mixture_name in STANDARD_MIXTURES:
                self._mixed_lambda = STANDARD_MIXTURES[mixture_name]
            else:
                self._mixed_lambda = tuple(self.p.get("mixed_lambda", (1/3, 1/3, 1/3)))
            # 将lambda信息注入params供AccumulationRegime.create使用
            self.p["mixed_lambda"] = self._mixed_lambda
            self.p["mixed_name"] = self.p.get("mixed_name", mixture_name)

        # 初始化状态
        self.state = SystemState(accumulation_regime=regime)

        # 初始化各模块
        self.sector_model = SectorModel(self.p)
        self.labor_market = LaborMarket(self.p)
        self.accumulation = AccumulationRegime.create(regime, self.p)
        self.digitalization = DigitalizationModel(self.p)
        self.tech_dynamics = TechnologyDynamics(self.p)
        self.tech_diffusion = TechnologyDiffusion(self.p)  # v2.1
        self.international = InternationalModule(self.p)
        self.welfare = WelfareModule(self.p)
        self.political_economy = PoliticalEconomyModule(self.p)  # v2.2 新增
        self.livelihood = LivelihoodModule(self.p)  # v3.0: 民生子模块
        self.environment = EnvironmentModule(self.p)  # v3.3: 气候/环境
        self.fracture_monitor = FractureMonitor(self.p)
        self.ews_monitor = EarlyWarningMonitor(
            self.p, window_size=self.config.ews_window_size
        )
        self.sfc_closure = SFCClosure(self.p)

        self.recorder = TimeSeriesRecorder(
            max_steps=int(self.config.T_max / self.config.dt_min) + 100
        )

        # v2.2: 追踪上次政治更新时间和总社会成本
        self._last_political_update = 0.0
        self._total_extra_unemployment = 0.0

        # v3.0: 多维评价体系初始化
        self.evaluator = FiveDimensionEvaluator(self.p)
        self.red_line_detector = RedLineDetector()
        self.cdi_calculator = CDICalculator(preset=self.config.cdi_weight_preset)
        self._last_evaluation_update = 0.0
        self._evaluation_history: list = []           # 历史 DimensionScores
        self._red_line_persistence: dict = {}          # 红线持续性追踪
        self._final_evaluation: CDIResult = None       # 最终评价结果

        # v3.2: 学术验证
        self.validator = AcademicValidator(self.p)
        # 追踪劳动力技能侵蚀
        self._initial_H_i = 0.0
        self._initial_H_b = 0.0

    def run(self) -> dict:
        """执行一次完整的模拟."""
        self._initialize()

        t = 0.0
        step = 0

        while t < self.config.T_max and not self.state.fracture_status.is_terminal:
            dt = self._compute_step_size()

            # === 主循环 (v2.2 增加政治经济学反馈) ===

            # 1. 更新外生变量
            self.international.step(self.state, dt)

            # 2. 计算部门产出
            self.sector_model.compute_all_production(self.state)

            # 3. 劳动力市场
            self.labor_market.step(self.state, dt)
            self.sector_model.compute_wages(self.state)
            self.sector_model.compute_profit_rates(self.state)

            # 4. 资本积累与投资分配
            self.accumulation.step(self.state, dt)

            # 5. 数字化与数据资产
            self.digitalization.step(self.state, dt)

            # 6. 技术动态
            self.tech_dynamics.step(self.state, dt)

            # 7. 知识扩散政策
            self.tech_diffusion.step(self.state, dt)

            # 8. 福利与再分配
            self.welfare.update_reallocation(self.state)
            self.welfare.step(self.state, dt)

            # 8.5. v3.0: 民生子模块（健康·教育·社保·安全感）
            self.livelihood.step(self.state, dt)

            # 8.6. v3.3: 气候/环境模块（碳排放+绿色转型压力）
            self.environment.step(self.state, dt)

            # 9. v2.2: 政治经济学反馈（按周期更新以减少计算量）
            if (t - self._last_political_update) >= self.config.political_update_interval:
                self.political_economy.step(self.state, dt)
                self._last_political_update = t

            # 10. SFC闭合
            self.sfc_closure.check_and_close(self.state)
            # v3.2: 记录SFC残差用于学术验证
            self.validator.record_sfc_residual(
                self.state.transaction_residual,
                self.state.gdp  # 用GDP作为交易规模的代理
            )
            # v3.2: 周期性记录状态快照（用于通用异常检测）
            self.validator.record_state_snapshot(self.state)

            # 11. EWS计算（在断裂检测之前）
            self.ews_monitor.feed(self.state)
            self.state.ews_signals = self.ews_monitor.compute_all(self.state)

            # 12. v2.2: 检查政治驱动的策略转变
            if self.config.political_feedback_enabled:
                regime_signal = self.political_economy.check_policy_regime_change(
                    self.state
                )
                if regime_signal == "emergency":
                    # 紧急政策转向：大幅降低提取率、增加福利投资
                    self._apply_emergency_policy_adjustment()

            # 12.5 v2.2: 尾部风险 — 所有策略都无法规避的意外事件
            self._check_tail_risks()

            # 12.6 v2.2: 策略特有的内在风险（命门）
            self._check_signature_vulnerability(dt)

            # 13. 断裂检测
            events = self.fracture_monitor.check(self.state)
            for evt in events:
                self._handle_fracture(evt)

            # 14. 记录
            self.state.t = t
            self.state.step = step
            # v3.2: 劳动力技能侵蚀（在记录前计算）
            self.state.skill_erosion_i = (
                self.state.sectors["i"].H / max(self._initial_H_i, 0.01)
                if self._initial_H_i > 0 else 1.0
            )
            self.state.skill_erosion_b = (
                self.state.sectors["b"].H / max(self._initial_H_b, 0.01)
                if self._initial_H_b > 0 else 1.0
            )
            self.recorder.record(self.state)

            # 15. v3.0: 多维评价（按周期更新以减少计算量）
            if self.config.evaluation_enabled:
                if (t - self._last_evaluation_update) >= self.config.evaluation_interval:
                    self._evaluate_current_state()
                    self._last_evaluation_update = t

            t += dt
            step += 1

        # v3.0: 最终评价（在循环结束后的完整评价）
        if self.config.evaluation_enabled:
            self._evaluate_current_state(final=True)

        return self._build_result()

    def _initialize(self):
        """初始化系统状态 (v3.1: 支持国情预设)."""
        state = self.state
        self.sector_model.initial_state(state)
        state.total_surplus = state.gdp * 0.15
        state.accumulation_fund = state.total_surplus * 0.8
        # v3.1: 技术差距从参数读取（不同国情起点不同）
        state.tech_gap = self.p.get("IC_tech_gap_initial", 3.0)
        # v3.3: 初始技术差距基准（用于内生封锁计算）
        state.initial_tech_gap = state.tech_gap
        state.effective_block_max = self.p.get("block_max", 0.5)
        state.effective_fdi_availability = self.p.get("fdi_availability", 1.0)
        # 记录初始人均GDP（用于增长率计算）
        state.initial_gdp_pc = state.gdp / max(state.households.total_population, 0.01)
        # v3.1: 初始化家庭群组人口结构（与部门劳动力对应）
        self._init_households_from_sectors(state)
        # v3.2: 记录初始技能水平（用于劳动力技能侵蚀追踪）
        self._initial_H_i = state.sectors["i"].H
        self._initial_H_b = state.sectors["b"].H

    def _compute_step_size(self) -> float:
        """自适应步长."""
        if not self.config.adaptive_stepping:
            return self.config.dt_base

        dt = self.config.dt_base

        # 检测快速变化
        if self.state.unemployment_rate > 0.15:
            dt = self.config.dt_base / 2
        if self.state.unemployment_rate > 0.20:
            dt = self.config.dt_base / 4

        # 断裂逼近
        if self.fracture_monitor.imminent(self.state, threshold=0.1):
            dt = min(dt, self.config.dt_min * 2)

        return max(self.config.dt_min, dt)

    def _handle_fracture(self, event: dict):
        """处理断裂事件."""
        fs = self.state.fracture_status
        ftype = event["type"]

        if ftype in fs.active_fractures:
            return

        fs.active_fractures.append(ftype)
        fs.fracture_time[ftype] = self.state.t

        if event["is_terminal"]:
            fs.is_terminal = True
            fs.terminal_type = ftype

    def _check_signature_vulnerability(self, dt: float):
        """v3.0: 每种策略独有的、不可消除的内在风险（命门）.

        混合策略：按 λ 权重混合所有适用的命门。
        纯策略A — 速度陷阱：世界前沿增长太快，渐进升级跟不上指数增长。
        纯策略B — 撕裂陷阱：赢家通吃导致的不平等最终引爆政治炸弹。
        纯策略C — 依附陷阱：外部依赖的三重诅咒（荷兰病+政策俘获+技术退化）。
        """
        import numpy as np
        g_world = self.p.get("g_world", 0.03)
        regime = self.regime

        # v3.0: 混合策略 — 按权重施加各纯策略的命门
        if self._is_mixed:
            la, lb, lc = self._mixed_lambda
            # A的命门（按la权重）
            if la > 0:
                self._check_A_vulnerability(g_world, dt, weight=la)
            # B的命门（按lb权重）
            if lb > 0:
                self._check_B_vulnerability(dt, weight=lb)
            # C的命门（按lc权重）
            if lc > 0:
                self._check_C_vulnerability(dt, weight=lc)
            return

        # 纯策略
        if regime == "A":
            self._check_A_vulnerability(g_world, dt)
        elif regime == "B":
            self._check_B_vulnerability(dt)
        elif regime == "C":
            self._check_C_vulnerability(dt)

    def _check_A_vulnerability(self, g_world: float, dt: float, weight: float = 1.0):
        """策略A命门：速度陷阱."""
        dd_i = self.state.sectors["i"].digital_density
        G = self.state.tech_gap
        if g_world > 0.05 and dd_i < 0.5 and G > 6.0:
            speed_penalty = (g_world - 0.05) * 2.0 * dt * weight
            self.state.tech_gap *= (1.0 + speed_penalty)
            drag = speed_penalty * 0.3
            self.state.sectors["i"].Y *= max(0.90, 1.0 - drag)

    def _check_B_vulnerability(self, dt: float, weight: float = 1.0):
        """策略B命门：撕裂陷阱."""
        cost_ratio = self.state.households.cost_ratio_industrial_vs_digital
        ps = self.state.political_support
        if cost_ratio > 2.5:
            tear_acceleration = (cost_ratio - 2.5) * 0.02 * dt * weight
            ps.support_index = max(0.05, ps.support_index - tear_acceleration)
        if cost_ratio > 4.0 and ps.support_index < 0.3:
            if hasattr(self.accumulation, '_regime_state'):
                self.accumulation._regime_state["feedback_failed_count"] += 1

    def _check_C_vulnerability(self, dt: float, weight: float = 1.0):
        """策略C命门：依附陷阱三合一 + v3.2消费主义陷阱 + v3.2人才外流."""
        ext_dep = self.state.external_dependency

        # 1. 荷兰病 (v3.3: 汇率渠道 + v3.2: 支出效应)
        resource_intensity = self.p.get("C_resource_export_intensity", 0.6)
        if resource_intensity > 0.5 and self.state.sectors["b"].Y > 0:
            # --- 汇率渠道 ---
            rer = self.state.real_exchange_rate
            export_share_i = 0.3
            competitiveness_loss = max(0.0, (rer - 1.0) * 0.5 * weight)
            self.state.sectors["i"].Y *= max(0.80, 1.0 - export_share_i * competitiveness_loss * dt)

            # --- v3.2: 支出效应 ---
            # 资源繁荣 → 国内非贸易品需求上升 → 劳动力从工业流向服务业
            # 机制：资源收入增加→国内消费能力上升→服务业（非贸易品）需求增长
            # → 服务业工资相对工业上升→工业劳动力流向服务业
            resource_boom = (resource_intensity - 0.5) * 2.0  # [0, 1]
            # 服务业需求因资源财富效应上升
            service_demand_boost = resource_boom * 0.08 * dt * weight
            self.state.sectors["s"].Y *= (1.0 + service_demand_boost)
            # 工业劳动力被服务业"拉走"（支出效应）
            labor_pull = resource_boom * 0.02 * dt * weight
            L_i_to_s = self.state.sectors["i"].L * labor_pull
            self.state.sectors["i"].L = max(0.5, self.state.sectors["i"].L - L_i_to_s)
            self.state.sectors["s"].L += L_i_to_s * 0.7  # 部分吸收
            # 流出工业的劳动力有部分进入低端非正式服务业
            self.state.informal_service_L += L_i_to_s * 0.3

        # 2. 政策俘获
        if ext_dep > 0.75:
            capture_effect = (ext_dep - 0.75) * 0.04 * dt * weight
            self.state.accumulation_fund *= (1.0 - capture_effect)

        # 3. 技术依附
        tech_imp = self.p.get("C_tech_import_preference", 0.6)
        if tech_imp > 0.6 and ext_dep > 0.6:
            atrophy = (tech_imp - 0.5) * 0.015 * dt * weight
            self.state.sectors["d"].A = max(0.1, self.state.sectors["d"].A * (1.0 - atrophy))

        # 4. v3.2: 消费主义陷阱 — FDI流向消费领域→储蓄率下降→国内积累能力减弱
        fdi_openness = self.p.get("C_fdi_openness", 0.4)
        # FDI总量中流向消费的估算比例：FDI开放度越高+国内储蓄率越低→消费型FDI越多
        domestic_savings_rate = self.state.accumulation_fund / max(self.state.gdp, 0.01)
        fdi_consumer_ratio = fdi_openness * (1.0 - min(1.0, domestic_savings_rate / 0.25))
        if fdi_consumer_ratio > 0.5:
            # 消费型FDI主导 → 储蓄率被进一步压低（消费信贷扩张）
            savings_drain = (fdi_consumer_ratio - 0.5) * 0.03 * dt * weight
            self.state.accumulation_fund *= max(0.70, 1.0 - savings_drain)
            # 长期效应：国内积累能力减弱 → 技术投资更依赖外部
            if fdi_consumer_ratio > 0.7:
                self.state.external_dependency = min(0.95,
                    self.state.external_dependency + 0.005 * dt * weight)

        # 5. v3.2: 人才外流 — 技术进口>60%且国内高技能岗位不足→H_d衰减
        if tech_imp > 0.6:
            # 数字部门高技能岗位是否充足：看数字部门就业占比
            digital_emp_share = self.state.sectors["d"].L / max(
                self.state.households.total_population, 1.0)
            # 当技术严重依赖进口+数字就业岗位占比不足 → 高技能人才流向海外
            if digital_emp_share < 0.08:  # 数字就业占比低于8%
                brain_drain_rate = (tech_imp - 0.5) * (0.08 - digital_emp_share) * 0.04 * dt * weight
                self.state.sectors["d"].H = max(0.3,
                    self.state.sectors["d"].H * (1.0 - brain_drain_rate))
                # 人才外流也削弱整体人力资本
                self.state.sectors["i"].H = max(0.3,
                    self.state.sectors["i"].H * (1.0 - brain_drain_rate * 0.5))

    def _apply_emergency_policy_adjustment(self):
        """v2.2: 政治紧急状态下的政策调整.

        当支持度崩溃时，政策被迫大幅转向。
        这本身就是一种"失败"——原策略被放弃，系统进入求生模式。
        """
        # 剧烈削减积累基金（相当于30%GDP）
        emergency_diversion = self.state.gdp * 0.30
        self.state.accumulation_fund = max(
            0, self.state.accumulation_fund - emergency_diversion
        )

        # 数字部门投资受到最大冲击（原策略的核心引擎停转）
        # 工业也受波及，但保留最低限度
        self.state.sectors["d"].K *= 0.85  # 数字资本部分荒废
        self.state.sectors["i"].K *= 0.95  # 工业略有损失

        # 失业急剧上升
        laid_off = self.state.sectors["d"].L * 0.25
        self.state.sectors["d"].L = max(1.0, self.state.sectors["d"].L - laid_off)
        self.state.households.unemployed.population += laid_off

        # 增加补偿（但远不足以弥补）
        self.state.households.unemployed.compensation_received += (
            emergency_diversion * 0.5
        )
        self.state.households.industrial_workers.compensation_received += (
            emergency_diversion * 0.2
        )

    def _check_tail_risks(self):
        """v2.2: 尾部风险 — 所有策略都可能遭遇的意外事件.

        校准原则：
        - 低频率（每年~0.5-1.5%）但真实影响
        - 有冷却期（同一风险不会连续触发）
        - 严重程度取决于系统当前状态（强者抗打击，弱者更脆弱）
        """
        import numpy as np

        # 冷却期追踪
        if not hasattr(self, '_tail_risk_cooldown'):
            self._tail_risk_cooldown = {"supply_chain": 0, "financial": 0,
                                         "tech_disruption": 0, "social_unrest": 0}

        # 冷却期递减
        for k in self._tail_risk_cooldown:
            self._tail_risk_cooldown[k] = max(0, self._tail_risk_cooldown[k] - 1)

        # 系统的整体抗风险能力
        resilience_factor = self.state.political_support.support_index
        inst_cap = self.p.get("institution_capacity", 0.6)

        # === 尾部风险1：供应链/基础设施冲击 ===
        if np.random.random() < 0.0015 and self._tail_risk_cooldown["supply_chain"] == 0:
            self._tail_risk_cooldown["supply_chain"] = 20  # 5年冷却
            # 严重程度取决于制度能力和外部依赖
            severity = np.random.uniform(0.03, 0.12) * (1.5 - inst_cap)
            self.state.sectors["b"].Y *= (1.0 - severity)
            self.state.sectors["i"].Y *= (1.0 - severity * 0.6)
            self.state.accumulation_fund = max(0, self.state.accumulation_fund * (1.0 - severity))

        # === 尾部风险2：国际金融冲击 ===
        if np.random.random() < 0.003 and self._tail_risk_cooldown["financial"] == 0:
            self._tail_risk_cooldown["financial"] = 16  # 4年冷却
            severity = np.random.uniform(0.05, 0.15)
            # 数字部门更脆弱（全球资本流动敏感）
            self.state.sectors["d"].K *= (1.0 - severity * 0.8)
            self.state.sectors["i"].Y *= (1.0 - severity * 0.3)
            # C策略（或含C成分的混合策略）更依赖外资 → 冲击更大
            if self.regime == "C" or (self._is_mixed and self._mixed_lambda[2] > 0.2):
                self.state.sectors["d"].K *= (1.0 - severity * 0.5)

        # === 尾部风险3：技术路线突变 ===
        if np.random.random() < 0.002 and self._tail_risk_cooldown["tech_disruption"] == 0:
            self._tail_risk_cooldown["tech_disruption"] = 24  # 6年冷却
            obsolescence = np.random.uniform(0.10, 0.20)
            # 数字密度高的部门受影响更小（技术基础好，转型快）
            dd_d = self.state.sectors["d"].digital_density
            effective_obs = obsolescence * (1.0 - dd_d * 0.5)
            self.state.sectors["d"].K *= (1.0 - effective_obs)
            self.state.tech_gap *= (1.0 + obsolescence * 0.3)

        # === 尾部风险4：社会动荡 ===
        # 这是最"条件化"的风险——只在政治脆弱时触发
        ps = self.state.political_support
        if ps.support_index < 0.4 and ps.protest_risk > 0.5:
            if (np.random.random() < 0.005 * (1.0 - ps.support_index) * ps.protest_risk
                    and self._tail_risk_cooldown["social_unrest"] == 0):
                self._tail_risk_cooldown["social_unrest"] = 12  # 3年冷却
                disruption = np.random.uniform(0.03, 0.10)
                self.state.sectors["i"].Y *= (1.0 - disruption)
                ps.support_index *= 0.94
                if self.regime == "C" or (self._is_mixed and self._mixed_lambda[2] > 0.2):
                    # 外资最怕社会动荡
                    self.state.sectors["d"].K *= (1.0 - disruption * 1.5)

    def _evaluate_current_state(self, final: bool = False):
        """v3.0: 对当前状态进行五维评价.

        Args:
            final: 是否为最终评价（在循环结束后调用）
        """
        from ..io.recorder import TimeSeriesRecorder

        # 从recorder获取时间序列数据
        t_series = {}
        if hasattr(self.recorder, 'arrays') and self.recorder.arrays:
            t_series = self.recorder.arrays

        # 计算五维得分
        scores = self.evaluator.evaluate(self.state, t_series=t_series)

        # 人道主义红线检测
        red_line = None
        if self.config.red_line_check_enabled:
            red_line = self.red_line_detector.check(
                self.state,
                persistence_tracker=self._red_line_persistence
            )

        # 计算CDI
        cdi_result = self.cdi_calculator.compute(
            scores,
            red_line=red_line,
            score_history=self._evaluation_history if not final else None
        )

        # 保存到评价历史
        self._evaluation_history.append(scores)
        # 限制历史长度
        if len(self._evaluation_history) > 200:
            self._evaluation_history = self._evaluation_history[-100:]

        # 如果是最终评价，保存为最终结果；否则记录到recorder
        if final:
            self._final_evaluation = cdi_result
        else:
            # 将评价结果记录到recorder（在record方法调用后追加）
            self.recorder.record_evaluation(scores, cdi_result)

    def _build_result(self) -> dict:
        """构建模拟结果 (v2.2 扩展)."""
        df = self.recorder.to_dataframe()
        fs = self.state.fracture_status

        summary = {
            "regime": self.regime,
            # v3.1: 国情信息
            "country_profile": self._country_profile,
            # v3.0: 混合策略信息
            "is_mixed": self._is_mixed,
            "lambda_vec": list(self._mixed_lambda) if self._is_mixed else None,
            "dominant_strategy": (["A", "B", "C"][np.argmax(self._mixed_lambda)]
                                  if self._is_mixed else self.regime),
            "survived": not fs.is_terminal,
            "survival_years": self.state.t,
            "terminal_fracture": fs.terminal_type,
            "total_fractures": len(fs.active_fractures),
            "fracture_types": fs.active_fractures,
            "fracture_times": fs.fracture_time,
            "final_gdp": self.state.gdp,
            "final_tech_gap": self.state.tech_gap,
            "peak_unemployment": df["unemployment_rate"].max(),
            "final_welfare": self.state.welfare_index,
            "final_gini": self.state.gini,
            "total_cost_unemployment": self.state.cost_metrics.unemployment_stock,
            "cost_concentration": self.state.cost_metrics.cost_concentration,
            "cost_ratio_industrial_digital": self.state.cost_metrics.cost_ratio_industrial_digital,
            "deind_type_final": self.state.cost_metrics.deind_type,
            "dependency_composite": self.state.dependency.composite,
            "ews_max_K_i": (
                df["ews_K_i_composite"].max()
                if "ews_K_i_composite" in df.columns else 0
            ),
            "ews_max_tech_gap": (
                df["ews_tech_gap_composite"].max()
                if "ews_tech_gap_composite" in df.columns else 0
            ),
            "final_data_producer": self.state.data_assets.producer_data,
            "final_data_consumer": self.state.data_assets.consumer_data,
            "sector_K_b": self.state.sectors["b"].K,
            "sector_K_i": self.state.sectors["i"].K,
            "sector_K_d": self.state.sectors["d"].K,
            "sector_K_s": self.state.sectors["s"].K,  # v3.3
            "sector_DL_b": self.state.sectors["b"].DL,
            "sector_DL_i": self.state.sectors["i"].DL,
            "sector_DL_d": self.state.sectors["d"].DL,
            "sector_DL_s": self.state.sectors["s"].DL,  # v3.3
            "total_steps": self.state.step,
            # v2.1
            "diffusion_cumulative_i": self.tech_diffusion.get_cumulative_diffusion("i"),
            "diffusion_cumulative_b": self.tech_diffusion.get_cumulative_diffusion("b"),
            "sector_gap_i": self.tech_diffusion.get_sector_gap("i"),
            "sector_gap_b": self.tech_diffusion.get_sector_gap("b"),
            # v2.2 新增
            "digital_density_i": self.state.sectors["i"].digital_density,
            "digital_density_b": self.state.sectors["b"].digital_density,
            "digital_density_d": self.state.sectors["d"].digital_density,
            "political_support": self.state.political_support.support_index,
            "political_protest_risk": self.state.political_support.protest_risk,
            "political_policy_constraint": self.state.political_support.policy_constraint,
            "political_employment_sat": self.state.political_support.employment_satisfaction,
            "political_wage_sat": self.state.political_support.wage_satisfaction,
            "political_inequality_tol": self.state.political_support.inequality_tolerance,
            # v2.2: EWS韧性指标
            "ews_resilience_K_i": self._get_ews_resilience("K_i_ratio"),
            "ews_resilience_tech_gap": self._get_ews_resilience("tech_gap"),
            # v2.2: B策略成本指标
            "bubble_pressure": (
                self.accumulation._regime_state.get("bubble_pressure", 0.0)
                if hasattr(self.accumulation, "_regime_state") else 0.0
            ),
            # v3.0: 五维评价
            "evaluation": self._build_evaluation_summary(),
            # v3.2: 学术验证报告
            "validation": self._build_validation_summary(),
            # v3.3: 盲区消除指标
            "v33_metrics": {
                "informal_service_L": self.state.informal_service_L,
                "informal_service_W": self.state.informal_service_W,
                "credit_rationing_ratio": self.state.credit_rationing_ratio,
                "real_exchange_rate": self.state.real_exchange_rate,
                "effective_block_max": self.state.effective_block_max,
                "effective_fdi_availability": self.state.effective_fdi_availability,
                "cumulative_carbon": self.state.cumulative_carbon,
                "green_transition_pressure": self.state.green_transition_pressure,
            },
        }

        return {
            "summary": summary,
            "dataframe": df,
        }

    def _init_households_from_sectors(self, state: SystemState):
        """v3.1: 从部门劳动力初始化家庭群组人口 (v3.2: 增加政治权重)."""
        # 数字高技能 = 数字部门劳动力
        hh_d = state.households.digital_high_skill
        hh_d.population = state.sectors["d"].L
        hh_d.employed = state.sectors["d"].L
        hh_d.avg_wage = state.sectors["d"].W
        hh_d.skill_level = state.sectors["d"].H
        hh_d.political_power = 0.30  # v3.2: 数字高技能政治影响力最高

        # 工业劳动者 = 工业部门劳动力
        hh_i = state.households.industrial_workers
        hh_i.population = state.sectors["i"].L
        hh_i.employed = state.sectors["i"].L
        hh_i.avg_wage = state.sectors["i"].W
        hh_i.skill_level = state.sectors["i"].H
        hh_i.political_power = 0.25  # v3.2: 工业劳动者—人口基数大+工会

        # 基础劳动者 = 基础部门劳动力
        hh_b = state.households.base_workers
        hh_b.population = state.sectors["b"].L
        hh_b.employed = state.sectors["b"].L
        hh_b.avg_wage = state.sectors["b"].W
        hh_b.skill_level = state.sectors["b"].H
        hh_b.political_power = 0.15  # v3.2: 基础劳动者—分散、组织度低

        # v3.3: 服务业劳动者初始化
        hh_s = state.households.service_workers
        hh_s.population = state.sectors["s"].L
        hh_s.employed = state.sectors["s"].L
        hh_s.avg_wage = state.sectors["s"].W
        hh_s.skill_level = state.sectors["s"].H
        hh_s.political_power = 0.15  # v3.2: 服务业劳动者—非正式、碎片化

        # 失业者初始为零
        hh_u = state.households.unemployed
        hh_u.population = 0.0
        hh_u.employed = 0.0
        hh_u.political_power = 0.10  # v3.2: 失业者政治影响力最低

        # 初始消费、储蓄和股权
        # v3.4: 股权初始分配 — 反映各群组的资本所有权
        total_K = sum(state.sectors[k].K for k in ALL_SECTORS)
        for h, sector_key, equity_share in [
            (state.households.digital_high_skill, "d", 0.40),   # 数字部门股权集中于高技能
            (state.households.industrial_workers, "i", 0.25),   # 工业股权部分由工人持有
            (state.households.base_workers, "b", 0.15),
            (state.households.service_workers, "s", 0.10),
            (state.households.unemployed, None, 0.0),           # 失业者无股权
        ]:
            if h.population > 0:
                h.income = h.avg_wage * h.population
                h.consumption = h.income * 0.85
                h.savings = h.income * 0.15
                h.debt = h.income * 0.05
                # v3.4: 初始股权 = 部门资本 × 群组所有权份额
                if sector_key:
                    h.equity = state.sectors[sector_key].K * equity_share
                else:
                    h.equity = 0.0

    def _get_ews_resilience(self, var_name: str) -> float:
        """获取EWS韧性指数."""
        ews = self.state.ews_signals
        if var_name in ews:
            return ews[var_name].resilience_index
        return 0.0

    def _build_validation_summary(self) -> dict:
        """v3.2: 构建学术验证摘要."""
        # 生成验证报告
        predefined = self.state.fracture_status.active_fractures
        val_report = self.validator.finalize(
            {"summary": {}},  # 简化版result
            predefined_fractures=predefined,
        )

        # 劳动力技能侵蚀
        current_H_i = self.state.sectors["i"].H
        current_H_b = self.state.sectors["b"].H
        skill_erosion_i = (current_H_i / max(self._initial_H_i, 0.01)
                           if self._initial_H_i > 0 else 1.0)
        skill_erosion_b = (current_H_b / max(self._initial_H_b, 0.01)
                           if self._initial_H_b > 0 else 1.0)

        return {
            "sfc_trustworthiness": val_report.sfc_trustworthiness,
            "sfc_median_residual": val_report.sfc_median_residual,
            "sfc_max_residual": val_report.sfc_max_residual,
            "sfc_pressure_ratio": val_report.sfc_pressure_residual_ratio,
            "generic_anomalies_detected": val_report.generic_anomalies_detected,
            "generic_anomaly_vars": val_report.generic_anomaly_variables,
            "predefined_fractures_count": val_report.predefined_fractures_triggered,
            "uncovered_anomalies": val_report.uncovered_anomalies,
            "unexpected_findings": val_report.unexpected_findings,
            # v3.2: 劳动力技能侵蚀
            "skill_erosion_industrial": skill_erosion_i,
            "skill_erosion_base": skill_erosion_b,
        }

    def _build_evaluation_summary(self) -> dict:
        """v3.0: 构建评价摘要."""
        if self._final_evaluation is None:
            return {"enabled": False}

        ev = self._final_evaluation
        result = {
            "enabled": True,
            "cdi": ev.cdi,
            "cdi_variants": ev.cdi_variants,
            "economy": ev.economy,
            "livelihood": ev.livelihood,
            "resilience": ev.resilience,
            "structure": ev.structure,
            "sovereignty": ev.sovereignty,
            "red_line_level": ev.red_line.level if ev.red_line else "PASS",
            "red_line_failures": ev.red_line.failures if ev.red_line else [],
            "red_line_warnings": ev.red_line.warnings if ev.red_line else [],
            "is_acceptable": ev.is_acceptable(),
            "recent_trend": ev.recent_trend,
            "volatility_penalty": ev.volatility_penalty,
            "weight_preset": self.config.cdi_weight_preset,
            # v3.0: 民生子模块详细数据
            "livelihood_details": self.livelihood.get_summary(),
        }
        return result
