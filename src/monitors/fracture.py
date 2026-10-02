"""断裂监测 —— 基于规则引擎的9种断裂模式检测 (v2.1 级联升级).

v2.1 升级：
- 断裂级联效应：已触发的断裂降低后续断裂的阈值
- tech_lock_in → 加速 industrial_withering（技术锁死削弱工业竞争力）
- premature_deindustrialization → 加速 social_collapse（去工业化→失业→社会崩溃）
- dependency_trap → 加速 tech_lock_in（依附固化→技术吸收能力丧失）
- 级联乘数随时间累积：断裂持续时间越长，对其他断裂的加速效应越强
"""

import numpy as np
from ..core.state import SystemState, FractureStatus


class FractureMonitor:
    """断裂检测器 —— 事后阈值 + 规则引擎 + 级联效应."""

    def __init__(self, params: dict):
        self.p = params
        self.mult = params.get("fracture_threshold_mult", 1.0)
        # v3.7.4: 可配置的福利崩溃阈值（替代硬编码0.30）
        self.welfare_threshold = params.get("welfare_collapse_threshold", 0.30)
        # 级联强度参数
        self.cascade_strength = params.get("cascade_strength", 0.15)
        # 记录断裂触发时间用于级联计算
        self._fracture_trigger_times: dict = {}

    def _cascade_multiplier(self, state: SystemState, target_fracture: str) -> float:
        """计算级联乘数：已激活的断裂对目标断裂的阈值降低效应.

        返回一个 [0, 1] 之间的乘数，1=无级联效应，越小=阈值越容易被触发.
        """
        active = state.fracture_status.active_fractures
        if not active:
            return 1.0

        cascade_factor = 1.0
        cs = self.cascade_strength

        # tech_lock_in → 降低 industrial_withering 阈值
        # 机制：技术锁死→工业部门无法获得数字赋能→竞争力下降→资本加速萎缩
        if "tech_lock_in" in active and target_fracture == "industrial_withering":
            duration = state.t - self._fracture_trigger_times.get("tech_lock_in", state.t)
            # 持续时间越长，级联效应越强
            cascade_factor -= cs * min(1.0, duration / 20.0) * 1.5

        # tech_lock_in → 降低 social_collapse 阈值
        # 机制：技术锁死→整体经济竞争力下降→失业上升+福利下降
        if "tech_lock_in" in active and target_fracture == "social_collapse":
            duration = state.t - self._fracture_trigger_times.get("tech_lock_in", state.t)
            cascade_factor -= cs * min(1.0, duration / 25.0) * 1.0

        # premature_deindustrialization → 降低 social_collapse 阈值
        # 机制：过早去工业化→大量工业劳动力失业/退回低端→社会矛盾激化
        if "premature_deindustrialization" in active and target_fracture == "social_collapse":
            duration = state.t - self._fracture_trigger_times.get(
                "premature_deindustrialization", state.t
            )
            cascade_factor -= cs * min(1.0, duration / 15.0) * 1.8

        # dependency_trap → 降低 tech_lock_in 阈值
        # 机制：过度依附→自主创新能力退化→技术差距加速拉大
        if "dependency_trap" in active and target_fracture == "tech_lock_in":
            duration = state.t - self._fracture_trigger_times.get("dependency_trap", state.t)
            cascade_factor -= cs * min(1.0, duration / 20.0) * 1.3

        # industrial_withering → 降低 fiscal_collapse 阈值
        # 机制：工业萎缩→税基缩小→政府收入下降
        if "industrial_withering" in active and target_fracture == "fiscal_collapse":
            duration = state.t - self._fracture_trigger_times.get("industrial_withering", state.t)
            cascade_factor -= cs * min(1.0, duration / 20.0) * 1.0

        # social_collapse → 降低 human_capital_erosion 阈值
        # 机制：社会崩溃→教育/健康投入不足→人力资本退化
        if "social_collapse" in active and target_fracture == "human_capital_erosion":
            duration = state.t - self._fracture_trigger_times.get("social_collapse", state.t)
            cascade_factor -= cs * min(1.0, duration / 15.0) * 1.2

        # jump_induced_deindustrialization → 降低 social_collapse 阈值
        # 机制：跳跃诱发去工业化→大规模工业失业+技能不匹配→社会矛盾快速激化
        if "jump_induced_deindustrialization" in active and target_fracture == "social_collapse":
            duration = state.t - self._fracture_trigger_times.get(
                "jump_induced_deindustrialization", state.t)
            cascade_factor -= cs * min(1.0, duration / 12.0) * 2.0

        # 多断裂并发→系统级崩溃风险
        if len(active) >= 3:
            cascade_factor -= cs * (len(active) - 2) * 0.8

        # v3.5: 新增级联路径

        # data_colonialism → 降低 tech_lock_in 阈值
        # 机制：数据被外国控制→技术路线由外国决定→技术锁死
        if "data_colonialism" in active and target_fracture == "tech_lock_in":
            duration = state.t - self._fracture_trigger_times.get("data_colonialism", state.t)
            cascade_factor -= cs * min(1.0, duration / 18.0) * 1.2

        # fiscal_collapse → 降低 human_capital_erosion 阈值
        # 机制：财政崩溃→教育医疗支出骤减→人力资本退化
        if "fiscal_collapse" in active and target_fracture == "human_capital_erosion":
            duration = state.t - self._fracture_trigger_times.get("fiscal_collapse", state.t)
            cascade_factor -= cs * min(1.0, duration / 10.0) * 1.5

        # dependency_trap → 降低 fiscal_collapse 阈值
        # 机制：过度依附→外部冲击传导→财政崩溃
        if "dependency_trap" in active and target_fracture == "fiscal_collapse":
            duration = state.t - self._fracture_trigger_times.get("dependency_trap", state.t)
            cascade_factor -= cs * min(1.0, duration / 20.0) * 1.0

        return max(0.25, cascade_factor)  # v3.5: 最多降低75%阈值（原70%）

    def check(self, state: SystemState) -> list:
        """检查所有断裂条件，返回触发的断裂事件列表."""
        events = []
        active = state.fracture_status.active_fractures
        m = self.mult

        # 1. 工业枯萎 (硬) — 受 tech_lock_in 级联影响
        K_i_ratio = state.sectors["i"].K / max(state.initial_K_i, 1.0)
        iw_threshold = 0.30 * m * self._cascade_multiplier(state, "industrial_withering")
        if K_i_ratio < iw_threshold and "industrial_withering" not in active:
            events.append(self._event("industrial_withering", "hard", True))

        # 2. 技术锁死 (软) — 受 dependency_trap 级联影响
        G = state.tech_gap
        tl_threshold = 10.0 * m * self._cascade_multiplier(state, "tech_lock_in")
        if G > tl_threshold and "tech_lock_in" not in active:
            events.append(self._event("tech_lock_in", "soft", False))

        # 3. 依附固化 (软)
        dep_threshold = 0.70 * m
        if state.external_dependency > dep_threshold and "dependency_trap" not in active:
            events.append(self._event("dependency_trap", "soft", False))

        # 4. 社会崩溃 (硬) — 受 tech_lock_in 和 premature_deind 级联影响
        sc_mult = self._cascade_multiplier(state, "social_collapse")
        if state.unemployment_rate > 0.25 * m * sc_mult and "social_collapse" not in active:
            events.append(self._event("social_collapse", "hard", True))

        if (state.welfare_index < self.welfare_threshold * m * sc_mult
                and state.welfare_index > 0
                and "social_collapse" not in active):
            events.append(self._event("social_collapse", "hard", True))

        # 5. 财政崩溃 (硬) — 受 industrial_withering 级联影响
        fc_mult = self._cascade_multiplier(state, "fiscal_collapse")
        if state.fiscal_debt_ratio > 1.50 * m * fc_mult and "fiscal_collapse" not in active:
            events.append(self._event("fiscal_collapse", "hard", True))

        # 6. 人力资本崩溃 (时间阈值) — 受 social_collapse 级联影响
        H_avg = np.mean([s.H for s in state.sectors.values()])
        hc_mult = self._cascade_multiplier(state, "human_capital_erosion")
        if H_avg < 0.50 * m * hc_mult and "human_capital_erosion" not in active:
            events.append(self._event("human_capital_erosion", "time", False))

        # 7. 数据殖民断裂 (v2.0)
        cons_data_abroad = state.data_assets.consumer_data * state.dependency.data_dependency
        total_data = state.data_assets.producer_data + state.data_assets.consumer_data
        if (total_data > 0
                and cons_data_abroad / total_data > 0.80 * m
                and "data_colonialism" not in active):
            events.append(self._event("data_colonialism", "soft", False))

        # 8. 过早去工业化断裂 (v2.0)
        if (state.cost_metrics.deind_type == "premature"
                and "premature_deindustrialization" not in active):
            events.append(self._event("premature_deindustrialization", "time", False))

        # 9. 跳跃诱发去工业化断裂 (v3.2) — 策略B过度提取的典型后果
        # 条件：工业资本萎缩>50% + 数字部门仍在扩张（说明提取仍在继续）
        K_i_ratio_initial = state.sectors["i"].K / max(state.initial_K_i, 1.0)
        # v3.7.3: 使用 v3.5 拆分后的 infra+app 初始资本, 而非 v3.4 遗留的 initial_K_d(=1.0)
        initial_K_d_total = state.initial_K_d_infra + state.initial_K_d_app
        K_d_ratio_initial = (state.sectors["d_infra"].K + state.sectors["d_app"].K) / max(initial_K_d_total, 1.0)
        if (K_i_ratio_initial < 0.50
                and K_d_ratio_initial > 1.20
                and "jump_induced_deindustrialization" not in active):
            events.append(self._event("jump_induced_deindustrialization", "hard", False))

        # 记录触发时间用于级联计算
        for evt in events:
            self._fracture_trigger_times[evt["type"]] = state.t

        return events

    def _event(self, ftype: str, severity: str, is_terminal: bool):
        return {
            "type": ftype,
            "severity": severity,
            "is_terminal": is_terminal,
        }

    def imminent(self, state: SystemState, threshold: float = 0.05) -> bool:
        """检测是否有断裂即将触发."""
        m = self.mult
        checks = [
            (state.sectors["i"].K / max(state.initial_K_i, 1.0)
             < 0.30 * m * (1 + threshold) * self._cascade_multiplier(state, "industrial_withering")),
            (state.tech_gap
             > 10.0 * m * (1 - threshold) * self._cascade_multiplier(state, "tech_lock_in")),
            (state.unemployment_rate
             > 0.25 * m * (1 - threshold) * self._cascade_multiplier(state, "social_collapse")),
        ]
        return any(checks)

    def summary(self) -> dict:
        """返回断裂监测摘要."""
        return {
            "active": list(self._fracture_trigger_times.keys()),
            "cascade_active": len(self._fracture_trigger_times) >= 2,
        }
