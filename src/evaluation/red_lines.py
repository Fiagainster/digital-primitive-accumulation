"""人道主义红线检测 —— 不可逾越的人类发展底线.

v3.0 核心升级：民生底线不能因为其他维度表现优异而被掩盖。
分为两级：
- 人道主义失败 (FAILURE)：路径被标记为"不可接受"，不参与策略排序
- 人道主义警告 (WARNING)：路径仍可排序，但报告中显式标注
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from ..core.state import SystemState


@dataclass
class RedLineResult:
    """红线检测结果."""

    # 总体判定
    is_acceptable: bool = True         # 是否通过了所有红线
    level: str = "PASS"                # "PASS" | "WARNING" | "FAILURE"

    # 触发的红线列表
    failures: List[str] = field(default_factory=list)    # 人道主义失败
    warnings: List[str] = field(default_factory=list)    # 人道主义警告

    # 详细信息
    details: Dict[str, dict] = field(default_factory=dict)

    # 记录触发时的时间点
    failure_times: Dict[str, float] = field(default_factory=dict)
    warning_times: Dict[str, float] = field(default_factory=dict)


class RedLineDetector:
    """人道主义红线检测器.

    红线独立于断裂检测——系统可能没有"崩溃"，
    但如果突破了某些人类发展的底线，依然被标记为不可接受。
    """

    def __init__(self):
        # 定义红线：每种红线的触发条件和严重级别
        self._red_lines = [
            {
                "name": "极端贫困",
                "check": self._check_extreme_poverty,
                "level": "FAILURE",
                "description": "超过40%人口处于贫困线以下",
            },
            {
                "name": "长期大规模失业",
                "check": self._check_persistent_unemployment,
                "level": "FAILURE",
                "description": "失业率超过20%持续15年以上",
            },
            {
                "name": "基本生存危机",
                "check": self._check_subsistence_crisis,
                "level": "FAILURE",
                "description": "基础部门人均产出低于生存线",
            },
            {
                "name": "代际掠夺",
                "check": self._check_intergenerational_plunder,
                "level": "WARNING",
                "description": "当代人代价超过后代收益3倍",
            },
            {
                "name": "群组灭绝性代价",
                "check": self._check_group_decimation,
                "level": "FAILURE",
                "description": "任一家庭群组人口下降超50%且福利低于0.20",
            },
            {
                "name": "大规模技能贬值",
                "check": self._check_skill_decimation,
                "level": "WARNING",
                "description": "工业劳动者技能持续贬值（>30%）且无再培训",
            },
            {
                "name": "儿童发展危机",
                "check": self._check_child_development,
                "level": "WARNING",
                "description": "教育投入/GDP持续低于2%且人力资本下降",
            },
        ]

    def check(self, state: SystemState,
              prev_state: Optional[SystemState] = None,
              persistence_tracker: Optional[dict] = None) -> RedLineResult:
        """对所有红线进行检测.

        Args:
            state: 当前系统状态
            prev_state: 前一步状态（用于检测变化趋势）
            persistence_tracker: 持续性追踪字典（用于检测"持续X年"的条件）

        Returns:
            RedLineResult: 检测结果
        """
        result = RedLineResult()
        if persistence_tracker is None:
            persistence_tracker = {}

        for rl in self._red_lines:
            check_result = rl["check"](state, prev_state, persistence_tracker)
            if check_result["triggered"]:
                if rl["level"] == "FAILURE":
                    result.failures.append(rl["name"])
                    result.failure_times[rl["name"]] = state.t
                else:
                    result.warnings.append(rl["name"])
                    result.warning_times[rl["name"]] = state.t
                result.details[rl["name"]] = {
                    "level": rl["level"],
                    "description": rl["description"],
                    "value": check_result.get("value", 0),
                    "threshold": check_result.get("threshold", 0),
                }

        # 判定最终级别
        if result.failures:
            result.is_acceptable = False
            result.level = "FAILURE"
        elif result.warnings:
            result.level = "WARNING"
        else:
            result.level = "PASS"

        return result

    # ─────────────────────────────────────────────
    # 各红线的检测逻辑
    # ─────────────────────────────────────────────

    def _check_extreme_poverty(self, state: SystemState,
                                prev_state, tracker: dict) -> dict:
        """极端贫困：超过40%人口在贫困线以下."""
        groups = state.households.all_groups()
        consumptions = []
        pops = []
        for h in groups:
            if h.population > 0:
                consumptions.append(h.consumption / h.population)
                pops.append(h.population)

        if not consumptions or sum(pops) == 0:
            return {"triggered": False, "value": 0, "threshold": 0.40}

        median_cons = np.median(consumptions)
        poverty_line = median_cons * 0.50
        poor_pop = sum(p for c, p in zip(consumptions, pops) if c < poverty_line)
        poverty_rate = poor_pop / sum(pops)

        return {
            "triggered": poverty_rate > 0.40,
            "value": poverty_rate,
            "threshold": 0.40,
        }

    def _check_persistent_unemployment(self, state: SystemState,
                                        prev_state, tracker: dict) -> dict:
        """长期大规模失业：失业率>20%持续15年以上."""
        u_rate = state.unemployment_rate
        key = "persistent_unemployment_years"

        if key not in tracker:
            tracker[key] = 0.0

        if u_rate > 0.20:
            tracker[key] += 0.25  # 每步0.25年
        else:
            tracker[key] = max(0.0, tracker[key] - 0.25)  # 缓慢恢复

        return {
            "triggered": tracker[key] > 15.0,
            "value": tracker[key],
            "threshold": 15.0,
        }

    def _check_subsistence_crisis(self, state: SystemState,
                                   prev_state, tracker: dict) -> dict:
        """基本生存危机：基础部门人均产出低于生存线."""
        food_per_cap = state.sectors["b"].Y / max(state.households.total_population, 1.0)
        # 绝对生存线（模型单位）
        subsistence_line = 0.20
        severity = subsistence_line / max(food_per_cap, 0.01)

        return {
            "triggered": food_per_cap < subsistence_line,
            "value": food_per_cap,
            "threshold": subsistence_line,
        }

    def _check_intergenerational_plunder(self, state: SystemState,
                                          prev_state, tracker: dict) -> dict:
        """代际掠夺：当代代价超过后代收益3倍."""
        total_cost = sum(h.cumulative_cost for h in state.households.all_groups())
        # 后代收益 = 累积GDP * 人力资本/当前消费
        future_benefit = (state.gdp * state.t
                          * (0.5 * (state.sectors["d_infra"].H + state.sectors["d_app"].H) + state.sectors["i"].H) / 2.0)
        if future_benefit > 0:
            ratio = total_cost / future_benefit
            return {
                "triggered": ratio > 3.0,
                "value": ratio,
                "threshold": 3.0,
            }
        return {"triggered": False, "value": 0, "threshold": 3.0}

    def _check_group_decimation(self, state: SystemState,
                                 prev_state, tracker: dict) -> dict:
        """群组灭绝性代价：任一家庭群组人口下降>50%且福利<0.20."""
        # 需要在初始化时记录初始人口
        if "initial_populations" not in tracker:
            tracker["initial_populations"] = {}
            for name, group in [
                ("digital_high_skill", state.households.digital_high_skill),
                ("industrial_workers", state.households.industrial_workers),
                ("base_workers", state.households.base_workers),
            ]:
                tracker["initial_populations"][name] = max(group.population, 1.0)

        for name, group in [
            ("digital_high_skill", state.households.digital_high_skill),
            ("industrial_workers", state.households.industrial_workers),
            ("base_workers", state.households.base_workers),
        ]:
            init_pop = tracker["initial_populations"].get(name, 1.0)
            pop_decline = 1.0 - group.population / init_pop
            if pop_decline > 0.50 and group.welfare_index < 0.20:
                return {
                    "triggered": True,
                    "value": pop_decline,
                    "threshold": 0.50,
                }

        return {"triggered": False, "value": 0, "threshold": 0.50}

    def _check_skill_decimation(self, state: SystemState,
                                 prev_state, tracker: dict) -> dict:
        """大规模技能贬值：工业劳动者技能持续贬值>30%且无再培训."""
        # 对比工业部门人力资本与数字部门
        skill_gap = (0.5 * (state.sectors["d_infra"].H + state.sectors["d_app"].H)
                     / max(state.sectors["i"].H, 0.01))
        # 技能差距过大 → 工业劳动者技能相对贬值严重
        is_triggered = skill_gap > 3.0  # 数字部门技能是工业的3倍以上

        return {
            "triggered": is_triggered,
            "value": skill_gap,
            "threshold": 3.0,
        }

    def _check_child_development(self, state: SystemState,
                                  prev_state, tracker: dict) -> dict:
        """儿童发展危机：教育投入/GDP持续低于2%且人力资本下降."""
        # 从培训投资近似教育投入
        training_invest = self._estimate_education_spending(state)
        edu_ratio = training_invest / max(state.gdp, 0.01)

        # 检查人力资本趋势
        avg_H = (state.sectors["b"].H + state.sectors["i"].H + 0.5 * (state.sectors["d_infra"].H + state.sectors["d_app"].H)) / 3.0
        key = "child_dev_h_prev"
        h_declining = False
        if key in tracker:
            h_declining = avg_H < tracker[key]
        tracker[key] = avg_H

        is_triggered = edu_ratio < 0.02 and h_declining

        return {
            "triggered": is_triggered,
            "value": edu_ratio,
            "threshold": 0.02,
        }

    @staticmethod
    def _estimate_education_spending(state: SystemState) -> float:
        """估计教育支出."""
        # 从培训投资和公共支出推断
        training_invest = state.gdp * 0.015  # 基础值
        # 加上社保相关支出
        training_invest += state.households.unemployed.compensation_received * 0.1
        return training_invest
