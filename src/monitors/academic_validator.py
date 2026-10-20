"""学术验证模块 —— v3.2 方法论防御性加固.

处理五个核心学术可信度问题：
1. 可证伪性 (Q1/Q11/Q12): 零假设基线 + 通用异常检测 + 意外发现追踪
2. SFC残差监控 (Q4): 残差占比追踪 + 告警
3. 断裂触发矩阵 (Q14): 每种断裂的触发率统计 + 休眠断裂标记
4. 五维相关性矩阵 (Q9): 维度独立性验证
5. 红线阈值敏感性 (Q10): 阈值扰动对结论的影响

设计原则：此模块不修改任何现有模块的行为逻辑，只做观测和记录。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings


@dataclass
class ValidationReport:
    """一次模拟运行后的学术验证报告."""

    # 1. 零假设检查
    null_hypothesis_passed: bool = True
    null_hypothesis_details: str = ""

    # 2. SFC残差
    sfc_residual_ratios: List[float] = field(default_factory=list)
    sfc_max_residual: float = 0.0
    sfc_median_residual: float = 0.0
    sfc_trustworthiness: str = "HIGH"     # HIGH | MEDIUM | LOW
    sfc_pressure_residual_ratio: float = 0.0  # 断裂逼近时的残差比

    # 3. 通用异常检测
    generic_anomalies_detected: int = 0
    generic_anomaly_variables: List[str] = field(default_factory=list)
    predefined_fractures_triggered: int = 0
    uncovered_anomalies: int = 0  # 通用检测发现但预定义断裂未覆盖的异常

    # 4. 五维相关性（跨模拟聚合，单次模拟不填充）
    dimension_correlations: Dict[str, float] = field(default_factory=dict)

    # 5. 意外发现标记
    unexpected_findings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        lines.append(f"SFC可信度: {self.sfc_trustworthiness} "
                     f"(中位数残差={self.sfc_median_residual:.4f}, "
                     f"最大={self.sfc_max_residual:.4f})")
        lines.append(f"通用异常检测: {self.generic_anomalies_detected}个异常, "
                     f"预定义断裂覆盖了{self.generic_anomalies_detected - self.uncovered_anomalies}个, "
                     f"未覆盖{self.uncovered_anomalies}个")
        if self.unexpected_findings:
            lines.append(f"意外发现: {len(self.unexpected_findings)}条")
        return "\n".join(lines)


class AcademicValidator:
    """学术验证器 —— 在模拟过程中和模拟结束后运行检测.

    使用方式:
        validator = AcademicValidator()
        # 在模拟过程中
        validator.record_sfc_residual(residual, total_tx)
        validator.record_state_snapshot(state)
        # 模拟结束后
        report = validator.finalize(result, predefined_fractures)
    """

    def __init__(self, params: dict = None):
        self.p = params or {}
        self._sfc_residuals: List[float] = []
        self._sfc_total_tx: List[float] = []
        self._state_snapshots: List[dict] = []  # 周期性快照
        self._snapshot_interval = 4  # 每4步记录一次快照
        self._step_counter = 0

    # ─────────────────────────────────────────────
    # 1. 零假设基线测试（静态方法，在批量实验中调用）
    # ─────────────────────────────────────────────
    @staticmethod
    def test_null_hypothesis(results_by_regime: Dict[str, List[dict]]) -> ValidationReport:
        """零假设：当三种策略参数完全相同时，输出应无统计差异.

        Args:
            results_by_regime: {"A": [result1, ...], "B": [...], "C": [...]}

        Returns:
            ValidationReport with null hypothesis check results
        """
        report = ValidationReport()

        # 提取各策略的CDI
        cdi_by_regime = {}
        for regime, results in results_by_regime.items():
            cdis = []
            for r in results:
                ev = r.get("summary", {}).get("evaluation", {})
                if ev.get("enabled"):
                    cdis.append(ev.get("cdi", 0))
            if cdis:
                cdi_by_regime[regime] = np.array(cdis)

        if len(cdi_by_regime) < 2:
            report.null_hypothesis_passed = True
            report.null_hypothesis_details = "策略数不足，无法检验"
            return report

        # 计算各策略CDI均值的最大差异
        means = {r: np.mean(cdis) for r, cdis in cdi_by_regime.items()}
        max_diff = max(means.values()) - min(means.values())

        # 简单检验：如果均值差异 < 0.05（CDI的5%），认为通过零假设
        # 这个阈值可以讨论——核心不是"完全相同"而是"差异小到无实际意义"
        if max_diff < 0.05:
            report.null_hypothesis_passed = True
            report.null_hypothesis_details = (
                f"零假设通过：三策略CDI最大差异={max_diff:.4f} < 0.05阈值。"
                f"均值: A={means.get('A', 0):.3f}, "
                f"B={means.get('B', 0):.3f}, "
                f"C={means.get('C', 0):.3f}"
            )
        else:
            report.null_hypothesis_passed = False
            report.null_hypothesis_details = (
                f"零假设未通过：三策略CDI最大差异={max_diff:.4f}。"
                f"这表明策略参数在'零假设'条件下仍然存在差异——"
                f"检查参数是否真的被设为完全相同。"
                f"均值: A={means.get('A', 0):.3f}, "
                f"B={means.get('B', 0):.3f}, "
                f"C={means.get('C', 0):.3f}"
            )

        return report

    # ─────────────────────────────────────────────
    # 2. SFC残差监控
    # ─────────────────────────────────────────────
    def record_sfc_residual(self, residual: float, total_transactions: float):
        """记录每一步的SFC残差."""
        self._sfc_residuals.append(abs(residual))
        self._sfc_total_tx.append(max(total_transactions, 0.01))

    def record_state_snapshot(self, state):
        """周期性记录状态快照（用于通用异常检测）."""
        self._step_counter += 1
        if self._step_counter % self._snapshot_interval == 0:
            self._state_snapshots.append({
                "t": state.t,
                "K_i": state.sectors["i"].K,
                "K_d": state.sectors["d_infra"].K + state.sectors["d_app"].K,
                "K_b": state.sectors["b"].K,
                "Y_i": state.sectors["i"].Y,
                "Y_d": state.sectors["d_infra"].Y + state.sectors["d_app"].Y,
                "Y_b": state.sectors["b"].Y,
                "tech_gap": state.tech_gap,
                "unemployment_rate": state.unemployment_rate,
                "gini": state.gini,
                "welfare_index": state.welfare_index,
                "gdp": state.gdp,
                "DD_i": state.sectors["i"].digital_density,
                "DD_d": 0.5 * (state.sectors["d_infra"].digital_density + state.sectors["d_app"].digital_density),
            })

    def _compute_sfc_stats(self) -> dict:
        """计算SFC残差统计."""
        if not self._sfc_residuals:
            return {"max": 0, "median": 0, "trustworthiness": "HIGH",
                    "pressure_ratio": 0}

        ratios = []
        for resid, tx in zip(self._sfc_residuals, self._sfc_total_tx):
            ratios.append(resid / tx)

        ratios = np.array(ratios)
        median_r = float(np.median(ratios))
        max_r = float(np.max(ratios))

        # 可信度判定
        if max_r > 0.10:
            trust = "LOW"
        elif max_r > 0.05:
            trust = "MEDIUM"
        else:
            trust = "HIGH"

        # 断裂逼近时的残差（最后25%的步骤）
        n = len(ratios)
        if n >= 4:
            pressure_r = float(np.mean(ratios[int(n*0.75):]))
        else:
            pressure_r = median_r

        return {
            "max": max_r,
            "median": median_r,
            "trustworthiness": trust,
            "pressure_ratio": pressure_r,
            "ratios": ratios,
        }

    # ─────────────────────────────────────────────
    # 3. 通用异常检测
    # ─────────────────────────────────────────────
    @staticmethod
    def detect_generic_anomalies(time_series: np.ndarray,
                                  window: int = 20,
                                  n_sigma: float = 3.0) -> List[int]:
        """不基于预定义阈值的通用异常检测.

        使用滑动窗口的均值和标准差，检测突变点。
        这回答"模型是否只检测到了预定义的断裂，还是有涌现的异常"。

        Args:
            time_series: 时间序列数据
            window: 滑动窗口大小
            n_sigma: 异常判定的标准差倍数

        Returns:
            异常点的索引列表
        """
        if len(time_series) < window + 1:
            return []

        anomalies = []
        for i in range(window, len(time_series)):
            window_data = time_series[i-window:i]
            mu = np.mean(window_data)
            sigma = np.std(window_data)
            if sigma < 1e-10:
                continue
            z_score = abs(time_series[i] - mu) / sigma
            if z_score > n_sigma:
                anomalies.append(i)

        # 合并连续的异常点
        if not anomalies:
            return []

        merged = [anomalies[0]]
        for a in anomalies[1:]:
            if a - merged[-1] > 2:  # 间隔超过2步 → 新的异常事件
                merged.append(a)

        return merged

    def detect_all_generic_anomalies(self) -> Dict[str, List[int]]:
        """对所有记录的状态变量执行通用异常检测."""
        if len(self._state_snapshots) < 30:
            return {}

        results = {}
        monitored_vars = [
            "K_i", "K_d", "tech_gap", "unemployment_rate",
            "gdp", "welfare_index", "gini", "DD_i",
        ]

        for var in monitored_vars:
            ts = np.array([s[var] for s in self._state_snapshots])
            anomalies = self.detect_generic_anomalies(ts)
            if anomalies:
                results[var] = anomalies

        return results

    # ─────────────────────────────────────────────
    # 4. 五维相关性矩阵（静态方法，在批量实验后调用）
    # ─────────────────────────────────────────────
    @staticmethod
    def compute_dimension_correlations(results: List[dict]) -> Dict[str, Dict[str, float]]:
        """计算五个评价维度之间的Pearson相关系数矩阵.

        Args:
            results: 批量实验结果列表

        Returns:
            5x5 相关系数矩阵字典
        """
        dims = ["economy", "livelihood", "resilience", "structure", "sovereignty"]
        # 收集所有有效的五维得分
        scores = {d: [] for d in dims}
        for r in results:
            ev = r.get("summary", {}).get("evaluation", {})
            if not ev.get("enabled"):
                continue
            for d in dims:
                scores[d].append(ev.get(d, 0))

        # 确保有足够数据
        for d in dims:
            if len(scores[d]) < 3:
                return {}

        # 计算相关系数
        corr_matrix = {}
        for d1 in dims:
            corr_matrix[d1] = {}
            arr1 = np.array(scores[d1])
            for d2 in dims:
                arr2 = np.array(scores[d2])
                if d1 == d2:
                    corr_matrix[d1][d2] = 1.0
                else:
                    # Pearson r
                    corr = np.corrcoef(arr1, arr2)[0, 1]
                    corr_matrix[d1][d2] = float(0.0 if np.isnan(corr) else corr)

        return corr_matrix

    @staticmethod
    def interpret_correlations(corr_matrix: Dict[str, Dict[str, float]]) -> List[str]:
        """解读五维相关性，输出有意义的发现."""
        findings = []
        if not corr_matrix:
            return findings

        dims = ["economy", "livelihood", "resilience", "structure", "sovereignty"]
        dim_labels = {
            "economy": "经济活力", "livelihood": "民生福祉",
            "resilience": "发展韧性", "structure": "结构健康",
            "sovereignty": "主权自主",
        }

        for i, d1 in enumerate(dims):
            for d2 in dims[i+1:]:
                r = corr_matrix.get(d1, {}).get(d2, 0)
                if abs(r) > 0.8:
                    findings.append(
                        f"高度{'正' if r > 0 else '负'}相关: "
                        f"{dim_labels[d1]} vs {dim_labels[d2]} (r={r:.3f})——"
                        f"这两个维度可能不是独立的"
                    )
                elif abs(r) < 0.1:
                    findings.append(
                        f"几乎独立: {dim_labels[d1]} vs {dim_labels[d2]} (r={r:.3f})——"
                        f"信息增益最大"
                    )
                elif r < -0.3:
                    findings.append(
                        f"显著负相关: {dim_labels[d1]} vs {dim_labels[d2]} (r={r:.3f})——"
                        f"这是模型最有价值的张力发现"
                    )

        return findings

    # ─────────────────────────────────────────────
    # 5. 红线敏感性分析（静态方法）
    # ─────────────────────────────────────────────
    @staticmethod
    def red_line_sensitivity_test(
        base_thresholds: Dict[str, float],
        perturbation: float = 0.25,
    ) -> Dict[str, List[float]]:
        """生成红线阈值的敏感性测试参数集.

        Args:
            base_thresholds: {"poverty_rate": 0.40, "unemployment_rate": 0.20, ...}
            perturbation: 扰动幅度（比例）

        Returns:
            {"poverty_rate": [0.30, 0.35, 0.40, 0.45, 0.50], ...}
        """
        result = {}
        for name, base in base_thresholds.items():
            delta = base * perturbation
            result[name] = [
                max(0.01, base - delta),
                max(0.01, base - delta/2),
                base,
                min(0.99, base + delta/2),
                min(0.99, base + delta),
            ]
        return result

    # ─────────────────────────────────────────────
    # 最终化：生成验证报告
    # ─────────────────────────────────────────────
    def finalize(self, result: dict,
                 predefined_fractures: List[str] = None) -> ValidationReport:
        """模拟结束后生成完整的验证报告.

        Args:
            result: 模拟结果字典
            predefined_fractures: 预定义断裂类型列表

        Returns:
            ValidationReport
        """
        report = ValidationReport()

        # 1. SFC残差统计
        sfc_stats = self._compute_sfc_stats()
        report.sfc_residual_ratios = list(sfc_stats.get("ratios", []))
        report.sfc_max_residual = sfc_stats["max"]
        report.sfc_median_residual = sfc_stats["median"]
        report.sfc_trustworthiness = sfc_stats["trustworthiness"]
        report.sfc_pressure_residual_ratio = sfc_stats["pressure_ratio"]

        # 2. 通用异常检测 vs 预定义断裂
        generic_anomalies = self.detect_all_generic_anomalies()
        report.generic_anomalies_detected = sum(len(v) for v in generic_anomalies.values())
        report.generic_anomaly_variables = list(generic_anomalies.keys())

        predefined_count = len(predefined_fractures) if predefined_fractures else 0
        report.predefined_fractures_triggered = predefined_count

        # 估算未覆盖的异常
        # 简化方法：如果通用检测发现了3个以上异常，但预定义断裂只有1个 → 可能有未覆盖的
        if (report.generic_anomalies_detected >= 3
                and predefined_count <= 1):
            report.uncovered_anomalies = max(
                0, report.generic_anomalies_detected - predefined_count * 2
            )
            report.unexpected_findings.append(
                f"通用异常检测发现{report.generic_anomalies_detected}个异常，"
                f"但预定义断裂只触发了{predefined_count}种——"
                f"可能存在未被预定义断裂覆盖的异常模式。"
                f"异常变量: {', '.join(report.generic_anomaly_variables)}"
            )

        # 3. SFC可信度告警
        if report.sfc_trustworthiness == "LOW":
            report.unexpected_findings.append(
                f"SFC可信度为LOW（最大残差比={report.sfc_max_residual:.4f}），"
                f"建议检查该模拟的核算闭合质量。"
            )

        # 4. 断裂逼近时残差异常
        if (report.sfc_pressure_residual_ratio
                > report.sfc_median_residual * 1.5):
            report.unexpected_findings.append(
                f"SFC残差在模拟后期（断裂逼近时）上升了"
                f"{report.sfc_pressure_residual_ratio/report.sfc_median_residual:.1f}倍——"
                f"核算框架可能在压力时刻充当了'减震器'。"
            )

        return report


# ─────────────────────────────────────────────
# 批量实验后分析工具
# ─────────────────────────────────────────────

def build_fracture_trigger_matrix(
    results: List[dict],
    regimes: List[str] = None,
    profiles: List[str] = None,
) -> Dict[str, Dict[str, int]]:
    """构建断裂触发频率矩阵.

    返回: {断裂类型: {策略A: 触发次数, 策略B: ..., 策略C: ...}}
    """
    fracture_counts: Dict[str, Dict[str, int]] = {}

    for r in results:
        summary = r["summary"]
        regime = summary.get("regime", "?")
        if regimes and regime not in regimes:
            continue

        for ft in summary.get("fracture_types", []):
            if ft not in fracture_counts:
                fracture_counts[ft] = {}
            fracture_counts[ft][regime] = fracture_counts[ft].get(regime, 0) + 1

    return fracture_counts


def identify_dormant_fractures(
    fracture_matrix: Dict[str, Dict[str, int]],
    n_total: int,
    threshold: float = 0.01,
) -> List[str]:
    """识别休眠断裂——触发率低于阈值的断裂模式.

    Args:
        fracture_matrix: build_fracture_trigger_matrix 的输出
        n_total: 模拟总次数
        threshold: 触发率阈值（默认1%）

    Returns:
        休眠断裂名称列表
    """
    dormant = []
    for ft_name, regime_counts in fracture_matrix.items():
        total_triggers = sum(regime_counts.values())
        rate = total_triggers / n_total
        if rate < threshold:
            dormant.append(
                f"{ft_name}（触发{total_triggers}次/{n_total}次，"
                f"触发率{rate*100:.1f}%）"
            )
    return dormant


def compute_labor_skill_erosion(
    initial_H_i: float,
    current_H_i: float,
) -> float:
    """计算工业留守者技能侵蚀指数.

    Returns:
        技能侵蚀比: 1.0 = 无侵蚀, < 0.5 = 严重侵蚀
    """
    return current_H_i / max(initial_H_i, 0.01)
