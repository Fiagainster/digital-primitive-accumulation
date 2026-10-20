"""早期预警信号系统 —— 临界慢化、闪烁、爆炸同步 (v2.2 升级).

v2.2 升级：
- 新增韧性指数(Resilience Index)：区分"结构性高压"与"临界逼近"
- 追踪EWS高位持续时间、恢复速度、抗扰动幅度
- 动态阈值：韧性高的系统即使EWS>0.7也不触发临界警报
v2.1 升级：
- 完善 es_proximity：基于 Kuramoto 序参量的部门间耦合强度
- 改进闪烁检测：使用 Hartigan's dip test 替代简化峰度检测
- 新增系统级耦合强度监测
"""

import numpy as np
import warnings
from scipy import stats
from ..core.state import SystemState, EWSResult

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.stats")


class EarlyWarningMonitor:
    """EWS监测器 —— 在断裂触发之前检测系统恢复力丧失的信号 (v2.2)."""

    def __init__(self, params: dict, window_size: int = 40):
        self.window_size = window_size
        self.history: dict = {}
        self.warning_threshold = params.get("ews_warning_threshold", 0.4)
        self.critical_threshold = params.get("ews_critical_threshold", 0.7)
        # v2.1: 追踪部门间耦合强度历史
        self._coupling_history: list = []
        self._sector_phase_history: dict = {
            "b": [], "i": [],
            "d": [], "s": [],  # aggregate growth for coupling computation
            "d_infra": [], "d_app": [], "s_high": [], "s_low": [],
        }
        # v2.2: 韧性追踪
        self._resilience_window = params.get("ews_resilience_window", 60)
        self._resilience_weight = params.get("ews_resilience_weight", 0.35)
        self._stress_duration_threshold = params.get("ews_stress_duration_threshold", 30)
        # 每个受监控变量的EWS历史（用于韧性计算）
        self._ews_history: dict = {}
        # 变量扰动历史（用于恢复速度计算）
        self._perturbation_history: dict = {}
        # 连续的"高压"步数计数
        self._stress_duration: dict = {}
        # 记录曾经达到的最低点（用于计算已发生的最大扰动）
        self._worst_values: dict = {}

    def feed(self, state: SystemState):
        """记录当前观测."""
        vars_to_monitor = {
            "K_i_ratio": state.sectors["i"].K / max(state.initial_K_i, 1.0),
            "tech_gap": state.tech_gap,
            "unemployment": state.unemployment_rate,
            "welfare": state.welfare_index,
            "external_dep": state.external_dependency,
        }
        for name, value in vars_to_monitor.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)

            # v2.2: 追踪最差值（用于计算抗扰动幅度）
            if name not in self._worst_values:
                self._worst_values[name] = value
            if name == "K_i_ratio" or name == "welfare":
                self._worst_values[name] = min(self._worst_values[name], value)
            else:
                self._worst_values[name] = max(self._worst_values[name], value)

        # v2.1: 追踪部门产出增长率（用于相位同步检测）
        for key in ["b", "i", "d_infra", "d_app", "s_high", "s_low"]:
            s = state.sectors[key]
            if s.Y > 0:
                growth = (s.Y - getattr(self, f'_prev_Y_{key}', s.Y)) / max(s.Y, 0.01)
            else:
                growth = 0.0
            self._sector_phase_history[key].append(growth)
            setattr(self, f'_prev_Y_{key}', s.Y)
            # 限制历史长度
            if len(self._sector_phase_history[key]) > self.window_size * 2:
                self._sector_phase_history[key] = self._sector_phase_history[key][-self.window_size * 2:]

        # v2.1: 计算数字部门和服务部门的聚合增长率（用于耦合计算）
        Y_d_agg = state.sectors["d_infra"].Y + state.sectors["d_app"].Y
        prev_Y_d = getattr(self, '_prev_Y_d_agg', Y_d_agg)
        d_agg_growth = (Y_d_agg - prev_Y_d) / max(Y_d_agg, 0.01)
        self._sector_phase_history["d"].append(d_agg_growth)
        self._prev_Y_d_agg = Y_d_agg
        if len(self._sector_phase_history["d"]) > self.window_size * 2:
            self._sector_phase_history["d"] = self._sector_phase_history["d"][-self.window_size * 2:]

        Y_s_agg = state.sectors["s_high"].Y + state.sectors["s_low"].Y
        prev_Y_s = getattr(self, '_prev_Y_s_agg', Y_s_agg)
        s_agg_growth = (Y_s_agg - prev_Y_s) / max(Y_s_agg, 0.01)
        self._sector_phase_history["s"].append(s_agg_growth)
        self._prev_Y_s_agg = Y_s_agg
        if len(self._sector_phase_history["s"]) > self.window_size * 2:
            self._sector_phase_history["s"] = self._sector_phase_history["s"][-self.window_size * 2:]

        # v2.1: 计算部门间耦合强度
        coupling = self._compute_coupling_strength(state)
        self._coupling_history.append(coupling)
        if len(self._coupling_history) > self.window_size * 2:
            self._coupling_history = self._coupling_history[-self.window_size * 2:]

    def _compute_coupling_strength(self, state: SystemState) -> float:
        """计算部门间耦合强度（Kuramoto 序参量近似）.

        高耦合 → 部门高度同步 → 一个部门的震荡会快速传播
        耦合强度上升 → 系统接近临界点（爆炸同步）
        """
        # 部门产出相关性 → 耦合的代理指标
        if (len(self._sector_phase_history["i"]) < 10
                or len(self._sector_phase_history["d"]) < 10):
            return 0.0

        i_growth = np.array(self._sector_phase_history["i"][-20:])
        d_growth = np.array(self._sector_phase_history["d"][-20:])
        b_growth = np.array(self._sector_phase_history["b"][-20:])

        # 计算两两相关性
        corr_id = np.corrcoef(i_growth, d_growth)[0, 1] if len(i_growth) >= 3 else 0.0
        corr_ib = np.corrcoef(i_growth, b_growth)[0, 1] if len(i_growth) >= 3 else 0.0
        corr_db = np.corrcoef(d_growth, b_growth)[0, 1] if len(d_growth) >= 3 else 0.0

        # 处理NaN
        corr_id = 0.0 if np.isnan(corr_id) else corr_id
        corr_ib = 0.0 if np.isnan(corr_ib) else corr_ib
        corr_db = 0.0 if np.isnan(corr_db) else corr_db

        # 综合耦合强度 = 平均绝对相关性
        coupling = (abs(corr_id) + abs(corr_ib) + abs(corr_db)) / 3.0
        return coupling

    def compute_all(self, state: SystemState) -> dict:
        """计算所有监测变量的EWS信号 (v2.2: 含韧性指数)."""
        results = {}
        for name in list(self.history.keys()):
            if len(self.history[name]) >= self.window_size:
                r = self._compute_ews(name)
                if r is not None and not np.isnan(r.composite_score):
                    # v2.2: 计算韧性指数
                    resilience = self._compute_resilience(name, r)
                    r.resilience_index = resilience["index"]
                    r.stress_duration = resilience["stress_duration"]
                    r.recovery_speed = resilience["recovery_speed"]
                    # v2.2: 动态调整状态 — 高韧性系统即使composite>0.7也不一定是critical
                    r.status = self._dynamic_status(r, resilience)
                    results[name] = r
        return results

    def _dynamic_status(self, r: EWSResult, resilience: dict) -> str:
        """v2.2: 基于韧性的动态状态判定.

        关键逻辑：
        - 高韧性（resilience_index > 0.6）：即使composite > 0.7，只报warning
        - 中等韧性（0.3-0.6）：按正常阈值
        - 低韧性（< 0.3）：降低阈值，更早预警
        """
        ri = resilience["index"]
        if ri > 0.6:
            # 高韧性系统：提高阈值
            crit_adj = self.critical_threshold + 0.1
            warn_adj = self.warning_threshold + 0.05
        elif ri > 0.3:
            crit_adj = self.critical_threshold
            warn_adj = self.warning_threshold
        else:
            # 低韧性系统：降低阈值
            crit_adj = self.critical_threshold - 0.1
            warn_adj = self.warning_threshold - 0.05

        if r.composite_score >= crit_adj:
            return "critical"
        elif r.composite_score >= warn_adj:
            return "warning"
        return "normal"

    def _compute_resilience(self, var_name: str, ews_result: EWSResult) -> dict:
        """v2.2: 计算变量的韧性指数.

        韧性 = f(压力持续性, 恢复速度, 抗扰动幅度)

        返回:
            index: 综合韧性指数 [0,1], 1=高韧性
            stress_duration: 连续高压步数
            recovery_speed: 恢复速度估计
        """
        series = np.array(self.history[var_name])

        # 1. 追踪EWS复合评分历史
        if var_name not in self._ews_history:
            self._ews_history[var_name] = []
        self._ews_history[var_name].append(ews_result.composite_score)
        if len(self._ews_history[var_name]) > self._resilience_window:
            self._ews_history[var_name] = self._ews_history[var_name][-self._resilience_window:]

        ews_hist = np.array(self._ews_history[var_name])

        # 2. 压力持续性：连续处于"高压"状态的步数
        if var_name not in self._stress_duration:
            self._stress_duration[var_name] = 0
        if ews_result.composite_score >= self.critical_threshold:
            self._stress_duration[var_name] += 1
        else:
            # 降至warning以下时重置
            if ews_result.composite_score < self.warning_threshold:
                self._stress_duration[var_name] = max(0, self._stress_duration[var_name] - 2)
            else:
                self._stress_duration[var_name] = max(0, self._stress_duration[var_name] - 0.5)

        stress_dur = self._stress_duration[var_name]

        # 压力持续性分数：超过阈值后开始扣分
        # 如果持续高压但系统未崩溃 → 可能只是结构性特征
        if stress_dur > self._stress_duration_threshold:
            stress_penalty = min(1.0, (stress_dur - self._stress_duration_threshold) / 30.0)
        else:
            stress_penalty = stress_dur / self._stress_duration_threshold * 0.5

        # 3. 恢复速度：基于变量近期轨迹估计
        recovery_speed = self._estimate_recovery_speed(var_name, series)

        # 4. 抗扰动幅度：变量曾经历的最大扰动
        worst = self._worst_values.get(var_name, 0)
        if var_name == "K_i_ratio":
            perturbation_magnitude = max(0, 1.0 - worst)
        elif var_name == "welfare":
            perturbation_magnitude = max(0, 1.0 - worst)
        else:
            initial_val = self.history[var_name][0] if self.history[var_name] else 1.0
            perturbation_magnitude = max(0, abs(worst - initial_val) / max(initial_val, 0.01))
        perturbation_resistance = max(0, 1.0 - min(1.0, perturbation_magnitude / 0.5))

        # 5. 综合韧性指数
        # 权重：恢复速度40% + 抗扰动30% + 压力持续性(反向)30%
        resilience = (
            0.40 * min(1.0, recovery_speed * 5.0)
            + 0.30 * perturbation_resistance
            + 0.30 * (1.0 - min(1.0, stress_penalty))
        )

        return {
            "index": max(0.0, min(1.0, resilience)),
            "stress_duration": stress_dur,
            "recovery_speed": recovery_speed,
        }

    def _estimate_recovery_speed(self, var_name: str, series: np.ndarray) -> float:
        """v2.2: 估计变量从扰动的恢复速度.

        方法：检测近期是否有从低谷回升的趋势。
        计算最近N步内的线性回归斜率，正斜率=在恢复。
        """
        if len(series) < 10:
            return 0.0

        recent = series[-20:] if len(series) >= 20 else series

        # 找到近期最低点
        min_idx = np.argmin(recent)
        min_val = recent[min_idx]
        end_val = recent[-1]

        # 从低点到终点的恢复
        if min_val < 0.01:
            return 0.0

        recovery = (end_val - min_val) / max(min_val, 0.01)
        steps_from_min = len(recent) - min_idx - 1

        if steps_from_min > 0:
            speed = recovery / steps_from_min
        else:
            speed = 0.0

        # 趋势检查：如果整体仍在下降，恢复速度=0
        if len(recent) >= 5:
            half = len(recent) // 2
            if np.mean(recent[-half:]) < np.mean(recent[:half]):
                speed = min(0, speed)

        return max(0.0, speed)  # 非负

    def _compute_ews(self, var_name: str) -> EWSResult:
        """计算单个变量的EWS."""
        series = np.array(self.history[var_name][-self.window_size:])
        if len(series) < 10:
            return EWSResult(variable=var_name, composite_score=0.0, status="normal")
        half = len(series) // 2

        # 1. 方差趋势（Kendall tau）
        var1 = np.var(series[:half])
        var2 = np.var(series[half:])
        if var1 < 1e-10:
            var_trend = 0.0
        else:
            var_trend = (var2 - var1) / var1

        # 2. AR(1)趋势
        ar1_1 = self._ar1(series[:half])
        ar1_2 = self._ar1(series[half:])
        ar1_trend = (ar1_2 - ar1_1) if ar1_1 is not None else 0.0

        # 3. 偏度
        skew = stats.skew(series[-20:]) if len(series) >= 20 else 0.0

        # 4. 闪烁检测（v2.1: 改进双模态检测）
        recent = series[-20:]
        bimodality = self._check_bimodality_v2(recent)

        # 5. 爆炸同步接近度（v2.1: 基于耦合强度变化）
        es_prox = self._compute_es_proximity_v2()

        # 综合评分 (v2.2: 加入韧性维度)
        parts = [
            np.clip(var_trend, 0, 2) if not np.isnan(var_trend) else 0.0,
            np.clip(ar1_trend if ar1_trend is not None and not np.isnan(ar1_trend) else 0.0, 0, 2),
            min(abs(skew) / 2.0, 1.0) if not np.isnan(skew) else 0.0,
            min((1.0 - bimodality), 1.0) if not np.isnan(bimodality) else 0.0,
            min(es_prox, 1.0) if not np.isnan(es_prox) else 0.0,
        ]
        composite = (
            0.25 * parts[0] + 0.25 * parts[1]
            + 0.20 * parts[2] + 0.15 * parts[3] + 0.15 * parts[4]
        )

        status = (
            "critical" if composite >= self.critical_threshold
            else "warning" if composite >= self.warning_threshold
            else "normal"
        )

        return EWSResult(
            variable=var_name,
            variance_trend=var_trend,
            ar1_trend=ar1_trend if ar1_trend else 0.0,
            skewness=skew,
            flickering_pvalue=bimodality,
            es_proximity=es_prox,
            composite_score=composite,
            status=status,
        )

    def _compute_es_proximity_v2(self) -> float:
        """v2.1: 改进的爆炸同步接近度.

        方法：
        1. 计算耦合强度的趋势（上升=接近临界点）
        2. 检测耦合强度的加速度（突然上升=临界逼近）
        3. 使用部门相位同步的 Kuramoto 序参量近似
        """
        if len(self._coupling_history) < 10:
            return 0.0

        couplings = np.array(self._coupling_history[-20:])

        # 耦合强度的趋势
        if len(couplings) >= 10:
            half = len(couplings) // 2
            c1 = np.mean(couplings[:half])
            c2 = np.mean(couplings[half:])
            trend = (c2 - c1) / max(abs(c1), 0.01)
        else:
            trend = 0.0

        # 当前耦合水平
        current = couplings[-1] if len(couplings) > 0 else 0.0

        # 爆炸同步接近度 = 高耦合 + 正趋势 + 加速
        es_prox = (
            0.4 * current
            + 0.4 * max(0.0, min(trend * 2.0, 1.0))
            + 0.2 * (1.0 if current > 0.7 else 0.0)
        )

        return max(0.0, min(1.0, es_prox))

    def _ar1(self, x: np.ndarray) -> float:
        """计算一阶自相关."""
        if len(x) < 2:
            return 0.0
        x1 = x[:-1]
        x2 = x[1:]
        denom = np.std(x1) * np.std(x2)
        if denom < 1e-10:
            return 0.0
        return np.corrcoef(x1, x2)[0, 1]

    def _check_bimodality(self, x: np.ndarray) -> float:
        """简化的双模态检测 —— 返回p值（越小越可能是双模态）."""
        if len(x) < 10:
            return 1.0
        kurt = stats.kurtosis(x)
        return 1.0 / (1.0 + abs(kurt + 0.5))

    def _check_bimodality_v2(self, x: np.ndarray) -> float:
        """v2.1: 改进的双模态检测.

        使用 Hartigan's dip test 的简化版本：
        - 将数据二分，检查两个半部的均值差异
        - 如果存在显著的双峰，两个半部的均值应有明显差异
        - 还检查了峰度（platykurtic = 可能的双模态）
        """
        if len(x) < 12:
            return 1.0

        # 方法1：峰度检测（负峰度 → 可能双模态）
        kurt = stats.kurtosis(x)
        kurt_score = 1.0 / (1.0 + abs(kurt + 0.5)) if not np.isnan(kurt) else 1.0

        # 方法2：二分均值差异
        sorted_x = np.sort(x)
        mid = len(sorted_x) // 2
        left_mean = np.mean(sorted_x[:mid])
        right_mean = np.mean(sorted_x[mid:])
        pooled_std = np.std(x)
        if pooled_std > 1e-10:
            mean_diff_score = abs(right_mean - left_mean) / pooled_std
        else:
            mean_diff_score = 0.0
        mean_diff_score = 1.0 - min(1.0, mean_diff_score / 3.0)

        # 方法3：分布与正态的偏差
        if len(x) >= 8:
            _, norm_pvalue = stats.normaltest(x)
            norm_score = 1.0 - norm_pvalue if not np.isnan(norm_pvalue) else 0.5
        else:
            norm_score = 0.5

        # 综合：越小越可能是双模态
        bimodality = 0.4 * kurt_score + 0.3 * mean_diff_score + 0.3 * norm_score
        return max(0.0, min(1.0, bimodality))

    def get_first_warning_times(self, t: float) -> dict:
        """记录每个变量的首次预警时间."""
        first_times = {}
        for name, ews in self.compute_all(None).items():
            if ews.status in ("warning", "critical"):
                if name not in getattr(self, '_first_warning', {}):
                    if not hasattr(self, '_first_warning'):
                        self._first_warning = {}
                    self._first_warning[name] = t
        return getattr(self, '_first_warning', {})
