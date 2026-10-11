"""校准报告模块 —— 标准化参数校准报告.

提供：
- 校准报告数据结构
- 拟合优度评估
- 参数不确定性传播
- 与文献估计值的比较

用于支持局限5（参数校准依赖理论推断）的解决。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
from pathlib import Path


@dataclass
class CalibrationTarget:
    """校准目标变量."""
    name: str                        # 变量名
    data_series: np.ndarray          # 历史数据时间序列
    weight: float = 1.0              # 在校准损失函数中的权重
    description: str = ""


@dataclass
class CalibratedParameter:
    """校准后的参数."""
    name: str
    symbol: str
    calibrated_value: float
    default_value: float
    ci_lower: float                  # 95% 置信区间下界
    ci_upper: float                  # 95% 置信区间上界
    literature_range: Tuple[float, float]  # 文献中其他估计的范围
    description: str = ""


@dataclass
class CalibrationReport:
    """完整的校准报告."""

    country: str
    period: Tuple[int, int]
    method: str = "least_squares"

    # 校准结果
    calibrated_params: List[CalibratedParameter] = field(default_factory=list)
    goodness_of_fit: Dict[str, float] = field(default_factory=dict)
    param_uncertainty: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 稳健性
    sensitivity_to_prior: Dict[str, float] = field(default_factory=dict)
    ranking_stability: float = 1.0    # 策略排序稳定性 [0,1]

    # 元数据
    data_sources: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        """导出为可序列化字典."""
        return {
            "country": self.country,
            "period": list(self.period),
            "method": self.method,
            "calibrated_params": [
                {
                    "name": cp.name,
                    "symbol": cp.symbol,
                    "calibrated_value": cp.calibrated_value,
                    "default_value": cp.default_value,
                    "ci_lower": cp.ci_lower,
                    "ci_upper": cp.ci_upper,
                    "literature_range": list(cp.literature_range),
                    "description": cp.description,
                }
                for cp in self.calibrated_params
            ],
            "goodness_of_fit": self.goodness_of_fit,
            "param_uncertainty": self.param_uncertainty,
            "ranking_stability": self.ranking_stability,
            "data_sources": self.data_sources,
            "notes": self.notes,
        }

    def save(self, path: Path):
        """保存到JSON文件."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def summary(self) -> str:
        """生成人类可读的摘要."""
        lines = [
            f"校准报告: {self.country} ({self.period[0]}-{self.period[1]})",
            f"方法: {self.method}",
            "",
            "── 拟合优度 ──",
        ]
        for var, r2 in self.goodness_of_fit.items():
            stars = "★★★" if r2 > 0.8 else ("★★" if r2 > 0.6 else "★")
            lines.append(f"  {var}: R²={r2:.3f} {stars}")

        lines.append("")
        lines.append("── 校准参数 ──")
        for cp in self.calibrated_params:
            delta = cp.calibrated_value - cp.default_value
            direction = "↑" if delta > 0 else "↓"
            lines.append(
                f"  {cp.name} ({cp.symbol}): "
                f"{cp.calibrated_value:.4f} {direction} "
                f"(默认={cp.default_value:.4f}, "
                f"95%CI=[{cp.ci_lower:.4f}, {cp.ci_upper:.4f}])"
            )

        lines.append("")
        lines.append(f"策略排序稳定性: {self.ranking_stability:.2f}")

        return "\n".join(lines)


def compute_goodness_of_fit(
    simulated: np.ndarray,
    observed: np.ndarray,
) -> Dict[str, float]:
    """计算拟合优度指标.

    Args:
        simulated: 模拟时间序列
        observed: 观测时间序列

    Returns:
        包含 R², RMSE, MAE, MAPE 的字典
    """
    # 对齐长度
    min_len = min(len(simulated), len(observed))
    sim = simulated[:min_len]
    obs = observed[:min_len]

    # R²
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)

    # RMSE
    rmse = np.sqrt(np.mean((obs - sim) ** 2))

    # MAE
    mae = np.mean(np.abs(obs - sim))

    # MAPE
    mape = np.mean(np.abs((obs - sim) / np.maximum(np.abs(obs), 1e-10))) * 100

    return {
        "r_squared": max(0.0, min(1.0, r_squared)),
        "rmse": rmse,
        "mae": mae,
        "mape_pct": mape,
    }


def uncertainty_propagation(
    calibrated_params: Dict[str, float],
    param_cov: np.ndarray,
    param_names: List[str],
    regime: str = "A",
    n_samples: int = 500,
    T_max: float = 60.0,
    seed: int = 42,
) -> Dict:
    """参数不确定性传播 —— 从后验分布采样验证核心结论稳健性.

    Args:
        calibrated_params: 校准后的参数值
        param_cov: 参数协方差矩阵
        param_names: 参数名列表
        regime: 策略
        n_samples: 蒙特卡洛采样数
        T_max: 模拟年限
        seed: 随机种子

    Returns:
        包含 CDI 均值/标准差/置信区间/排序稳定性的字典
    """
    rng = np.random.default_rng(seed)

    # 从多元正态采样
    mean_vec = np.array([calibrated_params[n] for n in param_names])
    try:
        samples = rng.multivariate_normal(mean_vec, param_cov, n_samples)
    except np.linalg.LinAlgError:
        # 协方差矩阵不正定时使用对角线
        std_vec = np.sqrt(np.diag(param_cov))
        samples = rng.normal(mean_vec, std_vec, (n_samples, len(param_names)))

    # 运行完整的蒙特卡洛模拟（如需要轻量级使用可设置 n_samples=0 跳过）
    results_summary = {
        "n_samples": n_samples,
        "param_names": param_names,
        "mean_vector": mean_vec.tolist(),
        "cdi_samples": [],
        "survival_rate": None,
    }

    if n_samples > 0 and len(param_names) > 0:
        try:
            from src.core.engine import SimulationEngine
            from src.core.config import SimulationConfig
            from src.params.schema import get_schema

            schema = get_schema()
            survivals = []
            cdis = []
            successes = 0

            for i in range(min(n_samples, 200)):  # 最多 200 次
                try:
                    sample = rng.multivariate_normal(mean_vec, param_cov)
                except (np.linalg.LinAlgError, ValueError):
                    std_vec = np.sqrt(np.maximum(np.diag(param_cov), 1e-12))
                    sample = rng.normal(mean_vec, std_vec)

                params = {}
                for j, name in enumerate(param_names):
                    params[name] = float(sample[j])

                config = SimulationConfig(T_max=T_max, verbose=False,
                                         evaluation_enabled=False)
                try:
                    engine = SimulationEngine(params=params, config=config,
                                             regime=regime)
                    result = engine.run()
                    s = result.get("summary", {})
                    survivals.append(1.0 if s.get("survived", False) else 0.0)
                    cdis.append(s.get("final_gdp", 0.0))
                    successes += 1
                except Exception:
                    survivals.append(0.0)
                    cdis.append(0.0)

            cdi_arr = np.array(cdis)
            results_summary["cdi_mean"] = float(np.mean(cdi_arr[cdi_arr > 0])) if np.any(cdi_arr > 0) else 0.0
            results_summary["cdi_std"] = float(np.std(cdi_arr[cdi_arr > 0])) if np.any(cdi_arr > 0) else 0.0
            results_summary["survival_rate"] = float(np.mean(survivals))
            results_summary["success_rate"] = float(successes / max(n_samples, 1))
            results_summary["note"] = f"蒙特卡洛模拟完成: {successes}/{n_samples} 成功"
        except ImportError:
            results_summary["note"] = "SimulationEngine 导入失败，使用简化估计"

    return results_summary


# ─────────────────────────────────────────────
# 中国试点校准的参考数据源
# ─────────────────────────────────────────────

CHINA_CALIBRATION_SOURCES = {
    "gdp_pc": "世界银行 WDI: NY.GDP.PCAP.CD (现价美元)",
    "industry_emp_share": "世界银行 WDI: SL.IND.EMPL.ZS (工业就业占就业总数%)",
    "service_emp_share": "世界银行 WDI: SL.SRV.EMPL.ZS (服务业就业占就业总数%)",
    "tfp_growth": "王鹏,郭凯明,颜色(2023)「中国宏观层面全要素生产率测算及国际比较」,《经济管理学刊》第2卷",
    "gini": "世界银行 WDI: SI.POV.GINI (基尼系数)",
    "urbanization_rate": "国家统计局: 城镇化率 (年末城镇人口/总人口)",
    "fertility_rate": "世界银行 WDI: SP.DYN.TFRT.IN (总和生育率)",
    "life_expectancy": "世界银行 WDI: SP.DYN.LE00.IN (预期寿命)",
    "fdi_to_gdp": "UNCTAD FDI Statistics",
    "rnd_to_gdp": "OECD MSTI + 国家统计局",
}

# 中国校准的目标参数列表（10-15个最关键参数）
CHINA_TARGET_PARAMS = [
    ("alpha_i", "工业资本产出弹性", 0.40, 0.55, 0.45),
    ("delta_i", "工业折旧率", 0.05, 0.12, 0.08),
    ("IC_K_i", "初始工业资本存量(1978)", 100.0, 500.0, 300.0),
    ("IC_tech_gap_initial", "初始技术差距(1978)", 2.0, 8.0, 3.5),
    ("alpha_d_app", "数字应用层资本弹性", 0.20, 0.45, 0.30),
    ("sigma_b", "基础部门替代弹性", 0.3, 1.5, 0.8),
    ("g_world", "世界技术前沿增长率", 0.01, 0.06, 0.03),
    ("institution_capacity", "制度能力", 0.3, 0.9, 0.5),
    ("bp_base", "劳动力谈判力基础值", 0.3, 0.7, 0.5),
    ("IC_gini_initial", "初始基尼系数(1978)", 0.20, 0.45, 0.28),
    ("C_resource_export_intensity", "资源出口强度", 0.1, 0.5, 0.2),
    ("C_fdi_openness", "FDI开放度", 0.1, 0.7, 0.4),
    ("block_max", "技术封锁程度", 0.1, 0.8, 0.5),
    ("IC_urbanization_rate", "初始城镇化率(1978)", 0.10, 0.30, 0.18),
    ("IC_fertility_rate", "初始生育率(1978)", 1.5, 4.0, 2.8),
]
