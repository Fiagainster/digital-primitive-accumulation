"""综合发展指数 (Composite Development Index, CDI).

v3.0 的核心聚合工具。
支持可配置的权重，四种预设方案，以及时间维度的评价。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .five_dimensions import DimensionScores
from .red_lines import RedLineResult


# ─────────────────────────────────────────────
# 预设权重方案
# ─────────────────────────────────────────────
WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "default": {
        # 民生偏高（v3.0默认）：论文的差异化定位
        "economy": 0.175,
        "livelihood": 0.30,
        "resilience": 0.175,
        "structure": 0.175,
        "sovereignty": 0.175,
    },
    "growth_first": {
        # 增长优先
        "economy": 0.35,
        "livelihood": 0.20,
        "resilience": 0.15,
        "structure": 0.15,
        "sovereignty": 0.15,
    },
    "livelihood_first": {
        # 民生优先
        "economy": 0.10,
        "livelihood": 0.45,
        "resilience": 0.15,
        "structure": 0.15,
        "sovereignty": 0.15,
    },
    "security_first": {
        # 安全优先
        "economy": 0.10,
        "livelihood": 0.20,
        "resilience": 0.25,
        "structure": 0.15,
        "sovereignty": 0.30,
    },
    "balanced": {
        # 均衡发展
        "economy": 0.20,
        "livelihood": 0.20,
        "resilience": 0.20,
        "structure": 0.20,
        "sovereignty": 0.20,
    },
}


@dataclass
class CDIResult:
    """CDI计算结果."""

    # 各维度得分
    economy: float = 0.0
    livelihood: float = 0.0
    resilience: float = 0.0
    structure: float = 0.0
    sovereignty: float = 0.0

    # CDI（默认权重）
    cdi: float = 0.0

    # 各种权重下的CDI
    cdi_variants: Dict[str, float] = field(default_factory=dict)

    # 人道主义红线
    red_line: Optional[RedLineResult] = None

    # 时间维度
    recent_trend: float = 0.0       # CDI最近的变化趋势
    volatility_penalty: float = 0.0  # 波动惩罚

    def is_acceptable(self) -> bool:
        """该路径是否可接受（通过了人道主义红线）."""
        if self.red_line is None:
            return True
        return self.red_line.is_acceptable

    def as_dict(self) -> dict:
        return {
            "economy": self.economy,
            "livelihood": self.livelihood,
            "resilience": self.resilience,
            "structure": self.structure,
            "sovereignty": self.sovereignty,
            "cdi": self.cdi,
            "cdi_variants": self.cdi_variants,
            "red_line_level": self.red_line.level if self.red_line else "PASS",
            "red_line_failures": self.red_line.failures if self.red_line else [],
            "red_line_warnings": self.red_line.warnings if self.red_line else [],
            "is_acceptable": self.is_acceptable(),
            "recent_trend": self.recent_trend,
            "volatility_penalty": self.volatility_penalty,
        }

    def format_report_line(self) -> str:
        """格式化为一行报告."""
        rl = "✓" if self.is_acceptable() else "✗"
        return (f"CDI={self.cdi:.3f} | E={self.economy:.2f} "
                f"L={self.livelihood:.2f} R={self.resilience:.2f} "
                f"S={self.structure:.2f} V={self.sovereignty:.2f} "
                f"[{rl}]")


class CDICalculator:
    """综合发展指数计算器.

    支持：
    - 五种预设权重方案
    - 自定义权重
    - 时间维度修正（趋势 + 波动惩罚）
    - 人道主义红线集成
    """

    def __init__(self, weights: Dict[str, float] = None,
                 preset: str = "default", discount_rate: float = 0.02):
        """
        Args:
            weights: 自定义权重字典（覆盖preset）
            preset: 预设方案名称
            discount_rate: v3.2: 时序贴现率δ（年化），默认0.02/年
        """
        if weights is not None:
            self.weights = self._validate_weights(weights)
        else:
            self.weights = WEIGHT_PRESETS.get(preset, WEIGHT_PRESETS["default"]).copy()

        self.preset_name = preset
        self.discount_rate = discount_rate  # v3.2: 时序贴现率

    def compute(self, scores: DimensionScores,
                red_line: RedLineResult = None,
                score_history: List[DimensionScores] = None) -> CDIResult:
        """计算CDI.

        Args:
            scores: 当前五维得分
            red_line: 人道主义红线检测结果
            score_history: 历史得分列表（用于计算趋势和波动）

        Returns:
            CDIResult: 完整的CDI计算结果
        """
        # 基本CDI（默认权重）
        cdi = (
            self.weights["economy"] * scores.economy
            + self.weights["livelihood"] * scores.livelihood
            + self.weights["resilience"] * scores.resilience
            + self.weights["structure"] * scores.structure
            + self.weights["sovereignty"] * scores.sovereignty
        )
        cdi = float(np.clip(cdi, 0.0, 1.0))

        # 各种预设权重下的CDI
        variants = {}
        for preset_name, w in WEIGHT_PRESETS.items():
            if preset_name == self.preset_name:
                continue
            v = (
                w["economy"] * scores.economy
                + w["livelihood"] * scores.livelihood
                + w["resilience"] * scores.resilience
                + w["structure"] * scores.structure
                + w["sovereignty"] * scores.sovereignty
            )
            variants[preset_name] = float(np.clip(v, 0.0, 1.0))
        # 也加上当前权重方案的CDI
        variants[self.preset_name] = cdi

        # 时间维度修正
        trend = 0.0
        volatility = 0.0
        temporal_weight = 1.0  # v3.2: 时序加权因子
        if score_history and len(score_history) >= 4:
            recent_cdis = []
            for hist_scores in score_history[-8:]:  # 最近8个快照
                h_cdi = (
                    self.weights["economy"] * hist_scores.economy
                    + self.weights["livelihood"] * hist_scores.livelihood
                    + self.weights["resilience"] * hist_scores.resilience
                    + self.weights["structure"] * hist_scores.structure
                    + self.weights["sovereignty"] * hist_scores.sovereignty
                )
                recent_cdis.append(h_cdi)

            if len(recent_cdis) >= 2:
                # 趋势：线性回归斜率
                x = np.arange(len(recent_cdis))
                trend = np.polyfit(x, recent_cdis, 1)[0]
                trend = float(np.clip(trend, -0.5, 0.5))

                # 波动惩罚
                if len(recent_cdis) >= 4:
                    sigma = np.std(recent_cdis)
                    # 波动超过0.1时开始惩罚
                    volatility = max(0.0, (sigma - 0.05) * 0.5)
                    volatility = min(0.1, volatility)

            # v3.2: 时序加权 — 指数贴现，近期权重大于远期
            # δ=0.02/年，评估间隔约1年，8个快照≈8年
            if len(recent_cdis) >= 2:
                n = len(recent_cdis)
                # 计算指数衰减权重：weight[i] ∝ exp(-δ * (n-1-i))
                # 最近的数据点权重最大
                ages = np.arange(n)[::-1]  # [n-1, n-2, ..., 0] — 最近在前
                raw_weights = np.exp(-self.discount_rate * ages)
                disc_weights = raw_weights / raw_weights.sum()
                # 加权CDI vs 简单平均CDI
                weighted_cdi = np.dot(disc_weights, np.array(recent_cdis))
                simple_mean = np.mean(recent_cdis)
                # 时序加权因子：如果近期CDI高于远期，加权后更高
                temporal_weight = weighted_cdi / max(simple_mean, 0.01)
                temporal_weight = float(np.clip(temporal_weight, 0.85, 1.15))

        # 最终CDI（含波动惩罚 + v3.2时序加权）
        final_cdi = max(0.0, cdi * temporal_weight - volatility)

        return CDIResult(
            economy=scores.economy,
            livelihood=scores.livelihood,
            resilience=scores.resilience,
            structure=scores.structure,
            sovereignty=scores.sovereignty,
            cdi=final_cdi,
            cdi_variants=variants,
            red_line=red_line,
            recent_trend=trend,
            volatility_penalty=volatility,
        )

    @staticmethod
    def _validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """验证并归一化权重."""
        required = ["economy", "livelihood", "resilience", "structure", "sovereignty"]
        for key in required:
            if key not in weights:
                raise ValueError(f"Missing weight for: {key}")

        total = sum(weights[k] for k in required)
        if abs(total - 1.0) > 0.01:
            # 自动归一化
            return {k: weights[k] / total for k in required}

        return {k: weights[k] for k in required}

    @staticmethod
    def list_presets() -> Dict[str, Dict[str, float]]:
        """列出所有预设权重方案."""
        return WEIGHT_PRESETS.copy()

    @staticmethod
    def compare_all_presets(scores: DimensionScores) -> Dict[str, float]:
        """用所有预设权重计算CDI并返回."""
        result = {}
        for preset_name, w in WEIGHT_PRESETS.items():
            cdi = (
                w["economy"] * scores.economy
                + w["livelihood"] * scores.livelihood
                + w["resilience"] * scores.resilience
                + w["structure"] * scores.structure
                + w["sovereignty"] * scores.sovereignty
            )
            result[preset_name] = float(np.clip(cdi, 0.0, 1.0))
        return result
