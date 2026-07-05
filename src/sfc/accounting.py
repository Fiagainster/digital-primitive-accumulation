"""SFC核算 —— 交易流矩阵、资产负债表、闭合检查 (v3.2 增强)."""

import numpy as np
import warnings
from ..core.state import SystemState, BalanceSheet, ALL_SECTORS


class SFCClosure:
    """存量-流量一致核算框架 (v3.2: 增加残差异常膨胀检测)."""

    def __init__(self, params: dict):
        self.tolerance = params.get("sfc_tolerance", 1e-4)
        # v3.2: 残差历史追踪用于异常检测
        self._residual_history: list = []
        self._anomaly_warned: bool = False

    def check_and_close(self, state: SystemState):
        """每步执行SFC闭合检查 (v3.2: 增加残差异常膨胀检测)."""
        # 确保所有部门有资产负债表
        for key in ALL_SECTORS:
            if key not in state.balance_sheets:
                state.balance_sheets[key] = BalanceSheet()
            bs = state.balance_sheets[key]
            s = state.sectors[key]
            # 更新实物资本
            bs.physical_capital = s.K
            # 数据资产
            if key == "d":
                bs.data_assets_value = state.data_assets.effective_stock * 0.1

        # 计算总储蓄和总投资
        total_investment = sum(s.I for s in state.sectors.values())
        total_saving = state.total_surplus + state.accumulation_fund * 0.2

        # 残差
        residual = abs(total_investment - total_saving) / max(state.gdp, 0.01)
        state.transaction_residual = residual

        # v3.2: 残差异常膨胀检测
        self._residual_history.append(residual)
        if len(self._residual_history) > 200:
            self._residual_history = self._residual_history[-100:]

        # 检查残差是否在系统压力下异常膨胀
        # 条件：最近的残差均值超过容忍度的10倍，且趋势在上升
        if len(self._residual_history) >= 20 and not self._anomaly_warned:
            recent = self._residual_history[-20:]
            early = self._residual_history[:min(20, len(self._residual_history) // 2)]
            recent_mean = np.mean(recent)
            early_mean = np.mean(early) if early else 0.0
            # 残差均值超过容忍度100倍或近期残差是早期的3倍以上
            anomaly_threshold = self.tolerance * 100
            if recent_mean > anomaly_threshold:
                warnings.warn(
                    f"SFC残差异常膨胀：近期均值={recent_mean:.2e}（容忍度={self.tolerance:.1e}）。"
                    f"这通常表示系统在压力下核算框架充当了'减震器'，"
                    f"会计可信度下降。",
                    RuntimeWarning, stacklevel=2
                )
                self._anomaly_warned = True
            elif early_mean > 0 and recent_mean / max(early_mean, 1e-10) > 3.0:
                warnings.warn(
                    f"SFC残差趋势上升：近期/早期={recent_mean:.2e}/{early_mean:.2e}。"
                    f"残差持续扩大表明核算闭合在退化。",
                    RuntimeWarning, stacklevel=2
                )
                self._anomaly_warned = True

        # 如不闭合，通过未计划库存调整
        if residual > self.tolerance:
            # 吸收到"未计划库存变化"
            state.accumulation_fund = max(0, state.accumulation_fund - residual * state.gdp * 0.1)

    def get_residual_stats(self) -> dict:
        """v3.2: 返回残差统计摘要."""
        if not self._residual_history:
            return {"mean": 0.0, "max": 0.0, "anomaly_triggered": False}
        return {
            "mean": float(np.mean(self._residual_history)),
            "max": float(np.max(self._residual_history)),
            "recent_mean": float(np.mean(self._residual_history[-20:])) if len(self._residual_history) >= 20 else float(np.mean(self._residual_history)),
            "anomaly_triggered": self._anomaly_warned,
        }

    def build_transaction_matrix(self, state: SystemState) -> np.ndarray:
        """构建简化的交易流矩阵（用于调试和验证）."""
        # 5部门(b, i, d, households, external) × 5交易类型
        mat = np.zeros((5, 5))

        # 行=支出方，列=收入方
        # 简化版：仅为会计闭合验证
        mat[3, 0] = state.sectors["b"].W * state.sectors["b"].L  # 家庭←基础工资
        mat[3, 1] = state.sectors["i"].W * state.sectors["i"].L  # 家庭←工业工资
        mat[3, 2] = state.sectors["d"].W * state.sectors["d"].L  # 家庭←数字工资

        return mat
