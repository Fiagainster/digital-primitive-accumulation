"""状态记录器 —— 预分配NumPy数组，高效记录全变量时间序列."""

import numpy as np
import pandas as pd
from ..core.state import SystemState


class TimeSeriesRecorder:
    """全变量时间序列记录器."""

    def __init__(self, max_steps: int = 2000):
        self.max_steps = max_steps
        self._idx = 0
        self._arrays: dict = {}
        self._init_arrays()

    def _init_arrays(self):
        """预分配所有记录数组."""
        n = self.max_steps
        keys = [
            "t",
            # 部门变量
            "K_b", "K_i", "K_d", "K_s",
            "H_b", "H_i", "H_d", "H_s",
            "L_b", "L_i", "L_d", "L_s",
            "DL_b", "DL_i", "DL_d", "DL_s",
            "Y_b", "Y_i", "Y_d", "Y_s",
            "W_b", "W_i", "W_d", "W_s",
            "r_b", "r_i", "r_d", "r_s",
            "I_b", "I_i", "I_d", "I_s",
            "A_b", "A_i", "A_d", "A_s",
            # v2.2: 数字密度
            "DD_b", "DD_i", "DD_d", "DD_s",
            # 宏观
            "gdp", "total_surplus", "accumulation_fund",
            "tech_gap", "competitiveness",
            "external_dependency",
            "welfare_index", "gini", "unemployment_rate",
            "fiscal_debt_ratio",
            # 数据资产
            "data_producer", "data_consumer", "data_public",
            # 依附
            "dep_infra", "dep_data", "dep_standards", "dep_composite",
            # 代价
            "cost_unemployment", "cost_skill_depreciation",
            "cost_sectoral_disparity", "cost_concentration",
            "cost_irreversibility", "cost_ratio_ind_dig",
            # 家庭群组
            "hh_digital_welfare", "hh_industrial_welfare",
            "hh_base_welfare", "hh_service_welfare",  # v3.3
            "hh_unemployed_welfare",
            "hh_digital_cost", "hh_industrial_cost",
            "hh_base_cost", "hh_service_cost",  # v3.3
            "hh_unemployed_cost",
            # 断裂
            "active_fractures_count", "is_terminal",
            # EWS
            "ews_K_i_composite", "ews_tech_gap_composite",
            "ews_unemployment_composite",
            # v2.2: EWS韧性
            "ews_K_i_resilience", "ews_tech_gap_resilience",
            # SFC
            "transaction_residual",
            # 去工业化
            "deind_quality",
            # v2.2: 政治经济学
            "political_support", "political_protest_risk",
            "political_policy_constraint", "political_reform_momentum",
            # v3.0: 五维评价（按年记录，频率低于主步长）
            "eval_economy", "eval_livelihood", "eval_resilience",
            "eval_structure", "eval_sovereignty", "eval_cdi",
            # v3.2: SFC残差监控 + 技能侵蚀
            "sfc_residual_ratio",
            "skill_erosion_industrial", "skill_erosion_base",
            # v3.3: 低端服务业 + 金融 + 国际 + 气候
            "informal_service_L", "informal_service_W",
            "credit_rationing_ratio", "real_exchange_rate",
            "effective_block_max", "effective_fdi_availability",
            "carbon_emissions", "cumulative_carbon",
            "green_transition_pressure",
        ]
        for k in keys:
            self._arrays[k] = np.zeros(n, dtype=np.float64)

    def record(self, state: SystemState):
        """记录当前状态到下一个时间槽."""
        if self._idx >= self.max_steps:
            return
        i = self._idx
        a = self._arrays

        a["t"][i] = state.t

        for key in ["b", "i", "d", "s"]:  # v3.3
            s = state.sectors[key]
            a[f"K_{key}"][i] = s.K
            a[f"H_{key}"][i] = s.H
            a[f"L_{key}"][i] = s.L
            a[f"DL_{key}"][i] = s.DL
            a[f"Y_{key}"][i] = s.Y
            a[f"W_{key}"][i] = s.W
            a[f"r_{key}"][i] = s.r
            a[f"I_{key}"][i] = s.I
            a[f"A_{key}"][i] = s.A
            a[f"DD_{key}"][i] = s.digital_density  # v2.2

        a["gdp"][i] = state.gdp
        a["total_surplus"][i] = state.total_surplus
        a["accumulation_fund"][i] = state.accumulation_fund
        a["tech_gap"][i] = state.tech_gap
        a["competitiveness"][i] = state.competitiveness
        a["external_dependency"][i] = state.external_dependency
        a["welfare_index"][i] = state.welfare_index
        a["gini"][i] = state.gini
        a["unemployment_rate"][i] = state.unemployment_rate
        a["fiscal_debt_ratio"][i] = state.fiscal_debt_ratio

        a["data_producer"][i] = state.data_assets.producer_data
        a["data_consumer"][i] = state.data_assets.consumer_data
        a["data_public"][i] = state.data_assets.public_data

        a["dep_infra"][i] = state.dependency.infrastructure_dependency
        a["dep_data"][i] = state.dependency.data_dependency
        a["dep_standards"][i] = state.dependency.standards_autonomy
        a["dep_composite"][i] = state.dependency.composite

        cm = state.cost_metrics
        a["cost_unemployment"][i] = cm.unemployment_stock
        a["cost_skill_depreciation"][i] = cm.skill_depreciation
        a["cost_sectoral_disparity"][i] = cm.sectoral_disparity
        a["cost_concentration"][i] = cm.cost_concentration
        a["cost_irreversibility"][i] = cm.irreversibility_index
        a["cost_ratio_ind_dig"][i] = cm.cost_ratio_industrial_digital

        hh = state.households
        a["hh_digital_welfare"][i] = hh.digital_high_skill.welfare_index
        a["hh_industrial_welfare"][i] = hh.industrial_workers.welfare_index
        a["hh_base_welfare"][i] = hh.base_workers.welfare_index
        a["hh_service_welfare"][i] = hh.service_workers.welfare_index  # v3.3
        a["hh_unemployed_welfare"][i] = hh.unemployed.welfare_index
        a["hh_digital_cost"][i] = hh.digital_high_skill.cumulative_cost
        a["hh_industrial_cost"][i] = hh.industrial_workers.cumulative_cost
        a["hh_base_cost"][i] = hh.base_workers.cumulative_cost
        a["hh_service_cost"][i] = hh.service_workers.cumulative_cost  # v3.3
        a["hh_unemployed_cost"][i] = hh.unemployed.cumulative_cost

        a["active_fractures_count"][i] = len(state.fracture_status.active_fractures)
        a["is_terminal"][i] = 1.0 if state.fracture_status.is_terminal else 0.0

        for name, ews in state.ews_signals.items():
            if name == "K_i_ratio":
                a["ews_K_i_composite"][i] = ews.composite_score
                a["ews_K_i_resilience"][i] = ews.resilience_index  # v2.2
            elif name == "tech_gap":
                a["ews_tech_gap_composite"][i] = ews.composite_score
                a["ews_tech_gap_resilience"][i] = ews.resilience_index  # v2.2
            elif name == "unemployment":
                a["ews_unemployment_composite"][i] = ews.composite_score

        a["transaction_residual"][i] = state.transaction_residual
        # v3.2: SFC残差比（用GDP作为交易规模代理）
        a["sfc_residual_ratio"][i] = (
            abs(state.transaction_residual) / max(state.gdp, 0.01)
        )
        a["deind_quality"][i] = cm.deind_quality

        # v2.2: 政治经济学变量
        ps = state.political_support
        a["political_support"][i] = ps.support_index
        a["political_protest_risk"][i] = ps.protest_risk
        a["political_policy_constraint"][i] = ps.policy_constraint
        a["political_reform_momentum"][i] = ps.reform_momentum

        # v3.2: 劳动力技能侵蚀
        a["skill_erosion_industrial"][i] = getattr(state, "skill_erosion_i", 1.0)
        a["skill_erosion_base"][i] = getattr(state, "skill_erosion_b", 1.0)

        # v3.3: 低端服务业
        a["informal_service_L"][i] = getattr(state, "informal_service_L", 0.0)
        a["informal_service_W"][i] = getattr(state, "informal_service_W", 0.0)
        # v3.3: 金融
        a["credit_rationing_ratio"][i] = getattr(state, "credit_rationing_ratio", 1.0)
        a["real_exchange_rate"][i] = getattr(state, "real_exchange_rate", 1.0)
        # v3.3: 内生国际
        a["effective_block_max"][i] = getattr(state, "effective_block_max", 0.5)
        a["effective_fdi_availability"][i] = getattr(state, "effective_fdi_availability", 1.0)
        # v3.3: 气候/环境
        a["carbon_emissions"][i] = getattr(state, "carbon_emissions", 0.0)
        a["cumulative_carbon"][i] = getattr(state, "cumulative_carbon", 0.0)
        a["green_transition_pressure"][i] = getattr(state, "green_transition_pressure", 0.0)

        self._idx += 1

    def to_dataframe(self) -> pd.DataFrame:
        """转换为Pandas DataFrame."""
        data = {}
        for k, v in self._arrays.items():
            data[k] = v[:self._idx]
        return pd.DataFrame(data)

    def get_array(self, key: str) -> np.ndarray:
        """获取单个变量的数组."""
        return self._arrays[key][:self._idx]

    def record_evaluation(self, scores, cdi_result) -> None:
        """v3.0: 记录五维评价结果（追加到当前时间槽之前的位置）.

        评价更新频率低于主步长，所以在最新记录的槽位写入评价数据。
        """
        if self._idx == 0:
            return
        i = self._idx - 1  # 写入最近一次record的槽位
        a = self._arrays

        a["eval_economy"][i] = scores.economy
        a["eval_livelihood"][i] = scores.livelihood
        a["eval_resilience"][i] = scores.resilience
        a["eval_structure"][i] = scores.structure
        a["eval_sovereignty"][i] = scores.sovereignty
        a["eval_cdi"][i] = cdi_result.cdi

    def __len__(self) -> int:
        return self._idx
