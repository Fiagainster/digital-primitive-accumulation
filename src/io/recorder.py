"""状态记录器 —— 预分配NumPy数组，高效记录全变量时间序列."""

import numpy as np
import pandas as pd
from ..core.state import SystemState, ALL_SECTORS


class TimeSeriesRecorder:
    """全变量时间序列记录器."""

    def __init__(self, max_steps: int = 2000):
        self.max_steps = max_steps
        self._idx = 0
        self._arrays: dict = {}
        self._init_arrays()

    def _init_arrays(self):
        """预分配所有记录数组 (v3.5: 六部门动态生成)."""
        n = self.max_steps
        # 部门变量 — 从 ALL_SECTORS 动态生成
        sector_vars = ["K", "H", "L", "DL", "Y", "W", "r", "I", "A", "DD"]
        keys = ["t"]
        for var in sector_vars:
            for s in ALL_SECTORS:
                keys.append(f"{var}_{s}")
        # 宏观
        keys += [
            "gdp", "total_surplus", "accumulation_fund",
            "tech_gap", "competitiveness",
            "external_dependency",
            "welfare_index", "gini", "unemployment_rate",
            "fiscal_debt_ratio",
            # 数据资产
            "data_producer", "data_consumer", "data_public",
            # 依附
            "dep_infra", "dep_data", "dep_standards", "dep_composite",
            # v3.5: 新增依附维度
            "dep_chip", "dep_capital", "dep_talent", "dep_ecosystem",
            # 代价
            "cost_unemployment", "cost_skill_depreciation",
            "cost_sectoral_disparity", "cost_concentration",
            "cost_irreversibility", "cost_ratio_ind_dig",
            # 家庭群组
            "hh_digital_welfare", "hh_industrial_welfare",
            "hh_base_welfare", "hh_service_welfare",
            "hh_unemployed_welfare",
            "hh_digital_cost", "hh_industrial_cost",
            "hh_base_cost", "hh_service_cost",
            "hh_unemployed_cost",
            # 断裂
            "active_fractures_count", "is_terminal",
            # EWS
            "ews_K_i_composite", "ews_tech_gap_composite",
            "ews_unemployment_composite",
            "ews_K_i_resilience", "ews_tech_gap_resilience",
            # SFC
            "transaction_residual",
            # 去工业化
            "deind_quality",
            # 政治经济学
            "political_support", "political_protest_risk",
            "political_policy_constraint", "political_reform_momentum",
            # 五维评价
            "eval_economy", "eval_livelihood", "eval_resilience",
            "eval_structure", "eval_sovereignty", "eval_cdi",
            # v3.2
            "sfc_residual_ratio",
            "skill_erosion_industrial", "skill_erosion_base",
            # v3.3 + v3.5
            "informal_service_L", "informal_service_W",
            "credit_rationing_ratio", "real_exchange_rate",
            "effective_block_max", "effective_fdi_availability",
            "carbon_emissions", "cumulative_carbon",
            "green_transition_pressure",
            "employment_quality_index",
            "tech_gap_infra", "tech_gap_app",
            "asset_price_digital",
            # v3.6: 平台垄断
            "digital_hhi", "platform_monopoly_rent", "data_network_moat",
            # v3.6: 人口动态
            "total_population", "working_age_pop", "dependency_ratio",
            "fertility_rate", "life_expectancy", "urbanization_rate",
            # v3.6: 空间维度
            "coastal_population_share", "spatial_gini",
            "agglomeration_bonus", "congestion_penalty",
            # v3.6: 政策执行
            "policy_execution_discount",
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

        for key in ALL_SECTORS:
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
        a["dep_chip"][i] = state.dependency.chip_dependency
        a["dep_capital"][i] = state.dependency.capital_dependency
        a["dep_talent"][i] = state.dependency.talent_dependency
        a["dep_ecosystem"][i] = state.dependency.ecosystem_dependency
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

        # v3.5: 新指标
        a["employment_quality_index"][i] = getattr(state, "employment_quality_index", 1.0)
        a["tech_gap_infra"][i] = getattr(state, "tech_gap_infra", state.tech_gap)
        a["tech_gap_app"][i] = getattr(state, "tech_gap_app", state.tech_gap)

        # v3.6: 平台垄断
        a["digital_hhi"][i] = getattr(state, "digital_hhi", 0.0)
        a["platform_monopoly_rent"][i] = getattr(state, "platform_monopoly_rent", 0.0)
        a["data_network_moat"][i] = getattr(state, "data_network_moat", 0.0)

        # v3.6: 人口动态
        a["total_population"][i] = getattr(state, "total_population", 0.0)
        a["working_age_pop"][i] = getattr(state, "working_age_pop", 0.0)
        a["dependency_ratio"][i] = getattr(state, "dependency_ratio", 0.5)
        a["fertility_rate"][i] = getattr(state, "fertility_rate", 2.5)
        a["life_expectancy"][i] = getattr(state, "life_expectancy", 65.0)
        a["urbanization_rate"][i] = getattr(state, "urbanization_rate", 0.35)

        # v3.6: 空间维度
        a["coastal_population_share"][i] = getattr(state, "coastal_population_share", 0.45)
        a["spatial_gini"][i] = getattr(state, "spatial_gini", 0.0)
        a["agglomeration_bonus"][i] = getattr(state, "agglomeration_bonus", 0.0)
        a["congestion_penalty"][i] = getattr(state, "congestion_penalty", 0.0)

        # v3.6: 政策执行
        a["policy_execution_discount"][i] = getattr(state, "policy_execution_discount", 1.0)

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
