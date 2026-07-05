"""五维评价体系 —— 从系统状态计算五个维度的标准化得分.

五个维度：
1. 经济活力 (Economic Vitality)
2. 民生福祉 (Livelihood & Human Welfare) ← 核心维度
3. 发展韧性 (Developmental Resilience)
4. 结构健康 (Structural Health)
5. 主权与自主 (Sovereignty & Autonomy)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..core.state import SystemState, ALL_SECTORS


@dataclass
class DimensionScores:
    """五维评价得分容器."""

    # 原始子指标（用于透明性和调试）
    raw_subs: Dict[str, float] = field(default_factory=dict)

    # 五个维度的标准化得分 [0, 1]
    economy: float = 0.0
    livelihood: float = 0.0
    resilience: float = 0.0
    structure: float = 0.0
    sovereignty: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "economy": self.economy,
            "livelihood": self.livelihood,
            "resilience": self.resilience,
            "structure": self.structure,
            "sovereignty": self.sovereignty,
        }

    def as_vector(self) -> np.ndarray:
        return np.array([self.economy, self.livelihood,
                         self.resilience, self.structure, self.sovereignty])


class FiveDimensionEvaluator:
    """从系统状态计算五维评价得分.

    所有子指标都标准化到 [0, 1] 区间，1=最好，0=最差。
    使用 sigmoid 或线性缩放，而非简单的 min-max，以保留尾部区分度。
    """

    def __init__(self, params: dict = None):
        self.p = params or {}

    def evaluate(self, state: SystemState,
                 history: Optional[List[dict]] = None,
                 t_series: Optional[Dict[str, np.ndarray]] = None) -> DimensionScores:
        """从当前状态和（可选的）时间序列历史计算五维得分.

        Args:
            state: 当前系统状态
            history: 历史状态快照列表（可选，用于计算趋势和波动）
            t_series: 时间序列数据字典（可选，从 recorder 获取）

        Returns:
            DimensionScores: 五个维度的标准化得分
        """
        scores = DimensionScores()

        # === 维度一：经济活力 ===
        scores.economy, eco_subs = self._evaluate_economy(state, t_series)

        # === 维度二：民生福祉 ===
        scores.livelihood, liv_subs = self._evaluate_livelihood(state, t_series)

        # === 维度三：发展韧性 ===
        scores.resilience, res_subs = self._evaluate_resilience(state, t_series)

        # === 维度四：结构健康 ===
        scores.structure, str_subs = self._evaluate_structure(state, t_series)

        # === 维度五：主权与自主 ===
        scores.sovereignty, sov_subs = self._evaluate_sovereignty(state, t_series)

        # 合并所有原始子指标
        scores.raw_subs = {**eco_subs, **liv_subs, **res_subs,
                           **str_subs, **sov_subs}

        return scores

    # ─────────────────────────────────────────────
    # 维度一：经济活力
    # ─────────────────────────────────────────────
    def _evaluate_economy(self, state: SystemState,
                          t_series: Optional[Dict[str, np.ndarray]] = None
                          ) -> Tuple[float, Dict[str, float]]:
        subs = {}

        # 1. 人均GDP年均增长率
        gdp_pc_initial = max(getattr(state, 'initial_gdp_pc', 1.0), 0.01)
        gdp_pc_current = state.gdp / max(state.households.total_population, 0.01)
        T = max(state.t, 1.0)
        g_gdp_pc = np.log(gdp_pc_current / gdp_pc_initial) / T
        # 标准化: 0%以下→0.1, 2%→0.5, 5%→0.8, 10%+→1.0
        subs["gdp_pc_growth"] = self._sigmoid_scale(g_gdp_pc, mid=0.03, steep=30)

        # 2. 投资效率 (ICOR的倒数)
        if t_series is not None and "gdp" in t_series and "K_total" in t_series:
            dY = np.diff(t_series["gdp"][-20:]) if len(t_series["gdp"]) >= 20 else [0]
            dK = np.diff(t_series["K_total"][-20:]) if len(t_series["K_total"]) >= 20 else [1]
            icor = np.mean(dK) / max(np.mean(np.abs(dY)), 1e-8)
            # ICOR < 3 → 高效 ~=1.0, ICOR > 10 → 低效 ~=0.2
            subs["investment_efficiency"] = max(0.0, min(1.0, 1.0 - (icor - 2.0) / 10.0))
        else:
            # 从吸收能力推断
            absorp_ratio = min(1.0, sum(s.absorption_cap for s in state.sectors.values())
                               / max(state.gdp * 0.2, 1.0))
            subs["investment_efficiency"] = absorp_ratio

        # 3. 产出波动性
        if t_series is not None and "gdp" in t_series and len(t_series["gdp"]) >= 20:
            gdp_series = t_series["gdp"][-20:]
            gdp_growth = np.diff(gdp_series) / np.maximum(np.abs(gdp_series[:-1]), 1e-8)
            sigma_Y = np.std(gdp_growth)
            subs["output_stability"] = max(0.0, 1.0 - sigma_Y / 0.15)
        else:
            subs["output_stability"] = 0.7  # 默认中等稳定

        # 4. 数字部门增加值占比（倒U型：太低或太高都不好）
        ds_share = state.sectors["d"].Y / max(state.gdp, 0.01)
        opt_share = 0.35
        subs["digital_share_quality"] = np.exp(-((ds_share - opt_share) ** 2) / 0.08)

        # 5. 出口竞争力
        subs["competitiveness"] = self._sigmoid_scale(
            state.competitiveness, mid=1.0, steep=3.0
        )

        # 加权聚合
        weights = {"gdp_pc_growth": 0.25, "investment_efficiency": 0.15,
                    "output_stability": 0.15, "digital_share_quality": 0.10,
                    "competitiveness": 0.15,
                    # 剩余0.20权重分配给一个综合判断
                    }
        # 实际使用简单平均，因为子指标在构造时已经考虑了重要性
        economy_raw = (
            0.25 * subs["gdp_pc_growth"]
            + 0.15 * subs["investment_efficiency"]
            + 0.15 * subs["output_stability"]
            + 0.10 * subs["digital_share_quality"]
            + 0.15 * subs["competitiveness"]
            + 0.20 * self._evaluate_total_gdp(state, t_series)
        )
        return float(np.clip(economy_raw, 0.0, 1.0)), subs

    def _evaluate_total_gdp(self, state: SystemState,
                            t_series: Optional[Dict[str, np.ndarray]] = None) -> float:
        """综合GDP规模判断."""
        # GDP规模: 按人均GDP粗略评价
        gdp_pc = state.gdp / max(state.households.total_population, 0.01)
        # 基准: 如果GDP_pc在1-10范围（模型单位），线性映射
        return self._sigmoid_scale(gdp_pc, mid=5.0, steep=1.0)

    # ─────────────────────────────────────────────
    # 维度二：民生福祉 ★ 核心维度
    # ─────────────────────────────────────────────
    def _evaluate_livelihood(self, state: SystemState,
                             t_series: Optional[Dict[str, np.ndarray]] = None
                             ) -> Tuple[float, Dict[str, float]]:
        subs = {}

        # 1. 综合福利指数
        subs["welfare_composite"] = state.welfare_index

        # 2. 贫困人口比例（用近似替代：消费低于中位数50%的群组占比）
        groups = state.households.all_groups()
        consumptions = []
        pops = []
        for h in groups:
            if h.population > 0:
                consumptions.append(h.consumption / h.population)
                pops.append(h.population)
        if consumptions and sum(pops) > 0:
            median_cons = np.median(consumptions)
            poverty_line = median_cons * 0.50
            poor_pop = sum(p for c, p in zip(consumptions, pops) if c < poverty_line)
            poverty_rate = poor_pop / sum(pops)
            # 贫困率 0% → 1.0,  20% → 0.3,  40% → 0.0
            subs["poverty_rate"] = max(0.0, 1.0 - poverty_rate / 0.40)
        else:
            subs["poverty_rate"] = 1.0

        # 3. 代价基尼系数（反向）
        cost_gini = state.households.cost_gini
        # 0→1.0, 0.5→0.3, 1.0→0.0
        subs["cost_equity"] = max(0.0, 1.0 - cost_gini)

        # 4. 社会流动性（劳动力从低技能向高技能转移）
        if t_series is not None and "L_d" in t_series and len(t_series["L_d"]) >= 20:
            dd_Ld = np.diff(t_series["L_d"][-20:])
            avg_Ld_growth = np.mean(dd_Ld) / max(np.mean(t_series["L_d"][-20:]), 0.01)
            subs["social_mobility"] = self._sigmoid_scale(avg_Ld_growth, mid=0.01, steep=100)
        else:
            # 从群组技能水平推断
            skill_ratio = (state.households.digital_high_skill.skill_level
                           / max(state.households.base_workers.skill_level, 0.01))
            subs["social_mobility"] = self._sigmoid_scale(skill_ratio - 1.0, mid=0.5, steep=3.0)

        # 5. 失业持续时间
        u_rate = state.unemployment_rate
        # 失业率为0 → 1.0, 5%→0.7, 10%→0.4, 20%→0.1
        subs["employment_health"] = max(0.0, 1.0 - u_rate / 0.25)

        # 6. 健康水平（从福利和基础部门产出推断）
        food_per_cap = state.sectors["b"].Y / max(state.households.total_population, 1.0)
        # 基础值（模型单位）
        subs["health_index"] = self._sigmoid_scale(food_per_cap, mid=0.5, steep=5.0)

        # 7. 教育公平度（群组间技能水平差距的反向）
        skill_levels = [h.skill_level for h in groups if h.population > 0]
        if len(skill_levels) >= 2:
            skill_gini = self._simple_gini(np.array(skill_levels))
            subs["edu_equity"] = max(0.0, 1.0 - skill_gini * 2.0)
        else:
            subs["edu_equity"] = 1.0

        # 8. 代际公平指数
        total_cost = sum(h.cumulative_cost for h in groups)
        total_gdp_cum = state.gdp * state.t  # 累积GDP近似
        if total_gdp_cum > 0:
            intergen_ratio = total_cost / total_gdp_cum
            # 代价/GDP < 10% → 1.0, 50% → 0.3, 100% → 0.0
            subs["intergen_fairness"] = max(0.0, 1.0 - intergen_ratio)
        else:
            subs["intergen_fairness"] = 1.0

        # 9. 心理/社会成本
        u_pop = state.households.unemployed.population
        subs["social_cost"] = max(0.0, 1.0 - u_pop / max(state.households.total_population, 1.0))

        # 加权聚合
        livelihood_raw = (
            0.20 * subs["welfare_composite"]
            + 0.15 * subs["poverty_rate"]
            + 0.12 * subs["cost_equity"]
            + 0.10 * subs["social_mobility"]
            + 0.10 * subs["employment_health"]
            + 0.08 * subs["health_index"]
            + 0.08 * subs["edu_equity"]
            + 0.07 * subs["intergen_fairness"]
            + 0.10 * subs["social_cost"]
        )
        return float(np.clip(livelihood_raw, 0.0, 1.0)), subs

    # ─────────────────────────────────────────────
    # 维度三：发展韧性
    # ─────────────────────────────────────────────
    def _evaluate_resilience(self, state: SystemState,
                             t_series: Optional[Dict[str, np.ndarray]] = None
                             ) -> Tuple[float, Dict[str, float]]:
        subs = {}

        # 1. EWS韧性指数
        resilience_val = 0.5
        for var_name in ["K_i_ratio", "tech_gap"]:
            if var_name in state.ews_signals:
                resilience_val = max(resilience_val,
                                     state.ews_signals[var_name].resilience_index)
        subs["resilience_index"] = resilience_val

        # 2. 技术自主度
        tech_import_share = state.external_dependency
        subs["tech_autonomy"] = max(0.0, 1.0 - tech_import_share)

        # 3. 财政缓冲空间
        debt_ratio = state.fiscal_debt_ratio
        subs["fiscal_space"] = max(0.0, 1.0 - debt_ratio / 1.50)

        # 4. 外汇储备充足率（用积累基金/GDP近似）
        fx_buffer = state.accumulation_fund / max(state.gdp, 0.01)
        subs["fx_adequacy"] = self._sigmoid_scale(fx_buffer, mid=0.3, steep=10.0)

        # 5. 粮食/能源自给率
        resource_self = state.sectors["b"].Y / max(
            state.households.total_population * 0.5, 1.0
        )
        subs["resource_self_sufficiency"] = self._sigmoid_scale(
            resource_self, mid=1.0, steep=3.0
        )

        # 6. 制度响应速度（用政策约束乘数的反向）
        policy_space = state.political_support.policy_constraint
        subs["institution_responsiveness"] = policy_space

        # 加权聚合（与文档中的权重一致）
        resilience_raw = (
            0.30 * subs["resilience_index"]
            + 0.20 * subs["tech_autonomy"]
            + 0.15 * subs["fiscal_space"]
            + 0.10 * subs["fx_adequacy"]
            + 0.10 * subs["resource_self_sufficiency"]
            + 0.15 * subs["institution_responsiveness"]
        )
        return float(np.clip(resilience_raw, 0.0, 1.0)), subs

    # ─────────────────────────────────────────────
    # 维度四：结构健康
    # ─────────────────────────────────────────────
    def _evaluate_structure(self, state: SystemState,
                            t_series: Optional[Dict[str, np.ndarray]] = None
                            ) -> Tuple[float, Dict[str, float]]:
        subs = {}

        # 1. 去工业化质量指数
        deind_quality = state.cost_metrics.deind_quality
        subs["deind_quality"] = deind_quality

        # 2. 工业数字密度
        dd_i = state.sectors["i"].digital_density
        subs["digital_density_i"] = dd_i  # 已在 [0,1]，越大越好

        # 3. 部门间工资差距
        wages = [state.sectors[k].W for k in ALL_SECTORS]
        w_max = max(wages)
        w_min = max(min(wages), 1e-6)
        wage_disparity = w_max / w_min - 1.0
        # 差距=0→1.0, 差距=3→0.3, 差距=10→0.0
        subs["wage_equity"] = max(0.0, 1.0 - wage_disparity / 10.0)

        # 4. 劳动力技能匹配度
        skill_match = min(
            state.sectors["i"].H / max(state.households.industrial_workers.skill_level, 0.01),
            state.sectors["d"].H / max(state.households.digital_high_skill.skill_level, 0.01),
            1.5  # 上限
        )
        subs["skill_match"] = self._sigmoid_scale(skill_match - 0.5, mid=0.3, steep=5.0)

        # 5. 产业结构多样化 (v3.3: 四部门)
        shares = np.array([state.sectors[k].Y for k in ALL_SECTORS])
        total = shares.sum()
        if total > 0:
            hhi = np.sum((shares / total) ** 2)
            # HHI: 1/4=完全均匀→高多样化, 1=完全垄断→低多样化
            n_sectors = len(ALL_SECTORS)
            hhi_min = 1.0 / n_sectors
            diversification = 1.0 - (hhi - hhi_min) / (1 - hhi_min) if hhi > hhi_min else 1.0
            subs["diversification"] = max(0.0, diversification)
        else:
            subs["diversification"] = 0.5

        # 6. 创新扩散速率
        # 从技术差距变化趋势推断
        if t_series is not None and "tech_gap" in t_series and len(t_series["tech_gap"]) >= 20:
            dG = np.diff(t_series["tech_gap"][-20:])
            diffusion_rate = -np.mean(dG)  # 负的差距变化=在追赶
            subs["diffusion_speed"] = self._sigmoid_scale(diffusion_rate, mid=0.02, steep=50)
        else:
            subs["diffusion_speed"] = 0.5

        # 加权聚合
        structure_raw = (
            0.25 * subs["deind_quality"]
            + 0.20 * subs["digital_density_i"]
            + 0.15 * subs["wage_equity"]
            + 0.15 * subs["skill_match"]
            + 0.10 * subs["diversification"]
            + 0.15 * subs["diffusion_speed"]
        )
        return float(np.clip(structure_raw, 0.0, 1.0)), subs

    # ─────────────────────────────────────────────
    # 维度五：主权与自主
    # ─────────────────────────────────────────────
    def _evaluate_sovereignty(self, state: SystemState,
                              t_series: Optional[Dict[str, np.ndarray]] = None
                              ) -> Tuple[float, Dict[str, float]]:
        subs = {}

        # 1. 综合自主指数（来自数字依附的反向）
        subs["autonomy_composite"] = max(0.0, 1.0 - state.dependency.composite)

        # 2. 数据主权度
        total_data = state.data_assets.effective_stock
        local_data = (state.data_assets.producer_data * 0.8
                      + state.data_assets.public_data * 1.0
                      + state.data_assets.consumer_data * 0.3)  # 消费者数据更多在国外
        data_sovereignty = local_data / max(total_data, 0.01)
        subs["data_sovereignty"] = min(1.0, data_sovereignty)

        # 3. 标准参与权（从制度能力 + 技术自主推断）
        subs["standards_participation"] = (
            0.5 * self.p.get("institution_capacity", 0.6)
            + 0.5 * subs["autonomy_composite"]
        )

        # 4. 政策自主空间
        subs["policy_autonomy"] = state.political_support.policy_constraint

        # 5. 关键供应链安全
        import_dep = state.external_dependency
        subs["supply_chain_security"] = max(0.0, 1.0 - import_dep)

        # 6. 资本账户自主
        # 检查外资在资本中的比例
        equity_foreign = 0.0
        total_equity = 1.0
        if state.balance_sheets:
            for bs in state.balance_sheets.values():
                equity_foreign += bs.equity_foreign
                total_equity += bs.net_worth
        foreign_control = equity_foreign / max(total_equity, 1.0)
        subs["capital_account_autonomy"] = max(0.0, 1.0 - foreign_control)

        # 加权聚合
        sovereignty_raw = (
            0.25 * subs["autonomy_composite"]
            + 0.20 * subs["data_sovereignty"]
            + 0.10 * subs["standards_participation"]
            + 0.20 * subs["policy_autonomy"]
            + 0.15 * subs["supply_chain_security"]
            + 0.10 * subs["capital_account_autonomy"]
        )
        return float(np.clip(sovereignty_raw, 0.0, 1.0)), subs

    # ─────────────────────────────────────────────
    # 工具函数
    # ─────────────────────────────────────────────
    @staticmethod
    def _sigmoid_scale(x: float, mid: float = 0.5, steep: float = 5.0) -> float:
        """Sigmoid标准化: 将任意实数映射到[0,1]."""
        return float(1.0 / (1.0 + np.exp(-steep * (x - mid))))

    @staticmethod
    def _simple_gini(arr: np.ndarray) -> float:
        """简易基尼系数."""
        if len(arr) < 2 or arr.sum() == 0:
            return 0.0
        arr = np.sort(arr)
        n = len(arr)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))
