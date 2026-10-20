"""v3.6 双区域空间模块 —— 沿海/内陆（城市/农村）分化.

核心机制：
- 劳动力空间流动（Harris-Todaro 期望工资差驱动）
- 产业空间集聚的自我强化（路径依赖）
- 集聚经济 vs 拥挤成本的正负外部性
- 区域不平等对政治稳定性的反馈

理论基础：
- Krugman (1991) 新经济地理学
- Harris & Todaro (1970) 城乡迁移模型
- World Bank (2009) 重塑经济地理
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from ..core.state import SystemState, ALL_SECTORS


@dataclass
class SpatialState:
    """双区域空间状态."""

    # 沿海/城市在各维度的份额
    coastal_industry_share: float = 0.60
    coastal_digital_infra_share: float = 0.75
    coastal_digital_app_share: float = 0.80
    coastal_service_high_share: float = 0.70
    coastal_service_low_share: float = 0.55
    coastal_base_share: float = 0.20       # 基础部门（农业）多在非沿海
    coastal_population_share: float = 0.45

    # 生产率
    urban_rural_productivity_ratio: float = 2.5

    # 劳动力流动
    migration_rate_net: float = 0.0        # 净迁移率（年）
    migration_cost_index: float = 0.30     # 流动摩擦 [0,1]

    # 集聚效应
    agglomeration_bonus: float = 0.0       # 集聚经济生产率增益 [0,1]
    congestion_penalty: float = 0.0        # 拥挤成本 [0,1]

    # 不平等
    coastal_inland_income_ratio: float = 2.5  # 沿海vs内陆人均收入比
    spatial_gini: float = 0.0                 # 空间基尼系数

    # 历史追踪
    years_high_inequality: float = 0.0     # 高区域不平等的累积年数


class SpatialModule:
    """双区域空间动态模块.

    追踪"沿海/城市"和"内陆/农村"两个区域之间的
    劳动力流动、产业集聚和经济分化。
    """

    def __init__(self, params: dict):
        self.p = params
        self._state = SpatialState()

        # 从参数读取初始值
        self._state.coastal_industry_share = params.get("IC_coastal_industry_share", 0.60)
        self._state.coastal_digital_infra_share = params.get("IC_coastal_digital_infra_share", 0.75)
        self._state.coastal_digital_app_share = params.get("IC_coastal_digital_app_share", 0.80)
        self._state.coastal_service_high_share = params.get("IC_coastal_service_high_share", 0.70)
        self._state.coastal_population_share = params.get("IC_coastal_population_share", 0.45)
        self._state.urban_rural_productivity_ratio = params.get("IC_urban_rural_productivity_ratio", 2.5)
        self._state.migration_cost_index = params.get("migration_cost_index", 0.30)

        # 追踪器
        self._prev_coastal_pop: float = self._state.coastal_population_share

    def step(self, state: SystemState, dt: float):
        """一个时间步的空间动态更新."""
        # 1. 劳动力空间流动
        self._update_labor_migration(state, dt)

        # 2. 产业空间分布演化
        self._update_industrial_distribution(state, dt)

        # 3. 集聚效应与拥挤成本
        self._update_agglomeration_effects(state, dt)

        # 4. 区域不平等度量
        self._update_spatial_inequality(state)

        # 5. 应用空间效应到经济系统
        self._apply_spatial_effects(state, dt)

        # 6. 同步到全局状态
        self._sync_to_state(state)

    # ─────────────────────────────────────────────
    # 劳动力空间流动
    # ─────────────────────────────────────────────
    def _update_labor_migration(self, state: SystemState, dt: float):
        """劳动力空间流动 —— Harris-Todaro 类型.

        迁移速率取决于沿海-内陆期望工资差，受迁移成本制约。
        """
        ss = self._state

        # 沿海期望工资（按就业份额加权）
        coastal_w = (
            ss.coastal_industry_share * state.sectors["i"].W
            + ss.coastal_digital_infra_share * state.sectors["d_infra"].W
            + ss.coastal_digital_app_share * state.sectors["d_app"].W
            + ss.coastal_service_high_share * state.sectors["s_high"].W
            + ss.coastal_service_low_share * state.sectors["s_low"].W
        )
        coastal_pop_weight = (
            ss.coastal_industry_share + ss.coastal_digital_infra_share
            + ss.coastal_digital_app_share + ss.coastal_service_high_share
            + ss.coastal_service_low_share
        )
        coastal_w /= max(coastal_pop_weight, 1.0)

        # 内陆工资（对称计算）
        inland_w = (
            (1 - ss.coastal_industry_share) * state.sectors["i"].W
            + (1 - ss.coastal_digital_infra_share) * state.sectors["d_infra"].W
            + (1 - ss.coastal_digital_app_share) * state.sectors["d_app"].W
            + (1 - ss.coastal_service_high_share) * state.sectors["s_high"].W
            + (1 - ss.coastal_service_low_share) * state.sectors["s_low"].W
        )
        inland_weight = 5.0 - coastal_pop_weight
        inland_w /= max(inland_weight, 1.0)

        # 工资比
        wage_gap = coastal_w / max(inland_w, 0.01)

        # 迁移速率（年化）
        # 工资差 > 1.5 倍时显著拉动人口流动
        if wage_gap > 1.0:
            migration_pressure = (wage_gap - 1.0) * 0.04
        else:
            migration_pressure = 0.0

        # 迁移成本打折
        migration_pressure *= (1.0 - ss.migration_cost_index)

        # 饱和效应：沿海人口占比越高，吸引力边际递减
        coastal_share = ss.coastal_population_share
        saturation_factor = max(0.05, 1.0 - coastal_share / 0.85)

        # 净迁移率
        net_migration = migration_pressure * saturation_factor * dt

        # 更新沿海人口份额
        self._prev_coastal_pop = ss.coastal_population_share
        ss.coastal_population_share = max(0.15, min(0.85,
            ss.coastal_population_share + net_migration
        ))
        ss.migration_rate_net = net_migration

    # ─────────────────────────────────────────────
    # 产业空间分布演化
    # ─────────────────────────────────────────────
    def _update_industrial_distribution(self, state: SystemState, dt: float):
        """产业空间分布的路径依赖演化.

        核心机制：
        - 数字部门天然倾向集聚（网络效应 + 人才池）
        - 服务业跟随人口分布
        - 工业受成本驱动可能向内陆扩散
        """
        ss = self._state

        # 1. 数字部门：自我强化集聚
        # app 层的网络效应最强 → 集聚倾向最明显
        d_app_agglomeration = (
            state.sectors["d_app"].Y / max(state.gdp, 0.01)
        ) * 0.02 * dt
        ss.coastal_digital_app_share = min(0.95,
            ss.coastal_digital_app_share + d_app_agglomeration
        )

        # infra 层随 app 层部分集聚但受物理基础设施限制
        ss.coastal_digital_infra_share = min(0.90,
            ss.coastal_digital_infra_share + d_app_agglomeration * 0.5
        )

        # 2. 服务业：跟随人口和数字部门
        ss.coastal_service_high_share = min(0.90,
            ss.coastal_service_high_share
            + (ss.coastal_population_share - ss.coastal_service_high_share) * 0.03 * dt
        )
        ss.coastal_service_low_share = min(0.85,
            ss.coastal_service_low_share
            + (ss.coastal_population_share - ss.coastal_service_low_share) * 0.05 * dt
        )

        # 3. 工业：成本驱动的分散化
        # 当沿海拥挤成本高时，工业向内陆转移
        if ss.congestion_penalty > 0.2:
            dispersion = ss.congestion_penalty * 0.01 * dt
            ss.coastal_industry_share = max(0.30,
                ss.coastal_industry_share - dispersion
            )
        else:
            # 工业也倾向沿海（出口便利）
            ss.coastal_industry_share = min(0.80,
                ss.coastal_industry_share
                + (ss.coastal_population_share * 0.8 - ss.coastal_industry_share) * 0.01 * dt
            )

    # ─────────────────────────────────────────────
    # 集聚效应与拥挤成本
    # ─────────────────────────────────────────────
    def _update_agglomeration_effects(self, state: SystemState, dt: float):
        """计算集聚经济正外部性和拥挤成本负外部性."""
        ss = self._state
        coastal_pop = ss.coastal_population_share

        # 1. 集聚经济：人口集中 → 知识溢出 + 厚劳动力市场 + 基础设施共享
        # 阈值：人口占比 > 0.35 时开始有显著集聚效应
        agglomeration_strength = self.p.get("agglomeration_strength", 0.15)
        if coastal_pop > 0.35:
            ss.agglomeration_bonus = min(
                0.25,
                (coastal_pop - 0.35) * agglomeration_strength * 2.0
            )
        else:
            ss.agglomeration_bonus = 0.0

        # 2. 拥挤成本：人口过度集中 → 房价/交通/环境压力
        congestion_threshold = self.p.get("congestion_threshold", 0.60)
        if coastal_pop > congestion_threshold:
            ss.congestion_penalty = min(
                0.20,
                (coastal_pop - congestion_threshold) * 0.30
            )
        else:
            ss.congestion_penalty = max(0.0,
                ss.congestion_penalty - 0.005 * dt
            )

    # ─────────────────────────────────────────────
    # 区域不平等度量
    # ─────────────────────────────────────────────
    def _update_spatial_inequality(self, state: SystemState):
        """计算区域间不平等指标."""
        ss = self._state

        # 沿海人均收入 vs 内陆人均收入
        coastal_income = (
            ss.coastal_industry_share * state.sectors["i"].Y
            + ss.coastal_digital_infra_share * state.sectors["d_infra"].Y
            + ss.coastal_digital_app_share * state.sectors["d_app"].Y
            + ss.coastal_service_high_share * state.sectors["s_high"].Y
            + ss.coastal_service_low_share * state.sectors["s_low"].Y
            + ss.coastal_base_share * state.sectors["b"].Y
        ) / max(ss.coastal_population_share, 0.01)

        inland_income = (
            (1 - ss.coastal_industry_share) * state.sectors["i"].Y
            + (1 - ss.coastal_digital_infra_share) * state.sectors["d_infra"].Y
            + (1 - ss.coastal_digital_app_share) * state.sectors["d_app"].Y
            + (1 - ss.coastal_service_high_share) * state.sectors["s_high"].Y
            + (1 - ss.coastal_service_low_share) * state.sectors["s_low"].Y
            + (1 - ss.coastal_base_share) * state.sectors["b"].Y
        ) / max(1 - ss.coastal_population_share, 0.01)

        ss.coastal_inland_income_ratio = coastal_income / max(inland_income, 0.01)

        # 空间基尼系数（简化：用沿海vs内陆的两点分布计算）
        # G = |coastal_share_of_pop - coastal_share_of_income|
        total_income = coastal_income * ss.coastal_population_share + inland_income * (1 - ss.coastal_population_share)
        coastal_income_share = (
            coastal_income * ss.coastal_population_share
        ) / max(total_income, 0.01)
        ss.spatial_gini = abs(coastal_income_share - ss.coastal_population_share)

        # 累积高不平等年数
        if ss.coastal_inland_income_ratio > 3.0:
            ss.years_high_inequality += 0.25  # 季度

    # ─────────────────────────────────────────────
    # 空间效应对经济系统的反馈
    # ─────────────────────────────────────────────
    def _apply_spatial_effects(self, state: SystemState, dt: float):
        """将空间结构变化反馈到经济系统."""
        ss = self._state

        # 1. 集聚经济 → 沿海数字部门生产率增益
        if ss.agglomeration_bonus > 0:
            for d_key in ["d_infra", "d_app", "s_high"]:
                coastal_share = getattr(
                    ss, f"coastal_{d_key}_share",
                    0.6
                )
                # 收益与沿海份额和集聚效应成正比
                productivity_boost = (
                    1.0 + ss.agglomeration_bonus * coastal_share * 0.5
                )
                state.sectors[d_key].Y *= productivity_boost

        # 2. 拥挤成本 → 生活成本上升 → 实际工资折扣
        if ss.congestion_penalty > 0:
            # 拥挤削弱了沿海高名义工资的实际购买力
            real_wage_discount = 1.0 - ss.congestion_penalty * 0.3
            for key in ALL_SECTORS:
                coastal_share = getattr(
                    ss, f"coastal_{key}_share",
                    ss.coastal_population_share
                )
                # 对沿海集中的部门施加更大的实际工资折扣
                if coastal_share > 0.5:
                    state.sectors[key].W *= (
                        1.0 - ss.congestion_penalty * coastal_share * 0.15
                    )

        # 3. 区域不平等 → 政治压力
        if ss.spatial_gini > 0.3:
            # 空间不平等加剧 → 内陆/农村居民的不满 → 政治支持度受压
            spatial_discontent = (ss.spatial_gini - 0.3) * 0.03 * dt
            state.political_support.support_index = max(
                0.05,
                state.political_support.support_index - spatial_discontent
            )

        # 4. 城镇化率联动
        state.urbanization_rate = ss.coastal_population_share

    # ─────────────────────────────────────────────
    # 同步到全局状态
    # ─────────────────────────────────────────────
    def _sync_to_state(self, state: SystemState):
        """将空间状态同步到全局 SystemState."""
        ss = self._state
        state.coastal_industry_share = ss.coastal_industry_share
        state.coastal_digital_infra_share = ss.coastal_digital_infra_share
        state.coastal_digital_app_share = ss.coastal_digital_app_share
        state.coastal_service_high_share = ss.coastal_service_high_share
        state.coastal_population_share = ss.coastal_population_share
        state.urban_rural_productivity_ratio = ss.urban_rural_productivity_ratio
        state.spatial_gini = ss.spatial_gini
        state.agglomeration_bonus = ss.agglomeration_bonus
        state.congestion_penalty = ss.congestion_penalty

    # ─────────────────────────────────────────────
    # 公共接口
    # ─────────────────────────────────────────────
    def get_state(self) -> SpatialState:
        """获取当前空间状态."""
        return self._state

    def get_summary(self) -> dict:
        """获取空间状态摘要."""
        ss = self._state
        return {
            "coastal_population_share": ss.coastal_population_share,
            "coastal_industry_share": ss.coastal_industry_share,
            "coastal_digital_app_share": ss.coastal_digital_app_share,
            "coastal_inland_income_ratio": ss.coastal_inland_income_ratio,
            "spatial_gini": ss.spatial_gini,
            "agglomeration_bonus": ss.agglomeration_bonus,
            "congestion_penalty": ss.congestion_penalty,
            "migration_rate_net": ss.migration_rate_net,
            "urban_rural_productivity_ratio": ss.urban_rural_productivity_ratio,
            "years_high_inequality": ss.years_high_inequality,
        }

    def check_spatial_red_lines(self) -> list:
        """检查与空间不平等相关的红线."""
        ss = self._state
        warnings = []

        if ss.coastal_inland_income_ratio > 5.0:
            warnings.append({
                "type": "extreme_spatial_inequality",
                "severity": "CRITICAL",
                "value": ss.coastal_inland_income_ratio,
                "threshold": 5.0,
                "description": "沿海-内陆收入比超过5倍，区域发展严重失衡",
            })

        if ss.spatial_gini > 0.4:
            warnings.append({
                "type": "high_spatial_gini",
                "severity": "WARNING",
                "value": ss.spatial_gini,
                "threshold": 0.4,
                "description": "空间基尼系数过高，区域不平等可能引发政治不稳定",
            })

        if ss.years_high_inequality > 20:
            warnings.append({
                "type": "persistent_spatial_inequality",
                "severity": "WARNING",
                "value": ss.years_high_inequality,
                "threshold": 20.0,
                "description": "高区域不平等持续超过20年，结构性固化风险",
            })

        return warnings
