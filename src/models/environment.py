"""v3.3 环境模块 —— 碳排放追踪 + 绿色转型压力.

三个轻量机制：
1. 碳排放追踪：工业产出 → 排放，数字密度高的工业更清洁
2. 绿色转型压力：国际碳边境调节 → 高碳工业出口成本上升
3. 搁浅资产风险：资源出口国面临化石燃料需求萎缩
"""

import numpy as np
from ..core.state import SystemState, ALL_SECTORS


class EnvironmentModule:
    """环境与气候模块."""

    def __init__(self, params: dict):
        self.p = params
        # 碳强度参数（每单位工业产出的碳排放）
        self._carbon_intensity_base = 0.5
        # 绿色转型压力参数（由场景设置）
        self._green_pressure_base = 0.0

    def step(self, state: SystemState, dt: float):
        """更新碳排放和气候损害."""
        # 1. 计算当期碳排放
        self._compute_emissions(state, dt)
        # 2. 更新累计碳排放
        state.cumulative_carbon += state.carbon_emissions * dt
        # 3. 计算气候损害
        self._compute_climate_damage(state)
        # 4. 绿色转型压力
        self._compute_green_pressure(state, dt)
        # 5. 应用损害到部门产出
        self._apply_damage(state, dt)

    def _compute_emissions(self, state: SystemState, dt: float):
        """计算当期碳排放.

        排放来源：
        - 工业部门：主要排放源，数字密度越高越清洁
        - 基础部门：农业和资源开采的排放
        - 数字部门：几乎零排放（电力间接排放忽略）
        """
        # 工业排放
        dd_i = state.sectors["i"].digital_density
        # 数字密度每提升0.1，碳强度下降约3%
        carbon_intensity_i = self._carbon_intensity_base * (1.0 - dd_i * 0.3)
        industrial_emissions = state.sectors["i"].Y * carbon_intensity_i

        # 基础部门排放（农业甲烷+资源开采）
        base_emissions = state.sectors["b"].Y * 0.15

        # 数字部门排放（几乎为零）
        digital_emissions = state.sectors["d"].Y * 0.02

        state.carbon_emissions = industrial_emissions + base_emissions + digital_emissions

    def _compute_climate_damage(self, state: SystemState):
        """计算气候损害乘数.

        简化损害函数：累计碳排放 → 损害乘数
        累计碳越高，对基础部门（农业）的损害越大。
        """
        # 损害与累计碳排放的非线性关系
        # 基础值100表示"正常"的累计碳排放水平
        carbon_ratio = state.cumulative_carbon / max(state.gdp * 10.0, 1.0)
        # 损害乘数：1.0=无损害，>1.0=损害
        state.carbon_damage_multiplier = 1.0 + carbon_ratio * 0.3
        state.carbon_damage_multiplier = min(1.5, state.carbon_damage_multiplier)

    def _compute_green_pressure(self, state: SystemState, dt: float):
        """计算绿色转型压力.

        压力来源：
        - 碳排放/GDP比越高 → 国际社会施加的碳边境调节越强
        - 场景参数中的绿色转型基线压力
        """
        # 碳强度 = 排放/GDP
        carbon_intensity = state.carbon_emissions / max(state.gdp, 0.01)

        # 绿色转型压力上升（如果碳强度高）
        target_pressure = min(1.0, carbon_intensity * 2.0)
        # 压力缓慢上升（制度惯性）
        state.green_transition_pressure += (
            (target_pressure - state.green_transition_pressure) * 0.1 * dt
        )
        state.green_transition_pressure = max(0.0, min(1.0, state.green_transition_pressure))

    def _apply_damage(self, state: SystemState, dt: float):
        """将气候损害和绿色转型压力应用到经济.

        受影响渠道：
        1. 基础部门（农业）：受气候损害直接影响
        2. 工业出口：碳边境调节增加成本
        3. 资源出口：化石燃料面临"搁浅资产"折价
        """
        # 1. 基础部门气候损害
        if state.carbon_damage_multiplier > 1.01:
            damage = (state.carbon_damage_multiplier - 1.0) * 0.1 * dt
            state.sectors["b"].Y *= max(0.85, 1.0 - damage)

        # 2. 碳边境调节：高碳工业出口成本上升
        if state.green_transition_pressure > 0.1:
            # 压力越大 → 碳边境税越高 → 工业出口竞争力下降
            carbon_tax_effect = state.green_transition_pressure * 0.05 * dt
            # 假设工业的30%是出口（受碳边境调节影响）
            state.sectors["i"].Y *= max(0.88, 1.0 - carbon_tax_effect * 0.3)

        # 3. v3.3: 搁浅资产风险 — 资源出口国面临化石燃料需求萎缩
        resource_intensity = self.p.get("C_resource_export_intensity", 0.3)
        if resource_intensity > 0.5 and state.green_transition_pressure > 0.3:
            # 高资源依赖 + 高绿色压力 → 资源出口价格折价
            stranded_discount = (state.green_transition_pressure - 0.3) * 0.03 * dt
            state.sectors["b"].Y *= max(0.85, 1.0 - stranded_discount * resource_intensity)
            # 资源出口收入减少 → 积累基金减少
            revenue_loss = state.sectors["b"].Y * stranded_discount * resource_intensity
            state.accumulation_fund = max(0.0, state.accumulation_fund - revenue_loss * 0.3)
