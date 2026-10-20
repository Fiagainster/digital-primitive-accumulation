"""v3.6 集成测试 —— 验证所有新模块的正确性.

测试覆盖:
- PlatformModule: 市场集中度演化、垄断租金、反垄断参数
- DemographyModule: 人口更新、生育率内生调节、年龄结构推进
- SpatialModule: 沿海/内陆分化、劳动力流动、集聚效应
- 政策执行折扣: 官僚体系内部摩擦
- 各模块在 SimulationEngine 中的集成
"""

import sys
import os
from pathlib import Path
import numpy as np

# 确保可以导入 src
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from src.core.engine import SimulationEngine
from src.core.config import SimulationConfig
from src.core.state import SystemState, ALL_SECTORS
from src.models.platform import PlatformModule, PlatformState
from src.models.demography import DemographyModule, DemographyState
from src.models.spatial import SpatialModule, SpatialState
from src.params.schema import get_schema


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def schema():
    return get_schema()


@pytest.fixture
def default_params(schema):
    """构建包含所有新参数的默认参数集."""
    params = schema.build_defaults("A")
    # 确保新参数存在
    params["anti_monopoly_policy"] = 0.3
    params["data_portability_mandate"] = 0.2
    params["platform_interoperability"] = 0.1
    params["IC_total_population"] = 7000.0
    params["IC_fertility_rate"] = 2.5
    params["IC_life_expectancy"] = 65.0
    params["IC_urbanization_rate"] = 0.35
    params["IC_coastal_population_share"] = 0.45
    params["migration_cost_index"] = 0.30
    return params


@pytest.fixture
def state():
    """构建包含所有新字段的初始状态."""
    from src.core.state import SectorState
    state = SystemState()
    for key in ALL_SECTORS:
        state.sectors[key] = SectorState(
            K=100.0, H=1.0, L=10.0, DL=0.1,
            Y=50.0, W=3.0, r=0.15, A=1.0,
        )
    state.gdp = 300.0
    state.total_population = 7000.0
    state.total_surplus = 45.0
    state.accumulation_fund = 36.0
    state.tech_gap = 3.0
    state.digital_hhi = 0.3
    state.coastal_population_share = 0.45
    return state


# ============================================================
# PlatformModule 测试
# ============================================================

class TestPlatformModule:
    """平台垄断模块测试."""

    def test_initialization(self, default_params):
        """测试模块正确初始化."""
        platform = PlatformModule(default_params)
        ps = platform.get_state()
        assert ps.anti_monopoly_policy == 0.3
        assert ps.data_portability_mandate == 0.2
        assert ps.platform_interoperability == 0.1
        assert 0.0 <= ps.digital_hhi <= 1.0

    def test_hhi_evolution_without_network_effects(self, default_params, state):
        """测试无网络效应时HHI不增长."""
        platform = PlatformModule(default_params)
        state.sectors["d_app"].L = 0.1  # 极少用户
        state.data_assets.consumer_data = 0.01  # 极少数据

        initial_hhi = platform.get_state().digital_hhi
        for _ in range(10):
            platform.step(state, dt=0.25)

        # 无用户和数据时，HHI应保持稳定或下降（竞争衰减 > 网络驱动力）
        final_hhi = platform.get_state().digital_hhi
        assert final_hhi <= initial_hhi + 0.05

    def test_hhi_evolution_with_strong_network_effects(self, default_params, state):
        """测试强网络效应推动HHI上升."""
        platform = PlatformModule(default_params)
        state.sectors["d_app"].L = 50.0  # 大量用户
        state.sectors["d_app"].DL = 0.6   # 高数字化
        state.data_assets.consumer_data = 50.0  # 大量数据

        initial_hhi = platform.get_state().digital_hhi
        for _ in range(40):  # 10年
            platform.step(state, dt=0.25)

        # 强网络效应 + 大量用户 → HHI 应上升
        final_hhi = platform.get_state().digital_hhi
        assert final_hhi > initial_hhi

    def test_anti_monopoly_policy_effect(self, default_params, state):
        """测试反垄断政策抑制HHI增长."""
        # 无反垄断
        params_no = default_params.copy()
        params_no["anti_monopoly_policy"] = 0.0
        platform_no = PlatformModule(params_no)
        state_no = SystemState()
        for key in ALL_SECTORS:
            state_no.sectors[key] = state.sectors[key]
        state_no.gdp = state.gdp
        state_no.data_assets.consumer_data = 50.0
        state_no.sectors["d_app"].L = 50.0
        state_no.sectors["d_app"].DL = 0.6
        state_no.total_population = state.total_population

        for _ in range(40):
            platform_no.step(state_no, dt=0.25)

        # 有反垄断
        params_yes = default_params.copy()
        params_yes["anti_monopoly_policy"] = 0.8
        platform_yes = PlatformModule(params_yes)
        state_yes = SystemState()
        for key in ALL_SECTORS:
            state_yes.sectors[key] = state.sectors[key]
        state_yes.gdp = state.gdp
        state_yes.data_assets.consumer_data = 50.0
        state_yes.sectors["d_app"].L = 50.0
        state_yes.sectors["d_app"].DL = 0.6
        state_yes.total_population = state.total_population

        for _ in range(40):
            platform_yes.step(state_yes, dt=0.25)

        # 反垄断政策应使HHI更低
        assert platform_yes.get_state().digital_hhi < platform_no.get_state().digital_hhi

    def test_monopoly_rent_nonlinear(self, default_params, state):
        """测试垄断租金的非线性增长."""
        platform = PlatformModule(default_params)
        ps = platform.get_state()

        # 模拟HHI手动设置来测试租金函数
        ps.digital_hhi = 0.3  # 低于阈值
        platform._update_monopoly_rent(state)
        assert ps.platform_monopoly_rent == 0.0  # 无租金

        ps.digital_hhi = 0.7  # 超过阈值
        platform._update_monopoly_rent(state)
        assert ps.platform_monopoly_rent > 0.25  # 应有显著租金

        ps.digital_hhi = 0.9  # 高度集中
        platform._update_monopoly_rent(state)
        assert ps.platform_monopoly_rent > 0.5

    def test_platform_risk_detection(self, default_params, state):
        """测试平台相关风险检测."""
        platform = PlatformModule(default_params)
        ps = platform.get_state()
        ps.digital_hhi = 0.85
        ps.platform_monopoly_rent = 0.7
        state.wealth_gini = 0.65

        risks = platform.check_platform_related_risks(state)
        assert len(risks) > 0
        assert any(r["type"] == "digital_feudalization" for r in risks)

    def test_monopoly_innovation_drag(self, default_params, state):
        """测试垄断对创新的抑制."""
        platform = PlatformModule(default_params)
        ps = platform.get_state()
        ps.digital_hhi = 0.8
        platform._update_innovation_drag(state)
        assert ps.monopoly_innovation_drag > 0.0

    def test_data_moat_with_portability(self, default_params, state):
        """测试数据可携带权削弱护城河."""
        platform = PlatformModule(default_params)
        state.data_assets.consumer_data = 80.0
        state.sectors["d_app"].DL = 0.7

        # 无数据可携带权 → 高护城河
        platform.get_state().data_portability_mandate = 0.0
        platform._update_data_moat(state, dt=1.0)
        moat_no_portability = platform.get_state().data_network_moat

        # 有数据可携带权 → 低护城河
        platform.get_state().data_portability_mandate = 0.8
        for _ in range(10):
            platform._update_data_moat(state, dt=1.0)
        moat_with_portability = platform.get_state().data_network_moat

        assert moat_with_portability < moat_no_portability


# ============================================================
# DemographyModule 测试
# ============================================================

class TestDemographyModule:
    """内生人口动态模块测试."""

    def test_initialization(self, default_params):
        """测试模块正确初始化."""
        demo = DemographyModule(default_params)
        ds = demo.get_state()
        assert ds.fertility_rate == 2.5
        assert ds.life_expectancy == 65.0
        assert ds.urbanization_rate == 0.35
        assert ds.total_population == 7000.0

    def test_fertility_declines_with_development(self, default_params, state):
        """测试经济发展导致生育率下降."""
        demo = DemographyModule(default_params)
        initial_tfr = demo.get_state().fertility_rate

        # 模拟高人均GDP
        state.gdp = 10000.0
        state.total_population = 7000.0
        state.sectors["s_high"].H = 2.0
        state.sectors["d_app"].H = 2.0

        demo._adjust_fertility(state)
        after_dev = demo.get_state().fertility_rate

        assert after_dev < initial_tfr  # 生育率应下降

    def test_life_expectancy_increases_with_health(self, default_params, state):
        """测试健康改善提升预期寿命."""
        demo = DemographyModule(default_params)
        initial_le = demo.get_state().life_expectancy

        state.gdp = 5000.0
        state.total_population = 7000.0
        state.sectors["i"].digital_density = 0.5  # 减少污染

        demo._adjust_mortality(state)
        after_health = demo.get_state().life_expectancy

        assert after_health > initial_le

    def test_urbanization_updates_in_quarterly_steps(self, default_params, state):
        """测试城镇化在季度步长下连续更新."""
        demo = DemographyModule(default_params)
        initial_urban = demo.get_state().urbanization_rate

        state.total_population = 7000.0
        state.sectors["i"].L = 30.0
        state.sectors["d_app"].L = 5.0
        state.sectors["s_high"].L = 8.0

        for _ in range(20):  # 5年
            demo._update_urbanization_continuous(state, dt=0.25)

        assert demo.get_state().urbanization_rate > initial_urban

    def test_age_structure_normalization(self, default_params, state):
        """测试年龄结构保持归一化."""
        demo = DemographyModule(default_params)
        ds = demo.get_state()

        assert abs(ds.age_structure.sum() - 1.0) < 1e-6

        # 运行多年
        for _ in range(20):  # 20年
            demo._adjust_fertility(state)
            demo._adjust_mortality(state)
            demo._cohort_component_step(state)

        assert abs(ds.age_structure.sum() - 1.0) < 1e-6

    def test_dependency_ratio_calculation(self, default_params, state):
        """测试抚养比计算."""
        demo = DemographyModule(default_params)

        demo._adjust_fertility(state)
        demo._adjust_mortality(state)
        demo._cohort_component_step(state)
        demo._update_labor_supply(state)

        ds = demo.get_state()
        assert ds.dependency_ratio > 0
        assert ds.working_age_pop > 0
        assert ds.old_age_dependency >= 0

    def test_demographic_dividend_detection(self, default_params, state):
        """测试人口红利检测."""
        demo = DemographyModule(default_params)
        ds = demo.get_state()

        # 设置有利的人口结构（高劳动年龄人口增长）
        ds.working_age_growth_rate = 0.02
        ds.demographic_dividend = 0.01

        assert demo.is_demographic_dividend()

        ds.demographic_dividend = -0.01
        assert not demo.is_demographic_dividend()

    def test_sync_to_state(self, default_params, state):
        """测试人口状态同步到全局状态."""
        demo = DemographyModule(default_params)
        demo._sync_to_state(state)

        assert state.total_population == 7000.0
        assert state.fertility_rate == 2.5
        assert state.life_expectancy > 0


# ============================================================
# SpatialModule 测试
# ============================================================

class TestSpatialModule:
    """双区域空间模块测试."""

    def test_initialization(self, default_params):
        """测试模块正确初始化."""
        spatial = SpatialModule(default_params)
        ss = spatial.get_state()
        assert 0.15 <= ss.coastal_population_share <= 0.85
        assert ss.migration_cost_index == 0.30

    def test_labor_migration_toward_higher_wages(self, default_params, state):
        """测试劳动力向高工资区域流动."""
        spatial = SpatialModule(default_params)
        initial_coastal = spatial.get_state().coastal_population_share

        # 设置沿海高工资
        state.sectors["d_app"].W = 10.0
        state.sectors["i"].W = 3.0

        for _ in range(20):  # 5年
            spatial.step(state, dt=0.25)

        # 沿海人口的份额应上升
        assert spatial.get_state().coastal_population_share > initial_coastal

    def test_high_migration_cost_restricts_flow(self, default_params, state):
        """测试高迁移成本限制人口流动."""
        # 低迁移成本
        params_low = default_params.copy()
        params_low["migration_cost_index"] = 0.05
        spatial_low = SpatialModule(params_low)
        state_low = SystemState()
        for key in ALL_SECTORS:
            state_low.sectors[key] = state.sectors[key]
        state_low.gdp = state.gdp
        state_low.total_population = state.total_population
        state_low.sectors["d_app"].W = 10.0

        for _ in range(20):
            spatial_low.step(state_low, dt=0.25)

        # 高迁移成本
        params_high = default_params.copy()
        params_high["migration_cost_index"] = 0.70
        spatial_high = SpatialModule(params_high)
        state_high = SystemState()
        for key in ALL_SECTORS:
            state_high.sectors[key] = state.sectors[key]
        state_high.gdp = state.gdp
        state_high.total_population = state.total_population
        state_high.sectors["d_app"].W = 10.0

        for _ in range(20):
            spatial_high.step(state_high, dt=0.25)

        # 低迁移成本应有更大的人口流动
        change_low = abs(spatial_low.get_state().coastal_population_share - 0.45)
        change_high = abs(spatial_high.get_state().coastal_population_share - 0.45)
        assert change_low > change_high

    def test_agglomeration_bonus_at_high_density(self, default_params, state):
        """测试高人口集中产生集聚效应."""
        spatial = SpatialModule(default_params)
        ss = spatial.get_state()
        ss.coastal_population_share = 0.70  # 高集中度

        spatial._update_agglomeration_effects(state, dt=1.0)

        assert ss.agglomeration_bonus > 0.0

    def test_congestion_penalty_at_very_high_density(self, default_params, state):
        """测试过度集中产生拥挤成本."""
        spatial = SpatialModule(default_params)
        ss = spatial.get_state()
        ss.coastal_population_share = 0.80  # 非常高集中度

        spatial._update_agglomeration_effects(state, dt=1.0)

        assert ss.congestion_penalty > 0.0

    def test_spatial_gini_calculation(self, default_params, state):
        """测试空间基尼系数计算."""
        spatial = SpatialModule(default_params)
        ss = spatial.get_state()
        ss.coastal_population_share = 0.50
        ss.coastal_industry_share = 0.80
        ss.coastal_digital_app_share = 0.90

        spatial._update_spatial_inequality(state)

        assert ss.spatial_gini > 0.0
        assert ss.coastal_inland_income_ratio > 1.0

    def test_spatial_red_lines(self, default_params, state):
        """测试空间红线检测."""
        spatial = SpatialModule(default_params)
        ss = spatial.get_state()
        ss.coastal_inland_income_ratio = 6.0
        ss.spatial_gini = 0.45
        ss.years_high_inequality = 25.0

        warnings = spatial.check_spatial_red_lines()
        assert len(warnings) > 0
        assert any(w["type"] == "extreme_spatial_inequality" for w in warnings)

    def test_digital_agglomeration_self_reinforcing(self, default_params, state):
        """测试数字部门集聚的自我强化."""
        spatial = SpatialModule(default_params)
        initial_d_app_coastal = spatial.get_state().coastal_digital_app_share

        state.gdp = 300.0
        state.sectors["d_app"].Y = 80.0  # 数字部门占比高

        for _ in range(20):
            spatial._update_industrial_distribution(state, dt=0.25)

        # 数字app应进一步向沿海集聚
        assert spatial.get_state().coastal_digital_app_share > initial_d_app_coastal


# ============================================================
# 政策执行折扣测试
# ============================================================

class TestPolicyExecutionDiscount:
    """政策执行折扣测试."""

    def test_execution_discount_baseline(self, default_params, state):
        """测试基础执行折扣."""
        from src.models.political_economy import PoliticalEconomyModule
        pe = PoliticalEconomyModule(default_params)

        discount = pe.compute_execution_discount(state, strategy_complexity=0.0)

        assert 0.5 <= discount <= 1.0

    def test_complexity_increases_discount(self, default_params, state):
        """测试策略复杂度增加执行折扣."""
        from src.models.political_economy import PoliticalEconomyModule
        pe = PoliticalEconomyModule(default_params)

        discount_simple = pe.compute_execution_discount(state, strategy_complexity=0.0)
        discount_complex = pe.compute_execution_discount(state, strategy_complexity=0.67)

        assert discount_complex < discount_simple  # 复杂策略执行效率更低

    def test_low_institution_capacity_reduces_execution(self, default_params, state):
        """测试低制度能力降低执行效率."""
        from src.models.political_economy import PoliticalEconomyModule
        params_low = default_params.copy()
        params_low["institution_capacity"] = 0.35
        pe_low = PoliticalEconomyModule(params_low)

        params_high = default_params.copy()
        params_high["institution_capacity"] = 0.90
        pe_high = PoliticalEconomyModule(params_high)

        discount_low = pe_low.compute_execution_discount(state, strategy_complexity=0.0)
        discount_high = pe_high.compute_execution_discount(state, strategy_complexity=0.0)

        assert discount_low < discount_high


# ============================================================
# Schema 测试
# ============================================================

class TestNewParameters:
    """新参数测试."""

    def test_all_new_params_registered(self, schema):
        """测试所有新参数已在schema中注册."""
        new_params = [
            "anti_monopoly_policy",
            "data_portability_mandate",
            "platform_interoperability",
            "IC_fertility_rate",
            "IC_life_expectancy",
            "IC_urbanization_rate",
            "IC_total_population",
            "migration_cost_index",
            "coastal_productivity_premium",
            "agglomeration_strength",
            "congestion_threshold",
            "IC_coastal_industry_share",
            "IC_coastal_population_share",
            "IC_urban_rural_productivity_ratio",
            "bureaucracy_complexity_penalty",
            "policy_execution_base_efficiency",
        ]
        for name in new_params:
            assert name in schema._params, f"参数 {name} 未在schema中注册"

    def test_new_params_have_valid_domains(self, schema):
        """测试新参数的定义域合理."""
        for name in ["anti_monopoly_policy", "IC_fertility_rate", "migration_cost_index"]:
            pdef = schema.get(name)
            assert pdef.domain[0] < pdef.domain[1]
            assert pdef.domain[0] <= pdef.default <= pdef.domain[1]


# ============================================================
# State 测试
# ============================================================

class TestNewStateFields:
    """新状态字段测试."""

    def test_platform_fields_exist(self):
        """测试平台垄断字段存在."""
        state = SystemState()
        assert hasattr(state, "digital_hhi")
        assert hasattr(state, "platform_monopoly_rent")
        assert hasattr(state, "data_network_moat")

    def test_demography_fields_exist(self):
        """测试人口动态字段存在."""
        state = SystemState()
        assert hasattr(state, "total_population")
        assert hasattr(state, "working_age_pop")
        assert hasattr(state, "dependency_ratio")
        assert hasattr(state, "fertility_rate")
        assert hasattr(state, "life_expectancy")
        assert hasattr(state, "urbanization_rate")

    def test_spatial_fields_exist(self):
        """测试空间维度字段存在."""
        state = SystemState()
        assert hasattr(state, "coastal_population_share")
        assert hasattr(state, "coastal_industry_share")
        assert hasattr(state, "spatial_gini")
        assert hasattr(state, "agglomeration_bonus")
        assert hasattr(state, "congestion_penalty")

    def test_policy_execution_field_exists(self):
        """测试政策执行字段存在."""
        state = SystemState()
        assert hasattr(state, "policy_execution_discount")

    def test_field_defaults_are_sensible(self):
        """测试字段默认值合理."""
        state = SystemState()
        assert 0.0 <= state.digital_hhi <= 1.0
        assert state.fertility_rate > 0
        assert state.life_expectancy > 0
        assert 0.0 <= state.coastal_population_share <= 1.0
        assert 0.5 <= state.policy_execution_discount <= 1.0


# ============================================================
# 集成测试
# ============================================================

class TestEngineIntegration:
    """引擎集成测试 —— 验证新模块在完整模拟中不破坏现有逻辑."""

    def test_engine_initializes_with_all_modules(self, default_params):
        """测试引擎正确初始化所有新模块."""
        config = SimulationConfig(
            T_max=10.0,
            platform_enabled=True,
            demography_enabled=True,
            spatial_enabled=True,
            policy_execution_discount_enabled=True,
            evaluation_enabled=False,
            political_feedback_enabled=False,
            verbose=False,
        )
        engine = SimulationEngine(params=default_params, config=config, regime="A")

        assert engine.platform is not None
        assert engine.demography is not None
        assert engine.spatial is not None

    def test_engine_runs_with_all_modules_enabled(self, default_params):
        """测试所有新模块启用时引擎正常运行."""
        config = SimulationConfig(
            T_max=5.0,
            platform_enabled=True,
            demography_enabled=True,
            demography_update_interval=1.0,
            spatial_enabled=True,
            policy_execution_discount_enabled=True,
            evaluation_enabled=False,
            political_feedback_enabled=False,
            dt_base=0.25,
            adaptive_stepping=False,
            verbose=False,
        )
        engine = SimulationEngine(params=default_params, config=config, regime="A")
        result = engine.run()

        assert result is not None
        summary = result.get("summary", {})
        assert "survived" in summary

    def test_engine_runs_with_modules_disabled(self, default_params):
        """测试所有新模块禁用时引擎正常运行（向后兼容）."""
        config = SimulationConfig(
            T_max=5.0,
            platform_enabled=False,
            demography_enabled=False,
            spatial_enabled=False,
            policy_execution_discount_enabled=False,
            evaluation_enabled=False,
            political_feedback_enabled=False,
            verbose=False,
        )
        engine = SimulationEngine(params=default_params, config=config, regime="A")
        result = engine.run()

        assert result is not None

    def test_result_contains_new_metrics(self, default_params):
        """测试结果包含所有新模块的指标."""
        config = SimulationConfig(
            T_max=5.0,
            platform_enabled=True,
            demography_enabled=True,
            spatial_enabled=True,
            policy_execution_discount_enabled=True,
            evaluation_enabled=False,
            political_feedback_enabled=False,
            verbose=False,
        )
        engine = SimulationEngine(params=default_params, config=config, regime="A")
        result = engine.run()
        summary = result.get("summary", {})

        # 平台垄断指标
        assert "platform_metrics" in summary
        pm = summary["platform_metrics"]
        assert "digital_hhi" in pm

        # 人口动态指标
        assert "demography_metrics" in summary
        dm = summary["demography_metrics"]
        assert "total_population" in dm

        # 空间维度指标
        assert "spatial_metrics" in summary
        sm = summary["spatial_metrics"]
        assert "coastal_population_share" in sm

        # 政策执行指标
        assert "policy_execution_metrics" in summary
        pem = summary["policy_execution_metrics"]
        assert "execution_discount" in pem

    def test_mixed_strategy_with_all_modules(self, default_params):
        """测试混合策略 + 所有新模块."""
        from src.models.mixed_strategy import STANDARD_MIXTURES
        params = default_params.copy()
        params["mixed_lambda"] = STANDARD_MIXTURES["A_dominant_B"]
        params["mixed_name"] = "A_dominant_B"

        config = SimulationConfig(
            T_max=5.0,
            platform_enabled=True,
            demography_enabled=True,
            spatial_enabled=True,
            policy_execution_discount_enabled=True,
            evaluation_enabled=False,
            political_feedback_enabled=False,
            verbose=False,
        )
        engine = SimulationEngine(params=params, config=config, regime="M")
        result = engine.run()

        assert result is not None
        assert result["summary"]["is_mixed"] is True


# ============================================================
# 配置测试
# ============================================================

class TestConfigNewFields:
    """配置新字段测试."""

    def test_new_config_fields_exist(self):
        """测试配置新增字段."""
        config = SimulationConfig()
        assert hasattr(config, "platform_enabled")
        assert hasattr(config, "demography_enabled")
        assert hasattr(config, "demography_update_interval")
        assert hasattr(config, "spatial_enabled")
        assert hasattr(config, "policy_execution_discount_enabled")

    def test_new_config_defaults(self):
        """测试配置新字段默认值."""
        config = SimulationConfig()
        assert config.platform_enabled is True
        assert config.demography_enabled is True
        assert config.spatial_enabled is True
        assert config.policy_execution_discount_enabled is True
        assert config.demography_update_interval == 1.0
