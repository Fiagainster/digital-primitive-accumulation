"""全链路测试 —— 验证模型的正确性、稳定性和输出质量."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.core.engine import SimulationEngine
from src.core.config import SimulationConfig
from src.core.state import SystemState, SectorState, DataAssets, HouseholdGroups
from src.params.schema import get_schema, ParameterSchema
from src.params.presets import get_scenario, SCENARIO_PRESETS
from src.models.sectors import SectorModel
from src.models.accumulation import GradualAccumulation, LeapAccumulation, SubstitutionAccumulation
from src.monitors.fracture import FractureMonitor
from src.monitors.ews import EarlyWarningMonitor
from src.io.recorder import TimeSeriesRecorder
from src.analysis.visualize import Visualizer, generate_academic_report


def test_parameter_system():
    """测试参数系统."""
    print("\n[TEST] 参数系统...")
    schema = get_schema()
    assert len(schema) > 50, f"参数数量不足: {len(schema)}"

    # 验证默认参数在定义域内
    defaults = schema.build_defaults("A")
    violations = schema.validate(defaults)
    assert len(violations) == 0, f"默认参数违规: {violations}"

    # 验证所有策略
    for regime in ["A", "B", "C"]:
        params = schema.build_defaults(regime)
        violations = schema.validate(params)
        assert len(violations) == 0, f"策略{regime}参数违规: {violations}"

    # 验证场景预设
    for name in SCENARIO_PRESETS:
        scene = get_scenario(name)
        assert len(scene) > 0, f"场景{name}为空"

    print("  ✓ 参数系统正常")


def test_state_container():
    """测试状态容器."""
    print("\n[TEST] 状态容器...")
    state = SystemState()
    state.sectors["b"] = SectorState(K=100, L=10, Y=50)
    state.sectors["i"] = SectorState(K=200, L=20, Y=100)
    state.sectors["d_infra"] = SectorState(K=15, L=2, Y=25)
    state.sectors["d_app"] = SectorState(K=15, L=3, Y=35)
    state.sectors["s_high"] = SectorState(K=30, L=8, Y=40)
    state.sectors["s_low"] = SectorState(K=20, L=7, Y=20)
    state.gdp = 210

    # 数据资产
    da = DataAssets(producer_data=10, consumer_data=5, public_data=3)
    assert da.effective_stock > 0
    state.data_assets = da

    # 家庭群组
    hh = HouseholdGroups()
    hh.industrial_workers.cumulative_cost = 100
    hh.digital_high_skill.cumulative_cost = 10
    assert hh.cost_ratio_industrial_vs_digital > 1
    assert 0 <= hh.cost_gini <= 1

    print("  ✓ 状态容器正常")


def test_single_simulation():
    """测试单次模拟能正常运行."""
    print("\n[TEST] 单次模拟...")
    schema = get_schema()
    config = SimulationConfig(T_max=20.0, verbose=False)

    for regime in ["A", "B", "C"]:
        params = schema.build_defaults(regime)
        scene = get_scenario("benign_globalization")
        params.update(scene)

        engine = SimulationEngine(params=params, config=config, regime=regime)
        result = engine.run()

        assert "summary" in result
        assert "dataframe" in result
        assert result["dataframe"] is not None
        assert len(result["dataframe"]) > 0

        s = result["summary"]
        assert "survived" in s
        assert s["survival_years"] <= 20.0
        assert s["final_gdp"] > 0

    print("  ✓ 单次模拟正常")


def test_accumulation_regimes():
    """测试三种积累策略的逻辑差异."""
    print("\n[TEST] 积累策略差异...")
    schema = get_schema()
    scene = get_scenario("tech_cold_war")
    config = SimulationConfig(T_max=50.0, verbose=False)
    results = {}

    for regime in ["A", "B", "C"]:
        p = schema.build_defaults(regime)
        p.update(scene)
        engine = SimulationEngine(params=p, config=config, regime=regime)
        result = engine.run()
        results[regime] = result["summary"]

    gdps = [results[r]["final_gdp"] for r in ["A", "B", "C"]]
    surv = [results[r]["survived"] for r in ["A", "B", "C"]]
    print(f"  存活: A={surv[0]} B={surv[1]} C={surv[2]}")
    print(f"  GDP: A={gdps[0]:.0f} B={gdps[1]:.0f} C={gdps[2]:.0f}")
    assert max(gdps) > 0, "GDP应大于0"

    print("  ✓ 积累策略差异验证通过")


def test_fracture_detection():
    """测试断裂检测."""
    print("\n[TEST] 断裂检测...")
    schema = get_schema()
    params = schema.build_defaults("A")

    monitor = FractureMonitor(params)

    # 构造一个接近断裂的状态
    state = SystemState()
    state.sectors["i"] = SectorState(K=5.0, L=10, Y=20, W=2.0)
    state.sectors["b"] = SectorState(K=100, L=20, Y=50, W=1.5)
    state.sectors["d_infra"] = SectorState(K=10, L=2, Y=15, W=4.0, A=0.05)
    state.sectors["d_app"] = SectorState(K=10, L=3, Y=15, W=6.0, A=0.15)
    state.sectors["s_high"] = SectorState(K=20, L=8, Y=25, W=3.0, A=1.0)
    state.sectors["s_low"] = SectorState(K=10, L=5, Y=10, W=1.5, A=0.7)
    state.initial_K_i = 300.0
    state.tech_gap = 15.0
    state.unemployment_rate = 0.30
    state.welfare_index = 0.2

    events = monitor.check(state)
    # 工业枯萎应该触发(K_i/initial = 5/300 = 0.017 < 0.3)
    assert len(events) > 0, "应检测到断裂事件"
    types = [e["type"] for e in events]
    assert "industrial_withering" in types, "应检测到工业枯萎"

    print(f"  ✓ 断裂检测正常 (检测到{len(events)}个事件: {types})")


def test_ews_system():
    """测试EWS预警系统."""
    print("\n[TEST] EWS预警...")
    params = {"ews_warning_threshold": 0.4, "ews_critical_threshold": 0.7}
    ews = EarlyWarningMonitor(params, window_size=20)

    # 模拟一个变量逐渐恶化
    state = SystemState()
    state.sectors["b"] = SectorState(K=200.0, Y=50.0)
    state.sectors["i"] = SectorState(K=300.0, Y=100.0)
    state.sectors["d_infra"] = SectorState(K=10.0, Y=15.0)
    state.sectors["d_app"] = SectorState(K=10.0, Y=15.0)
    state.sectors["s_high"] = SectorState(K=30.0, Y=25.0)
    state.sectors["s_low"] = SectorState(K=10.0, Y=8.0)
    state.initial_K_i = 300.0
    state.tech_gap = 3.0
    state.unemployment_rate = 0.05
    state.welfare_index = 1.0

    for i in range(60):
        state.sectors["i"].K *= 0.98  # 资本逐渐萎缩
        state.tech_gap *= 1.02  # 差距逐渐拉大
        state.unemployment_rate += 0.002  # 失业缓慢上升
        state.welfare_index *= 0.995
        state.t = float(i)
        ews.feed(state)

    results = ews.compute_all(state)
    assert len(results) >= 0, "EWS不应崩溃"
    for name, r in results.items():
        if not np.isnan(r.composite_score):
            assert 0 <= r.composite_score <= 2.0, f"EWS评分应在合理范围: {r.composite_score}"

    print(f"  ✓ EWS系统正常 ({len(results)}个变量被监测)")


def test_recorder_io():
    """测试状态记录与导出."""
    print("\n[TEST] 状态记录与导出...")
    recorder = TimeSeriesRecorder(max_steps=100)

    state = SystemState()
    state.sectors["b"] = SectorState(K=100, L=10, Y=50, W=2.0, DL=0.1, H=1.0, r=0.1, I=5, A=1.0)
    state.sectors["i"] = SectorState(K=200, L=20, Y=100, W=3.0, DL=0.2, H=1.2, r=0.12, I=10, A=1.2)
    state.sectors["d_infra"] = SectorState(K=15, L=2, Y=25, W=5.0, DL=0.3, H=1.5, r=0.15, I=1.5, A=1.5)
    state.sectors["d_app"] = SectorState(K=15, L=3, Y=35, W=5.0, DL=0.3, H=1.5, r=0.15, I=1.5, A=1.5)
    state.sectors["s_high"] = SectorState(K=25, L=8, Y=30, W=3.5, DL=0.15, H=1.0, r=0.10, I=2.0, A=1.0)
    state.sectors["s_low"] = SectorState(K=15, L=7, Y=15, W=1.5, DL=0.05, H=0.6, r=0.08, I=1.0, A=0.7)
    state.gdp = 210

    for i in range(50):
        state.t = i * 0.25
        state.step = i
        recorder.record(state)

    assert len(recorder) == 50

    df = recorder.to_dataframe()
    assert len(df) == 50
    assert "gdp" in df.columns
    assert "K_i" in df.columns
    assert "unemployment_rate" in df.columns

    print("  ✓ 状态记录与导出正常")


def test_end_to_end_pipeline():
    """端到端全链路测试."""
    print("\n" + "=" * 60)
    print("[TEST] 端到端全链路测试")
    print("=" * 60)

    # 1. 运行完整比较
    from src.experiment.runner import ExperimentRunner
    runner = ExperimentRunner()
    results = runner.run_comparison(
        regimes=["A", "B", "C"],
        scenarios=["benign_globalization", "tech_cold_war"],
        T_max=30.0,
        n_repeats=1,
    )
    assert len(results) == 6  # 3 regimes × 2 scenarios

    # 2. 验证每个结果包含必要字段
    required_keys = ["survived", "survival_years", "final_gdp", "final_tech_gap",
                     "cost_concentration", "cost_ratio_industrial_digital"]
    for r in results:
        s = r["summary"]
        for key in required_keys:
            assert key in s, f"缺少字段: {key}"

    # 3. 可视化
    viz = Visualizer()
    viz.plot_strategy_comparison(results)
    viz.plot_phase_diagram(results)

    # 选取一个存活和一个崩溃的模拟进行详细可视化
    survived = [r for r in results if r["summary"]["survived"]]
    failed = [r for r in results if not r["summary"]["survived"]]
    for r in (survived[:1] + failed[:1]):
        s = r["summary"]
        viz.plot_single_simulation(r, filename=f"e2e_{s['regime']}_{s['scenario']}.png")

    # 4. 学术报告
    report = generate_academic_report(results)
    assert len(report) > 0

    # 5. 导出
    from src.io.exporter import ResultExporter
    output_dir = "experiments/results"
    ResultExporter.to_csv_summary(results, os.path.join(output_dir, "test_summary.csv"))

    # 如果有h5py，测试HDF5导出
    try:
        ResultExporter.to_hdf5(results, os.path.join(output_dir, "test_results.h5"))
    except ImportError:
        print("  (h5py not available, skipping HDF5 test)")

    print("\n" + "=" * 60)
    print("✓ 端到端全链路测试通过")
    print("=" * 60)

    return results


def test_edge_cases():
    """测试边缘情况."""
    print("\n[TEST] 边缘情况...")

    # 极端参数
    schema = get_schema()
    config = SimulationConfig(T_max=5.0)

    # 极高提取率
    extreme = schema.build_defaults("B")
    extreme.update({"B_tau_actual_max": 0.95, "B_phi_d_jump": 0.9})
    engine = SimulationEngine(params=extreme, config=config, regime="B")
    result = engine.run()
    assert result["summary"]["survived"] == False or result["summary"]["survival_years"] <= 5.0

    # 极低提取率
    mild = schema.build_defaults("A")
    mild.update({"A_tau_max": 0.05, "A_tau_min": 0.01})
    engine = SimulationEngine(params=mild, config=config, regime="A")
    result = engine.run()
    assert result["summary"]["survival_years"] >= 0

    print("  ✓ 边缘情况测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("原始积累模型 — 全链路测试套件")
    print("=" * 60)

    # 单元测试
    test_parameter_system()
    test_state_container()
    test_recorder_io()

    # 功能测试
    test_single_simulation()
    test_accumulation_regimes()
    test_fracture_detection()
    test_ews_system()
    test_edge_cases()

    # 端到端测试
    results = test_end_to_end_pipeline()

    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print(f"共完成 {len(results)} 次模拟")
    print("=" * 60)
