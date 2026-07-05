"""主入口 —— 命令行接口 (v3.1 升级)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.core.engine import SimulationEngine
from src.core.config import SimulationConfig
from src.params.schema import get_schema
from src.params.presets import get_scenario
from src.params.country_profiles import (
    list_country_profiles, get_country_profile, get_profile_names,
)
from src.experiment.runner import ExperimentRunner
from src.analysis.visualize import Visualizer, generate_academic_report
from src.models.mixed_strategy import STANDARD_MIXTURES, MixedStrategy


def run_single_demo(profile: str = "manufacturing_giant"):
    """运行单次模拟演示 (v3.1 含国情预设)."""
    print("=" * 60)
    print(f"原始积累模型 v3.1 — 国情: {profile}")
    print("=" * 60)

    schema = get_schema()
    config = SimulationConfig(T_max=60.0, verbose=False)

    for regime in ["A", "B", "C"]:
        print(f"\n--- 策略 {regime} ---")
        params = schema.build_from_profile(
            profile_name=profile,
            regime=regime,
            scenario="benign_globalization",
        )

        engine = SimulationEngine(params=params, config=config, regime=regime)
        result = engine.run()
        s = result["summary"]

        status = "✓ 存活" if s["survived"] else f"✗ 崩溃 [{s['terminal_fracture']}]"
        print(f"  {status} | {s['survival_years']:.1f}年")
        print(f"  GDP: {s['final_gdp']:.0f} | 技术差距: {s['final_tech_gap']:.2f}")

        # v3.0: 五维评价
        ev = s.get("evaluation", {})
        if ev.get("enabled"):
            print(f"  === 五维评价 (v3.0) ===")
            print(f"  CDI={ev['cdi']:.3f} | 经济={ev['economy']:.2f} "
                  f"民生={ev['livelihood']:.2f} 韧性={ev['resilience']:.2f} "
                  f"结构={ev['structure']:.2f} 主权={ev['sovereignty']:.2f}")
            print(f"  红线: {ev['red_line_level']} | 可接受: {ev['is_acceptable']}")
            if ev.get("red_line_failures"):
                print(f"    ✗ 失败: {', '.join(ev['red_line_failures'])}")
            if ev.get("red_line_warnings"):
                print(f"    ⚡ 警告: {', '.join(ev['red_line_warnings'])}")

        print(f"  失业率峰值: {s['peak_unemployment']:.3f} | "
              f"代价基尼: {s['cost_concentration']:.3f}")
        print(f"  工业/数字代价比: {s['cost_ratio_industrial_digital']:.1f}")
        print(f"  断裂: {s['total_fractures']}个 {s['fracture_types']}")

        # 可视化
        viz = Visualizer()
        viz.plot_single_simulation(result, f"策略{regime} — 模拟结果",
                                   filename=f"demo_regime_{regime}.png")

    # v3.0: 生成五维雷达图
    all_results = []
    for regime in ["A", "B", "C"]:
        params = schema.build_from_profile(
            profile_name=profile,
            regime=regime,
            scenario="benign_globalization",
        )
        engine = SimulationEngine(params=params, config=config, regime=regime)
        result = engine.run()
        all_results.append(result)

    viz = Visualizer()
    viz.plot_five_dimension_radar(all_results)
    viz.plot_cost_waterfall(all_results)

    print("\n✓ 单次模拟演示完成。图表已保存至 experiments/results/figures/")


def run_mixed_demo(profile: str = "manufacturing_giant"):
    """v3.1: 混合策略演示（支持国情预设）."""
    print("\n" + "=" * 60)
    print(f"原始积累模型 v3.1 — 混合策略演示 | 国情: {profile}")
    print("=" * 60)

    schema = get_schema()
    config = SimulationConfig(T_max=50.0, verbose=False)

    # 选取关键混合配比
    mixtures_to_test = [
        ("pure_A", "A"),
        ("pure_B", "B"),
        ("pure_C", "C"),
        ("B_dominant_A", "M"),
        ("B_dominant_C", "M"),
        ("C_dominant_A", "M"),
        ("A_B_balanced", "M"),
    ]

    results = []
    for name, base_regime in mixtures_to_test:
        regime_str = base_regime if base_regime != "M" else f"M_{name}"
        # 使用国情预设构建参数
        params = schema.build_from_profile(
            profile_name=profile,
            regime="A" if base_regime == "M" else base_regime,
            scenario="benign_globalization",
        )

        if base_regime == "M":
            mixture = STANDARD_MIXTURES.get(name, (1/3, 1/3, 1/3))
            params["mixed_lambda"] = mixture
            params["mixed_name"] = name
        else:
            mixture = None

        print(f"\n--- {name} (λ={mixture if mixture else '纯'}) ---")
        engine = SimulationEngine(params=params, config=config, regime=regime_str)
        result = engine.run()
        s = result["summary"]

        status = "✓ 存活" if s["survived"] else f"✗ 崩溃 [{s['terminal_fracture']}]"
        print(f"  {status} | {s['survival_years']:.1f}年")

        ev = s.get("evaluation", {})
        if ev.get("enabled"):
            print(f"  CDI={ev['cdi']:.3f} | E={ev['economy']:.2f} "
                  f"L={ev['livelihood']:.2f} R={ev['resilience']:.2f} "
                  f"S={ev['structure']:.2f} V={ev['sovereignty']:.2f}")
            print(f"  红线: {ev['red_line_level']} | 可接受: {ev['is_acceptable']}")

        results.append(result)

    # 雷达图比较
    viz = Visualizer()
    viz.plot_five_dimension_radar(results)
    viz.plot_cost_waterfall(results)

    # 生成报告
    generate_academic_report(results)

    print("\n✓ 混合策略演示完成。")


def run_comparison_demo():
    """运行策略比较演示."""
    print("\n" + "=" * 60)
    print("原始积累模型 — 策略比较演示")
    print("=" * 60)

    runner = ExperimentRunner()
    results = runner.run_comparison(
        regimes=["A", "B", "C"],
        scenarios=[
            "benign_globalization", "tech_cold_war", "resource_crash",
            "full_containment", "tech_explosion", "resource_war",
        ],
        T_max=50.0,
        n_repeats=1,
    )

    # 可视化比较
    viz = Visualizer()
    viz.plot_strategy_comparison(results)
    viz.plot_five_dimension_radar(results)  # v3.0: 五维雷达图
    viz.plot_cost_waterfall(results)        # v3.0: 代价瀑布图

    # 所有模拟的时间序列
    for r in results:
        s = r["summary"]
        viz.plot_single_simulation(r, filename=f"comp_{s['regime']}_{s.get('scenario','?')}.png")

    # 学术报告
    report = generate_academic_report(results)

    print("\n✓ 策略比较演示完成。")
    return results


def run_batch_experiment():
    """运行批量实验（LHS采样）."""
    print("\n" + "=" * 60)
    print("原始积累模型 — 批量实验")
    print("=" * 60)

    runner = ExperimentRunner()
    results = runner.run_experiment(
        name="screening_run",
        regimes=["A", "B", "C"],
        n_samples=50,
        scenario="benign_globalization",
        T_max=50.0,
    )

    viz = Visualizer()
    viz.plot_strategy_comparison(results)
    viz.plot_phase_diagram(results, "final_tech_gap", "survival_years")
    viz.plot_phase_diagram(results, "cost_concentration", "survival_years")
    viz.plot_five_dimension_radar(results)  # v3.0

    generate_academic_report(results)

    print("\n✓ 批量实验完成。")
    return results


def run_comparison_demo():
    """运行策略比较演示."""
    print("\n" + "=" * 60)
    print("原始积累模型 — 策略比较演示")
    print("=" * 60)

    runner = ExperimentRunner()
    results = runner.run_comparison(
        regimes=["A", "B", "C"],
        scenarios=[
            "benign_globalization", "tech_cold_war", "resource_crash",
            "full_containment", "tech_explosion", "resource_war",
        ],
        T_max=50.0,
        n_repeats=1,
    )

    # 可视化比较
    viz = Visualizer()
    viz.plot_strategy_comparison(results)

    # 所有模拟的时间序列
    for r in results:
        s = r["summary"]
        viz.plot_single_simulation(r, filename=f"comp_{s['regime']}_{s['scenario']}.png")

    # 学术报告
    report = generate_academic_report(results)

    print("\n✓ 策略比较演示完成。")
    return results


def run_batch_experiment():
    """运行批量实验（LHS采样）."""
    print("\n" + "=" * 60)
    print("原始积累模型 — 批量实验")
    print("=" * 60)

    runner = ExperimentRunner()
    results = runner.run_experiment(
        name="screening_run",
        regimes=["A", "B", "C"],
        n_samples=50,
        scenario="benign_globalization",
        T_max=50.0,
    )

    viz = Visualizer()
    viz.plot_strategy_comparison(results)
    viz.plot_phase_diagram(results, "final_tech_gap", "survival_years")
    viz.plot_phase_diagram(results, "cost_concentration", "survival_years")

    generate_academic_report(results)

    print("\n✓ 批量实验完成。")
    return results


def run_profiles_command():
    """v3.1: 列出所有国情预设."""
    print("=" * 70)
    print("国情预设列表 (Country Profiles)")
    print("=" * 70)
    profiles = list_country_profiles()
    for p in profiles:
        print(f"\n  [{p['name']}] {p['label']}")
        print(f"    典型参照: {p['typical_references']}")
        print(f"    {p['description']}")
        print(f"    自然策略倾向: {p['natural_strategy']}")
        print(f"    特有风险: {p['strategy_risks']}")
        print(f"    {p['summary']}")
    print("\n" + "=" * 70)
    print("使用方式: python main.py demo --profile <name>")
    print("          python main.py compare --profiles <name1>,<name2>,...")


def parse_profile_arg() -> str:
    """从命令行参数中解析 --profile."""
    for i, arg in enumerate(sys.argv):
        if arg == "--profile" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "manufacturing_giant"  # 默认


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
    else:
        cmd = "demo"

    if cmd == "demo":
        profile = parse_profile_arg()
        run_single_demo(profile=profile)
    elif cmd == "compare":
        run_comparison_demo()
    elif cmd == "batch":
        run_batch_experiment()
    elif cmd == "mixed":
        run_mixed_demo()  # v3.0: 混合策略演示
    elif cmd == "profiles":
        run_profiles_command()  # v3.1: 列出国情预设
    else:
        print(f"Usage: python main.py [demo|compare|batch|mixed|profiles]")
        print(f"  demo     - 单次模拟演示（三种纯策略 + 五维评价）")
        print(f"             --profile <name>  指定国情预设（默认: manufacturing_giant）")
        print(f"  compare  - 策略×场景交叉比较")
        print(f"  batch    - LHS批量实验")
        print(f"  mixed    - v3.0 混合策略演示（纯策略+混合策略）")
        print(f"  profiles - v3.1 列出所有国情预设")
