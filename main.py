"""主入口 —— 命令行接口 (v3.7 升级).

v3.7: 新增 --config 参数，支持从YAML配置文件加载所有参数和权重。
"""

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


def parse_profile_arg() -> str:
    """从命令行参数中解析 --profile."""
    for i, arg in enumerate(sys.argv):
        if arg == "--profile" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "manufacturing_giant"  # 默认


def parse_config_arg() -> str:
    """从命令行参数中解析 --config (YAML配置文件路径)."""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def run_single_demo(profile: str = "manufacturing_giant",
                    config_path: str = None):
    """运行单次模拟演示 (v3.7 含YAML配置支持)."""
    print("=" * 60)
    config_tag = f" | 配置: {config_path}" if config_path else ""
    print(f"原始积累模型 v3.7 — 国情: {profile}{config_tag}")
    print("=" * 60)

    schema = get_schema()
    sim_config = SimulationConfig(T_max=60.0, verbose=False)

    for regime in ["A", "B", "C"]:
        print(f"\n--- 策略 {regime} ---")
        # v3.7: 如果有YAML配置，使用YAML加载
        if config_path:
            params = schema.build_from_yaml(
                config_path=config_path,
                profile_name=profile,
                regime=regime,
                scenario="benign_globalization",
            )
        else:
            params = schema.build_from_profile(
                profile_name=profile,
                regime=regime,
                scenario="benign_globalization",
            )

        engine = SimulationEngine(
            params=params, config=sim_config, regime=regime,
            config_path=config_path,
        )
        result = engine.run()
        s = result["summary"]

        status = "[OK] survived" if s["survived"] else f"[FAIL] collapsed [{s['terminal_fracture']}]"
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
                print(f"    [FAIL] red line failures: {', '.join(ev['red_line_failures'])}")
            if ev.get("red_line_warnings"):
                print(f"    ⚡ 警告: {', '.join(ev['red_line_warnings'])}")

        print(f"  失业率峰值: {s['peak_unemployment']:.3f} | "
              f"代价基尼: {s['cost_concentration']:.3f}")
        print(f"  工业/数字代价比: {s['cost_ratio_industrial_digital']:.1f}")
        print(f"  断裂: {s['total_fractures']}个 {s['fracture_types']}")

        # v3.5: 六部门结构指标
        print(f"  === 六部门结构 (v3.5) ===")
        print(f"  K_d_infra: {s.get('sector_K_d_infra', 'N/A'):.1f} | K_d_app: {s.get('sector_K_d_app', 'N/A'):.1f}")
        print(f"  DL_d_infra: {s.get('sector_DL_d_infra', 'N/A'):.3f} | DL_d_app: {s.get('sector_DL_d_app', 'N/A'):.3f}")
        print(f"  K_s_high: {s.get('sector_K_s_high', 'N/A'):.1f} | K_s_low: {s.get('sector_K_s_low', 'N/A'):.1f}")
        # v3.5: 新指标
        v35 = s.get('v35_metrics', {})
        print(f"  就业质量指数: {v35.get('employment_quality_index', 'N/A')}")
        print(f"  技术差距(基建): {v35.get('tech_gap_infra', 'N/A')} | 技术差距(应用): {v35.get('tech_gap_app', 'N/A')}")
        print(f"  依赖综合指数: {s.get('dependency_composite', 'N/A')}")
        if v35:
            print(f"  v3.5综合指标: {v35}")

        # 可视化
        viz = Visualizer()
        viz.plot_single_simulation(result, f"策略{regime} — 模拟结果",
                                   filename=f"demo_regime_{regime}.png")

    # v3.0: 生成五维雷达图
    all_results = []
    for regime in ["A", "B", "C"]:
        if config_path:
            params = schema.build_from_yaml(
                config_path=config_path,
                profile_name=profile,
                regime=regime,
                scenario="benign_globalization",
            )
        else:
            params = schema.build_from_profile(
                profile_name=profile,
                regime=regime,
                scenario="benign_globalization",
            )
        engine = SimulationEngine(
            params=params, config=sim_config, regime=regime,
            config_path=config_path,
        )
        result = engine.run()
        all_results.append(result)

    viz = Visualizer()
    viz.plot_five_dimension_radar(all_results)
    viz.plot_cost_waterfall(all_results)

    print("\n[Done] Single simulation demo completed. Figures saved to experiments/results/figures/")


def run_mixed_demo(profile: str = "manufacturing_giant",
                   config_path: str = None):
    """v3.7: 混合策略演示（支持YAML配置）."""
    print("\n" + "=" * 60)
    config_tag = f" | 配置: {config_path}" if config_path else ""
    print(f"原始积累模型 v3.7 — 混合策略演示 | 国情: {profile}{config_tag}")
    print("=" * 60)

    schema = get_schema()
    sim_config = SimulationConfig(T_max=50.0, verbose=False)

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
        # 使用国情预设构建参数 (v3.7: YAML优先)
        if config_path:
            params = schema.build_from_yaml(
                config_path=config_path,
                profile_name=profile,
                regime="A" if base_regime == "M" else base_regime,
                scenario="benign_globalization",
            )
        else:
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
        engine = SimulationEngine(
            params=params, config=sim_config, regime=regime_str,
            config_path=config_path,
        )
        result = engine.run()
        s = result["summary"]

        status = "[OK] survived" if s["survived"] else f"[FAIL] collapsed [{s['terminal_fracture']}]"
        print(f"  {status} | {s['survival_years']:.1f}年")

        ev = s.get("evaluation", {})
        if ev.get("enabled"):
            print(f"  CDI={ev['cdi']:.3f} | E={ev['economy']:.2f} "
                  f"L={ev['livelihood']:.2f} R={ev['resilience']:.2f} "
                  f"S={ev['structure']:.2f} V={ev['sovereignty']:.2f}")
            print(f"  红线: {ev['red_line_level']} | 可接受: {ev['is_acceptable']}")

        # v3.5: 六部门结构指标
        print(f"  === 六部门结构 (v3.5) ===")
        print(f"  K_d_infra: {s.get('sector_K_d_infra', 'N/A'):.1f} | K_d_app: {s.get('sector_K_d_app', 'N/A'):.1f}")
        print(f"  DL_d_infra: {s.get('sector_DL_d_infra', 'N/A'):.3f} | DL_d_app: {s.get('sector_DL_d_app', 'N/A'):.3f}")
        print(f"  K_s_high: {s.get('sector_K_s_high', 'N/A'):.1f} | K_s_low: {s.get('sector_K_s_low', 'N/A'):.1f}")
        # v3.5: 新指标
        v35 = s.get('v35_metrics', {})
        print(f"  就业质量指数: {v35.get('employment_quality_index', 'N/A')}")
        print(f"  技术差距(基建): {v35.get('tech_gap_infra', 'N/A')} | 技术差距(应用): {v35.get('tech_gap_app', 'N/A')}")
        print(f"  依赖综合指数: {s.get('dependency_composite', 'N/A')}")
        if v35:
            print(f"  v3.5综合指标: {v35}")

        results.append(result)

    # 雷达图比较
    viz = Visualizer()
    viz.plot_five_dimension_radar(results)
    viz.plot_cost_waterfall(results)

    # 生成报告
    generate_academic_report(results)

    print("\n[Done] Mixed strategy demo completed.")


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
    viz.plot_five_dimension_radar(results)  # v3.5: 五维雷达图
    viz.plot_cost_waterfall(results)        # v3.5: 代价瀑布图

    # 所有模拟的时间序列
    for r in results:
        s = r["summary"]
        viz.plot_single_simulation(r, filename=f"comp_{s['regime']}_{s.get('scenario','?')}.png")

    # 学术报告
    report = generate_academic_report(results)

    print("\n[Done] Strategy comparison demo completed.")
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
    viz.plot_five_dimension_radar(results)  # v3.5

    generate_academic_report(results)

    print("\n[Done] Batch experiment completed.")
    return results


def run_profiles_command():
    """v3.5: 列出所有国情预设."""
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
    else:
        cmd = "demo"

    # v3.7: 解析YAML配置文件路径
    config_path = parse_config_arg()
    profile = parse_profile_arg()

    if config_path:
        print(f"[config] 从YAML加载参数: {config_path}")

    if cmd == "demo":
        run_single_demo(profile=profile, config_path=config_path)
    elif cmd == "compare":
        run_comparison_demo()
    elif cmd == "batch":
        run_batch_experiment()
    elif cmd == "mixed":
        run_mixed_demo(profile=profile, config_path=config_path)
    elif cmd == "profiles":
        run_profiles_command()
    else:
        print(f"Usage: python main.py [demo|compare|batch|mixed|profiles] [options]")
        print(f"  demo     - 单次模拟演示（三种纯策略 + 五维评价 + 六部门结构）")
        print(f"             --profile <name>  指定国情预设（默认: manufacturing_giant）")
        print(f"             --config <path>   v3.7: 从YAML加载参数配置")
        print(f"  compare  - 策略×场景交叉比较")
        print(f"  batch    - LHS批量实验")
        print(f"  mixed    - v3.7 混合策略演示（纯策略+混合策略）")
        print(f"             --profile <name>  指定国情预设")
        print(f"             --config <path>   v3.7: 从YAML加载参数配置")
        print(f"  profiles - v3.7 列出所有国情预设")
