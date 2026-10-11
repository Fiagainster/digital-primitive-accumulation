#!/usr/bin/env python
"""
=============================================================
  第四阶段：机制深挖 (Mechanism Deep Dive)
=============================================================

研究问题:
    在前三阶段识别的关键区域中，深入分析:
    1. 代价在不同群体间的分布（谁承担代价？）
    2. 金融不稳定性传导（资产泡沫/资本外逃/信贷错配的联动）
    3. 数字依附的自我强化与逆转条件
    4. 混合策略能否克服纯策略的结构性缺陷？

实验设计:
    - 代价分布: LHS采样，追踪五组家庭时间序列
    - 金融稳定性: LHS采样，关注金融指标的联动
    - 依赖动态: 参数扫描（制度能力×封锁强度）
    - 混合策略: 全谱系混合配比比较

预计耗时: 40-90 分钟

使用方法:
    python run_phase4_mechanisms.py                        # 默认全部运行
    python run_phase4_mechanisms.py --skip cost            # 跳过代价分布
    python run_phase4_mechanisms.py --quick                # 快速模式
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from shared_config import (
    print_header, save_phase_results,
    phase_started, phase_completed, load_phase_summary,
    RESULT_DIR, PHASE4_N_LHS, PHASE4_T_MAX,
    get_timestamp,
)

_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from research_scripts.common.runner import (
    quick_run, build_param_sets, param_grid_explore, param_2d_grid,
    BatchRunner, get_optimal_workers, ALL_MIXTURES,
)
from src.core.config import SimulationConfig


def run_phase4(skip=None, T_max=None, n_workers=None, quick=False):
    """
    执行第四阶段机制深挖.

    Args:
        skip: 要跳过的分析 (如 "cost,dependency")
        T_max: 模拟年限
        n_workers: 并行worker数
        quick: 快速模式
    """
    skip = set(skip.split(",")) if skip else set()
    T_max = T_max or (60 if quick else PHASE4_T_MAX)
    n_lhs = 20 if quick else PHASE4_N_LHS

    print_header("第四阶段：机制深挖")
    print(f"  T_max = {T_max}, LHS样本数 = {n_lhs}")
    print(f"  跳过: {skip if skip else '无'}")
    print()

    record = phase_started("phase4_mechanisms")
    run_id = f"phase4_{get_timestamp()}"
    run_dir = RESULT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    start_time = time.time()

    # ---- 4a. 代价分布分析 ----
    if "cost" not in skip:
        all_results["cost_distribution"] = _run_cost_distribution(
            T_max, n_lhs, n_workers, quick)
        _print_section_summary("代价分布", all_results["cost_distribution"])

    # ---- 4b. 金融稳定性联动 ----
    if "financial" not in skip:
        all_results["financial_stability"] = _run_financial_stability(
            T_max, n_lhs, n_workers, quick)
        _print_section_summary("金融稳定性", all_results["financial_stability"])

    # ---- 4c. 数字依附动态 ----
    if "dependency" not in skip:
        all_results["dependency_dynamics"] = _run_dependency_dynamics(
            T_max, n_workers, quick)
        _print_section_summary("数字依附动态", all_results["dependency_dynamics"])

    # ---- 4d. 混合策略比较 ----
    if "mixed" not in skip:
        all_results["mixed_strategies"] = _run_mixed_strategy_analysis(
            T_max, n_workers, quick)
        _print_section_summary("混合策略", all_results["mixed_strategies"])

    # ---- 汇总 ----
    total_elapsed = time.time() - start_time
    analysis = _analyze_phase4(all_results)

    save_phase_results("phase4_mechanisms", {
        "run_id": run_id,
        "parameters": {"T_max": T_max, "n_lhs": n_lhs, "quick": quick},
        "elapsed_minutes": total_elapsed / 60,
        "results": all_results,
        "analysis": analysis,
    }, f"{run_id}_results.json")

    phase_completed("phase4_mechanisms", analysis)
    _print_phase4_report(analysis)

    print(f"\n  总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"  结果目录: {run_dir}")
    return all_results, analysis


def _run_cost_distribution(T_max, n_lhs, n_workers, quick):
    """代价分布分析."""
    print("\n  [4a] 代价分布分析...")
    key_combos = [
        ("A", "manufacturing_giant", "benign_globalization"),
        ("B", "digital_leapfrogger", "benign_globalization"),
        ("B", "digital_leapfrogger", "tech_cold_war"),
        ("C", "resource_rich", "benign_globalization"),
    ]

    results = {}
    for regime, profile, scenario in key_combos:
        combo = f"{regime}_{profile}_{scenario}"
        print(f"    {combo} ...", end=" ", flush=True)

        base = build_param_sets(regime=regime, profile=profile, scenario=scenario)
        # 对成本相关参数做LHS采样
        cost_params = [
            "B_bubble_sensitivity", "capital_flight_sensitivity",
            "C_consumption_trap", "C_dutch_disease",
            "A_incumbent_resistance", "A_middle_income_threshold",
            "welfare_share", "social_security_level",
        ]

        rng = np.random.default_rng(hash(combo) % (2**31))
        param_sets = []
        for _ in range(n_lhs):
            p = base.copy()
            for pid in cost_params:
                if pid in p:
                    p[pid] = rng.uniform(0.2, 0.9)
            param_sets.append(p)

        config_obj = SimulationConfig(T_max=T_max)
        runner = BatchRunner(n_workers=n_workers)
        sim_results = runner.run(param_sets, regime=regime, config=config_obj, desc=combo)

        # 提取代价分布指标
        cost_metrics = []
        for r in sim_results:
            summ = r.get("summary", {})
            cost_metrics.append({
                "survived": summ.get("survived", False),
                "survival_years": summ.get("survival_years", 0),
                "employment_quality": summ.get("employment_quality_index"),
                "cost_concentration": summ.get("cost_concentration"),
                "gini": summ.get("gini"),
                "welfare_share": summ.get("welfare_share"),
            })

        results[combo] = cost_metrics
        survived = sum(1 for m in cost_metrics if m["survived"])
        print(f"存活率={survived}/{len(cost_metrics)}")

    return {"combos_tested": list(results.keys()), "metrics": results}


def _run_financial_stability(T_max, n_lhs, n_workers, quick):
    """金融稳定性联动分析."""
    print("\n  [4b] 金融稳定性联动分析...")
    key_combos = [
        ("A", "manufacturing_giant", "benign_globalization"),
        ("B", "digital_leapfrogger", "benign_globalization"),
        ("C", "resource_rich", "resource_crash"),
    ]

    results = {}
    for regime, profile, scenario in key_combos:
        combo = f"{regime}_{profile}_{scenario}"
        print(f"    {combo} ...", end=" ", flush=True)

        base = build_param_sets(regime=regime, profile=profile, scenario=scenario)
        fin_params = [
            "capital_flight_sensitivity", "B_bubble_sensitivity",
            "credit_allocation_bias", "financial_repression",
            "exchange_rate_regime",
        ]

        rng = np.random.default_rng(hash(combo + "_fin") % (2**31))
        param_sets = []
        for _ in range(n_lhs):
            p = base.copy()
            for pid in fin_params:
                if pid in p:
                    p[pid] = rng.uniform(0.2, 0.9)
            param_sets.append(p)

        config_obj = SimulationConfig(T_max=T_max)
        runner = BatchRunner(n_workers=n_workers)
        sim_results = runner.run(param_sets, regime=regime, config=config_obj, desc=combo)

        fin_metrics = []
        for r in sim_results:
            summ = r.get("summary", {})
            ts = r.get("time_series", {})
            fin_metrics.append({
                "survived": summ.get("survived", False),
                "asset_bubble": summ.get("asset_bubble"),
                "capital_flight": summ.get("capital_flight"),
                "credit_mismatch": summ.get("credit_mismatch"),
                "employment_quality": summ.get("employment_quality_index"),
            })

        results[combo] = fin_metrics
        survived = sum(1 for m in fin_metrics if m["survived"])
        print(f"存活率={survived}/{len(fin_metrics)}")

    # 检测联动模式
    linkages = _detect_financial_linkages(results)
    return {"combos_tested": list(results.keys()), "metrics": results, "linkages": linkages}


def _detect_financial_linkages(results):
    """检测金融变量之间的联动."""
    linkages = []
    # 简化：检查泡沫和资本外逃是否同时出现
    for combo, metrics in results.items():
        bubble_flight_corr = np.corrcoef(
            [m.get("asset_bubble", 0) or 0 for m in metrics],
            [m.get("capital_flight", 0) or 0 for m in metrics],
        )[0, 1]
        if abs(bubble_flight_corr) > 0.3:
            linkages.append({
                "combo": combo,
                "type": "bubble_flight",
                "correlation": float(bubble_flight_corr),
            })
    return linkages


def _run_dependency_dynamics(T_max, n_workers, quick):
    """数字依附动态."""
    print("\n  [4c] 数字依附动态分析...")
    # 扫描制度能力 × 封锁强度 对依附的影响
    combos = [
        ("B", "digital_leapfrogger", "tech_cold_war"),
        ("C", "resource_rich", "tech_cold_war"),
    ]

    results = {}
    for regime, profile, scenario in combos:
        combo = f"{regime}_{profile}_{scenario}"
        print(f"    {combo} ...", flush=True)

        base = build_param_sets(regime=regime, profile=profile, scenario=scenario)

        ic_values = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        block_values = [0.0, 0.20, 0.40, 0.60, 0.80, 0.95]
        param_sets = param_2d_grid(
            base, "institution_capacity", ic_values,
            "block_max", block_values,
        )

        config_obj = SimulationConfig(T_max=T_max)
        runner = BatchRunner(n_workers=n_workers)
        sim_results = runner.run(param_sets, regime=regime, config=config_obj, desc=combo)

        # 聚合为二维矩阵
        dep_matrix = np.zeros((len(ic_values), len(block_values)))
        surv_matrix = np.zeros((len(ic_values), len(block_values)))
        for i, ic in enumerate(ic_values):
            for j, bm in enumerate(block_values):
                idx = i * len(block_values) + j
                if idx < len(sim_results):
                    r = sim_results[idx]
                    summ = r.get("summary", {})
                    dep_matrix[i, j] = summ.get("composite_dependency", 0) or 0
                    surv_matrix[i, j] = 1 if summ.get("survived") else 0

        results[combo] = {
            "ic_values": ic_values,
            "block_values": block_values,
            "dependency_matrix": dep_matrix.tolist(),
            "survival_matrix": surv_matrix.tolist(),
            "dependency_range": [float(dep_matrix.min()), float(dep_matrix.max())],
            "survival_rate": float(surv_matrix.mean()),
        }

        print(f"      依附范围: [{dep_matrix.min():.3f}, {dep_matrix.max():.3f}] "
              f"存活率: {surv_matrix.mean():.1%}")

    return results


def _run_mixed_strategy_analysis(T_max, n_workers, quick):
    """混合策略分析."""
    print("\n  [4d] 混合策略比较...")
    mixtures = ["pure_A", "pure_B", "pure_C",
                "A_dominant_B", "B_dominant_A", "A_B_balanced", "uniform"]
    profiles = ["manufacturing_giant", "digital_leapfrogger"]
    scenarios = ["benign_globalization", "tech_cold_war"]

    results = {}
    for profile in profiles:
        for scenario in scenarios:
            combo = f"MIXED_{profile}_{scenario}"
            print(f"    {profile} × {scenario} ...", end=" ", flush=True)

            combo_results = []
            for mix_name in mixtures:
                try:
                    r = quick_run(
                        regime="M",
                        profile=profile,
                        scenario=scenario,
                        mixture_name=mix_name,
                        T_max=T_max,
                    )
                    summ = r.get("summary", {})
                    combo_results.append({
                        "mixture": mix_name,
                        "survived": summ.get("survived", False),
                        "survival_years": summ.get("survival_years", 0),
                        "cdi": summ.get("cdi"),
                        "final_gdp": summ.get("final_gdp", 0),
                        "final_tech_gap": summ.get("final_tech_gap", 999),
                        "incompatibility_cost": summ.get("incompatibility_cost"),
                    })
                except Exception as e:
                    combo_results.append({
                        "mixture": mix_name,
                        "survived": False,
                        "error": str(e),
                    })

            results[combo] = combo_results
            survived = sum(1 for r in combo_results if r.get("survived"))
            print(f"存活={survived}/{len(combo_results)}")

    # 寻找优于纯策略的混合策略
    better_than_pure = []
    for combo, mixes in results.items():
        pure_a = next((m for m in mixes if m["mixture"] == "pure_A"), None)
        pure_b = next((m for m in mixes if m["mixture"] == "pure_B"), None)
        pure_c = next((m for m in mixes if m["mixture"] == "pure_C"), None)
        for m in mixes:
            if m["mixture"].startswith("pure_"):
                continue
            cdi = m.get("cdi")
            # 检查是否优于所有纯策略
            better = True
            for pure in [pure_a, pure_b, pure_c]:
                if pure and pure.get("cdi") and cdi and pure["cdi"] >= cdi:
                    better = False
                    break
            if better and cdi:
                better_than_pure.append({
                    "combo": combo,
                    "mixture": m["mixture"],
                    "cdi": cdi,
                })

    return {
        "profiles_tested": profiles,
        "scenarios_tested": scenarios,
        "mixtures_tested": mixtures,
        "results": results,
        "mixtures_better_than_pure": better_than_pure,
    }


def _analyze_phase4(all_results: dict) -> dict:
    findings = []

    # 代价分布发现
    cost = all_results.get("cost_distribution", {})
    if cost:
        for combo, metrics in cost.get("metrics", {}).items():
            survived_metrics = [m for m in metrics if m["survived"]]
            dead_metrics = [m for m in metrics if not m["survived"]]
            if survived_metrics and dead_metrics:
                surv_eq = np.mean([m.get("employment_quality", 0) or 0 for m in survived_metrics])
                dead_eq = np.mean([m.get("employment_quality", 0) or 0 for m in dead_metrics])
                if abs(surv_eq - dead_eq) > 0.1:
                    findings.append(f"{combo}: 存活/断裂的就业质量差异={surv_eq:.2f} vs {dead_eq:.2f}")

    # 金融联动发现
    fin = all_results.get("financial_stability", {})
    for linkage in fin.get("linkages", []):
        findings.append(f"{linkage['combo']}: 泡沫-外逃相关性={linkage['correlation']:.2f}")

    # 混合策略发现
    mixed = all_results.get("mixed_strategies", {})
    better = mixed.get("mixtures_better_than_pure", [])
    for b in better:
        findings.append(f"{b['combo']}: {b['mixture']} 优于所有纯策略 CDI={b['cdi']:.3f}")

    return {"key_findings": findings}


def _print_section_summary(title, data):
    """打印各子节摘要."""
    pass  # 已在各函数内部打印


def _print_phase4_report(analysis: dict):
    print("\n" + "=" * 60)
    print("  第四阶段关键发现")
    print("=" * 60)
    findings = analysis.get("key_findings", [])
    if findings:
        print("\n  [机制发现]")
        for f in findings:
            print(f"    • {f}")
    else:
        print("\n  (未检测到显著性差异，可能需要增大样本量)")


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="第四阶段：机制深挖")
    parser.add_argument("--skip", type=str, default=None,
                       help="跳过的分析: cost,financial,dependency,mixed (逗号分隔)")
    parser.add_argument("--T-max", type=float, default=None,
                       help="模拟年限")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行worker数")
    parser.add_argument("--quick", action="store_true",
                       help="快速模式")

    args = parser.parse_args()
    run_phase4(skip=args.skip, T_max=args.T_max, n_workers=args.workers, quick=args.quick)
