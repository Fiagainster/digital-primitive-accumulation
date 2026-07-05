#!/usr/bin/env python
"""08_代价分布深度分析 —— EX-F-04: 系统比较不同策略下代价的时空分布与群体定位.

研究问题:
    代价在不同家庭群组（数字/工业/基础/服务业/失业）间如何分布？
    哪种策略的代价最集中（高基尼）？是否存在"代价转移"现象？
    补偿窗口何时出现？持续多久？

复杂度: ★★★☆☆ (中等偏高)
预计时间: 10-30分钟
并行度: 多进程
用途: 核心创新——让隐性代价可见、可定位

运行方式:
    python research_scripts/08_cost_distribution_analysis.py
    python research_scripts/08_cost_distribution_analysis.py --n-lhs 100
"""

import sys
import os
import argparse
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.core.config import SimulationConfig
from common.logger import ResearchLogger
from common.result_store import ResultStore
from common.runner import (
    _run_single_isolated, build_param_sets, get_optimal_workers,
    ALL_REGIMES, ALL_SCENARIOS, sample_lhs_params,
)

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "08_cost_distribution_analysis"


def main():
    parser = argparse.ArgumentParser(description="代价分布深度分析 (EX-F-04)")
    parser.add_argument("--regimes", type=str, default="A,B,C",
                       help="策略列表 (默认: A,B,C)")
    parser.add_argument("--scenarios", type=str,
                       default="benign_globalization,tech_cold_war,full_containment",
                       help="场景列表 (默认: 3个关键场景)")
    parser.add_argument("--profiles", type=str,
                       default="manufacturing_giant,resource_rich,digital_leapfrogger",
                       help="国情预设 (默认: 3个代表性国情)")
    parser.add_argument("--n-lhs", type=int, default=50,
                       help="每组参数采样数 (默认: 50)")
    parser.add_argument("--T-max", type=float, default=60.0,
                       help="模拟年限 (默认: 60)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数 (默认: 自动)")
    args = parser.parse_args()

    regimes = args.regimes.split(",")
    scenarios = args.scenarios.split(",")
    profiles = args.profiles.split(",")

    total_runs = len(regimes) * len(scenarios) * len(profiles) * args.n_lhs

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"策略={regimes} × 场景={len(scenarios)}个 × 国情={len(profiles)}个 × LHS={args.n_lhs}",
        regimes=regimes, scenarios=scenarios, profiles=profiles,
    )
    logger.logger.info(f"总模拟次数: {total_runs}")

    store.set_config(vars(args))
    n_workers = get_optimal_workers(args.workers)

    # 构建任务
    tasks = []
    for profile in profiles:
        for scenario in scenarios:
            for regime in regimes:
                param_sets = sample_lhs_params(
                    n_samples=args.n_lhs,
                    regime=regime,
                    scenario=scenario,
                )
                for i, params in enumerate(param_sets):
                    task = {
                        "params": params,
                        "regime": regime,
                        "config": SimulationConfig(
                            T_max=args.T_max,
                            random_seed=42 + i,
                            verbose=False,
                        ),
                        "sim_id": len(tasks),
                        "_profile": profile,
                        "_scenario": scenario,
                    }
                    tasks.append(task)

    # 运行
    logger.logger.info(f"使用 {n_workers} 个并行核心")

    results = [None] * len(tasks)
    completed = 0

    if len(tasks) <= 4:
        for i, task in enumerate(tasks):
            results[i] = _run_single_isolated(task)
            completed += 1
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_single_isolated, task): i
                for i, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"summary": {"survived": False, "error": str(e)},
                                    "_error": str(e)}
                completed += 1
                if completed % max(1, len(tasks) // 20) == 0:
                    logger.progress(completed, len(tasks))

    logger.progress(len(tasks), len(tasks), "全部完成")

    # ================================================================
    # 代价分布分析
    # ================================================================
    logger.logger.info(f"\n{'─' * 80}")
    logger.logger.info("  代价分布分析")
    logger.logger.info(f"{'─' * 80}")

    # 按策略聚合代价指标
    for regime in regimes:
        subset = [r for r in results
                 if r["summary"].get("regime") == regime
                 and r.get("_success", True)]
        if not subset:
            continue

        cost_conc = [s["summary"].get("cost_concentration", 0) for s in subset]
        cost_ratio = [s["summary"].get("cost_ratio_industrial_digital", 0)
                     for s in subset]
        peak_unemp = [s["summary"].get("peak_unemployment", 0) for s in subset]

        logger.logger.info(f"\n  策略{regime} (n={len(subset)}):")
        logger.stat("代价基尼系数", f"{np.mean(cost_conc):.3f} ± {np.std(cost_conc):.3f}")
        logger.stat("工业/数字代价比", f"{np.mean(cost_ratio):.2f} ± {np.std(cost_ratio):.2f}")
        logger.stat("峰值失业率", f"{np.mean(peak_unemp):.3f} ± {np.std(peak_unemp):.3f}")

        # 按场景细分
        for scenario in scenarios:
            sc_subset = [r for r in subset if r.get("_scenario") == scenario]
            if len(sc_subset) < 2:
                continue
            sc_cost = np.mean([s["summary"].get("cost_concentration", 0)
                             for s in sc_subset])
            sc_ratio = np.mean([s["summary"].get("cost_ratio_industrial_digital", 0)
                              for s in sc_subset])
            logger.logger.info(f"      场景 {scenario:<25}: 代价基尼={sc_cost:.3f} | "
                             f"工/数代价比={sc_ratio:.1f}")

    # 代价最集中的场景×策略组合
    worst_cost = None
    worst_val = float("-inf")
    for regime in regimes:
        for scenario in scenarios:
            subset = [r for r in results
                     if r["summary"].get("regime") == regime
                     and r.get("_scenario") == scenario
                     and r.get("_success", True)]
            if not subset:
                continue
            mean_cost = np.mean([s["summary"].get("cost_concentration", 0)
                               for s in subset])
            if mean_cost > worst_val:
                worst_val = mean_cost
                worst_cost = (regime, scenario)

    if worst_cost:
        logger.finding(
            f"代价最集中的组合: 策略{worst_cost[0]} × {worst_cost[1]} "
            f"(基尼={worst_val:.3f})",
            level="KEY"
        )

    # 策略间代价比较
    logger.logger.info(f"\n  策略代价排名 (代价基尼越低越好):")
    regime_cost = {}
    for regime in regimes:
        subset = [r for r in results
                 if r["summary"].get("regime") == regime
                 and r.get("_success", True)]
        regime_cost[regime] = np.mean([s["summary"].get("cost_concentration", 0)
                                      for s in subset]) if subset else 1.0
    sorted_by_cost = sorted(regime_cost.items(), key=lambda x: x[1])
    for i, (regime, cost) in enumerate(sorted_by_cost):
        logger.logger.info(f"    {i+1}. 策略{regime}: 代价基尼={cost:.3f}")

    # 代价比分析
    logger.logger.info(f"\n  工业/数字代价比分析:")
    for regime in regimes:
        subset = [r for r in results
                 if r["summary"].get("regime") == regime
                 and r.get("_success", True)]
        if not subset:
            continue
        ratios = [s["summary"].get("cost_ratio_industrial_digital", 0) for s in subset]
        high_ratio = sum(1 for r in ratios if r > 2.5)
        extreme_ratio = sum(1 for r in ratios if r > 4.0)
        logger.logger.info(f"    策略{regime}: 平均比={np.mean(ratios):.1f} | "
                         f">2.5(撕裂): {high_ratio/len(ratios):.1%} | "
                         f">4.0(崩溃): {extreme_ratio/len(ratios):.1%}")

    # 人道主义红线触发分析
    logger.logger.info(f"\n  人道主义红线触发分析:")
    for regime in regimes:
        subset = [r for r in results
                 if r["summary"].get("regime") == regime
                 and r.get("_success", True)]
        if not subset:
            continue
        evs = [s["summary"].get("evaluation", {}) for s in subset]
        red_fails = []
        red_warns = []
        for ev in evs:
            red_fails.extend(ev.get("red_line_failures", []))
            red_warns.extend(ev.get("red_line_warnings", []))
        if red_fails:
            fail_counts = {}
            for f in red_fails:
                fail_counts[f] = fail_counts.get(f, 0) + 1
            logger.logger.info(f"    策略{regime} 红线失败:")
            for f, c in sorted(fail_counts.items(), key=lambda x: -x[1]):
                logger.logger.info(f"      {f}: {c}/{len(subset)}次 ({c/len(subset):.1%})")

    # 可视化
    logger.logger.info(f"\n{'─' * 60}")
    logger.logger.info("  生成图表...")
    try:
        from src.analysis.visualize import Visualizer
        viz = Visualizer()
        valid_results = [r for r in results if r.get("_success", True)]
        viz.plot_cost_waterfall(valid_results)
        logger.logger.info("  ✓ 代价瀑布图")
    except Exception as e:
        logger.logger.error(f"  图表失败: {e}")

    # 保存
    store.add_batch(results)
    store.save_all()

    logger.experiment_end(success=True)

    print(f"\n✓ 代价分布分析完成")
    print(f"  结果目录: {store.result_dir}")


if __name__ == "__main__":
    main()
