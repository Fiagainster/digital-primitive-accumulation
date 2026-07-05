#!/usr/bin/env python
"""04_混合策略全谱系扫描 —— 13种标准混合配比的系统性比较 (v3.5: 不兼容矩阵).

研究问题:
    混合策略（主导策略+次要策略）能否克服纯策略的内在缺陷？
    是否存在"最优"混合比例？混合的内生成本有多大？
    v3.5新增: 策略对不兼容性惩罚分析（AB/BC/AC冲突矩阵）

复杂度: ★★☆☆☆ (中等)
预计时间: 3-15分钟
并行度: 多进程
用途: 探索"第三条道路"的可能性空间

运行方式:
    python research_scripts/04_mixed_strategy_scan.py
    python research_scripts/04_mixed_strategy_scan.py --scenarios all --profiles all
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
from src.models.mixed_strategy import STANDARD_MIXTURES
from common.logger import ResearchLogger
from common.result_store import ResultStore
from common.runner import (
    build_param_sets, _run_single_isolated, get_optimal_workers,
    ALL_SCENARIOS,
)

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "04_mixed_strategy_scan"

MIXTURE_NAMES = list(STANDARD_MIXTURES.keys())


def main():
    parser = argparse.ArgumentParser(description="混合策略全谱系扫描")
    parser.add_argument("--scenarios", type=str,
                       default="benign_globalization,tech_cold_war,full_containment",
                       help="场景列表，逗号分隔或 'all'")
    parser.add_argument("--profiles", type=str,
                       default="manufacturing_giant,resource_rich,digital_leapfrogger",
                       help="国情预设，逗号分隔或 'all'")
    parser.add_argument("--mixtures", type=str, default=None,
                       help="混合配比列表，逗号分隔或 'all' (默认: all)")
    parser.add_argument("--T-max", type=float, default=50.0,
                       help="模拟年限 (默认: 50)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数 (默认: 自动)")
    parser.add_argument("--custom-lambda", type=str, default=None,
                       help="自定义λ向量，如 '0.6,0.25,0.15'")
    args = parser.parse_args()

    # 解析
    scenarios = (ALL_SCENARIOS if args.scenarios == "all"
                 else args.scenarios.split(","))
    profiles = (["manufacturing_giant", "resource_rich", "digital_leapfrogger",
                 "small_open_economy", "post_soviet_industrial", "weak_industrial_large_pop"]
                if args.profiles == "all"
                else args.profiles.split(","))
    mixture_names = (MIXTURE_NAMES if args.mixtures in (None, "all")
                     else args.mixtures.split(","))

    # 添加自定义混合
    custom_mixtures = {}
    if args.custom_lambda:
        parts = tuple(float(x) for x in args.custom_lambda.split(","))
        if len(parts) == 3 and abs(sum(parts) - 1.0) < 0.01:
            custom_mixtures["custom"] = parts
            mixture_names.append("custom")

    total_runs = len(scenarios) * len(profiles) * len(mixture_names)

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"混合策略={len(mixture_names)}种 × 场景={len(scenarios)}个 × 国情={len(profiles)}个",
        mixtures=mixture_names, scenarios=scenarios, profiles=profiles,
    )
    logger.logger.info(f"总模拟次数: {total_runs}")

    store.set_config(vars(args))
    n_workers = get_optimal_workers(args.workers)

    # 构建任务
    config = SimulationConfig(T_max=args.T_max, verbose=False)

    tasks = []
    for profile in profiles:
        for scenario in scenarios:
            for mix_name in mixture_names:
                # 混合策略以 A 为基础构建参数，然后覆盖 λ
                params = build_param_sets(
                    regime="A",
                    scenario=scenario,
                    profile=profile,
                )

                if mix_name == "custom" and custom_mixtures:
                    mixture = custom_mixtures["custom"]
                else:
                    mixture = STANDARD_MIXTURES.get(mix_name, (1.0, 0.0, 0.0))

                params["mixed_lambda"] = mixture
                params["mixed_name"] = mix_name

                task = {
                    "params": params,
                    "regime": "M",
                    "config": config,
                    "sim_id": len(tasks),
                    "_profile": profile,
                    "_scenario": scenario,
                    "_mixture": mix_name,
                    "_lambda": mixture,
                }
                tasks.append(task)

    # 运行
    logger.logger.info(f"使用 {n_workers} 个并行核心")

    results = [None] * len(tasks)
    completed = 0

    if len(tasks) <= 4:
        for i, task in enumerate(tasks):
            results[i] = _run_single_isolated(task)
            s = results[i]["summary"]
            logger.sim_end(i, s)
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
    # 分析：按混合配比聚合
    # ================================================================
    logger.logger.info(f"\n{'─' * 80}")
    logger.logger.info("  混合策略排名 (按CDI)")
    logger.logger.info(f"{'─' * 80}")

    mix_stats = {}
    for mix_name in mixture_names:
        subset = [r for r in results
                 if r.get("_mixture") == mix_name
                 and r.get("_success", True)]
        if not subset:
            continue

        surv = np.mean([s["summary"]["survived"] for s in subset])
        yrs = np.mean([s["summary"]["survival_years"] for s in subset])
        gdp = np.median([s["summary"]["final_gdp"] for s in subset])
        gap = np.mean([s["summary"]["final_tech_gap"] for s in subset])

        cdi_vals = [s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                    for s in subset]
        cdi_mean = np.nanmean(cdi_vals)

        liv_vals = [s["summary"].get("evaluation", {}).get("livelihood", float("nan"))
                    for s in subset]
        liv_mean = np.nanmean(liv_vals)

        # 红线触发率
        red_fails = sum(1 for s in subset
                       if s["summary"].get("evaluation", {}).get("red_line_failures"))
        is_ok = sum(1 for s in subset
                   if s["summary"].get("evaluation", {}).get("is_acceptable", True))

        lambda_val = subset[0].get("_lambda", (0, 0, 0))
        mix_stats[mix_name] = {
            "lambda": lambda_val,
            "survival_rate": surv,
            "mean_years": yrs,
            "median_gdp": gdp,
            "mean_tech_gap": gap,
            "mean_cdi": cdi_mean,
            "mean_livelihood": liv_mean,
            "red_line_failure_rate": red_fails / len(subset),
            "acceptable_rate": is_ok / len(subset),
            "n": len(subset),
        }

    # 排名
    sorted_by_cdi = sorted(mix_stats.items(), key=lambda x: -x[1]["mean_cdi"])

    logger.table("混合策略排名 (按CDI)",
                 ["混合配比", "λ(A,B,C)", "CDI", "民生", "存活率", "可接受率"],
                 [[name,
                   f"({stats['lambda'][0]:.0%},{stats['lambda'][1]:.0%},{stats['lambda'][2]:.0%})",
                   f"{stats['mean_cdi']:.3f}",
                   f"{stats['mean_livelihood']:.3f}",
                   f"{stats['survival_rate']:.1%}",
                   f"{stats['acceptable_rate']:.1%}"]
                  for name, stats in sorted_by_cdi])

    # 最佳混合策略
    if sorted_by_cdi:
        best_name, best_stats = sorted_by_cdi[0]
        logger.finding(
            f"最佳混合策略: {best_name} λ={best_stats['lambda']} "
            f"CDI={best_stats['mean_cdi']:.3f} 民生={best_stats['mean_livelihood']:.3f}",
            level="KEY"
        )

    # 混合vs纯策略比较
    pure_cdi = {}
    for r in results:
        if r.get("_mixture") in ("pure_A", "pure_B", "pure_C"):
            pure_cdi[r["_mixture"]] = r["summary"].get("evaluation", {}).get("cdi", float("nan"))

    best_mixed_cdi = sorted_by_cdi[0][1]["mean_cdi"] if sorted_by_cdi else 0
    logger.logger.info(f"\n  纯策略A CDI: {pure_cdi.get('pure_A', float('nan')):.3f}")
    logger.logger.info(f"  纯策略B CDI: {pure_cdi.get('pure_B', float('nan')):.3f}")
    logger.logger.info(f"  纯策略C CDI: {pure_cdi.get('pure_C', float('nan')):.3f}")
    logger.logger.info(f"  最佳混合 CDI: {best_mixed_cdi:.3f}")

    best_pure = max(pure_cdi.values()) if pure_cdi else 0
    if best_mixed_cdi > best_pure:
        logger.finding("混合策略优于所有纯策略！", level="KEY")
    else:
        logger.finding("混合策略未超越最佳纯策略，但在某些维度可能有优势")

    # v3.5: 策略不兼容性惩罚分析
    logger.logger.info(f"\n{'─' * 60}")
    logger.logger.info("  v3.5 策略不兼容性分析")
    logger.logger.info(f"{'─' * 60}")

    # 收集不兼容信息
    incompat_data = {}
    for r in results:
        if not r.get("_success", True):
            continue
        mix_name = r.get("_mixture", "unknown")
        incompat = r["summary"].get("incompatibility_info", {})
        if incompat:
            if mix_name not in incompat_data:
                incompat_data[mix_name] = []
            incompat_data[mix_name].append(incompat)

    if incompat_data:
        for mix_name, infos in incompat_data.items():
            avg_penalty = np.mean([i.get("penalty", 0) for i in infos])
            avg_AB = np.mean([i.get("AB_conflict", 0) for i in infos])
            avg_BC = np.mean([i.get("BC_conflict", 0) for i in infos])
            avg_AC = np.mean([i.get("AC_conflict", 0) for i in infos])
            logger.logger.info(
                f"  {mix_name:<20}: 总惩罚={avg_penalty:.3f} | "
                f"AB={avg_AB:.3f} | BC={avg_BC:.3f} | AC={avg_AC:.3f}"
            )
    else:
        logger.logger.info("  (不兼容数据需从引擎接入，当前summary中暂未直接包含)")

    # v3.5: 混合策略的六部门分析
    logger.logger.info(f"\n{'─' * 60}")
    logger.logger.info("  v3.5 六部门资本分布 (混合策略)")
    logger.logger.info(f"{'─' * 60}")
    for mix_name in mixture_names[:5]:  # 只展示前5个避免输出过多
        subset = [r for r in results
                 if r.get("_mixture") == mix_name
                 and r.get("_success", True)]
        if not subset:
            continue
        sector_info = subset[0]["summary"].get("sector_info", {})
        if sector_info:
            logger.logger.info(f"  {mix_name}:")
            for s_key in ["d_infra", "d_app", "s_high", "s_low"]:
                k_val = sector_info.get(f"K_{s_key}", "N/A")
                logger.logger.info(f"    K_{s_key}={k_val}")

    # ================================================================
    # 可视化
    # ================================================================
    logger.logger.info(f"\n{'─' * 60}")
    logger.logger.info("  生成图表...")

    viz_path = None
    try:
        from src.analysis.visualize import Visualizer
        viz = Visualizer()
        valid_results = [r for r in results if r.get("_success", True)]
        viz.plot_five_dimension_radar(valid_results)
        logger.logger.info("  ✓ 五维雷达图")
    except Exception as e:
        logger.logger.error(f"  图表失败: {e}")

    # 保存
    store.add_batch(results)
    store.set_parameters({"mixture_stats": mix_stats})
    store.save_all()

    logger.experiment_end(success=True)

    print(f"\n✓ 混合策略扫描完成")
    print(f"  最佳: {best_name if sorted_by_cdi else 'N/A'} "
          f"CDI={best_mixed_cdi:.3f}")


if __name__ == "__main__":
    main()
