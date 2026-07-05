#!/usr/bin/env python
"""06_基准生存扫描 —— EX-S-01: 三种纯策略在大范围LHS参数空间中的生存边界.

研究问题:
    在合理的参数范围内，每种策略的存活率是多少？
    主要断裂模式是什么？是否存在"全空间崩溃"的参数区域？

复杂度: ★★★☆☆ (中等偏高)
预计时间: 20-60分钟（取决于采样量）
并行度: 多进程（默认使用所有可用核心）
用途: 建立策略生存能力的基本认知，为后续聚焦实验提供参数范围

运行方式:
    python research_scripts/06_baseline_survival_scan.py
    python research_scripts/06_baseline_survival_scan.py --n-samples 500 --regimes A,B,C
    python research_scripts/06_baseline_survival_scan.py --n-samples 200 --T-max 100
"""

import sys
import os
import argparse
import json
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.core.config import SimulationConfig
from common.logger import ResearchLogger
from common.result_store import ResultStore
from common.runner import (
    _run_single_isolated, get_optimal_workers, sample_lhs_params, ALL_REGIMES,
)

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "06_baseline_survival_scan"


def compute_batch_stats(results: list) -> dict:
    """计算一批结果的综合统计."""
    summaries = [r["summary"] for r in results if r.get("_success", True)]
    n = len(summaries)
    if n == 0:
        return {"n": 0}

    survived = [s.get("survived", False) for s in summaries]
    surv_years = [s.get("survival_years", 0) for s in summaries]
    tech_gaps = [s.get("final_tech_gap", 0) for s in summaries]
    gdps = [s.get("final_gdp", 0) for s in summaries]
    cost_conc = [s.get("cost_concentration", 0) for s in summaries]
    total_fract = [s.get("total_fractures", 0) for s in summaries]
    peak_unemp = [s.get("peak_unemployment", 0) for s in summaries]

    cdi_vals = [s.get("evaluation", {}).get("cdi", float("nan"))
                for s in summaries]
    cdi_clean = [c for c in cdi_vals if not np.isnan(c)]

    # 断裂模式频率
    fracture_freq = {}
    for s in summaries:
        for ft in s.get("fracture_types", []):
            fracture_freq[ft] = fracture_freq.get(ft, 0) + 1

    # 多断裂并发（级联，≥3种）
    multi_fracture = sum(1 for s in summaries if s.get("total_fractures", 0) >= 3)

    ev = [s.get("evaluation", {}) for s in summaries]
    red_failures = sum(1 for e in ev if e.get("red_line_failures"))

    return {
        "n": n,
        "survival_rate": np.mean(survived) if survived else 0,
        "mean_survival_years": np.mean(surv_years),
        "std_survival_years": np.std(surv_years),
        "mean_tech_gap": np.mean(tech_gaps),
        "std_tech_gap": np.std(tech_gaps),
        "mean_gdp": np.mean(gdps),
        "median_gdp": np.median(gdps),
        "std_gdp": np.std(gdps),
        "mean_cost_concentration": np.mean(cost_conc),
        "mean_total_fractures": np.mean(total_fract),
        "multi_fracture_count": multi_fracture,
        "multi_fracture_rate": multi_fracture / n,
        "mean_cdi": np.mean(cdi_clean) if cdi_clean else float("nan"),
        "red_line_failure_rate": red_failures / n,
        "mean_peak_unemployment": np.mean(peak_unemp),
        "fracture_frequency": fracture_freq,
        "min_survival_years": min(surv_years),
        "max_survival_years": max(surv_years),
    }


def main():
    parser = argparse.ArgumentParser(description="基准生存扫描 (EX-S-01)")
    parser.add_argument("--regimes", type=str, default="A,B,C",
                       help="策略列表 (默认: A,B,C)")
    parser.add_argument("--scenario", default="benign_globalization",
                       help="国际场景 (默认: benign_globalization)")
    parser.add_argument("--profile", default="manufacturing_giant",
                       help="国情预设 (默认: manufacturing_giant)")
    parser.add_argument("--n-samples", type=int, default=200,
                       help="每个策略的LHS采样数 (默认: 200)")
    parser.add_argument("--T-max", type=float, default=100.0,
                       help="模拟年限 (默认: 100)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数 (默认: 自动)")
    parser.add_argument("--save-time-series", action="store_true",
                       help="保存完整时间序列（需要大量磁盘空间）")
    args = parser.parse_args()

    regimes = args.regimes.split(",")
    total_runs = len(regimes) * args.n_samples

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"策略={regimes} × LHS={args.n_samples} × T_max={args.T_max}",
        regimes=regimes, n_samples=args.n_samples, T_max=args.T_max,
    )
    logger.logger.info(f"总模拟次数: {total_runs}")

    store.set_config(vars(args))
    n_workers = get_optimal_workers(args.workers)

    # ================================================================
    # 逐策略运行
    # ================================================================
    all_results = []
    regime_stats = {}

    for regime_idx, regime in enumerate(regimes):
        logger.phase_start(
            f"策略 {regime} ({regime_idx + 1}/{len(regimes)})",
            regime_idx + 1, len(regimes)
        )

        # LHS采样
        logger.logger.info(f"LHS采样 {args.n_samples} 组参数...")
        param_sets = sample_lhs_params(
            n_samples=args.n_samples,
            regime=regime,
            scenario=args.scenario,
        )

        # 构建任务
        tasks = []
        for i, params in enumerate(param_sets):
            task = {
                "params": params,
                "regime": regime,
                "config": SimulationConfig(
                    T_max=args.T_max,
                    random_seed=42 + i,
                    verbose=False,
                ),
                "sim_id": len(all_results) + i,
            }
            tasks.append(task)

        # 运行
        logger.logger.info(f"开始运行 {len(tasks)} 次模拟 ({n_workers} 核心)...")

        results = [None] * len(tasks)
        completed = 0

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
                if completed % max(1, len(tasks) // 10) == 0:
                    logger.progress(completed, len(tasks),
                                   f"策略{regime} 进行中")

        # 统计
        batch_stats = compute_batch_stats(results)
        regime_stats[regime] = batch_stats

        # 输出当前策略结果
        logger.logger.info(f"\n  策略{regime} 统计 (n={batch_stats['n']}):")
        logger.stat("存活率", f"{batch_stats['survival_rate']:.1%}")
        logger.stat("平均存活年数", f"{batch_stats['mean_survival_years']:.1f} ± "
                   f"{batch_stats['std_survival_years']:.1f}")
        logger.stat("平均技术差距", batch_stats["mean_tech_gap"], "")
        logger.stat("中位数GDP", batch_stats["median_gdp"], "")
        logger.stat("平均CDI", batch_stats["mean_cdi"], "")
        logger.stat("平均断裂数", batch_stats["mean_total_fractures"], "")
        logger.stat("多断裂并发率(≥3)", f"{batch_stats['multi_fracture_rate']:.1%}")
        logger.stat("红线失败率", f"{batch_stats['red_line_failure_rate']:.1%}")

        if batch_stats["fracture_frequency"]:
            logger.logger.info("  断裂模式频率:")
            for ft, count in sorted(batch_stats["fracture_frequency"].items(),
                                   key=lambda x: -x[1]):
                logger.logger.info(f"    {ft:<35} {count:>4}次 ({100*count/batch_stats['n']:.1f}%)")

        all_results.extend(results)
        logger.progress(len(all_results), total_runs, f"策略{regime} 完成")

    # ================================================================
    # 跨策略比较
    # ================================================================
    logger.logger.info(f"\n{'─' * 80}")
    logger.logger.info("  策略综合比较")
    logger.logger.info(f"{'─' * 80}")

    headers = ["指标", "A(渐进)", "B(跳跃)", "C(替代)"]
    metrics = [
        ("存活率", "survival_rate", "{:.1%}"),
        ("平均存活年数", "mean_survival_years", "{:.1f}"),
        ("平均技术差距", "mean_tech_gap", "{:.1f}"),
        ("中位数GDP", "median_gdp", "{:.0f}"),
        ("平均代价基尼", "mean_cost_concentration", "{:.3f}"),
        ("多断裂并发率", "multi_fracture_rate", "{:.1%}"),
        ("红线失败率", "red_line_failure_rate", "{:.1%}"),
        ("平均CDI", "mean_cdi", "{:.3f}"),
    ]

    rows = []
    for label, key, fmt in metrics:
        row = [label]
        for r in regimes:
            stats = regime_stats.get(r, {})
            val = stats.get(key, float("nan"))
            try:
                row.append(fmt.format(val))
            except (ValueError, TypeError):
                row.append("N/A")
        rows.append(row)

    logger.table("策略综合比较", headers, rows)

    # 发现
    for regime in regimes:
        stats = regime_stats.get(regime, {})
        if stats.get("survival_rate", 0) < 0.5:
            logger.finding(f"策略{regime}存活率<50%——在默认场景下可能是不可行的", level="WARN")
        if stats.get("multi_fracture_rate", 0) > 0.3:
            logger.finding(f"策略{regime}有{stats['multi_fracture_rate']:.1%}的模拟出现≥3断裂并发", level="WARN")

    # 保存
    store.add_batch(all_results)
    store.set_parameters({"regime_stats": regime_stats})
    store.save_all()

    logger.experiment_end(success=True, summary=(
        f"A存活率={regime_stats.get('A',{}).get('survival_rate',0):.1%} | "
        f"B存活率={regime_stats.get('B',{}).get('survival_rate',0):.1%} | "
        f"C存活率={regime_stats.get('C',{}).get('survival_rate',0):.1%}"
    ))

    print(f"\n✓ 基准生存扫描完成")
    print(f"  结果目录: {store.result_dir}")


if __name__ == "__main__":
    main()
