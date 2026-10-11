#!/usr/bin/env python
"""11_技术追赶临界条件 —— EX-F-03: 找到"追赶"与"锁死"的参数边界 (v3.5: infra/app分层).

研究问题:
    在什么条件下后发国家能成功追赶技术前沿？什么条件下会被锁死在低技术陷阱？
    自主创新vs技术吸收的最优混合比例是什么？
    前沿增长速度对追赶窗口有何影响？

复杂度: ★★★★☆ (高)
预计时间: 30分钟-2小时
并行度: 多进程
用途: 回答"技术封锁如何影响后发国家命运"这一核心问题

运行方式:
    python research_scripts/11_tech_catchup_critical.py
    python research_scripts/11_tech_catchup_critical.py --n-lhs 1000
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
EXPERIMENT_NAME = "11_tech_catchup_critical"


def is_catchup_success(summary: dict, threshold: float = 0.3) -> bool:
    """判断技术追赶是否成功."""
    return (summary.get("final_tech_gap", 999) < threshold
            and summary.get("survived", False))


def is_tech_lock_in(summary: dict, threshold: float = 5.0) -> bool:
    """判断是否陷入技术锁死."""
    return summary.get("final_tech_gap", 0) > threshold


def main():
    parser = argparse.ArgumentParser(description="技术追赶临界条件 (EX-F-03)")
    parser.add_argument("--regimes", type=str, default="A,B,C",
                       help="策略列表")
    parser.add_argument("--scenarios", type=str,
                       default="benign_globalization,tech_boom,tech_cold_war,tech_explosion",
                       help="场景列表（关注与技术相关的场景）")
    parser.add_argument("--profile", default="manufacturing_giant",
                       help="国情预设")
    parser.add_argument("--n-lhs", type=int, default=300,
                       help="每个策略×场景组合的LHS采样数 (默认: 300)")
    parser.add_argument("--T-max", type=float, default=100.0,
                       help="模拟年限 (默认: 100)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数")
    parser.add_argument("--focus-params", type=str, default=None,
                       help="聚焦的技术参数，逗号分隔（如 'g_world,chi_d,zeta_d'）")
    args = parser.parse_args()

    regimes = args.regimes.split(",")
    scenarios = args.scenarios.split(",")

    total_runs = len(regimes) * len(scenarios) * args.n_lhs

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"策略={regimes} × 场景={len(scenarios)}个 × LHS={args.n_lhs} | T_max={args.T_max}",
        regimes=regimes, scenarios=scenarios, n_lhs=args.n_lhs,
    )
    logger.logger.info(f"总模拟次数: {total_runs}")

    store.set_config(vars(args))
    n_workers = get_optimal_workers(args.workers)

    # 构建任务
    tasks = []
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
                    "_scenario": scenario,
                }
                tasks.append(task)

    # 运行
    logger.logger.info(f"使用 {n_workers} 个并行核心")

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
            if completed % max(1, len(tasks) // 20) == 0:
                logger.progress(completed, len(tasks))

    logger.progress(len(tasks), len(tasks), "全部完成")

    # ================================================================
    # 技术追赶分析
    # ================================================================
    logger.logger.info(f"\n{'─' * 80}")
    logger.logger.info("  技术追赶分析")
    logger.logger.info(f"{'─' * 80}")

    # 追赶成功、追赶中、锁死的分类
    for regime in regimes:
        regime_results = [r for r in results
                         if r["summary"].get("regime") == regime
                         and r.get("_success", True)]
        if not regime_results:
            continue

        succeeded = sum(1 for r in regime_results
                       if is_catchup_success(r["summary"]))
        locked = sum(1 for r in regime_results
                    if is_tech_lock_in(r["summary"]))
        middling = len(regime_results) - succeeded - locked

        avg_gap = np.mean([s["summary"]["final_tech_gap"] for s in regime_results])
        avg_growth = np.mean([s["summary"]["final_gdp"] for s in regime_results])

        logger.logger.info(f"\n  策略{regime} (n={len(regime_results)}):")
        logger.stat("追赶成功(G<0.3)", f"{succeeded} ({succeeded/len(regime_results):.1%})")
        logger.stat("追赶中(0.3≤G≤5)", f"{middling} ({middling/len(regime_results):.1%})")
        logger.stat("锁死(G>5)", f"{locked} ({locked/len(regime_results):.1%})")
        logger.stat("平均技术差距", avg_gap)

        # 按场景细分
        for scenario in scenarios:
            sc_sub = [r for r in regime_results if r.get("_scenario") == scenario]
            if not sc_sub:
                continue
            sc_succ = sum(1 for r in sc_sub if is_catchup_success(r["summary"]))
            sc_locked = sum(1 for r in sc_sub if is_tech_lock_in(r["summary"]))
            sc_gap = np.mean([s["summary"]["final_tech_gap"] for s in sc_sub])
            logger.logger.info(f"      场景 {scenario:<25}: 成功={sc_succ/len(sc_sub):.1%} | "
                             f"锁死={sc_locked/len(sc_sub):.1%} | 平均差距={sc_gap:.1f}")

    # 技术追赶的场景依赖性 (v3.5: infra/app分层)
    logger.logger.info(f"\n  技术追赶的场景依赖 (v3.5 infra/app分层):")
    for scenario in scenarios:
        sc_all = [r for r in results
                 if r.get("_scenario") == scenario and r.get("_success", True)]
        if not sc_all:
            continue
        sc_succ = sum(1 for r in sc_all if is_catchup_success(r["summary"]))
        sc_gap = np.mean([s["summary"]["final_tech_gap"] for s in sc_all])
        sc_gap_infra = np.mean([s["summary"].get("final_tech_gap_infra",
                               s["summary"].get("final_tech_gap", 0))
                              for s in sc_all])
        sc_gap_app = np.mean([s["summary"].get("final_tech_gap_app",
                             s["summary"].get("final_tech_gap", 0))
                            for s in sc_all])
        logger.logger.info(f"    {scenario:<25}: 追赶成功率={sc_succ/len(sc_all):.1%} | "
                         f"平均差距={sc_gap:.1f} | "
                         f"infra差距={sc_gap_infra:.1f} | app差距={sc_gap_app:.1f}")

    # 关键发现
    # 技术爆发场景下的表现
    if "tech_explosion" in scenarios:
        te_all = [r for r in results
                 if r.get("_scenario") == "tech_explosion"
                 and r.get("_success", True)]
        te_succ = sum(1 for r in te_all if is_catchup_success(r["summary"]))
        if te_succ / max(len(te_all), 1) < 0.2:
            logger.finding(
                f"技术爆发(tech_explosion)场景下追赶成功率仅{te_succ/max(len(te_all),1):.1%}——"
                "前沿指数增长使后发国家几乎无法追赶",
                level="KEY"
            )

    # 技术冷战下的策略表现
    if "tech_cold_war" in scenarios:
        tc_results = [r for r in results
                     if r.get("_scenario") == "tech_cold_war"
                     and r.get("_success", True)]
        for regime in regimes:
            tc_regime = [r for r in tc_results
                        if r["summary"].get("regime") == regime]
            if not tc_regime:
                continue
            tc_succ = sum(1 for r in tc_regime if is_catchup_success(r["summary"]))
            tc_locked = sum(1 for r in tc_regime if is_tech_lock_in(r["summary"]))
            logger.finding(
                f"技术冷战下策略{regime}: 追赶成功={tc_succ/len(tc_regime):.1%} | "
                f"锁死={tc_locked/len(tc_regime):.1%}"
            )

    # 保存
    store.add_batch(results)
    store.save_all()

    logger.experiment_end(success=True)

    print(f"\n✓ 技术追赶临界分析完成")
    print(f"  结果目录: {store.result_dir}")


if __name__ == "__main__":
    main()
