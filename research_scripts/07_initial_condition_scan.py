#!/usr/bin/env python
"""07_初始条件扫描 —— EX-S-02: 探索初始条件（工业基础、资源、人力资本）对策略可行性的影响.

研究问题:
    在不同类型的后发国家（制造业大国/资源富国/数字跳跃者等）中，三种策略的可行性如何？
    哪种国情最适合哪种策略？"国情不适配"的策略会有什么不良后果？

复杂度: ★★★☆☆ (中等偏高)
预计时间: 10-40分钟
并行度: 多进程
用途: 为"不同国情应选择不同策略"提供模型证据

运行方式:
    python research_scripts/07_initial_condition_scan.py
    python research_scripts/07_initial_condition_scan.py --profiles all --n-lhs 50
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
from src.params.country_profiles import get_country_profile, get_profile_names
from common.logger import ResearchLogger
from common.result_store import ResultStore
from common.runner import (
    _run_single_isolated, build_param_sets, get_optimal_workers, ALL_REGIMES,
)

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "07_initial_condition_scan"

ALL_PROFILES = get_profile_names()


def main():
    parser = argparse.ArgumentParser(description="初始条件扫描 (EX-S-02)")
    parser.add_argument("--regimes", type=str, default="A,B,C",
                       help="策略列表 (默认: A,B,C)")
    parser.add_argument("--profiles", type=str, default="all",
                       help="国情预设，逗号分隔或 'all'")
    parser.add_argument("--scenario", default="benign_globalization",
                       help="国际场景 (默认: benign_globalization)")
    parser.add_argument("--n-lhs", type=int, default=30,
                       help="每个策略×国情组合的策略参数采样数 (默认: 30)")
    parser.add_argument("--T-max", type=float, default=60.0,
                       help="模拟年限 (默认: 60)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数 (默认: 自动)")
    args = parser.parse_args()

    regimes = args.regimes.split(",")
    profiles = ALL_PROFILES if args.profiles == "all" else args.profiles.split(",")

    total_runs = len(regimes) * len(profiles) * args.n_lhs

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"策略={regimes} × 国情={len(profiles)}种 × LHS={args.n_lhs}",
        regimes=regimes, profiles=profiles, n_lhs=args.n_lhs,
    )
    logger.logger.info(f"总模拟次数: {total_runs}")

    store.set_config(vars(args))
    n_workers = get_optimal_workers(args.workers)

    # 打印国情信息
    logger.logger.info("\n  国情预设概要:")
    for pname in profiles:
        p = get_country_profile(pname)
        if p:
            logger.logger.info(
                f"    [{pname:<30}] 自然策略倾向={p.get('natural_strategy','?')} | "
                f"K_i={p.get('K_i_initial','?')} | R={p.get('R_initial','?')} | "
                f"制度={p.get('institution_capacity','?')}"
            )

    # 构建所有任务
    tasks = []
    for profile in profiles:
        for regime in regimes:
            for i in range(args.n_lhs):
                params = build_param_sets(
                    regime=regime,
                    scenario=args.scenario,
                    profile=profile,
                )
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
                }
                tasks.append(task)

    # 运行
    logger.logger.info(f"\n使用 {n_workers} 个并行核心")

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
    # 国情×策略矩阵分析
    # ================================================================
    logger.logger.info(f"\n{'─' * 80}")
    logger.logger.info("  国情×策略 适配矩阵")
    logger.logger.info(f"{'─' * 80}")

    # 计算每个国情×策略组合的统计
    profile_regime_stats = {}
    for profile in profiles:
        profile_regime_stats[profile] = {}
        profile_info = get_country_profile(profile)

        for regime in regimes:
            subset = [r for r in results
                     if r.get("_profile") == profile
                     and r["summary"].get("regime") == regime
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
            fractures = np.mean([s["summary"]["total_fractures"] for s in subset])

            red_fails = sum(1 for s in subset
                          if s["summary"].get("evaluation", {}).get("red_line_failures"))

            profile_regime_stats[profile][regime] = {
                "surv": surv, "yrs": yrs, "gdp": gdp,
                "gap": gap, "cdi": cdi_mean, "fractures": fractures,
                "red_fail_rate": red_fails / len(subset),
                "n": len(subset),
            }

    # 输出矩阵
    headers = ["国情预设", "自然倾向",
               "A存活率", "A-CDI", "B存活率", "B-CDI", "C存活率", "C-CDI"]
    rows = []
    for profile in profiles:
        pinfo = get_country_profile(profile)
        natural = pinfo.get("natural_strategy", "?")
        ps = profile_regime_stats.get(profile, {})
        row = [profile[:25], natural]
        for regime in regimes:
            rd = ps.get(regime, {})
            row.append(f"{rd.get('surv', 0):.1%}")
            row.append(f"{rd.get('cdi', float('nan')):.3f}")
        rows.append(row)

    logger.table("国情×策略 适配矩阵", headers, rows)

    # 最佳策略推荐
    logger.logger.info("\n  最佳策略推荐 (按CDI):")
    for profile in profiles:
        ps = profile_regime_stats.get(profile, {})
        if not ps:
            continue
        best_regime = max(ps.keys(), key=lambda r: ps[r]["cdi"])
        natural = get_country_profile(profile).get("natural_strategy", "?")
        match_str = "✓ 与自然倾向一致" if best_regime == natural else "✗ 与自然倾向不同"

        best_cdi = ps[best_regime]["cdi"]
        natural_cdi = ps.get(natural, {}).get("cdi", float("nan"))
        improvement = ""
        if not np.isnan(natural_cdi) and best_regime != natural:
            imp = best_cdi - natural_cdi
            improvement = f" (vs自然倾向{natural}: +{imp:.3f})"

        logger.logger.info(f"    {profile:<30} → 策略{best_regime} "
                          f"CDI={best_cdi:.3f}{improvement} {match_str}")

    # 不匹配警告
    for profile in profiles:
        pinfo = get_country_profile(profile)
        natural = pinfo.get("natural_strategy", "")
        risks = pinfo.get("strategy_risks", "")
        ps = profile_regime_stats.get(profile, {})
        worst = min(ps.keys(), key=lambda r: ps[r]["cdi"]) if ps else None
        if worst and worst != natural:
            logger.finding(
                f"{profile}: 最差策略={worst} (自然倾向={natural}) | 风险: {risks}",
                level="WARN"
            )

    # 保存
    store.add_batch(results)
    store.set_parameters({"profile_regime_stats": profile_regime_stats})
    store.save_all()

    logger.experiment_end(success=True)

    print(f"\n✓ 初始条件扫描完成")
    print(f"  结果目录: {store.result_dir}")


if __name__ == "__main__":
    main()
