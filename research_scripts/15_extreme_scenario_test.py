#!/usr/bin/env python
"""15_极端场景极限测试 —— EX-D-05: 在极端参数下测试系统行为边界 (v3.5).

研究问题:
    系统的绝对崩溃条件是什么？在极端条件下"存活"所需的最低条件是什么？
    多冲击叠加（封锁+资源枯竭+技术爆发+人口断崖）是否会产生乘数效应？

复杂度: ★★★★☆ (高)
预计时间: 10-30分钟
并行度: 多进程
用途: 识别系统不可逾越的物理约束、验证边界行为合理性

运行方式:
    python research_scripts/15_extreme_scenario_test.py
    python research_scripts/15_extreme_scenario_test.py --all-combinations
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
)

# ============================================================
# 极端场景定义
# ============================================================
EXPERIMENT_NAME = "15_extreme_scenario_test"

EXTREME_OVERRIDES = {
    "baseline": {
        "label": "基准对照",
        "overrides": {},
    },
    "total_blockade": {
        "label": "技术彻底封锁 (block=1.0)",
        "overrides": {"block_strength": 1.0, "fdi_accessibility": 0.0},
    },
    "resource_exhaustion": {
        "label": "资源瞬间枯竭 (R→0)",
        "overrides": {"R_initial": 0.01},
    },
    "population_crash": {
        "label": "人口断崖 (劳动力年减5%)",
        "overrides": {"population_growth": -0.05},
    },
    "ai_singularity": {
        "label": "AI奇点 (g_world→0.50)",
        "overrides": {"g_world": 0.50},
    },
    "financial_collapse": {
        "label": "金融崩溃 (信贷冻结)",
        "overrides": {"credit_accessibility": 0.0, "debt_gdp_threshold": 0.3},
    },
    "climate_catastrophe": {
        "label": "气候灾难 (碳排放失控+碳关税)",
        "overrides": {"carbon_price": 500, "green_transition_pressure": 0.95},
    },
    "double_whammy": {
        "label": "双重打击: 封锁+资源枯竭",
        "overrides": {"block_strength": 1.0, "R_initial": 0.01, "fdi_accessibility": 0.0},
    },
    "triple_whammy": {
        "label": "三重打击: 封锁+资源枯竭+AI奇点",
        "overrides": {"block_strength": 1.0, "R_initial": 0.01,
                      "g_world": 0.30, "fdi_accessibility": 0.0},
    },
    "perfect_storm": {
        "label": "完美风暴: 全部四种打击",
        "overrides": {"block_strength": 1.0, "R_initial": 0.01,
                      "g_world": 0.30, "population_growth": -0.03,
                      "fdi_accessibility": 0.0, "credit_accessibility": 0.1},
    },
    "low_institution_baseline": {
        "label": "低制度能力基准",
        "overrides": {"institution_capacity": 0.2},
    },
    "low_institution_blockade": {
        "label": "低制度能力+封锁",
        "overrides": {"institution_capacity": 0.2, "block_strength": 1.0,
                      "fdi_accessibility": 0.0},
    },
}


def main():
    parser = argparse.ArgumentParser(description="极端场景极限测试 (EX-D-05)")
    parser.add_argument("--regimes", type=str, default="A,B,C",
                       help="策略列表")
    parser.add_argument("--scenarios", type=str, default="all",
                       help="要测试的极端场景，逗号分隔或 'all'")
    parser.add_argument("--profile", default="manufacturing_giant",
                       help="国情预设")
    parser.add_argument("--T-max", type=float, default=60.0,
                       help="模拟年限")
    parser.add_argument("--n-repeats", type=int, default=3,
                       help="每个组合的重复次数 (默认: 3)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数")
    parser.add_argument("--list-scenarios", action="store_true",
                       help="列出所有极端场景")
    args = parser.parse_args()

    if args.list_scenarios:
        print("\n极端场景列表:")
        for key, info in EXTREME_OVERRIDES.items():
            print(f"  [{key:<30}] {info['label']}")
        return

    regimes = args.regimes.split(",")
    extreme_keys = (list(EXTREME_OVERRIDES.keys()) if args.scenarios == "all"
                    else args.scenarios.split(","))

    total_runs = len(regimes) * len(extreme_keys) * args.n_repeats

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"极端场景={len(extreme_keys)}个 × 策略={regimes} × 重复={args.n_repeats}",
        extreme_scenarios=extreme_keys, regimes=regimes,
    )
    logger.logger.info(f"总模拟次数: {total_runs}")

    store.set_config(vars(args))
    n_workers = get_optimal_workers(args.workers)

    # 构建任务
    tasks = []
    for extreme_key in extreme_keys:
        extreme_info = EXTREME_OVERRIDES[extreme_key]
        for regime in regimes:
            for rep in range(args.n_repeats):
                params = build_param_sets(
                    regime=regime,
                    scenario="benign_globalization",
                    profile=args.profile,
                )
                params.update(extreme_info["overrides"])

                task = {
                    "params": params,
                    "regime": regime,
                    "config": SimulationConfig(
                        T_max=args.T_max,
                        random_seed=42 + rep,
                        verbose=False,
                    ),
                    "sim_id": len(tasks),
                    "_extreme_key": extreme_key,
                    "_extreme_label": extreme_info["label"],
                    "_rep": rep,
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
                if completed % max(1, len(tasks) // 10) == 0:
                    logger.progress(completed, len(tasks))

    logger.progress(len(tasks), len(tasks), "全部完成")

    # ================================================================
    # 分析
    # ================================================================
    logger.logger.info(f"\n{'─' * 80}")
    logger.logger.info("  极端场景生存矩阵")
    logger.logger.info(f"{'─' * 80}")

    # 构建极端场景×策略矩阵
    matrix = {}
    for extreme_key in extreme_keys:
        label = EXTREME_OVERRIDES[extreme_key]["label"]
        matrix[extreme_key] = {"label": label}
        for regime in regimes:
            subset = [r for r in results
                     if r.get("_extreme_key") == extreme_key
                     and r["summary"].get("regime") == regime
                     and r.get("_success", True)]
            if not subset:
                matrix[extreme_key][regime] = {"surv": 0.0, "n": 0}
                continue

            surv = np.mean([s["summary"]["survived"] for s in subset])
            yrs = np.mean([s["summary"]["survival_years"] for s in subset])
            gap = np.mean([s["summary"]["final_tech_gap"] for s in subset])
            gdp = np.median([s["summary"]["final_gdp"] for s in subset])
            cdi_vals = [s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                        for s in subset]
            cdi_mean = np.nanmean(cdi_vals)
            fractures = np.mean([s["summary"]["total_fractures"] for s in subset])

            matrix[extreme_key][regime] = {
                "surv": surv, "yrs": yrs, "gap": gap,
                "gdp": gdp, "cdi": cdi_mean, "fractures": fractures,
                "n": len(subset),
            }

    # 输出矩阵
    headers = ["极端场景", "A存活率", "A-CDI", "B存活率", "B-CDI", "C存活率", "C-CDI"]
    rows = []
    for extreme_key in extreme_keys:
        label = matrix[extreme_key]["label"][:30]
        row = [label]
        for regime in regimes:
            rd = matrix[extreme_key].get(regime, {})
            row.append(f"{rd.get('surv', 0):.1%}")
            row.append(f"{rd.get('cdi', float('nan')):.3f}")
        rows.append(row)

    logger.table("极端场景生存矩阵", headers, rows)

    # 绝对崩溃条件
    logger.logger.info(f"\n  最致命场景:")
    for regime in regimes:
        worst_key = min(extreme_keys,
                       key=lambda k: matrix[k].get(regime, {}).get("surv", 1))
        worst_surv = matrix[worst_key].get(regime, {}).get("surv", 0)
        worst_label = EXTREME_OVERRIDES[worst_key]["label"]
        logger.logger.info(f"    策略{regime}: {worst_label} → 存活率={worst_surv:.1%}")

    # 存活的最低条件
    survivors = [r for r in results
                if r.get("_success", True)
                and r["summary"].get("survived", False)]
    if survivors:
        # 找出存活的最极端场景
        most_extreme_survivor = None
        most_extreme_stress = float("inf")
        for r in survivors:
            stress_score = (
                r.get("_extreme_key", "").count("_")
                + (1 if r.get("_extreme_key") == "perfect_storm" else 0)
            )
            if most_extreme_survivor is None:
                most_extreme_survivor = r
                most_extreme_stress = stress_score
        if most_extreme_survivor:
            logger.finding(
                f"最极端存活: 策略{most_extreme_survivor['summary']['regime']} "
                f"在 {most_extreme_survivor['_extreme_label']} 下存活",
                level="KEY"
            )

    # 多冲击乘数效应
    single_hit = matrix.get("total_blockade", {}).get("A", {}).get("surv", 1.0)
    double_hit = matrix.get("double_whammy", {}).get("A", {}).get("surv", 1.0)
    triple_hit = matrix.get("triple_whammy", {}).get("A", {}).get("surv", 1.0)
    perfect = matrix.get("perfect_storm", {}).get("A", {}).get("surv", 1.0)

    logger.logger.info(f"\n  多冲击乘数效应 (策略A):")
    logger.logger.info(f"    单独封锁: 存活率={single_hit:.1%}")
    logger.logger.info(f"    双重打击: 存活率={double_hit:.1%} (乘数≈{single_hit/double_hit if double_hit>0 else float('inf'):.1f}x)")
    logger.logger.info(f"    三重打击: 存活率={triple_hit:.1%}")
    logger.logger.info(f"    完美风暴: 存活率={perfect:.1%}")

    if perfect < 0.1:
        logger.finding("完美风暴(封锁+资源枯竭+AI奇点+人口断崖)下几乎无人能存活——"
                      "这定义了系统的绝对边界")

    # 保存
    store.add_batch(results)
    store.set_parameters({"extreme_matrix": matrix})
    store.save_all()

    logger.experiment_end(success=True)

    print(f"\n✓ 极端场景测试完成")
    print(f"  结果目录: {store.result_dir}")


if __name__ == "__main__":
    main()
