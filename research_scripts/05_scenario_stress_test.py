#!/usr/bin/env python
"""05_国际场景压力测试 —— 8种国际场景下三种策略的生存能力矩阵 (v3.5).

研究问题:
    不同策略对国际环境的敏感度如何？哪些策略是"场景稳健"的，哪些是"场景脆弱"的？
    在极端封锁（full_containment）或技术爆发（tech_explosion）下，各策略表现如何？
    v3.5新增: effective_block_max 对 infra vs app 的分层影响分析

复杂度: ★★☆☆☆ (中等)
预计时间: 5-20分钟
并行度: 多进程
用途: 识别策略的场景依赖性、发现"反直觉"的场景×策略组合

运行方式:
    python research_scripts/05_scenario_stress_test.py
    python research_scripts/05_scenario_stress_test.py --n-lhs 100
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
from src.params.presets import SCENARIO_PRESETS
from common.logger import ResearchLogger
from common.result_store import ResultStore
from common.runner import (
    build_param_sets, _run_single_isolated, get_optimal_workers,
    ALL_SCENARIOS, ALL_REGIMES, sample_lhs_params,
)

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "05_scenario_stress_test"


def main():
    parser = argparse.ArgumentParser(description="国际场景压力测试")
    parser.add_argument("--regimes", type=str, default="A,B,C",
                       help="策略列表 (默认: A,B,C)")
    parser.add_argument("--scenarios", type=str, default="all",
                       help="场景列表或 'all'")
    parser.add_argument("--profiles", type=str, default="manufacturing_giant",
                       help="国情预设 (默认: manufacturing_giant)")
    parser.add_argument("--n-lhs", type=int, default=50,
                       help="每个策略×场景组合的LHS采样数 (默认: 50)")
    parser.add_argument("--T-max", type=float, default=50.0,
                       help="模拟年限 (默认: 50)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数 (默认: 自动)")
    args = parser.parse_args()

    regimes = args.regimes.split(",")
    scenarios = ALL_SCENARIOS if args.scenarios == "all" else args.scenarios.split(",")

    total_runs = len(regimes) * len(scenarios) * args.n_lhs

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"策略={regimes} × 场景={len(scenarios)}个 × LHS={args.n_lhs}",
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
    # 场景×策略存活率矩阵
    # ================================================================
    logger.logger.info(f"\n{'─' * 80}")
    logger.logger.info("  场景×策略 存活率矩阵")
    logger.logger.info(f"{'─' * 80}")

    matrix_data = {}
    scene_descriptions = {}

    for scenario in scenarios:
        matrix_data[scenario] = {}
        scene_info = SCENARIO_PRESETS.get(scenario, {})
        scene_descriptions[scenario] = scene_info.get("description", scenario)

        for regime in regimes:
            subset = [r for r in results
                     if r.get("_scenario") == scenario
                     and r["summary"].get("regime") == regime
                     and r.get("_success", True)]
            if not subset:
                matrix_data[scenario][regime] = {"surv": 0, "n": 0}
                continue

            surv = np.mean([s["summary"]["survived"] for s in subset])
            yrs = np.mean([s["summary"]["survival_years"] for s in subset])
            gdp = np.median([s["summary"]["final_gdp"] for s in subset])
            gap = np.mean([s["summary"]["final_tech_gap"] for s in subset])
            fractures = np.mean([s["summary"]["total_fractures"] for s in subset])

            cdi_vals = [s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                        for s in subset]
            cdi_mean = np.nanmean(cdi_vals)

            matrix_data[scenario][regime] = {
                "surv": surv, "yrs": yrs, "gdp": gdp,
                "gap": gap, "fractures": fractures,
                "cdi": cdi_mean, "n": len(subset),
            }

    # 输出矩阵
    headers = ["场景", "A存活率", "A-CDI", "B存活率", "B-CDI", "C存活率", "C-CDI"]
    rows = []
    for scenario in scenarios:
        d = matrix_data[scenario]
        row = [scenario[:20]]
        for regime in regimes:
            rd = d.get(regime, {})
            row.append(f"{rd.get('surv', 0):.1%}")
            row.append(f"{rd.get('cdi', float('nan')):.3f}")
        rows.append(row)

    logger.table("存活率 × CDI 矩阵", headers, rows)

    # 策略的场景稳健性排名
    logger.logger.info(f"\n  策略场景稳健性:")
    for regime in regimes:
        surv_rates = [matrix_data[s][regime]["surv"] for s in scenarios]
        cdi_vals = [matrix_data[s][regime]["cdi"] for s in scenarios]
        logger.logger.info(
            f"    策略{regime}: 平均存活率={np.mean(surv_rates):.1%} "
            f"(最差={np.min(surv_rates):.1%}) | "
            f"平均CDI={np.nanmean(cdi_vals):.3f}"
        )

    # 最危险场景
    for regime in regimes:
        worst = min(scenarios, key=lambda s: matrix_data[s][regime]["surv"])
        worst_rate = matrix_data[worst][regime]["surv"]
        logger.finding(f"策略{regime}最危险场景: {worst} (存活率={worst_rate:.1%})")

    # 反直觉发现
    for scenario in scenarios:
        survs = {r: matrix_data[scenario][r]["surv"] for r in regimes}
        best = max(survs, key=survs.get)
        worst = min(survs, key=survs.get)
        if best == "C" and scenario in ("full_containment", "tech_cold_war"):
            logger.finding(
                f"反直觉: {scenario}下策略C存活率最高({survs['C']:.1%})——替代策略在封锁下反而有韧性",
                level="KEY"
            )

    # v3.5: 封锁对infra/app分层的差异影响
    logger.logger.info(f"\n{'─' * 60}")
    logger.logger.info("  v3.5 技术封锁的分层影响 (infra vs app)")
    logger.logger.info(f"{'─' * 60}")
    for scenario in scenarios:
        sc_results = [r for r in results
                     if r.get("_scenario") == scenario
                     and r.get("_success", True)]
        if not sc_results:
            continue
        gap_infra = np.mean([s["summary"].get("final_tech_gap_infra",
                            s["summary"].get("final_tech_gap", 0))
                           for s in sc_results])
        gap_app = np.mean([s["summary"].get("final_tech_gap_app",
                          s["summary"].get("final_tech_gap", 0))
                         for s in sc_results])
        emp_q = np.mean([s["summary"].get("employment_quality_index", 1.0)
                       for s in sc_results])
        logger.logger.info(f"  {scenario:<25}: gap_infra={gap_infra:.1f} | "
                          f"gap_app={gap_app:.1f} | emp_quality={emp_q:.3f}")
        if gap_infra > gap_app + 1.0:
            logger.finding(
                f"  {scenario}: infra差距({gap_infra:.1f})显著大于app({gap_app:.1f})"
                "——基础设施层更易受封锁影响",
                level="INFO"
            )

    # 保存
    store.add_batch(results)
    store.set_parameters({"matrix": matrix_data})
    store.save_all()

    logger.experiment_end(success=True)

    print(f"\n✓ 场景压力测试完成")
    print(f"  结果目录: {store.result_dir}")


if __name__ == "__main__":
    main()
