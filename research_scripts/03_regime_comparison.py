#!/usr/bin/env python
"""03_三种策略全维度比较 —— 纯策略A/B/C在多场景下的全面比较 (v3.5: 六部门评估).

研究问题:
    在相同的初始条件和外部环境下，三种纯策略（A渐进/B跳跃/C替代）
    在五个评价维度上各有什么优劣？命门在什么条件下被触发？
    v3.5新增: 分层技术差距(infra/app)、就业质量指数、六维依附评估

复杂度: ★★☆☆☆ (中等)
预计时间: 2-10分钟
并行度: 多进程
用途: 建立对三种策略的基本认知、论文核心图表

运行方式:
    python research_scripts/03_regime_comparison.py
    python research_scripts/03_regime_comparison.py --profiles all --scenarios all
    python research_scripts/03_regime_comparison.py --T-max 80
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
from src.analysis.visualize import Visualizer, generate_academic_report
from common.logger import ResearchLogger
from common.result_store import ResultStore
from common.runner import (
    build_param_sets, _run_single_isolated, get_optimal_workers,
    ALL_SCENARIOS, ALL_REGIMES,
)

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "03_regime_comparison"


def main():
    parser = argparse.ArgumentParser(description="三种纯策略全维度比较")
    parser.add_argument("--regimes", type=str, default="A,B,C",
                       help="策略列表，逗号分隔 (默认: A,B,C)")
    parser.add_argument("--scenarios", type=str, default="benign_globalization,tech_cold_war,resource_crash,full_containment,tech_explosion,resource_war",
                       help="场景列表，逗号分隔或 'all'")
    parser.add_argument("--profiles", type=str, default="manufacturing_giant",
                       help="国情预设，逗号分隔或 'all'")
    parser.add_argument("--T-max", type=float, default=50.0,
                       help="模拟年限 (默认: 50)")
    parser.add_argument("--n-repeats", type=int, default=3,
                       help="每组参数重复次数 (默认: 3, 用于评估随机性)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数 (默认: 自动)")
    parser.add_argument("--no-charts", action="store_true",
                       help="跳过图表生成")
    args = parser.parse_args()

    # 解析参数
    regimes = args.regimes.split(",") if args.regimes != "all" else ALL_REGIMES
    scenarios = (ALL_SCENARIOS if args.scenarios == "all"
                 else args.scenarios.split(","))
    profiles = (["manufacturing_giant", "resource_rich", "digital_leapfrogger",
                 "small_open_economy", "post_soviet_industrial", "weak_industrial_large_pop"]
                if args.profiles == "all"
                else args.profiles.split(","))

    total_runs = len(regimes) * len(scenarios) * len(profiles) * args.n_repeats

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"策略={regimes} × 场景={len(scenarios)}个 × 国情={len(profiles)}个 × 重复={args.n_repeats}",
        regimes=regimes, scenarios=scenarios, profiles=profiles,
    )
    logger.logger.info(f"总模拟次数: {total_runs}")

    store.set_config(vars(args))
    n_workers = get_optimal_workers(args.workers)

    # 构建所有任务
    config = SimulationConfig(T_max=args.T_max, verbose=False)

    tasks = []
    for profile in profiles:
        for scenario in scenarios:
            for regime in regimes:
                for rep in range(args.n_repeats):
                    params = build_param_sets(
                        regime=regime,
                        scenario=scenario,
                        profile=profile,
                    )
                    task = {
                        "params": params,
                        "regime": regime,
                        "config": SimulationConfig(
                            T_max=args.T_max,
                            random_seed=42 + rep,
                            verbose=False,
                        ),
                        "sim_id": len(tasks),
                        "_profile": profile,
                        "_scenario": scenario,
                        "_rep": rep,
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
    # 聚合分析
    # ================================================================
    logger.logger.info(f"\n{'─' * 80}")
    logger.logger.info("  汇总分析")
    logger.logger.info(f"{'─' * 80}")

    # 按策略分组
    for regime in regimes:
        regime_results = [r for r in results
                         if r["summary"].get("regime", "") == regime
                         and r.get("_success", True)]
        if not regime_results:
            continue

        surv = np.mean([s["summary"]["survived"] for s in regime_results])
        yrs = np.mean([s["summary"]["survival_years"] for s in regime_results])
        gdp = np.median([s["summary"]["final_gdp"] for s in regime_results])
        gap = np.mean([s["summary"]["final_tech_gap"] for s in regime_results])
        # v3.5: 分层技术差距
        gap_infra = np.mean([s["summary"].get("final_tech_gap_infra", s["summary"].get("final_tech_gap", 0))
                           for s in regime_results])
        gap_app = np.mean([s["summary"].get("final_tech_gap_app", s["summary"].get("final_tech_gap", 0))
                         for s in regime_results])
        fractures = np.mean([s["summary"]["total_fractures"] for s in regime_results])
        # v3.5: 就业质量
        emp_quality = np.mean([s["summary"].get("employment_quality_index", 1.0)
                             for s in regime_results])

        cdi_vals = [s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                    for s in regime_results]
        cdi_mean = np.nanmean(cdi_vals)

        logger.logger.info(
            f"  策略{regime}: 存活率={surv:.1%} | 平均{yrs:.0f}年 | "
            f"GDP中位数={gdp:.0f} | 技术差距={gap:.1f} | "
            f"断裂={fractures:.1f} | CDI={cdi_mean:.3f}"
        )
        logger.logger.info(
            f"          v3.5: 技术差距(infra)={gap_infra:.1f} | "
            f"技术差距(app)={gap_app:.1f} | 就业质量={emp_quality:.3f}"
        )

    # 按场景×策略交叉矩阵
    for scenario in scenarios:
        logger.logger.info(f"\n  场景: {scenario}")
        for regime in regimes:
            subset = [r for r in results
                     if r.get("_scenario") == scenario
                     and r["summary"].get("regime") == regime
                     and r.get("_success", True)]
            if not subset:
                continue
            surv = np.mean([s["summary"]["survived"] for s in subset])
            logger.logger.info(f"    策略{regime}: 存活率={surv:.1%}")

    # ================================================================
    # 可视化
    # ================================================================
    if not args.no_charts:
        logger.logger.info(f"\n{'─' * 60}")
        logger.logger.info("  生成图表...")

        viz = Visualizer()

        # 雷达图（五维）
        try:
            # 取每个策略的第一组结果做雷达图
            sample_for_radar = []
            for regime in regimes:
                for r in results:
                    if (r["summary"].get("regime") == regime
                            and r.get("_profile") == profiles[0]
                            and r.get("_scenario") == scenarios[0]
                            and r.get("_success", True)):
                        sample_for_radar.append(r)
                        break

            viz.plot_five_dimension_radar(sample_for_radar)
            logger.logger.info("  ✓ 五维雷达图")
        except Exception as e:
            logger.logger.error(f"  雷达图失败: {e}")

        # 代价瀑布图
        try:
            viz.plot_cost_waterfall(sample_for_radar)
            logger.logger.info("  ✓ 代价瀑布图")
        except Exception as e:
            logger.logger.error(f"  瀑布图失败: {e}")

        # 策略比较图
        try:
            viz.plot_strategy_comparison([r for r in results if r.get("_success", True)])
            logger.logger.info("  ✓ 策略比较图")
        except Exception as e:
            logger.logger.error(f"  策略比较图失败: {e}")

        # 学术报告
        try:
            generate_academic_report([r for r in results if r.get("_success", True)])
            logger.logger.info("  ✓ 学术报告")
        except Exception as e:
            logger.logger.error(f"  学术报告生成失败: {e}")

    # ================================================================
    # 保存
    # ================================================================
    store.add_batch(results)
    store.save_all()

    logger.experiment_end(success=True)


if __name__ == "__main__":
    main()
