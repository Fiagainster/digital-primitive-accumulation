#!/usr/bin/env python
"""14_制度调节效应分析 —— EX-D-03: 量化制度变量对策略效果的调节作用.

研究问题:
    制度能力（再分配力度、规划强度、社保覆盖率）能否改变策略的命运？
    好的制度能否弥补差的策略选择？坏的制度能否毁掉好的策略？
    是否存在"制度替代"效应——某些制度配置可以替代策略选择？

复杂度: ★★★★★ (极高)
预计时间: 1-5小时
并行度: 多进程
用途: 政策含义——回答"制度改革能否救场"的核心问题

运行方式:
    python research_scripts/14_institutional_moderation.py
    python research_scripts/14_institutional_moderation.py --grid-size 8  # 快速版
    python research_scripts/14_institutional_moderation.py --grid-size 12  # 标准版
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
    ALL_SCENARIOS,
)

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "14_institutional_moderation"


def main():
    parser = argparse.ArgumentParser(description="制度调节效应分析 (EX-D-03)")
    parser.add_argument("--regimes", type=str, default="A,B,C",
                       help="策略列表")
    parser.add_argument("--scenarios", type=str,
                       default="benign_globalization,tech_cold_war,full_containment",
                       help="场景列表")
    parser.add_argument("--profile", default="manufacturing_giant",
                       help="国情预设")
    parser.add_argument("--grid-size", type=int, default=10,
                       help="每维度的网格点数 (默认: 10, 总计10³×len(scenarios)×len(regimes) = 27000)")
    parser.add_argument("--T-max", type=float, default=60.0,
                       help="模拟年限")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数")
    args = parser.parse_args()

    regimes = args.regimes.split(",")
    scenarios = args.scenarios.split(",")

    # 三个制度参数
    redistribution_values = np.linspace(0.0, 0.5, args.grid_size)
    planning_intensity_values = np.linspace(0.0, 1.0, args.grid_size)
    social_coverage_values = np.linspace(0.0, 1.0, args.grid_size)

    total_runs = (len(regimes) * len(scenarios)
                  * args.grid_size ** 3)

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"制度3参数 × 网格={args.grid_size}³ × 策略={regimes} × 场景={len(scenarios)}个",
        regimes=regimes, scenarios=scenarios, grid_size=args.grid_size,
    )
    logger.logger.info(f"总模拟次数: {total_runs}")
    logger.logger.info("制度参数: redistribution × planning_intensity × social_coverage")

    store.set_config(vars(args))
    n_workers = get_optimal_workers(args.workers)

    # 构建任务
    tasks = []
    for scenario in scenarios:
        for regime in regimes:
            for redist in redistribution_values:
                for plan in planning_intensity_values:
                    for social in social_coverage_values:
                        params = build_param_sets(
                            regime=regime,
                            scenario=scenario,
                            profile=args.profile,
                        )
                        # 覆盖制度参数
                        params["redistribution_intensity"] = redist
                        params["planning_intensity"] = plan
                        params["social_coverage"] = social

                        task = {
                            "params": params,
                            "regime": regime,
                            "config": SimulationConfig(
                                T_max=args.T_max,
                                random_seed=42,
                                verbose=False,
                            ),
                            "sim_id": len(tasks),
                            "_scenario": scenario,
                            "_redist": redist,
                            "_plan": plan,
                            "_social": social,
                        }
                        tasks.append(task)

    # 运行
    logger.logger.info(f"\n使用 {n_workers} 个并行核心")

    results = [None] * len(tasks)
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 分批提交，避免内存问题
        batch_size = 5000
        for start in range(0, len(tasks), batch_size):
            end = min(start + batch_size, len(tasks))
            batch_tasks = tasks[start:end]

            futures = {
                executor.submit(_run_single_isolated, task): i + start
                for i, task in enumerate(batch_tasks)
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
    # 制度调节效应分析
    # ================================================================
    logger.logger.info(f"\n{'─' * 80}")
    logger.logger.info("  制度调节效应分析")
    logger.logger.info(f"{'─' * 80}")

    valid = [r for r in results if r.get("_success", True)]

    # 制度的"拯救效应"
    # 极端制度 vs 无制度的存活差异
    for regime in regimes:
        regime_valid = [r for r in valid
                       if r["summary"].get("regime") == regime]

        # 无制度组 (redist=0, plan=0, social=0)
        no_inst = [r for r in regime_valid
                  if r["_redist"] < 0.02 and r["_plan"] < 0.02
                  and r["_social"] < 0.02]

        # 强制度组 (top 10%)
        high_redist = np.percentile([r["_redist"] for r in regime_valid], 90)
        high_plan = np.percentile([r["_plan"] for r in regime_valid], 90)
        high_social = np.percentile([r["_social"] for r in regime_valid], 90)
        strong_inst = [r for r in regime_valid
                      if r["_redist"] >= high_redist
                      and r["_plan"] >= high_plan
                      and r["_social"] >= high_social]

        if no_inst and strong_inst:
            no_surv = np.mean([s["summary"]["survived"] for s in no_inst])
            strong_surv = np.mean([s["summary"]["survived"] for s in strong_inst])
            no_cdi = np.nanmean([s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                               for s in no_inst])
            strong_cdi = np.nanmean([s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                                   for s in strong_inst])

            logger.logger.info(f"\n  策略{regime} 制度拯救效应:")
            logger.stat("无制度: 存活率", f"{no_surv:.1%}")
            logger.stat("强制度: 存活率", f"{strong_surv:.1%}")
            logger.stat("无制度: CDI", no_cdi)
            logger.stat("强制度: CDI", strong_cdi)
            logger.stat("存活率提升", f"{strong_surv - no_surv:.1%}")
            logger.stat("CDI提升", strong_cdi - no_cdi)

            if strong_surv - no_surv > 0.2:
                logger.finding(
                    f"制度显著拯救策略{regime}：存活率提升{strong_surv-no_surv:.1%}",
                    level="KEY"
                )

    # 制度的边际收益
    logger.logger.info(f"\n  制度参数的边际收益 (按CDI):")
    for param_name, param_key in [
        ("再分配力度", "_redist"),
        ("规划强度", "_plan"),
        ("社保覆盖率", "_social"),
    ]:
        bins = np.linspace(0, 1, 11)
        logger.logger.info(f"\n    {param_name}:")
        for i in range(len(bins) - 1):
            bin_center = (bins[i] + bins[i+1]) / 2
            bin_results = [r for r in valid
                         if bins[i] <= r[param_key] < bins[i+1]]
            if len(bin_results) < 5:
                continue
            bin_cdi = np.nanmean([s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                                for s in bin_results])
            bin_surv = np.mean([s["summary"]["survived"] for s in bin_results])
            logger.logger.info(f"      [{bins[i]:.1f}-{bins[i+1]:.1f}]: "
                             f"CDI={bin_cdi:.3f} | 存活率={bin_surv:.1%} | n={len(bin_results)}")

    # 制度替代效应
    # 检查：差策略+好制度 vs 好策略+差制度
    logger.logger.info(f"\n  制度替代效应:")
    for regime in regimes:
        regime_valid = [r for r in valid
                       if r["summary"].get("regime") == regime]
        for other_regime in regimes:
            if other_regime == regime:
                continue
            other_valid = [r for r in valid
                          if r["summary"].get("regime") == other_regime]

            # 差策略+好制度
            bad_strategy_good_inst = [r for r in regime_valid
                                     if r["_redist"] > 0.3
                                     and r["_plan"] > 0.5
                                     and r["_social"] > 0.5]
            # 好策略+差制度
            good_strategy_bad_inst = [r for r in other_valid
                                     if r["_redist"] < 0.1
                                     and r["_plan"] < 0.2
                                     and r["_social"] < 0.2]

            if bad_strategy_good_inst and good_strategy_bad_inst:
                bad_good_cdi = np.nanmean([s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                                         for s in bad_strategy_good_inst])
                good_bad_cdi = np.nanmean([s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                                         for s in good_strategy_bad_inst])

                if bad_good_cdi > good_bad_cdi:
                    logger.finding(
                        f"制度替代: 策略{regime}+好制度(CDI={bad_good_cdi:.3f}) > "
                        f"策略{other_regime}+差制度(CDI={good_bad_cdi:.3f})——"
                        "制度可以部分替代策略选择",
                        level="KEY"
                    )

    # 保存
    store.add_batch(results)
    store.save_all()

    logger.experiment_end(success=True)

    print(f"\n✓ 制度调节效应分析完成")
    print(f"  结果目录: {store.result_dir}")


if __name__ == "__main__":
    main()
