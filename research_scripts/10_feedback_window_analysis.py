#!/usr/bin/env python
"""10_反哺窗口分析 —— EX-F-02: 策略B"数字反哺工业"承诺的可实现性系统扫描 (v3.5).

研究问题:
    跳跃式积累（策略B）承诺"先牺牲工业、数字成熟后反哺"。
    但在什么条件下反哺会兑现？什么条件下会落空？
    反哺窗口何时开启？何时关闭？反哺失败后系统会怎样？

复杂度: ★★★★☆ (高)
预计时间: 20分钟-1.5小时
并行度: 多进程
用途: 聚焦策略B最核心的承诺——反哺机制的可行性

运行方式:
    python research_scripts/10_feedback_window_analysis.py
    python research_scripts/10_feedback_window_analysis.py --grid-size 20
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
# 配置
# ============================================================
EXPERIMENT_NAME = "10_feedback_window_analysis"


def main():
    parser = argparse.ArgumentParser(description="反哺窗口分析 (EX-F-02)")
    parser.add_argument("--grid-size", type=int, default=15,
                       help="每维度的网格点数 (默认: 15, 总计 15^3=3375次)")
    parser.add_argument("--scenarios", type=str,
                       default="benign_globalization,tech_cold_war,full_containment",
                       help="场景列表")
    parser.add_argument("--profile", default="digital_leapfrogger",
                       help="国情预设 (默认: digital_leapfrogger, 最可能选B的国家)")
    parser.add_argument("--T-max", type=float, default=80.0,
                       help="模拟年限 (默认: 80, 给反哺充分时间)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数 (默认: 自动)")
    args = parser.parse_args()

    scenarios = args.scenarios.split(",")

    # 扫描参数空间
    feedback_trigger_range = np.linspace(0.3, 0.8, args.grid_size)
    feedback_ratio_range = np.linspace(0.05, 0.40, args.grid_size)
    industry_tolerance_range = np.linspace(0.2, 0.8, args.grid_size)

    total_runs = len(scenarios) * args.grid_size ** 3

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"网格={args.grid_size}³ × 场景={len(scenarios)}个 | T_max={args.T_max}",
        grid_size=args.grid_size, scenarios=scenarios,
    )
    logger.logger.info(f"总模拟次数: {total_runs}")
    logger.logger.info(f"扫描参数: feedback_trigger × feedback_ratio × industry_tolerance")

    store.set_config(vars(args))
    n_workers = get_optimal_workers(args.workers)

    # 构建所有任务
    tasks = []
    for scenario in scenarios:
        for trigger in feedback_trigger_range:
            for ratio in feedback_ratio_range:
                for tolerance in industry_tolerance_range:
                    params = build_param_sets(
                        regime="B",
                        scenario=scenario,
                        profile=args.profile,
                    )
                    params["feedback_trigger"] = trigger
                    params["feedback_ratio"] = ratio
                    params["industry_tolerance"] = tolerance

                    task = {
                        "params": params,
                        "regime": "B",
                        "config": SimulationConfig(
                            T_max=args.T_max,
                            random_seed=42,
                            verbose=False,
                        ),
                        "sim_id": len(tasks),
                        "_scenario": scenario,
                        "_trigger": trigger,
                        "_ratio": ratio,
                        "_tolerance": tolerance,
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
    # 分析：反哺有效性
    # ================================================================
    logger.logger.info(f"\n{'─' * 80}")
    logger.logger.info("  反哺窗口分析")
    logger.logger.info(f"{'─' * 80}")

    # 提取反哺相关信息
    feedback_triggered = []
    feedback_succeeded = []
    no_feedback_failed = []

    for r in results:
        if not r.get("_success", True):
            continue
        s = r["summary"]
        # 检查是否有反哺迹象：技术差距改善 + 工业资本回升
        # (具体指标取决于engine中的记录方式)
        # 使用CDI和存活率作为代理
        if s["survived"]:
            ev = s.get("evaluation", {})
            # 工业/数字代价比低说明反哺在起作用
            cost_ratio = s.get("cost_ratio_industrial_digital", 999)
            if cost_ratio < 1.5:
                feedback_succeeded.append(r)
            else:
                feedback_triggered.append(r)
        else:
            no_feedback_failed.append(r)

    n_valid = len([r for r in results if r.get("_success", True)])
    logger.logger.info(f"  总有效模拟: {n_valid}")
    logger.logger.info(f"  反哺成功(代价比<1.5): {len(feedback_succeeded)} "
                      f"({len(feedback_succeeded)/n_valid*100:.1f}%)")
    logger.logger.info(f"  存活但代价比高: {len(feedback_triggered)} "
                      f"({len(feedback_triggered)/n_valid*100:.1f}%)")
    logger.logger.info(f"  崩溃(反哺失败): {len(no_feedback_failed)} "
                      f"({len(no_feedback_failed)/n_valid*100:.1f}%)")

    # 反哺成功 vs 失败 的参数分布
    if feedback_succeeded and no_feedback_failed:
        success_triggers = [r["_trigger"] for r in feedback_succeeded]
        fail_triggers = [r["_trigger"] for r in no_feedback_failed]
        success_ratios = [r["_ratio"] for r in feedback_succeeded]
        fail_ratios = [r["_ratio"] for r in no_feedback_failed]
        success_tolerances = [r["_tolerance"] for r in feedback_succeeded]
        fail_tolerances = [r["_tolerance"] for r in no_feedback_failed]

        logger.logger.info(f"\n  反哺成功 vs 失败 参数对比:")
        logger.logger.info(f"    {'参数':<25} {'成功均值':<12} {'失败均值':<12} {'差异':<10}")
        logger.logger.info(f"    {'反馈触发':<25} {np.mean(success_triggers):<12.3f} "
                         f"{np.mean(fail_triggers):<12.3f} "
                         f"{np.mean(success_triggers)-np.mean(fail_triggers):+.3f}")
        logger.logger.info(f"    {'反馈比例':<25} {np.mean(success_ratios):<12.3f} "
                         f"{np.mean(fail_ratios):<12.3f} "
                         f"{np.mean(success_ratios)-np.mean(fail_ratios):+.3f}")
        logger.logger.info(f"    {'工业容忍度':<25} {np.mean(success_tolerances):<12.3f} "
                         f"{np.mean(fail_tolerances):<12.3f} "
                         f"{np.mean(success_tolerances)-np.mean(fail_tolerances):+.3f}")

    # 按场景分析
    for scenario in scenarios:
        sc_results = [r for r in results
                     if r.get("_scenario") == scenario and r.get("_success", True)]
        if not sc_results:
            continue
        surv = np.mean([s["summary"]["survived"] for s in sc_results])
        feedback_ok = sum(1 for r in sc_results
                        if r["summary"].get("cost_ratio_industrial_digital", 999) < 1.5
                        and r["summary"]["survived"])
        logger.logger.info(f"\n  场景 {scenario}: 存活率={surv:.1%} | "
                         f"反哺成功={feedback_ok}/{len(sc_results)} "
                         f"({feedback_ok/len(sc_results)*100:.1f}%)")

    # 最优反哺参数
    all_valid = [r for r in results if r.get("_success", True)]
    if all_valid:
        best = max(all_valid,
                  key=lambda r: r["summary"].get("evaluation", {}).get("cdi", -999))
        logger.finding(
            f"最优反哺参数: trigger={best['_trigger']:.2f}, "
            f"ratio={best['_ratio']:.2f}, tolerance={best['_tolerance']:.2f} "
            f"→ CDI={best['summary'].get('evaluation', {}).get('cdi', float('nan')):.3f}",
            level="KEY"
        )

    # 保存
    store.add_batch(results)
    store.save_all()

    logger.experiment_end(success=True)

    print(f"\n✓ 反哺窗口分析完成")
    print(f"  结果目录: {store.result_dir}")


if __name__ == "__main__":
    main()
