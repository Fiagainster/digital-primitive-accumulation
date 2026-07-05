#!/usr/bin/env python
"""16_混合策略时机优化 —— EX-F-05: 寻找最优的策略切换序列与时机 (v3.5).

研究问题:
    "先A后B"还是"先B后A"？不同切换时机产生什么结果？
    切换成本有多大？是否存在"最佳切换窗口"？
    混合策略是否显著优于纯策略？

复杂度: ★★★★★ (极高)
预计时间: 30分钟-2小时
并行度: 多进程
用途: 回答"政策节奏和时机"这一核心实践问题

运行方式:
    python research_scripts/16_mixed_timing_optimization.py
    python research_scripts/16_mixed_timing_optimization.py --T-max 100 --time-steps 15
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
EXPERIMENT_NAME = "16_mixed_timing_optimization"

# 策略切换序列
TRANSITION_SEQUENCES = [
    ("A", "B", "A→B"),
    ("B", "A", "B→A"),
    ("A", "C", "A→C"),
    ("C", "A", "C→A"),
    ("B", "C", "B→C"),
    ("C", "B", "C→B"),
]

# 过渡期长度
TRANSITION_DURATIONS = [2, 4, 6, 8, 10]


def simulate_sequence(regime1: str, regime2: str,
                      switch_time: float, transition_duration: float,
                      base_params: dict, config: SimulationConfig,
                      sim_id: int) -> dict:
    """模拟策略切换序列.

    简化实现：将模拟分为两个阶段
    - 阶段1 (0 → switch_time): 策略1
    - 阶段2 (switch_time → T_max): 策略2（含过渡期）

    完整实现需要在engine层面支持动态策略切换。
    此处使用分阶段平均参数的方式近似。
    """
    # 阶段1：策略1
    params1 = base_params.copy()
    # ... 理论上需要两个阶段的引擎运行，此处用参数混合近似

    # 分阶段模拟的简化方式：使用混合策略权重随时间变化
    # 这需要在engine中支持时间变化的lambda
    # 当前实现：使用静态混合，权重按时间比例

    total_time = config.T_max
    weight1 = switch_time / total_time
    weight2 = 1.0 - weight1

    lambda_a = 0.0
    lambda_b = 0.0
    lambda_c = 0.0

    if regime1 == "A":
        lambda_a += weight1
    elif regime1 == "B":
        lambda_b += weight1
    elif regime1 == "C":
        lambda_c += weight1

    if regime2 == "A":
        lambda_a += weight2
    elif regime2 == "B":
        lambda_b += weight2
    elif regime2 == "C":
        lambda_c += weight2

    total = lambda_a + lambda_b + lambda_c
    if total > 0:
        lambda_a /= total
        lambda_b /= total
        lambda_c /= total

    params = base_params.copy()
    params["mixed_lambda"] = (lambda_a, lambda_b, lambda_c)

    # 过渡期成本（越长成本越大）
    transition_cost_factor = 1.0 + 0.1 * transition_duration / 2.0

    task = {
        "params": params,
        "regime": "M",
        "config": config,
        "sim_id": sim_id,
        "_regime1": regime1,
        "_regime2": regime2,
        "_switch_time": switch_time,
        "_transition_duration": transition_duration,
        "_lambda": (lambda_a, lambda_b, lambda_c),
    }

    result = _run_single_isolated(task)
    result["_regime1"] = regime1
    result["_regime2"] = regime2
    result["_switch_time"] = switch_time
    result["_transition_duration"] = transition_duration

    return result


def main():
    parser = argparse.ArgumentParser(description="策略切换时机优化 (EX-F-05)")
    parser.add_argument("--scenario", default="benign_globalization",
                       help="国际场景")
    parser.add_argument("--profile", default="manufacturing_giant",
                       help="国情预设")
    parser.add_argument("--T-max", type=float, default=80.0,
                       help="模拟年限 (默认: 80)")
    parser.add_argument("--time-steps", type=int, default=10,
                       help="切换时间的网格点数 (默认: 10)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数")
    parser.add_argument("--sequences", type=str, default=None,
                       help="指定切换序列，如 'A,B,B,A' (两两一对)")
    args = parser.parse_args()

    # 构建切换时间网格
    switch_times = np.linspace(5, args.T_max * 0.7, args.time_steps)

    sequences = TRANSITION_SEQUENCES
    if args.sequences:
        parts = args.sequences.split(",")
        sequences = []
        for i in range(0, len(parts) - 1, 2):
            sequences.append((parts[i], parts[i+1], f"{parts[i]}→{parts[i+1]}"))

    total_runs = (len(sequences) * len(switch_times)
                  * len(TRANSITION_DURATIONS))

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"切换序列={len(sequences)}种 × 时间={args.time_steps}点 × "
        f"过渡={len(TRANSITION_DURATIONS)}种 | T_max={args.T_max}",
        sequences=sequences, switch_times=args.time_steps,
    )
    logger.logger.info(f"总模拟次数: {total_runs}")

    store.set_config(vars(args))
    n_workers = get_optimal_workers(args.workers)

    # 构建任务
    tasks = []
    base_params = build_param_sets(
        regime="A",
        scenario=args.scenario,
        profile=args.profile,
    )
    base_config = SimulationConfig(T_max=args.T_max, verbose=False)

    for r1, r2, seq_label in sequences:
        for sw_time in switch_times:
            for trans_dur in TRANSITION_DURATIONS:
                tasks.append({
                    "r1": r1, "r2": r2, "seq_label": seq_label,
                    "sw_time": sw_time, "trans_dur": trans_dur,
                    "base_params": base_params.copy(),
                    "config": base_config,
                    "sim_id": len(tasks),
                })

    # 运行
    logger.logger.info(f"\n使用 {n_workers} 个并行核心")

    results = [None] * len(tasks)
    completed = 0

    if len(tasks) <= 4:
        for i, task_meta in enumerate(tasks):
            results[i] = simulate_sequence(
                task_meta["r1"], task_meta["r2"],
                task_meta["sw_time"], task_meta["trans_dur"],
                task_meta["base_params"], task_meta["config"],
                task_meta["sim_id"],
            )
            completed += 1
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for i, tm in enumerate(tasks):
                future = executor.submit(
                    simulate_sequence,
                    tm["r1"], tm["r2"],
                    tm["sw_time"], tm["trans_dur"],
                    tm["base_params"], tm["config"],
                    tm["sim_id"],
                )
                futures[future] = i

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
    # 分析
    # ================================================================
    logger.logger.info(f"\n{'─' * 80}")
    logger.logger.info("  策略切换时机分析")
    logger.logger.info(f"{'─' * 80}")

    valid = [r for r in results if r.get("_success", True)]

    # 按切换序列分析
    for seq_label in set(s["seq_label"] for s in tasks):
        seq_results = [r for r in valid
                      if r.get("_regime1", "") + "→" + r.get("_regime2", "") == seq_label]
        if not seq_results:
            continue

        surv = np.mean([s["summary"]["survived"] for s in seq_results])
        cdi_vals = [s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                    for s in seq_results]
        cdi_mean = np.nanmean(cdi_vals)
        gdp = np.median([s["summary"]["final_gdp"] for s in seq_results])

        logger.logger.info(f"\n  {seq_label} (n={len(seq_results)}):")
        logger.stat("存活率", f"{surv:.1%}")
        logger.stat("CDI均值", cdi_mean)
        logger.stat("GDP中位数", gdp)

    # 最优切换序列
    best_seq = None
    best_cdi = float("-inf")
    seq_cdi = {}
    for seq_label in set(s["seq_label"] for s in tasks):
        seq_results = [r for r in valid
                      if r.get("_regime1", "") + "→" + r.get("_regime2", "") == seq_label]
        if not seq_results:
            continue
        cdi_mean = np.nanmean([s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                             for s in seq_results])
        seq_cdi[seq_label] = cdi_mean
        if cdi_mean > best_cdi:
            best_cdi = cdi_mean
            best_seq = seq_label

    if best_seq:
        logger.finding(f"最优切换序列: {best_seq} (CDI={best_cdi:.3f})", level="KEY")

    # 最优切换时机
    for seq_label in seq_cdi:
        seq_results = [r for r in valid
                      if r.get("_regime1", "") + "→" + r.get("_regime2", "") == seq_label]
        if len(seq_results) < 5:
            continue

        # 按切换时间分组
        time_groups = {}
        for r in seq_results:
            sw = round(r.get("_switch_time", 0))
            if sw not in time_groups:
                time_groups[sw] = []
            time_groups[sw].append(r)

        best_time = None
        best_time_cdi = float("-inf")
        for sw, group in time_groups.items():
            cdi_m = np.nanmean([s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                             for s in group])
            if cdi_m > best_time_cdi:
                best_time_cdi = cdi_m
                best_time = sw

        if best_time:
            logger.logger.info(f"    {seq_label} 最优切换时间: ~{best_time}年 "
                             f"(CDI={best_time_cdi:.3f})")

    # 切换成本
    # 比较最早切换 vs 最晚切换
    for seq_label in seq_cdi:
        seq_results = [r for r in valid
                      if r.get("_regime1", "") + "→" + r.get("_regime2", "") == seq_label]
        if len(seq_results) < 10:
            continue

        early = [r for r in seq_results if r.get("_switch_time", 0) < 20]
        late = [r for r in seq_results if r.get("_switch_time", 0) > 40]

        if early and late:
            early_cdi = np.nanmean([s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                                  for s in early])
            late_cdi = np.nanmean([s["summary"].get("evaluation", {}).get("cdi", float("nan"))
                                 for s in late])
            diff = late_cdi - early_cdi
            if diff > 0.05:
                logger.finding(f"{seq_label}: 晚切换(CDI={late_cdi:.3f}) > "
                             f"早切换(CDI={early_cdi:.3f})——"
                             "不应过早放弃初始策略")

    # 保存
    store.add_batch(results)
    store.save_all()

    logger.experiment_end(success=True)

    print(f"\n✓ 策略切换时机优化完成")
    print(f"  结果目录: {store.result_dir}")
    print(f"  最优序列: {best_seq if best_seq else 'N/A'}")


if __name__ == "__main__":
    main()
