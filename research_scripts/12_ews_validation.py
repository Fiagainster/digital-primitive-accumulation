#!/usr/bin/env python
"""12_EWS预警有效性验证 —— EX-F-06: 验证早期预警信号系统是否有效 (v3.5).

研究问题:
    EWS系统能否在断裂触发前给出有效预警？
    预警提前量有多大？"黄金响应窗口"有多宽？
    误报率和漏报率是多少？哪个EWS指标最有效？

复杂度: ★★★★☆ (高)
预计时间: 30分钟-2小时
并行度: 多进程
用途: 验证模型的方法论贡献——预警系统的可靠性

运行方式:
    python research_scripts/12_ews_validation.py
    python research_scripts/12_ews_validation.py --n-lhs 500 --T-max 120
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
    ALL_REGIMES, sample_lhs_params,
)

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "12_ews_validation"


def analyze_ews(result: dict) -> dict:
    """分析单次模拟的EWS表现."""
    summary = result["summary"]
    time_series = result.get("time_series", {})
    fractures = summary.get("fracture_history", [])

    ews_info = {
        "had_fracture": not summary.get("survived", True)
                       or summary.get("total_fractures", 0) > 0,
        "first_fracture_time": None,
        "first_ews_warning_time": None,
        "first_ews_critical_time": None,
        "warning_lead_time": None,  # 预警提前量
        "critical_lead_time": None,
        "false_alarm": False,       # 有预警但未断裂
        "missed": False,            # 断裂但无预警
        "ews_k_i_max": summary.get("ews_max_K_i", 0),
        "ews_tech_gap_max": summary.get("ews_max_tech_gap", 0),
        "resilience_min": summary.get("ews_min_resilience", 1.0),
    }

    # 首次断裂时间
    if fractures:
        ews_info["first_fracture_time"] = min(
            f.get("time", float("inf")) for f in fractures
        )

    # EWS信号（从时间序列中提取）
    ews_ts = time_series.get("ews_composite", [])
    ews_times = time_series.get("time", [])

    if len(ews_ts) > 0 and len(ews_times) > 0:
        # 找到首次warning和critical的时间
        for i, (t, val) in enumerate(zip(ews_times, ews_ts)):
            if ews_info["first_ews_warning_time"] is None and val > 0.4:
                ews_info["first_ews_warning_time"] = t
            if ews_info["first_ews_critical_time"] is None and val > 0.7:
                ews_info["first_ews_critical_time"] = t

    # 计算预警提前量
    if ews_info["first_fracture_time"] is not None:
        if ews_info["first_ews_warning_time"] is not None:
            ews_info["warning_lead_time"] = (
                ews_info["first_fracture_time"]
                - ews_info["first_ews_warning_time"]
            )
        else:
            ews_info["missed"] = True  # 断裂但无预警

        if ews_info["first_ews_critical_time"] is not None:
            ews_info["critical_lead_time"] = (
                ews_info["first_fracture_time"]
                - ews_info["first_ews_critical_time"]
            )
    elif ews_info["first_ews_warning_time"] is not None:
        # 有预警但未断裂 → 误报
        ews_info["false_alarm"] = True

    return ews_info


def main():
    parser = argparse.ArgumentParser(description="EWS预警有效性验证 (EX-F-06)")
    parser.add_argument("--regimes", type=str, default="A,B,C",
                       help="策略列表")
    parser.add_argument("--scenarios", type=str,
                       default="benign_globalization,tech_cold_war,full_containment,tech_explosion",
                       help="场景列表（选可能触发断裂的场景）")
    parser.add_argument("--profile", default="manufacturing_giant",
                       help="国情预设")
    parser.add_argument("--n-lhs", type=int, default=200,
                       help="每个策略×场景的LHS采样数 (默认: 200)")
    parser.add_argument("--T-max", type=float, default=100.0,
                       help="模拟年限 (默认: 100)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数")
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
    logger.logger.info("EWS配置: warning_threshold=0.4, critical_threshold=0.7")

    store.set_config(vars(args))
    n_workers = get_optimal_workers(args.workers)

    # 构建任务（需要记录时间序列）
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
                        ews_window_size=40,
                    ),
                    "sim_id": len(tasks),
                    "_scenario": scenario,
                }
                tasks.append(task)

    # 运行
    logger.logger.info(f"\n使用 {n_workers} 个并行核心")

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
    # EWS分析
    # ================================================================
    logger.logger.info(f"\n{'─' * 80}")
    logger.logger.info("  EWS预警有效性分析")
    logger.logger.info(f"{'─' * 80}")

    valid_results = [r for r in results if r.get("_success", True)]
    ews_analyses = [analyze_ews(r) for r in valid_results]

    # 总体统计
    n_with_fracture = sum(1 for a in ews_analyses if a["had_fracture"])
    n_warning = sum(1 for a in ews_analyses
                   if a["first_ews_warning_time"] is not None)
    n_critical = sum(1 for a in ews_analyses
                    if a["first_ews_critical_time"] is not None)
    n_false_alarm = sum(1 for a in ews_analyses if a["false_alarm"])
    n_missed = sum(1 for a in ews_analyses if a["missed"])

    logger.logger.info(f"\n  总模拟: {len(valid_results)}")
    logger.logger.info(f"  发生断裂: {n_with_fracture} ({n_with_fracture/len(valid_results):.1%})")
    logger.logger.info(f"  EWS Warning触发: {n_warning} ({n_warning/len(valid_results):.1%})")
    logger.logger.info(f"  EWS Critical触发: {n_critical} ({n_critical/len(valid_results):.1%})")

    # 预警提前量分析
    lead_times = [a["warning_lead_time"] for a in ews_analyses
                  if a["warning_lead_time"] is not None and a["warning_lead_time"] > 0]
    if lead_times:
        logger.logger.info(f"\n  预警提前量 (Warning → 断裂):")
        logger.stat("均值", f"{np.mean(lead_times):.1f}", "年")
        logger.stat("中位数", f"{np.median(lead_times):.1f}", "年")
        logger.stat("最小值", f"{np.min(lead_times):.1f}", "年")
        logger.stat("最大值", f"{np.max(lead_times):.1f}", "年")
        logger.stat("标准差", f"{np.std(lead_times):.1f}", "年")

        # 提前量分布
        bins = [0, 2, 5, 10, 20, 50, 100]
        logger.logger.info("  提前量分布:")
        for i in range(len(bins) - 1):
            count = sum(1 for t in lead_times if bins[i] <= t < bins[i+1])
            logger.logger.info(f"    {bins[i]}-{bins[i+1]}年: {count}次 "
                             f"({count/len(lead_times):.1%})")

    # 误报率和漏报率
    false_alarm_rate = n_false_alarm / max(n_warning, 1)
    missed_rate = n_missed / max(n_with_fracture, 1)

    logger.logger.info(f"\n  EWS可靠性:")
    logger.stat("误报率(有预警但未断裂)", f"{false_alarm_rate:.1%}")
    logger.stat("漏报率(断裂但无预警)", f"{missed_rate:.1%}")

    if false_alarm_rate < 0.2 and missed_rate < 0.2:
        logger.finding("EWS系统可靠性良好：误报率和漏报率均<20%", level="KEY")
    elif missed_rate > 0.3:
        logger.finding(f"EWS漏报率高({missed_rate:.1%})——某些断裂模式可能缺乏事前信号", level="WARN")

    # EWS指标相对重要性
    k_i_max_vals = [a["ews_k_i_max"] for a in ews_analyses]
    gap_max_vals = [a["ews_tech_gap_max"] for a in ews_analyses]

    logger.logger.info(f"\n  EWS指标强度:")
    logger.stat("平均EWS(K_i)max", np.mean(k_i_max_vals))
    logger.stat("平均EWS(G)max", np.mean(gap_max_vals))

    # 按策略分析EWS
    for regime in regimes:
        regime_ews = [a for r, a in zip(valid_results, ews_analyses)
                     if r["summary"].get("regime") == regime]
        if not regime_ews:
            continue
        regime_fract = sum(1 for a in regime_ews if a["had_fracture"])
        regime_warn = sum(1 for a in regime_ews if a["first_ews_warning_time"] is not None)
        regime_missed = sum(1 for a in regime_ews if a["missed"])
        logger.logger.info(f"\n  策略{regime}: 断裂率={regime_fract/len(regime_ews):.1%} | "
                         f"预警率={regime_warn/len(regime_ews):.1%} | "
                         f"漏报率={regime_missed/max(regime_fract,1):.1%}")

    # 保存
    store.add_batch(results)
    store.set_parameters({"ews_analyses": ews_analyses})
    store.save_all()

    logger.experiment_end(success=True,
                         summary=f"预警提前量均值={np.mean(lead_times) if lead_times else 0:.1f}年 | "
                                f"误报率={false_alarm_rate:.1%} | 漏报率={missed_rate:.1%}")

    print(f"\n✓ EWS验证完成")
    print(f"  结果目录: {store.result_dir}")


if __name__ == "__main__":
    main()
