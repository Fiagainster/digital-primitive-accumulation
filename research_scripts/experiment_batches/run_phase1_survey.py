#!/usr/bin/env python
"""
=============================================================
  第一阶段：策略空间普查 (Strategy Space Survey)
=============================================================

研究问题:
    在国情×策略×场景的三维空间中，哪些组合下系统存活/断裂？
    不同策略产生怎样的多维后果？是否存在反事实轨迹？

实验设计:
    3 国情 × 3 策略 × 3 场景 = 27 个核心组合
    每个组合运行 1 次完整模拟 (T=80年)
    输出完整诊断报告 + 汇总对比矩阵

预计耗时: 10-20 分钟

使用方法:
    python run_phase1_survey.py                          # 默认运行全部 27 组合
    python run_phase1_survey.py --quick                  # 快速版: 9 组合, T=40
    python run_phase1_survey.py --profiles manufacturing_giant,digital_leapfrogger
    python run_phase1_survey.py --skip-completed          # 跳过已完成的组合
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from shared_config import (
    print_header, summarize_simulations, save_phase_results,
    phase_started, phase_completed, phase_status,
    OUTPUT_DIR, RESULT_DIR, LOG_DIR, FIGURE_DIR,
    PHASE1_CORE_PROFILES, PHASE1_CORE_SCENARIOS,
    ALL_REGIMES, PHASE1_T_MAX,
    get_timestamp,
)

# 导入模型基础设施
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from research_scripts.common.runner import quick_run


def run_phase1(profiles=None, scenarios=None, regimes=None, T_max=None,
               quick=False, skip_completed=False):
    """
    执行第一阶段策略空间普查.

    Args:
        profiles: 国情列表 (默认: PHASE1_CORE_PROFILES)
        scenarios: 场景列表 (默认: PHASE1_CORE_SCENARIOS)
        regimes: 策略列表 (默认: ALL_REGIMES)
        T_max: 模拟年限 (默认: PHASE1_T_MAX)
        quick: 快速模式 (减少组合数)
        skip_completed: 跳过已完成的组合
    """
    profiles = profiles or PHASE1_CORE_PROFILES
    scenarios = scenarios or PHASE1_CORE_SCENARIOS
    regimes = regimes or ALL_REGIMES
    T_max = T_max or (40 if quick else PHASE1_T_MAX)

    if quick:
        # 快速模式：核心国情 × 核心场景 × 全部策略
        profiles = ["manufacturing_giant", "digital_leapfrogger"]
        scenarios = ["benign_globalization", "tech_cold_war"]
        T_max = 40

    total = len(profiles) * len(scenarios) * len(regimes)

    print_header("第一阶段：策略空间普查")
    print(f"  实验规模: {len(profiles)}国情 × {len(scenarios)}场景 × {len(regimes)}策略 = {total} 组合")
    print(f"  模拟年限: T_max = {T_max}")
    print(f"  模式: {'快速' if quick else '标准'}")
    print(f"  预计耗时: ~{total * 0.3:.0f}-{total * 0.8:.0f} 分钟")
    print()

    # ---- 记录阶段开始 ----
    record = phase_started("phase1_survey")

    # ---- 输出目录（本次运行） ----
    run_id = f"phase1_{get_timestamp()}"
    run_dir = RESULT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 逐组合运行 ----
    all_results = []
    all_summaries = []
    errors = []
    completed_count = 0
    start_time = time.time()

    for i_p, profile in enumerate(profiles):
        for i_s, scenario in enumerate(scenarios):
            for i_r, regime in enumerate(regimes):
                combo_id = f"{profile}_{regime}_{scenario}"
                idx = len(all_results) + 1

                print(f"[{idx}/{total}] {profile} × 策略{regime} × {scenario} ... ", end="", flush=True)

                try:
                    result = quick_run(
                        regime=regime,
                        profile=profile,
                        scenario=scenario,
                        T_max=T_max,
                    )

                    summary = result.get("summary", {})
                    survived = summary.get("survived", False)
                    surv_years = summary.get("survival_years", 0)
                    final_gdp = summary.get("final_gdp", 0)
                    cdi = summary.get("cdi", None)
                    tech_gap = summary.get("final_tech_gap", 999)
                    n_fractures = summary.get("total_fractures", 0)
                    fracture_types = summary.get("fracture_types", [])

                    status = "存活" if survived else f"断裂@{surv_years:.1f}年"
                    print(f" [{status}] CDI={cdi:.3f}" if cdi else f" [{status}]")

                    entry = {
                        "combo_id": combo_id,
                        "profile": profile,
                        "regime": regime,
                        "scenario": scenario,
                        "T_max": T_max,
                        "survived": survived,
                        "survival_years": surv_years,
                        "final_gdp": final_gdp,
                        "final_tech_gap": tech_gap,
                        "total_fractures": n_fractures,
                        "fracture_types": fracture_types,
                        "cdi": cdi,
                        # 五维分解（如果有）
                        "five_dims": summary.get("five_dimensions", None),
                        # 数字依附（如果有）
                        "composite_dependency": summary.get("composite_dependency", None),
                    }
                    all_summaries.append(entry)
                    all_results.append(result)
                    completed_count += 1

                except Exception as e:
                    print(f" [错误] {e}")
                    errors.append({"combo_id": combo_id, "error": str(e)})
                    all_summaries.append({
                        "combo_id": combo_id,
                        "profile": profile,
                        "regime": regime,
                        "scenario": scenario,
                        "survived": False,
                        "error": str(e),
                    })

        # 每完成一个国情，打印进度
        elapsed = time.time() - start_time
        remaining = (elapsed / max(completed_count, 1)) * (total - len(all_results))
        print(f"  --- 进度: {len(all_results)}/{total} | "
              f"已用: {elapsed/60:.1f}分 | 预计剩余: {remaining/60:.1f}分 ---")

    # ---- 汇总分析 ----
    total_elapsed = time.time() - start_time
    summary_stats = _analyze_phase1(all_summaries)

    # 保存
    save_phase_results("phase1_survey", {
        "run_id": run_id,
        "parameters": {
            "profiles": profiles,
            "scenarios": scenarios,
            "regimes": regimes,
            "T_max": T_max,
            "quick": quick,
        },
        "total_combinations": total,
        "completed": completed_count,
        "errors": errors,
        "elapsed_minutes": total_elapsed / 60,
        "summaries": all_summaries,
        "analysis": summary_stats,
    }, f"{run_id}_results.json")

    # ---- 记录阶段完成 ----
    phase_completed("phase1_survey", summary_stats)

    # ---- 打印关键发现 ----
    _print_phase1_report(summary_stats, all_summaries)

    print(f"\n  总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"  结果目录: {run_dir}")
    return all_summaries, summary_stats


def _analyze_phase1(summaries: list) -> dict:
    """对第一阶段结果做分层统计."""
    import numpy as np

    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return {}
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    # 按策略分组
    by_regime = {}
    for r in ["A", "B", "C"]:
        subset = [s for s in summaries if s.get("regime") == r]
        survived = [s for s in subset if s.get("survived")]
        by_regime[r] = {
            "total": len(subset),
            "survived": len(survived),
            "survival_rate": len(survived) / max(len(subset), 1),
            "cdi": stats([s.get("cdi") for s in subset if s.get("cdi")]),
            "tech_gap": stats([s.get("final_tech_gap") for s in subset if s.get("final_tech_gap")]),
            "gdp": stats([s.get("final_gdp") for s in subset]),
        }

    # 按国情分组
    by_profile = {}
    for p in set(s.get("profile") for s in summaries if s.get("profile")):
        subset = [s for s in summaries if s.get("profile") == p]
        survived = [s for s in subset if s.get("survived")]
        by_profile[p] = {
            "total": len(subset),
            "survived": len(survived),
            "survival_rate": len(survived) / max(len(subset), 1),
        }

    # 按场景分组
    by_scenario = {}
    for sc in set(s.get("scenario") for s in summaries if s.get("scenario")):
        subset = [s for s in summaries if s.get("scenario") == sc]
        survived = [s for s in subset if s.get("survived")]
        by_scenario[sc] = {
            "total": len(subset),
            "survived": len(survived),
            "survival_rate": len(survived) / max(len(subset), 1),
        }

    # 交互效应：国情×策略二维矩阵
    interaction = {}
    for p in set(s.get("profile") for s in summaries if s.get("profile")):
        interaction[p] = {}
        for r in ["A", "B", "C"]:
            subset = [s for s in summaries
                      if s.get("profile") == p and s.get("regime") == r]
            survived = [s for s in subset if s.get("survived")]
            interaction[p][r] = {
                "survival_rate": len(survived) / max(len(subset), 1),
                "avg_cdi": stats([s.get("cdi") for s in subset if s.get("cdi")]).get("mean"),
            }

    return {
        "by_regime": by_regime,
        "by_profile": by_profile,
        "by_scenario": by_scenario,
        "interaction_profile_regime": interaction,
        "overall_survival_rate": len([s for s in summaries if s.get("survived")]) / max(len(summaries), 1),
    }


def _print_phase1_report(stats: dict, summaries: list):
    """打印第一阶段关键发现."""
    print("\n" + "=" * 60)
    print("  第一阶段关键发现")
    print("=" * 60)

    print(f"\n  总体存活率: {stats['overall_survival_rate']:.1%}")

    # 策略排名
    print("\n  [策略存活率排名]")
    by_r = stats["by_regime"]
    for r in sorted(by_r, key=lambda x: by_r[x]["survival_rate"], reverse=True):
        info = by_r[r]
        avg_cdi = info["cdi"].get("mean", 0)
        print(f"    策略{r}: 存活率={info['survival_rate']:.1%} "
              f"({info['survived']}/{info['total']}) 平均CDI={avg_cdi:.3f}" if avg_cdi else "")

    # 国情×策略交互
    print("\n  [国情×策略 存活率矩阵]")
    interaction = stats["interaction_profile_regime"]
    profiles = list(interaction.keys())
    print(f"    {'':>25s}  {'A':>8s}  {'B':>8s}  {'C':>8s}")
    print(f"    {'-' * 50}")
    for p in profiles:
        row = f"    {p:<25s}"
        for r in ["A", "B", "C"]:
            sr = interaction[p][r]["survival_rate"]
            row += f"  {sr:7.1%}"
        print(row)

    # 断裂模式分布
    print("\n  [常见断裂类型]")
    from collections import Counter
    all_fractures = []
    for s in summaries:
        for ft in s.get("fracture_types", []):
            all_fractures.append(ft)
    if all_fractures:
        for ft, count in Counter(all_fractures).most_common(8):
            print(f"    {ft}: {count}次")

    # 异常/值得关注的组合
    print("\n  [值得关注的发现]")
    for s in summaries:
        if s.get("survived") and s.get("total_fractures", 0) >= 4:
            print(f"    ⚠ {s['combo_id']}: 存活但有{s['total_fractures']}次断裂")
        if s.get("final_tech_gap", 0) < 2.0 and not s.get("survived"):
            print(f"    ⚡ {s['combo_id']}: 技术差距较小({s['final_tech_gap']:.1f})但系统已断裂")
        cdi = s.get("cdi")
        if cdi and cdi > 0.75:
            print(f"    ★ {s['combo_id']}: 高CDI={cdi:.3f} 策略{s.get('regime')}")


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="第一阶段：策略空间普查")
    parser.add_argument("--quick", action="store_true",
                       help="快速模式 (2国情×2场景×3策略, T=40)")
    parser.add_argument("--T-max", type=float, default=None,
                       help="模拟年限")
    parser.add_argument("--profiles", type=str, default=None,
                       help="国情列表，逗号分隔 (如 manufacturing_giant,digital_leapfrogger)")
    parser.add_argument("--scenarios", type=str, default=None,
                       help="场景列表，逗号分隔")
    parser.add_argument("--regimes", type=str, default=None,
                       help="策略列表，逗号分隔 (如 A,B,C)")
    parser.add_argument("--skip-completed", action="store_true",
                       help="跳过已完成的组合 (暂未实现)")

    args = parser.parse_args()

    profiles = args.profiles.split(",") if args.profiles else None
    scenarios = args.scenarios.split(",") if args.scenarios else None
    regimes = args.regimes.split(",") if args.regimes else None

    run_phase1(
        profiles=profiles,
        scenarios=scenarios,
        regimes=regimes,
        T_max=args.T_max,
        quick=args.quick,
    )
