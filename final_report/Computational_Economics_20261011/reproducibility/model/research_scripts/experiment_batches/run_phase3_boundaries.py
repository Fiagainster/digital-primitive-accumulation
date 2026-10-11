#!/usr/bin/env python
"""
=============================================================
  第三阶段：生存边界测绘 (Survival Boundary Mapping)
=============================================================

研究问题:
    在策略参数空间的大范围内，系统的存活边界在哪里？
    不同策略在不同国情下的生存概率如何分布？

实验设计:
    对策略参数进行拉丁超立方采样 (LHS)，每个参数组合重复运行
    以获得稳健的存活概率估计。

    - 采样: LHS (n_samples 个参数组合)
    - 范围: 策略核心参数的 param_domains 定义域
    - 重复: 每个组合 n_repeats 次 (用于稳健性)
    - 输出: 存活概率曲面 + 参数敏感性排名

预计耗时: n_samples=200, n_repeats=2 → ~400次模拟 → 30-60分钟

使用方法:
    python run_phase3_boundaries.py                           # 默认参数
    python run_phase3_boundaries.py --n-samples 100           # 更快但更粗糙
    python run_phase3_boundaries.py --regimes A,B --profiles digital_leapfrogger
    python run_phase3_boundaries.py --quick                   # 快速版: n=50, n_repeats=1
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from shared_config import (
    print_header, save_phase_results,
    phase_started, phase_completed, load_phase_summary,
    RESULT_DIR, PHASE3_N_SAMPLES, PHASE3_T_MAX, PHASE3_N_REPEATS,
    get_timestamp,
)

_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from research_scripts.common.runner import (
    build_param_sets, BatchRunner, get_optimal_workers,
)
from src.core.config import SimulationConfig
from src.params.schema import get_schema
from src.params.presets import get_scenario


def run_phase3(n_samples=None, n_repeats=None, T_max=None, n_workers=None,
               regimes=None, profiles=None, scenarios=None, quick=False):
    """
    执行第三阶段生存边界测绘.
    """
    n_samples = n_samples or (50 if quick else PHASE3_N_SAMPLES)
    n_repeats = n_repeats or (1 if quick else PHASE3_N_REPEATS)
    T_max = T_max or (60 if quick else PHASE3_T_MAX)

    # 默认扫描所有策略×核心国情×核心场景
    regimes = regimes or ["A", "B", "C"]
    profiles = profiles or ["manufacturing_giant", "digital_leapfrogger", "resource_rich"]
    scenarios = scenarios or ["benign_globalization", "tech_cold_war"]

    total_combos = len(regimes) * len(profiles) * len(scenarios)
    total_sims = total_combos * n_samples * n_repeats

    print_header("第三阶段：生存边界测绘")
    print(f"  实验规模: {len(regimes)}策略 × {len(profiles)}国情 × {len(scenarios)}场景")
    print(f"  LHS采样: {n_samples} 个参数组合")
    print(f"  重复次数: {n_repeats}")
    print(f"  总模拟次数: {total_sims}")
    print(f"  T_max = {T_max}")
    print(f"  模式: {'快速' if quick else '标准'}")
    print(f"  预计耗时: ~{total_sims * 1.5 / 60:.0f}-{total_sims * 3 / 60:.0f} 分钟")
    print()

    # ---- 检查前序阶段 ----
    phase2_summary = load_phase_summary("phase2_parameters")
    if phase2_summary:
        thresholds = phase2_summary.get("thresholds_detected", {})
        if thresholds:
            print("  [提示] 第二阶段检测到阈值效应，本阶段将重点验证:")
            for key, info in list(thresholds.items())[:3]:
                print(f"    - {info.get('param', key)}")
            print()

    record = phase_started("phase3_boundaries")
    run_id = f"phase3_{get_timestamp()}"
    run_dir = RESULT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    schema = get_schema()
    all_results = {}
    start_time = time.time()
    total_done = 0

    for regime in regimes:
        for profile in profiles:
            for scenario in scenarios:
                combo_key = f"{regime}_{profile}_{scenario}"
                combo_label = f"策略{regime} × {profile} × {scenario}"
                print(f"\n{'─' * 60}")
                print(f"  [{combo_label}]")
                print(f"{'─' * 60}")

                # 构建LHS参数集
                base = build_param_sets(regime=regime, profile=profile, scenario=scenario)

                # 从 param_domains 获取策略参数的定义域
                try:
                    param_ids, domains = _get_strategy_params_with_domains(schema, regime)
                    if len(param_ids) < 3:
                        print(f"    可用参数不足，跳过")
                        continue
                except Exception as e:
                    print(f"    参数域获取失败: {e}，使用默认参数集")
                    param_ids = None
                    domains = None

                # 生成LHS样本
                rng = np.random.default_rng(hash(combo_key) % (2**31))
                if param_ids and domains:
                    n_params = min(len(param_ids), 12)  # 限制参数数量
                    param_ids = param_ids[:n_params]
                    domains = domains[:n_params]

                    lhs_samples = _lhs_sample(n_samples, len(param_ids), rng)
                    param_sets = []
                    for i in range(n_samples):
                        p = base.copy()
                        for j, pid in enumerate(param_ids):
                            lo, hi = domains[j]
                            p[pid] = lo + (hi - lo) * lhs_samples[i, j]
                        param_sets.append(p)
                else:
                    # 回退：使用 schema 的 LHS 采样
                    from research_scripts.common.runner import sample_lhs_params
                    param_sets = sample_lhs_params(n_samples, regime, scenario)

                # 重复运行
                all_param_sets = param_sets * n_repeats

                config_obj = SimulationConfig(T_max=T_max)
                runner = BatchRunner(n_workers=n_workers)
                results = runner.run(all_param_sets, regime=regime, config=config_obj,
                                    desc=combo_label)

                total_done += len(results)

                # 按参数组合聚合存活率
                aggregated = _aggregate_survival(results, n_repeats, n_samples)

                # 参数重要性排名
                if param_ids:
                    importance = _param_importance(results, param_ids, param_sets)
                else:
                    importance = {}

                all_results[combo_key] = {
                    "regime": regime,
                    "profile": profile,
                    "scenario": scenario,
                    "n_samples": n_samples,
                    "n_repeats": n_repeats,
                    "n_params": len(param_ids) if param_ids else 0,
                    "param_ids": param_ids,
                    "total_survived": sum(1 for r in results
                                         if r.get("summary", {}).get("survived")),
                    "survival_rate": sum(1 for r in results
                                        if r.get("summary", {}).get("survived")) / max(len(results), 1),
                    "survival_by_sample": aggregated,
                    "param_importance": importance,
                }

                print(f"    存活率: {all_results[combo_key]['survival_rate']:.1%}")
                if importance:
                    top3 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"    关键参数: {', '.join(f'{k}({v:.2f})' for k, v in top3)}")

                # 阶段性保存（防止崩溃丢失）
                elapsed = time.time() - start_time
                remaining_rate = (total_done / total_sims) if total_sims > 0 else 0
                if remaining_rate > 0:
                    eta = elapsed / remaining_rate - elapsed
                    print(f"    进度: {total_done}/{total_sims} | 预计剩余: {eta/60:.1f}分")

    # ---- 汇总 ----
    total_elapsed = time.time() - start_time
    analysis = _analyze_phase3(all_results)

    save_phase_results("phase3_boundaries", {
        "run_id": run_id,
        "parameters": {
            "n_samples": n_samples,
            "n_repeats": n_repeats,
            "T_max": T_max,
            "regimes": regimes,
            "profiles": profiles,
            "scenarios": scenarios,
        },
        "total_simulations": total_done,
        "elapsed_minutes": total_elapsed / 60,
        "results": all_results,
        "analysis": analysis,
    }, f"{run_id}_results.json")

    phase_completed("phase3_boundaries", analysis)

    _print_phase3_report(analysis, all_results)

    print(f"\n  总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"  总模拟次数: {total_done}")
    print(f"  结果目录: {run_dir}")
    return all_results, analysis


def _get_strategy_params_with_domains(schema, regime: str):
    """从schema中提取策略相关参数及其定义域."""
    param_ids = []
    domains = []
    # 获取 param_domains
    try:
        domain_defs = schema._data.get("param_domains", {})
    except:
        domain_defs = {}

    all_ids = [pid for pid in schema.all_ids() if schema.get(pid).layer == "strategy"]
    # 过滤与当前策略相关的参数
    relevant_prefixes = []
    if regime in ("A", "B", "C"):
        relevant_prefixes.append(regime + "_")
    relevant_prefixes.extend(["common_", "diff_", "institution_", "platform_"])

    for pid in all_ids:
        if any(pid.startswith(p) for p in relevant_prefixes):
            param_ids.append(pid)
            if pid in domain_defs:
                d = domain_defs[pid]
                domains.append((d.get("min", 0), d.get("max", 1)))
            else:
                # 默认范围
                domains.append((0.0, 1.0))

    # 限制数量
    if len(param_ids) > 15:
        param_ids = param_ids[:15]
        domains = domains[:15]

    return param_ids, domains


def _lhs_sample(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """拉丁超立方采样 (简单实现)."""
    samples = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        samples[:, j] = (perm + rng.random(n)) / n
    return samples


def _aggregate_survival(results: list, n_repeats: int, n_samples: int) -> list:
    """按参数组合聚合存活信息."""
    aggregated = []
    for i in range(0, len(results), n_repeats):
        chunk = results[i:i + n_repeats]
        survived = sum(1 for r in chunk
                      if r.get("summary", {}).get("survived", False))
        aggregated.append({
            "sample_idx": i // n_repeats,
            "survived": survived,
            "total": len(chunk),
            "survival_rate": survived / max(len(chunk), 1),
        })
    return aggregated


def _param_importance(results: list, param_ids: list, param_sets: list) -> dict:
    """简单的参数重要性度量：存活/死亡组间参数均值差异."""
    survived_results = [r for r in results if r.get("summary", {}).get("survived")]
    dead_results = [r for r in results if not r.get("summary", {}).get("survived")]

    if not survived_results or not dead_results:
        return {}

    importance = {}
    for i, pid in enumerate(param_ids):
        surv_vals = [s["params"].get(pid, 0) for s in survived_results
                    if s.get("params") and pid in s["params"]]
        dead_vals = [s["params"].get(pid, 0) for s in dead_results
                    if s.get("params") and pid in s["params"]]
        if surv_vals and dead_vals:
            # Cohen's d 效应量
            mean_diff = abs(np.mean(surv_vals) - np.mean(dead_vals))
            pooled_std = np.sqrt((np.var(surv_vals) + np.var(dead_vals)) / 2)
            if pooled_std > 0:
                importance[pid] = mean_diff / pooled_std
            else:
                importance[pid] = 0

    return importance


def _analyze_phase3(all_results: dict) -> dict:
    """分析第三阶段结果."""
    # 综合存活率矩阵
    survival_matrix = {}
    param_rankings = {}

    for key, result in all_results.items():
        survival_matrix[key] = {
            "regime": result["regime"],
            "profile": result["profile"],
            "scenario": result["scenario"],
            "survival_rate": result["survival_rate"],
        }

        # 聚合参数重要性
        for pid, imp in result.get("param_importance", {}).items():
            if pid not in param_rankings:
                param_rankings[pid] = []
            param_rankings[pid].append(imp)

    # 全局参数排名
    global_param_importance = {
        pid: float(np.mean(imps))
        for pid, imps in param_rankings.items()
    }
    top_params = sorted(global_param_importance.items(),
                       key=lambda x: x[1], reverse=True)[:10]

    return {
        "survival_matrix": survival_matrix,
        "top_params": [{"param": p, "importance": i} for p, i in top_params],
        "global_param_importance": global_param_importance,
    }


def _print_phase3_report(analysis: dict, all_results: dict):
    """打印第三阶段报告."""
    print("\n" + "=" * 60)
    print("  第三阶段关键发现")
    print("=" * 60)

    # 生存率矩阵
    print("\n  [生存率矩阵]")
    matrix = analysis.get("survival_matrix", {})
    if matrix:
        # 按存活率排序
        sorted_items = sorted(matrix.items(), key=lambda x: x[1]["survival_rate"], reverse=True)
        for key, info in sorted_items[:10]:
            print(f"    {info['regime']} × {info['profile']} × {info['scenario']}: "
                  f"存活率={info['survival_rate']:.1%}")

    # 关键参数排名
    print("\n  [全局参数重要性排名 (Top 5)]")
    top = analysis.get("top_params", [])[:5]
    for item in top:
        print(f"    {item['param']}: importance={item['importance']:.3f}")

    # 高风险组合
    print("\n  [需重点关注的组合]")
    for key, info in sorted_items:
        if 0.3 <= info["survival_rate"] <= 0.7:
            print(f"    ⚡ {key}: 高不确定性 存活率={info['survival_rate']:.1%}")


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="第三阶段：生存边界测绘")
    parser.add_argument("--n-samples", type=int, default=None,
                       help="LHS采样数量 (默认: 200)")
    parser.add_argument("--n-repeats", type=int, default=None,
                       help="每个参数组合的重复次数 (默认: 2)")
    parser.add_argument("--T-max", type=float, default=None,
                       help="模拟年限")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行worker数")
    parser.add_argument("--regimes", type=str, default=None,
                       help="策略，逗号分隔")
    parser.add_argument("--profiles", type=str, default=None,
                       help="国情，逗号分隔")
    parser.add_argument("--scenarios", type=str, default=None,
                       help="场景，逗号分隔")
    parser.add_argument("--quick", action="store_true",
                       help="快速模式")

    args = parser.parse_args()

    regimes = args.regimes.split(",") if args.regimes else None
    profiles = args.profiles.split(",") if args.profiles else None
    scenarios = args.scenarios.split(",") if args.scenarios else None

    run_phase3(
        n_samples=args.n_samples,
        n_repeats=args.n_repeats,
        T_max=args.T_max,
        n_workers=args.workers,
        regimes=regimes,
        profiles=profiles,
        scenarios=scenarios,
        quick=args.quick,
    )
