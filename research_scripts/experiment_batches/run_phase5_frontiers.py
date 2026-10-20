#!/usr/bin/env python
"""
=============================================================
  第五阶段：前沿边界 (Synthesis & Frontiers)
=============================================================

研究问题 (整合前四阶段发现):
    1. 断裂临界边界精确绘制 — 在关键参数对上精确标定活/死边界
    2. 技术追赶临界条件 — 追赶与锁死的参数相图
    3. 反哺窗口分析 — 策略B数字反哺工业何时兑现？
    4. EWS早期预警验证 — 预警信号在断裂前的行为

实验设计:
    - 断裂边界: 2D高精度网格扫描（基于前序阶段识别的关键参数对）
    - 技术追赶: LHS大样本扫描
    - 反哺窗口: 参数网格×时间窗口
    - EWS验证: 断裂轨迹上的预警信号回溯

预计耗时: 1-4 小时（取决于参数选择）

使用方法:
    python run_phase5_frontiers.py                          # 使用前序阶段建议的参数
    python run_phase5_frontiers.py --focus fracture         # 仅断裂边界
    python run_phase5_frontiers.py --grid-size 30           # 更高精度
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from shared_config import (
    print_header, save_phase_results,
    phase_started, phase_completed, load_phase_summary,
    RESULT_DIR, PHASE5_GRID_SIZE, PHASE5_N_LHS,
    get_timestamp,
)

_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from research_scripts.common.runner import (
    build_param_sets, param_2d_grid, param_grid_explore,
    BatchRunner, get_optimal_workers,
)
from src.core.config import SimulationConfig


def run_phase5(focus="all", grid_size=None, n_workers=None, T_max=80):
    """
    执行第五阶段前沿边界分析.

    Args:
        focus: 'fracture', 'catchup', 'feedback', 'ews', 'all'
        grid_size: 网格精度
        n_workers: 并行worker数
        T_max: 模拟年限
    """
    grid_size = grid_size or PHASE5_GRID_SIZE

    # ---- 从前期结果提取关键参数对 ----
    phase1 = load_phase_summary("phase1_survey")
    phase2 = load_phase_summary("phase2_parameters")
    phase3 = load_phase_summary("phase3_boundaries")

    # 自动确定关键参数对
    key_param_pairs = _determine_key_pairs(phase2, phase3)

    foci = ["fracture", "catchup", "feedback", "ews"] if focus == "all" else [focus]

    print_header("第五阶段：前沿边界")
    print(f"  焦点: {foci}")
    print(f"  网格精度: {grid_size}")
    print(f"  关键参数对: {[(p['name'], p['pair']) for p in key_param_pairs[:3]]}")
    print(f"  T_max = {T_max}")
    print()

    record = phase_started("phase5_frontiers")
    run_id = f"phase5_{get_timestamp()}"
    run_dir = RESULT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    start_time = time.time()

    # ---- 5a. 断裂边界映射 ----
    if "fracture" in foci:
        all_results["fracture_boundaries"] = _run_fracture_boundaries(
            key_param_pairs, grid_size, T_max, n_workers)

    # ---- 5b. 技术追赶临界 ----
    if "catchup" in foci:
        all_results["tech_catchup"] = _run_tech_catchup(
            T_max, n_workers)

    # ---- 5c. 反哺窗口分析 ----
    if "feedback" in foci:
        all_results["feedback_windows"] = _run_feedback_windows(
            T_max, n_workers)

    # ---- 5d. EWS验证 ----
    if "ews" in foci:
        all_results["ews_validation"] = _run_ews_validation(
            T_max, n_workers)

    # ---- 汇总 ----
    total_elapsed = time.time() - start_time
    analysis = _analyze_phase5(all_results)

    save_phase_results("phase5_frontiers", {
        "run_id": run_id,
        "parameters": {
            "focus": foci,
            "grid_size": grid_size,
            "T_max": T_max,
            "key_param_pairs": key_param_pairs,
        },
        "elapsed_minutes": total_elapsed / 60,
        "results": all_results,
        "analysis": analysis,
    }, f"{run_id}_results.json")

    phase_completed("phase5_frontiers", analysis)
    _print_phase5_report(analysis)

    print(f"\n  总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"  结果目录: {run_dir}")
    return all_results, analysis


def _determine_key_pairs(phase2_summary, phase3_summary):
    """从前序阶段确定关键参数对."""
    pairs = []

    # 默认参数对（若前序数据不足则使用）
    default_pairs = [
        {
            "name": "xi_infra_to_app × block_max (策略B, digital_leapfrogger)",
            "pair": ["xi_infra_to_app", "block_max"],
            "regime": "B",
            "profile": "digital_leapfrogger",
            "scenario": "tech_cold_war",
            "range1": [0.05, 0.50],
            "range2": [0.0, 0.95],
        },
        {
            "name": "institution_capacity × B_g_target (策略B, manufacturing_giant)",
            "pair": ["institution_capacity", "B_g_target"],
            "regime": "B",
            "profile": "manufacturing_giant",
            "scenario": "tech_cold_war",
            "range1": [0.20, 0.90],
            "range2": [0.10, 0.40],
        },
        {
            "name": "g_world × xi_infra_to_app (策略A, manufacturing_giant)",
            "pair": ["g_world", "xi_infra_to_app"],
            "regime": "A",
            "profile": "manufacturing_giant",
            "scenario": "benign_globalization",
            "range1": [0.01, 0.12],
            "range2": [0.05, 0.50],
        },
        {
            "name": "institution_capacity × C_tech_import_preference (策略C, resource_rich)",
            "pair": ["institution_capacity", "C_tech_import_preference"],
            "regime": "C",
            "profile": "resource_rich",
            "scenario": "benign_globalization",
            "range1": [0.20, 0.90],
            "range2": [0.10, 0.80],
        },
        {
            "name": "capital_flight_sensitivity × block_max (策略B, digital_leapfrogger)",
            "pair": ["capital_flight_sensitivity", "block_max"],
            "regime": "B",
            "profile": "digital_leapfrogger",
            "scenario": "tech_cold_war",
            "range1": [0.10, 0.90],
            "range2": [0.0, 0.95],
        },
    ]

    # 如果第二阶段有阈值发现，优先使用
    thresholds = phase2_summary.get("thresholds_detected", {})
    if thresholds:
        print("  [提示] 基于第二阶段阈值检测结果调整参数对")
        # 将检测到的阈值参数排在前面
        threshold_params = set()
        for info in thresholds.values():
            threshold_params.add(info.get("param", ""))

        # 将包含阈值参数的配对提前
        for pair in default_pairs:
            if any(p in threshold_params for p in pair["pair"]):
                pairs.insert(0, pair)
            else:
                pairs.append(pair)
    else:
        pairs = default_pairs

    # 如果第三阶段有参数重要性排名，补充到 pairing 信息中
    top_params = phase3_summary.get("top_params", [])
    if top_params:
        top_names = [p["param"] for p in top_params[:5]]
        print(f"  [提示] 第三阶段识别的Top参数: {top_names}")

    return pairs


def _run_fracture_boundaries(key_pairs, grid_size, T_max, n_workers):
    """断裂边界映射."""
    print("\n  [5a] 断裂边界映射...")
    results = {}

    for i, pair_info in enumerate(key_pairs[:3]):  # 最多做3对
        name = pair_info["name"]
        p1, p2 = pair_info["pair"]
        r1 = pair_info.get("range1", [0.0, 1.0])
        r2 = pair_info.get("range2", [0.0, 1.0])

        print(f"    [{i+1}/{min(len(key_pairs), 3)}] {name}")
        print(f"      网格: {grid_size}×{grid_size} = {grid_size**2} 次模拟")

        base = build_param_sets(
            regime=pair_info["regime"],
            profile=pair_info["profile"],
            scenario=pair_info["scenario"],
        )

        v1 = np.linspace(r1[0], r1[1], grid_size)
        v2 = np.linspace(r2[0], r2[1], grid_size)
        param_sets = param_2d_grid(base, p1, list(v1), p2, list(v2))

        config_obj = SimulationConfig(T_max=T_max)
        runner = BatchRunner(n_workers=n_workers)
        sim_results = runner.run(param_sets, regime=pair_info["regime"],
                                config=config_obj, desc=name)

        # 构建存活矩阵
        surv_matrix = np.zeros((grid_size, grid_size))
        fracture_count = np.zeros((grid_size, grid_size))
        for i_idx in range(grid_size):
            for j_idx in range(grid_size):
                idx = i_idx * grid_size + j_idx
                if idx < len(sim_results):
                    r = sim_results[idx]
                    summ = r.get("summary", {})
                    surv_matrix[i_idx, j_idx] = 1 if summ.get("survived") else 0
                    fracture_count[i_idx, j_idx] = summ.get("total_fractures", 0)

        # 找到断裂边界：存活率从 1→0 的过渡带
        boundary_zone = _find_boundary_zone(surv_matrix, v1, v2, p1, p2)

        results[name] = {
            "param_pair": [p1, p2],
            "regime": pair_info["regime"],
            "profile": pair_info["profile"],
            "scenario": pair_info["scenario"],
            "grid_size": grid_size,
            "range1": r1,
            "range2": r2,
            "values1": list(v1),
            "values2": list(v2),
            "survival_matrix": surv_matrix.tolist(),
            "fracture_count_matrix": fracture_count.tolist(),
            "survival_rate": float(surv_matrix.mean()),
            "boundary_zone": boundary_zone,
        }

        print(f"      存活率: {surv_matrix.mean():.1%}")
        if boundary_zone:
            print(f"      边界区域: {boundary_zone['description']}")

    return results


def _find_boundary_zone(surv_matrix, v1, v2, p1, p2):
    """在存活矩阵中寻找活/死过渡带."""
    n1, n2 = surv_matrix.shape
    # 简化方法：找存活率梯度最大的区域
    from scipy.ndimage import sobel
    try:
        grad = np.sqrt(sobel(surv_matrix, axis=0)**2 + sobel(surv_matrix, axis=1)**2)
        max_grad_idx = np.unravel_index(np.argmax(grad), grad.shape)
        return {
            "description": f"最大梯度在 {p1}={v1[max_grad_idx[0]]:.3f}, {p2}={v2[max_grad_idx[1]]:.3f}",
            "gradient_value": float(grad[max_grad_idx]),
            "location": [float(v1[max_grad_idx[0]]), float(v2[max_grad_idx[1]])],
        }
    except Exception:
        return None


def _run_tech_catchup(T_max, n_workers):
    """技术追赶临界条件."""
    print("\n  [5b] 技术追赶临界条件...")
    n_lhs = PHASE5_N_LHS

    combos = [
        ("B", "digital_leapfrogger", "tech_cold_war"),
        ("A", "manufacturing_giant", "benign_globalization"),
    ]

    results = {}
    for regime, profile, scenario in combos:
        combo = f"{regime}_{profile}_{scenario}"
        print(f"    {combo} ...", end=" ", flush=True)

        base = build_param_sets(regime=regime, profile=profile, scenario=scenario)
        tech_params = [
            "xi_infra_to_app", "diff_policy_intensity", "B_g_target",
            "institution_capacity", "block_max",
        ]

        rng = np.random.default_rng(hash(combo + "_tech") % (2**31))
        param_sets = []
        for _ in range(n_lhs):
            p = base.copy()
            for pid in tech_params:
                if pid in p:
                    p[pid] = rng.uniform(0.1, 0.9)
            param_sets.append(p)

        config_obj = SimulationConfig(T_max=T_max)
        runner = BatchRunner(n_workers=n_workers)
        sim_results = runner.run(param_sets, regime=regime, config=config_obj, desc=combo)

        # 分类：追赶成功 vs 锁死
        catchup = []
        lockin = []
        for r in sim_results:
            summ = r.get("summary", {})
            tg = summ.get("final_tech_gap", 999)
            if tg < 2.0:  # 技术差距缩小到2倍以内
                catchup.append(r)
            else:
                lockin.append(r)

        catchup_rate = len(catchup) / max(len(sim_results), 1)

        # 识别区分追赶/锁死的关键参数
        importance = {}
        if catchup and lockin:
            for pid in tech_params:
                c_vals = [r.get("params", {}).get(pid, 0) for r in catchup if r.get("params")]
                l_vals = [r.get("params", {}).get(pid, 0) for r in lockin if r.get("params")]
                if c_vals and l_vals:
                    importance[pid] = abs(np.mean(c_vals) - np.mean(l_vals)) / max(
                        np.std(c_vals) + np.std(l_vals), 1e-8)

        results[combo] = {
            "n_simulations": len(sim_results),
            "catchup": len(catchup),
            "lockin": len(lockin),
            "catchup_rate": catchup_rate,
            "discriminating_params": sorted(importance.items(),
                                           key=lambda x: x[1], reverse=True)[:5],
        }

        print(f"追赶率={catchup_rate:.1%} ({len(catchup)}/{len(sim_results)})")
        if importance:
            top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"      分类参数: {', '.join(f'{k}({v:.2f})' for k, v in top)}")

    return results


def _run_feedback_windows(T_max, n_workers):
    """反哺窗口分析."""
    print("\n  [5c] 反哺窗口分析...")

    base = build_param_sets(regime="B", profile="digital_leapfrogger",
                           scenario="benign_globalization")

    # 扫描反哺阈值和反哺比例
    thresholds = np.linspace(0.20, 0.70, 10)
    ratios = np.linspace(0.05, 0.30, 8)

    param_sets = param_2d_grid(
        base, "B_feedback_threshold", list(thresholds),
        "B_feedback_ratio", list(ratios),
    )

    print(f"    网格: {len(thresholds)} × {len(ratios)} = {len(param_sets)} 次模拟")

    config_obj = SimulationConfig(T_max=T_max)
    runner = BatchRunner(n_workers=n_workers)
    sim_results = runner.run(param_sets, regime="B", config=config_obj,
                            desc="反馈窗口分析")

    # 记录反哺是否触发及其时机
    fb_matrix = np.zeros((len(thresholds), len(ratios)))
    fb_timing = np.zeros((len(thresholds), len(ratios)))
    for i in range(len(thresholds)):
        for j in range(len(ratios)):
            idx = i * len(ratios) + j
            if idx < len(sim_results):
                r = sim_results[idx]
                summ = r.get("summary", {})
                fb_matrix[i, j] = summ.get("feedback_triggered", 0) or 0
                fb_timing[i, j] = summ.get("feedback_timing", T_max) or T_max

    # 找到反哺成功区域
    success_zone = np.where(fb_matrix > 0)
    results = {
        "threshold_values": list(thresholds),
        "ratio_values": list(ratios),
        "feedback_triggered_matrix": fb_matrix.tolist(),
        "feedback_timing_matrix": fb_timing.tolist(),
        "feedback_rate": float(fb_matrix.mean()),
        "mean_timing": float(fb_timing[fb_timing < T_max].mean()) if fb_timing[fb_timing < T_max].size > 0 else None,
    }

    print(f"    反哺触发率: {fb_matrix.mean():.1%}")
    if results["mean_timing"]:
        print(f"    平均触发时点: {results['mean_timing']:.1f}年")

    return results


def _run_ews_validation(T_max, n_workers):
    """EWS早期预警验证."""
    print("\n  [5d] EWS预警验证...")

    base = build_param_sets(regime="B", profile="digital_leapfrogger",
                           scenario="tech_cold_war")

    rng = np.random.default_rng(42)
    n_samples = 50
    ews_params = [
        "B_extraction_ceiling", "B_g_target", "B_feedback_threshold",
        "xi_infra_to_app", "block_max",
    ]

    param_sets = []
    for _ in range(n_samples):
        p = base.copy()
        for pid in ews_params:
            if pid in p:
                p[pid] = rng.uniform(0.1, 0.9)
        param_sets.append(p)

    config_obj = SimulationConfig(T_max=T_max, ews_enabled=True)  # 启用EWS
    runner = BatchRunner(n_workers=n_workers)
    sim_results = runner.run(param_sets, regime="B", config=config_obj,
                            desc="EWS验证")

    # 分析EWS信号
    ews_alerts = []
    fractures = []
    for r in sim_results:
        summ = r.get("summary", {})
        ews = r.get("ews", {})
        if ews:
            ews_alerts.append(ews)
        if not summ.get("survived"):
            fractures.append({"time": summ.get("survival_years"), "ews": ews})

    # 计算预警性能（简化的混淆矩阵）
    # True positive: EWS alert + fracture
    # False positive: EWS alert + survived
    tp = sum(1 for r in sim_results
             if r.get("ews", {}).get("alert", False)
             and not r.get("summary", {}).get("survived"))
    fp = sum(1 for r in sim_results
             if r.get("ews", {}).get("alert", False)
             and r.get("summary", {}).get("survived"))
    fn = sum(1 for r in sim_results
             if not r.get("ews", {}).get("alert", False)
             and not r.get("summary", {}).get("survived"))

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + sum(1 for r in sim_results
                           if not r.get("ews", {}).get("alert", False)
                           and r.get("summary", {}).get("survived")), 1)

    results = {
        "n_simulations": len(sim_results),
        "ews_enabled": True,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "tpr": tpr,
        "fpr": fpr,
        "n_fractures": len(fractures),
    }

    print(f"    TPR (检出率): {tpr:.1%}")
    print(f"    FPR (误报率): {fpr:.1%}")
    print(f"    断裂事件: {len(fractures)}/{len(sim_results)}")

    return results


def _analyze_phase5(all_results: dict) -> dict:
    """汇总第五阶段分析."""
    findings = []

    # 断裂边界
    fb = all_results.get("fracture_boundaries", {})
    for name, result in fb.items():
        bz = result.get("boundary_zone")
        if bz:
            findings.append(f"断裂边界: {bz.get('description', '')}")
        if result["survival_rate"] < 0.3:
            findings.append(f"极端区域: {name} 存活率仅 {result['survival_rate']:.1%}")

    # 技术追赶
    tc = all_results.get("tech_catchup", {})
    for combo, result in tc.items():
        cr = result.get("catchup_rate", 0)
        findings.append(f"技术追赶: {combo} 追赶率={cr:.1%}")
        dp = result.get("discriminating_params", [])
        if dp:
            findings.append(f"  关键区分类参数: {dp[0][0]}")

    # 反哺窗口
    fw = all_results.get("feedback_windows", {})
    mt = fw.get("mean_timing")
    if mt:
        findings.append(f"反哺平均触发时点: {mt:.1f}年")

    # EWS
    ew = all_results.get("ews_validation", {})
    findings.append(f"EWS: TPR={ew.get('tpr', 0):.1%}, FPR={ew.get('fpr', 0):.1%}")

    return {"key_findings": findings}


def _print_phase5_report(analysis: dict):
    print("\n" + "=" * 60)
    print("  第五阶段关键发现 — 综合前沿")
    print("=" * 60)
    findings = analysis.get("key_findings", [])
    for f in findings:
        print(f"    • {f}")


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="第五阶段：前沿边界")
    parser.add_argument("--focus", type=str, default="all",
                       choices=["fracture", "catchup", "feedback", "ews", "all"],
                       help="分析焦点")
    parser.add_argument("--grid-size", type=int, default=None,
                       help="断裂边界网格精度 (默认: 20, 全量: 50)")
    parser.add_argument("--T-max", type=float, default=80,
                       help="模拟年限")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行worker数")

    args = parser.parse_args()
    run_phase5(focus=args.focus, grid_size=args.grid_size,
              n_workers=args.workers, T_max=args.T_max)
