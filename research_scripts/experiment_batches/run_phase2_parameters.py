#!/usr/bin/env python
"""
=============================================================
  第二阶段：关键参数窗口扫描 (Key Parameter Windows)
  【待第一批结果后配置】
=============================================================

⚠ 本脚本需要在第一批实验完成后，根据实际发现来配置扫描方向。

典型配置示例（根据第一批发现选择一个或多个方向）:
    python run_phase2_parameters.py --focus direction_1   # 某策略生存条件
    python run_phase2_parameters.py --focus direction_2   # 某断裂触发边界
    python run_phase2_parameters.py --focus all           # 全部已配置方向

如何配置:
    编辑本文件中的 SWEEP_CONFIGS 字典，根据第一批数据中的关键发现
    替换 example_* 条目为实际的扫描配置。
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
    RESULT_DIR, get_timestamp,
)

_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from research_scripts.common.runner import (
    build_param_sets, param_grid_explore, param_2d_grid,
    BatchRunner, get_optimal_workers,
)
from src.core.config import SimulationConfig


# ================================================================
# Phase-1-Adapted 扫描配置
# ================================================================

# 第一批中的四个"可存活"国情（排除 small_open 和 populous_weak_industry）
FOUR_VIABLE_PROFILES = [
    "manufacturing_giant",
    "resource_rich",
    "digital_leapfrogger",
    "post_soviet",
]

SWEEP_CONFIGS = {
    # ================================================================
    # 方向1: A策略的速度陷阱量化
    # 第一批发现: manuf_giant×A在g≤3%存活, g≥5%死亡
    # 问题: g_world的存活阈值在哪？制度能力能否推高阈值？
    # ================================================================
    "a_speed_trap": {
        "description": "A策略速度陷阱 — g_world阈值与制度调节",
        "sweeps": [
            # 基准场景: g_world × institution_capacity
            {
                "name": "A: g_world × institution_capacity (基准场景)",
                "profiles": ["manufacturing_giant"],
                "regime": "A",
                "scenario": "benign_globalization",
                "T_max": 50,
                "param1": "g_world",
                "values1": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
                "param2": "institution_capacity",
                "values2": [0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95],
            },
            # 压力场景重复: 验证外部封锁是否降低A的g_world耐受上限
            {
                "name": "A: g_world × institution_capacity (tech_cold_war)",
                "profiles": ["manufacturing_giant"],
                "regime": "A",
                "scenario": "tech_cold_war",
                "T_max": 50,
                "param1": "g_world",
                "values1": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
                "param2": "institution_capacity",
                "values2": [0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95],
            },
            # 跨国情验证: A在非制造业大国是否完全不可能存活？
            {
                "name": "A: g_world × institution_capacity (digital_leapfrogger)",
                "profiles": ["digital_leapfrogger"],
                "regime": "A",
                "scenario": "benign_globalization",
                "T_max": 50,
                "param1": "g_world",
                "values1": [0.01, 0.02, 0.03, 0.04, 0.05],
                "param2": "institution_capacity",
                "values2": [0.50, 0.65, 0.80, 0.90, 0.95],
            },
        ],
    },

    # ================================================================
    # 方向2: B策略的国情分叉机制
    # 第一批发现: B在resource_rich/digital_leapfrogger=100%, manuf_giant/post_soviet≈0%
    # 假说: 工业资本存量的差异决定了B的代价是否可控
    # ================================================================
    "b_bifurcation": {
        "description": "B策略国情分叉 — 跨4个可存活国情扫描B核心参数",
        "sweeps": [
            # B_g_target × institution_capacity: 跨4个国情
            {
                "name": "B: B_g_target × institution_capacity (跨4国情)",
                "profiles": FOUR_VIABLE_PROFILES,
                "regime": "B",
                "scenario": "benign_globalization",
                "T_max": 50,
                "param1": "B_g_target",
                "values1": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
                "param2": "institution_capacity",
                "values2": [0.20, 0.35, 0.50, 0.65, 0.80, 0.90],
            },
            # B_feedback_threshold × B_feedback_ratio: 仅B死亡的两个国情
            {
                "name": "B: feedback_threshold × feedback_ratio (manuf_giant + post_soviet)",
                "profiles": ["manufacturing_giant", "post_soviet"],
                "regime": "B",
                "scenario": "benign_globalization",
                "T_max": 50,
                "param1": "B_feedback_threshold",
                "values1": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
                "param2": "B_feedback_ratio",
                "values2": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            },
        ],
    },

    # ================================================================
    # 方向3: C策略在resource_rich的脆弱性边界
    # 第一批发现: resource_rich×C=62.5%，在tech_cold_war/resource_crash/full_containment死亡
    # 问题: 制度能力或参数调整能否挽救C在这些场景的死亡？
    # ================================================================
    "c_fragility": {
        "description": "C策略脆弱性 — resource_rich在封锁/资源崩盘/全封锁下的生存边界",
        "sweeps": [
            # tech_cold_war: 技术进口断裂
            {
                "name": "C: tech_import_preference × institution_capacity (tech_cold_war)",
                "profiles": ["resource_rich"],
                "regime": "C",
                "scenario": "tech_cold_war",
                "T_max": 50,
                "param1": "C_tech_import_preference",
                "values1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "param2": "institution_capacity",
                "values2": [0.20, 0.35, 0.50, 0.65, 0.80, 0.90],
            },
            # resource_crash: 资源出口收入断裂
            {
                "name": "C: resource_intensity × institution_capacity (resource_crash)",
                "profiles": ["resource_rich"],
                "regime": "C",
                "scenario": "resource_crash",
                "T_max": 50,
                "param1": "C_resource_intensity",
                "values1": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "param2": "institution_capacity",
                "values2": [0.20, 0.35, 0.50, 0.65, 0.80, 0.90],
            },
            # full_containment: 技术与资本双重断裂
            {
                "name": "C: block_max × institution_capacity (full_containment)",
                "profiles": ["resource_rich"],
                "regime": "C",
                "scenario": "full_containment",
                "T_max": 50,
                "param1": "block_max",
                "values1": [0.0, 0.20, 0.40, 0.60, 0.80, 0.95],
                "param2": "institution_capacity",
                "values2": [0.20, 0.35, 0.50, 0.65, 0.80, 0.90],
            },
        ],
    },

    # ================================================================
    # 方向4: post_soviet的"只有C能活"之谜
    # 第一批发现: post_soviet (inst=0.40) 只有C存活，A和B全灭
    # 问题: 参数调整能否让A或B在此国情存活？需要多高的制度能力？
    # ================================================================
    "post_soviet_puzzle": {
        "description": "post_soviet之谜 — A和B在post_soviet是否有存活可能？",
        "sweeps": [
            # A在post_soviet: 制度能力 × 全球增速
            {
                "name": "post_soviet×A: institution_capacity × g_world",
                "profiles": ["post_soviet"],
                "regime": "A",
                "scenario": "benign_globalization",
                "T_max": 50,
                "param1": "institution_capacity",
                "values1": [0.20, 0.35, 0.50, 0.65, 0.80, 0.90],
                "param2": "g_world",
                "values2": [0.01, 0.02, 0.03, 0.05, 0.07],
            },
            # B在post_soviet: 制度能力 × 提取率目标
            {
                "name": "post_soviet×B: institution_capacity × B_g_target",
                "profiles": ["post_soviet"],
                "regime": "B",
                "scenario": "benign_globalization",
                "T_max": 50,
                "param1": "institution_capacity",
                "values1": [0.20, 0.35, 0.50, 0.65, 0.80, 0.90],
                "param2": "B_g_target",
                "values2": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            },
        ],
    },

    # ================================================================
    # 第三批方向2: A策略相变边界高精度测绘
    # 第二批发现: g_world在[0.01,0.02]有43%存活率跳变, inst在[0.65,0.80]有80%跳变
    # 目标: 加密g_world和inst的网格精度，绘制精确相变边界
    # ================================================================
    "a_phase_boundary": {
        "description": "A策略相变边界 — 高精度(g_world, inst)网格",
        "sweeps": [
            # manuf_giant高精度网格
            {
                "name": "A相变边界: g_world × inst (manuf_giant, 高精度)",
                "profiles": ["manufacturing_giant"],
                "regime": "A",
                "scenario": "benign_globalization",
                "T_max": 50,
                "param1": "g_world",
                "values1": [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050],
                "param2": "institution_capacity",
                "values2": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            },
            # post_soviet验证网格
            {
                "name": "A相变边界: g_world × inst (post_soviet, 验证)",
                "profiles": ["post_soviet"],
                "regime": "A",
                "scenario": "benign_globalization",
                "T_max": 50,
                "param1": "g_world",
                "values1": [0.010, 0.015, 0.020, 0.030, 0.050],
                "param2": "institution_capacity",
                "values2": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            },
        ],
    },
}


# ================================================================
# 数值异常检测
# ================================================================

GDP_EXPLOSION_THRESHOLD = 1e15   # GDP超过此值视为数值爆炸
TECH_GAP_ALERT = 200.0           # 技术差距超过此值发出警告


def _flag_anomalies(results: list) -> dict:
    """检测模拟结果中的数值异常."""
    anomalies = {
        "gdp_explosions": 0,
        "extreme_tech_gaps": 0,
        "nan_results": 0,
        "details": [],
    }
    for r in results:
        summ = r.get("summary", {})
        gdp = summ.get("final_gdp", 0)
        tg = summ.get("final_tech_gap", 0)

        if gdp > GDP_EXPLOSION_THRESHOLD:
            anomalies["gdp_explosions"] += 1
            anomalies["details"].append({
                "type": "gdp_explosion",
                "gdp": float(gdp),
                "regime": summ.get("regime"),
                "survived": summ.get("survived"),
            })
        if tg > TECH_GAP_ALERT:
            anomalies["extreme_tech_gaps"] += 1
        if gdp != gdp or tg != tg:  # NaN check
            anomalies["nan_results"] += 1

    return anomalies


# ================================================================
# 主函数
# ================================================================

def run_phase2(focus="all", T_max=None, n_workers=None):
    """
    Phase 2 主入口.
    Args:
        focus: 'b_paradox', 'c_lockin', 'a_survival', 'all'
    """
    if focus == "all":
        foci = list(SWEEP_CONFIGS.keys())
    else:
        foci = [focus]

    # ---- 载入Phase 1结果 ----
    phase1_summary = load_phase_summary("phase1_survey")
    if phase1_summary:
        print("  [Phase 1 回顾]")
        p1 = phase1_summary
        print(f"    总体存活率: {p1.get('overall_survival_rate', '?'):.1%}" if isinstance(p1.get('overall_survival_rate'), float) else f"    总体存活率: {p1.get('overall_survival_rate', '?')}")
        by_r = p1.get("by_regime", {})
        for r in ["A", "B", "C"]:
            info = by_r.get(r, {})
            print(f"    策略{r}: 存活率={info.get('survival_rate', 0):.1%}")
        print()

    print_header(f"第二阶段：关键参数窗口扫描 (focus={focus})")
    print(f"  扫描组: {foci}")
    print(f"  并行worker: {get_optimal_workers(n_workers)}")
    print(f"  数值爆炸阈值: GDP > {GDP_EXPLOSION_THRESHOLD:.0e}")
    print()

    record = phase_started("phase2_parameters")
    run_id = f"phase2_{get_timestamp()}"
    run_dir = RESULT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_anomalies = {}
    total_simulations = 0
    start_time = time.time()

    for focus_key in foci:
        if focus_key not in SWEEP_CONFIGS:
            print(f"  ⚠ 未知焦点 '{focus_key}'，跳过")
            continue

        config = SWEEP_CONFIGS[focus_key]
        print(f"\n{'=' * 60}")
        print(f"  [{focus_key}] {config['description']}")
        print(f"{'=' * 60}")

        focus_results = []
        focus_anomalies = []

        for sweep in config["sweeps"]:
            profiles = sweep.get("profiles", [sweep.get("profile", "manufacturing_giant")])
            regime = sweep["regime"]
            scenario = sweep["scenario"]

            for profile in profiles:
                name = f"{sweep['name']} | {profile}"
                print(f"\n  {name}")
                print(f"    策略={regime} 场景={scenario}")

                base = build_param_sets(
                    regime=regime, profile=profile, scenario=scenario,
                )

                # 1D 还是 2D 扫描？
                if "param2" in sweep and "values2" in sweep:
                    param_sets = param_2d_grid(
                        base, sweep["param1"], sweep["values1"],
                        sweep["param2"], sweep["values2"],
                    )
                    n = len(param_sets)
                    n1, n2 = len(sweep["values1"]), len(sweep["values2"])
                    print(f"    2D网格: {n1}×{n2} = {n} 次模拟")
                else:
                    param_sets = param_grid_explore(
                        base, sweep["param1"], sweep["values1"],
                    )
                    n = len(param_sets)
                    print(f"    1D扫描: {n} 次模拟")

                total_simulations += n

                _T = sweep.get("T_max", T_max or 80)
                config_obj = SimulationConfig(T_max=_T)
                runner = BatchRunner(n_workers=n_workers)
                results = runner.run(param_sets, regime=regime,
                                    config=config_obj, desc=name)

                # ---- 提取结果 ----
                survived = [r for r in results
                           if r.get("summary", {}).get("survived", False)]

                # 提取额外指标 (CDI, 技术差距, GDP)
                cdi_values = []
                tech_gaps = []
                gdps = []
                dependencies = []
                for r in results:
                    ev = r.get("evaluation", {})
                    summ = r.get("summary", {})
                    if ev.get("enabled"):
                        cdi_values.append(ev.get("cdi"))
                    tech_gaps.append(summ.get("final_tech_gap", 0))
                    gdps.append(summ.get("final_gdp", 0))
                    dependencies.append(summ.get("dependency_composite"))

                # 沿param1的存活率曲线
                p1_vals = sweep["values1"]
                if "values2" in sweep:
                    # 2D：对每个param1值聚合所有param2的结果
                    surv_rates = []
                    for i in range(len(p1_vals)):
                        chunk = results[i * len(sweep["values2"]):(i+1) * len(sweep["values2"])]
                        sr = len([r for r in chunk
                                 if r.get("summary", {}).get("survived")]) / len(chunk)
                        surv_rates.append({"value": p1_vals[i], "survival_rate": sr})
                else:
                    surv_rates = [
                        {"value": p1_vals[i],
                         "survival_rate": (1 if results[i].get("summary", {}).get("survived")
                                          else 0)}
                        for i in range(len(p1_vals))
                    ]

                # 数值异常检测
                anomalies = _flag_anomalies(results)
                if anomalies["gdp_explosions"] > 0:
                    print(f"    ⚠ 检测到 {anomalies['gdp_explosions']} 次GDP数值爆炸!")
                if anomalies["nan_results"] > 0:
                    print(f"    ⚠ 检测到 {anomalies['nan_results']} 次NaN结果!")
                if anomalies["extreme_tech_gaps"] > 0:
                    print(f"    ⚠ 检测到 {anomalies['extreme_tech_gaps']} 次极端技术差距(>{TECH_GAP_ALERT})")

                # 存活率
                sr = len(survived) / max(n, 1)
                print(f"    存活率: {len(survived)}/{n} = {sr:.1%}")

                # 存活率随参数的变化范围
                if surv_rates:
                    nonzero = [s for s in surv_rates if s["survival_rate"] > 0]
                    if nonzero:
                        min_sr = min(surv_rates, key=lambda x: x["survival_rate"])
                        max_sr = max(surv_rates, key=lambda x: x["survival_rate"])
                        print(f"    存活率范围: [{min_sr['survival_rate']:.0%} @{min_sr['value']}] → [{max_sr['survival_rate']:.0%} @{max_sr['value']}]")
                        if max_sr["survival_rate"] > 0.5:
                            print(f"    ★ 存在存活率>50%的参数区域！")

                # CDI范围
                cdi_clean = [c for c in cdi_values if c is not None]
                if cdi_clean:
                    print(f"    CDI范围: [{min(cdi_clean):.3f}, {max(cdi_clean):.3f}]")

                # 技术差距范围
                if tech_gaps:
                    tg_clean = [t for t in tech_gaps if t is not None and t < TECH_GAP_ALERT]
                    if tg_clean:
                        print(f"    技术差距: [{min(tg_clean):.1f}, {max(tg_clean):.1f}]")

                # 存储
                sweep_result = {
                    "name": name,
                    "regime": regime,
                    "profile": profile,
                    "scenario": scenario,
                    "param1": sweep["param1"],
                    "param2": sweep.get("param2"),
                    "values1": sweep["values1"],
                    "values2": sweep.get("values2"),
                    "total_sims": n,
                    "survived_count": len(survived),
                    "survival_rate": sr,
                    "survival_by_value1": surv_rates,
                    "cdi_range": [float(min(cdi_clean)), float(max(cdi_clean))] if cdi_clean else None,
                    "tech_gap_range": [float(min(tech_gaps)), float(max(tech_gaps))] if tech_gaps else None,
                    "gdp_range": [float(min(gdps)), float(max(gdps))] if gdps else None,
                    "anomalies": anomalies,
                }
                focus_results.append(sweep_result)
                if anomalies["gdp_explosions"] > 0 or anomalies["nan_results"] > 0:
                    focus_anomalies.append(sweep_result)

        all_results[focus_key] = focus_results
        all_anomalies[focus_key] = focus_anomalies

        # 焦点小结
        n_survived_total = sum(r["survived_count"] for r in focus_results)
        n_total = sum(r["total_sims"] for r in focus_results)
        print(f"\n  [{focus_key} 小结] 总存活率: {n_survived_total}/{n_total} = {n_survived_total/max(n_total,1):.1%}")

    # ---- 汇总分析 ----
    total_elapsed = time.time() - start_time
    analysis = _analyze_phase2(all_results, all_anomalies)

    save_phase_results("phase2_parameters", {
        "run_id": run_id,
        "focus": foci,
        "total_simulations": total_simulations,
        "elapsed_minutes": total_elapsed / 60,
        "anomalies_detected": sum(len(a) for a in all_anomalies.values()),
        "results": all_results,
        "analysis": analysis,
        "phase1_context": phase1_summary,
    }, f"{run_id}_results.json")

    phase_completed("phase2_parameters", analysis)

    _print_phase2_report(analysis, all_anomalies)

    print(f"\n  总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"  总模拟次数: {total_simulations}")
    print(f"  数值异常检测: {sum(len(a) for a in all_anomalies.values())} 项")
    print(f"  结果目录: {run_dir}")
    return all_results, analysis


def _analyze_phase2(all_results: dict, all_anomalies: dict) -> dict:
    """分析Phase 2结果."""
    findings = []
    thresholds = {}

    for focus_key, focus_results in all_results.items():
        for result in focus_results:
            surv_by_val = result.get("survival_by_value1", [])
            if len(surv_by_val) < 3:
                continue

            param_name = result["param1"]
            profile = result.get("profile", "?")
            values = [s["value"] for s in surv_by_val]
            rates = [s["survival_rate"] for s in surv_by_val]

            # 存活率跳变检测 (>30%)
            for i in range(1, len(rates)):
                jump = abs(rates[i] - rates[i-1])
                if jump > 0.3:
                    key = f"{focus_key}_{param_name}_{profile}"
                    thresholds[key] = {
                        "param": param_name,
                        "profile": profile,
                        "jump_magnitude": float(jump),
                        "jump_interval": [float(values[i-1]), float(values[i])],
                    }
                    findings.append(
                        f"[阈值] {param_name}@{profile}: 存活率在"
                        f" [{values[i-1]:.3f}, {values[i]:.3f}] 跳变 {jump:.0%}"
                    )

            # 是否存在 >0 的存活率（对A策略特别重要）
            if result["regime"] == "A" and max(rates) > 0:
                findings.append(
                    f"★ [A可存活!] {profile}: {param_name}={values[rates.index(max(rates))]:.3f} "
                    f"时存活率达 {max(rates):.0%}"
                )

            # 数值异常
            anomalies = result.get("anomalies", {})
            if anomalies.get("gdp_explosions", 0) > 0:
                findings.append(
                    f"⚠ [数值爆炸] {profile}×{result['regime']}: "
                    f"{anomalies['gdp_explosions']}次 GDP>{GDP_EXPLOSION_THRESHOLD:.0e}"
                )

    # 异常汇总
    anomaly_summary = {}
    for focus_key, anomalies in all_anomalies.items():
        for a in anomalies:
            anomaly_summary[f"{a['profile']}_{a['regime']}_{a['param1']}"] = a.get("anomalies", {})

    return {
        "thresholds_detected": thresholds,
        "key_findings": findings,
        "n_thresholds": len(thresholds),
        "anomaly_summary": anomaly_summary,
    }


def _print_phase2_report(analysis: dict, all_anomalies: dict):
    """打印Phase 2报告."""
    print("\n" + "=" * 64)
    print("  第二阶段关键发现")
    print("=" * 64)

    findings = analysis.get("key_findings", [])
    if findings:
        # 分组打印
        threshold_findings = [f for f in findings if f.startswith("[阈值]")]
        survival_findings = [f for f in findings if "可存活" in f]
        anomaly_findings = [f for f in findings if f.startswith("⚠")]

        if survival_findings:
            print("\n  [策略A的生存条件]")
            for f in survival_findings:
                print(f"    {f}")

        if threshold_findings:
            print("\n  [参数阈值效应]")
            for f in threshold_findings[:10]:
                print(f"    {f}")

        if anomaly_findings:
            print("\n  [数值异常警告]")
            for f in anomaly_findings:
                print(f"    {f}")
    else:
        print("\n  (未检测到显著发现)")

    total_anomalies = sum(len(a) for a in all_anomalies.values())
    if total_anomalies > 0:
        print(f"\n  ⚠ 共检测到 {total_anomalies} 组数值异常，详情见结果JSON")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="第二阶段：关键参数窗口扫描 (Phase 1 适配版)")
    parser.add_argument("--focus", type=str, default="all",
                       help="扫描焦点: 根据SWEEP_CONFIGS中的键名 (如 direction_1,direction_2 或 all)")
    parser.add_argument("--T-max", type=float, default=None,
                       help="模拟年限")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行worker数")

    args = parser.parse_args()
    run_phase2(focus=args.focus, T_max=args.T_max, n_workers=args.workers)
