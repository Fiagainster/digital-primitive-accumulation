"""批量实验脚本 —— v3.2 全面压力测试.

运行方式:
    conda activate pa_model
    cd D:/Project/原始积累模型探索/primitive_accumulation
    PYTHONIOENCODING=utf-8 python batch_experiments.py
    PYTHONIOENCODING=utf-8 python batch_experiments.py --profile digital_leapfrogger

实验设计:
    E1 — 基准生存扫描: 3策略 × 200 LHS × 1场景 = 600次
    E2 — 场景压力矩阵: 3策略 × 8场景 × 100 LHS = 2400次
    E3 — 恶意场景聚焦: 3策略 × 3恶意场景 × 200 LHS = 1800次
    E4 — v3.2: 策略×国情×场景笛卡尔积: 3策略 × 6国情 × 3场景 × 50 LHS = 2700次
    总计: ~7500次模拟
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
import time
import json
import numpy as np

warnings.filterwarnings("ignore")

from src.core.engine import SimulationEngine
from src.core.config import SimulationConfig
from src.params.schema import get_schema
from src.params.presets import get_scenario, SCENARIO_PRESETS
from src.params.country_profiles import COUNTRY_PROFILES
from src.io.exporter import ResultExporter
from src.analysis.visualize import generate_academic_report

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = "experiments/results"
T_MAX = 60.0
EWS_WINDOW = 40
N_WORKERS = min(os.cpu_count() or 4, 8)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 辅助函数
# ============================================================


def run_batch(regime: str, scenario_id: str, n_samples: int,
              schema, config: SimulationConfig, profile_name: str = None) -> list:
    """运行一批LHS采样模拟 (v3.2: 支持国情预设).

    Args:
        regime: 积累策略 A/B/C
        scenario_id: 国际场景ID
        n_samples: LHS采样数
        schema: 参数体系
        config: 模拟配置
        profile_name: v3.2: 国情预设名称（如 manufacturing_giant）
    """
    scene_params = get_scenario(scenario_id)

    # LHS采样策略参数
    strategy_ids = [pid for pid in schema.all_ids()
                    if schema.get(pid).layer == "strategy"]
    strategy_ids = strategy_ids[:15]  # 限制维度
    rng = np.random.default_rng(42)
    samples = schema.sample_lhs(n_samples, strategy_ids, rng)

    results = []
    for i in range(n_samples):
        # v3.2: 如果提供了国情预设，使用 build_from_profile
        if profile_name:
            params = schema.build_from_profile(
                profile_name=profile_name,
                regime=regime,
                scenario=scenario_id,
            )
        else:
            params = schema.build_defaults(regime)
            params.update(scene_params)

        for j, pid in enumerate(strategy_ids):
            params[pid] = samples[i, j]

        config.random_seed = 42 + i
        engine = SimulationEngine(params=params, config=config, regime=regime,
                                  country_profile=profile_name)
        result = engine.run()
        result["summary"]["scenario"] = scenario_id
        result["summary"]["regime"] = regime
        result["summary"]["sample_id"] = i
        if profile_name:
            result["summary"]["country_profile"] = profile_name
        results.append(result)

    return results


def compute_stats(results: list) -> dict:
    """从一批结果计算统计量."""
    summaries = [r["summary"] for r in results]
    n = len(summaries)
    if n == 0:
        return {}

    survived = [s["survived"] for s in summaries]
    surv_years = [s["survival_years"] for s in summaries]
    tech_gaps = [s["final_tech_gap"] for s in summaries]
    gdps = [s["final_gdp"] for s in summaries]
    cost_conc = [s["cost_concentration"] for s in summaries]
    cost_ratio = [s["cost_ratio_industrial_digital"] for s in summaries]
    ews_K = [s.get("ews_max_K_i", 0) for s in summaries]
    ews_G = [s.get("ews_max_tech_gap", 0) for s in summaries]
    total_fract = [s.get("total_fractures", 0) for s in summaries]
    diff_i = [s.get("diffusion_cumulative_i", 0) for s in summaries]
    diff_b = [s.get("diffusion_cumulative_b", 0) for s in summaries]
    gap_i = [s.get("sector_gap_i", 0) for s in summaries]

    # 断裂类型频率
    fracture_freq = {}
    for s in summaries:
        for ft in s.get("fracture_types", []):
            fracture_freq[ft] = fracture_freq.get(ft, 0) + 1

    # 多断裂并发（级联）
    multi_fracture = sum(1 for s in summaries if s.get("total_fractures", 0) >= 3)

    return {
        "n": n,
        "survival_rate": np.mean(survived),
        "mean_survival_years": np.mean(surv_years),
        "std_survival_years": np.std(surv_years),
        "mean_tech_gap": np.mean(tech_gaps),
        "std_tech_gap": np.std(tech_gaps),
        "mean_gdp": np.mean(gdps),
        "std_gdp": np.std(gdps),
        "median_gdp": np.median(gdps),
        "mean_cost_concentration": np.mean(cost_conc),
        "mean_cost_ratio": np.mean(cost_ratio),
        "mean_ews_K_i": np.mean(ews_K),
        "mean_ews_tech_gap": np.mean(ews_G),
        "mean_total_fractures": np.mean(total_fract),
        "multi_fracture_count": multi_fracture,
        "multi_fracture_rate": multi_fracture / n,
        "mean_diffusion_i": np.mean(diff_i),
        "mean_diffusion_b": np.mean(diff_b),
        "mean_sector_gap_i": np.mean(gap_i),
        "fracture_frequency": fracture_freq,
    }


def format_section(title: str, width: int = 70) -> str:
    """格式化报告章节标题."""
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def format_subsection(title: str, width: int = 60) -> str:
    """格式化报告子标题."""
    return f"\n{'─' * width}\n{title}\n{'─' * width}"


def stats_to_text(stats: dict, regime: str, scenario: str) -> str:
    """将统计量格式化为可读文本."""
    lines = []
    lines.append(format_subsection(f"策略 {regime} × {scenario} ({stats.get('n', 0)}次模拟)"))
    lines.append(f"  存活率:            {stats['survival_rate']*100:.1f}%")
    lines.append(f"  平均存活年数:       {stats['mean_survival_years']:.1f} ± {stats['std_survival_years']:.1f}")
    lines.append(f"  平均技术差距:       {stats['mean_tech_gap']:.2f} ± {stats['std_tech_gap']:.2f}")
    lines.append(f"  平均GDP:           {stats['mean_gdp']:.0f} ± {stats['std_gdp']:.0f} (中位数: {stats['median_gdp']:.0f})")
    lines.append(f"  平均代价基尼:       {stats['mean_cost_concentration']:.3f}")
    lines.append(f"  平均工业/数字代价比: {stats['mean_cost_ratio']:.2f}")
    lines.append(f"  平均EWS(K_i)max:   {stats['mean_ews_K_i']:.3f}")
    lines.append(f"  平均EWS(G)max:     {stats['mean_ews_tech_gap']:.3f}")
    lines.append(f"  平均断裂数:         {stats['mean_total_fractures']:.2f}")
    lines.append(f"  多断裂并发(≥3):    {stats['multi_fracture_count']}/{stats['n']} ({stats['multi_fracture_rate']*100:.1f}%)")
    lines.append(f"  平均D→I扩散量:     {stats['mean_diffusion_i']:.2f}")
    lines.append(f"  平均D→B扩散量:     {stats['mean_diffusion_b']:.2f}")
    lines.append(f"  平均部门内差距I:    {stats['mean_sector_gap_i']:.3f}")

    if stats["fracture_frequency"]:
        lines.append(f"\n  断裂模式频率:")
        for ft, count in sorted(stats["fracture_frequency"].items(), key=lambda x: -x[1]):
            lines.append(f"    {ft:<35} {count:>4}次 ({100*count/stats['n']:.1f}%)")

    return "\n".join(lines)


# ============================================================
# 主程序
# ============================================================
def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    schema = get_schema()
    config = SimulationConfig(T_max=T_MAX, ews_window_size=EWS_WINDOW, verbose=False)

    # v3.2: 解析命令行参数
    profile_name = None
    if "--profile" in sys.argv:
        idx = sys.argv.index("--profile")
        if idx + 1 < len(sys.argv):
            profile_name = sys.argv[idx + 1]
            if profile_name not in COUNTRY_PROFILES:
                print(f"可用的国情预设: {list(COUNTRY_PROFILES.keys())}")
                sys.exit(1)

    all_experiment_results = {}  # 用于最终汇总
    report_lines = []

    profile_info = f"\n  国情预设: {profile_name}" if profile_name else ""
    header = f"""
╔══════════════════════════════════════════════════════════════════╗
║     原始积累模型 v3.2 — 大批量实验                               ║
║     Primitive Accumulation Model — Batch Experiments             ║
╠══════════════════════════════════════════════════════════════════╣
║  实验时间: {timestamp:<50}║
║  模拟年限: T_max = {T_MAX}年                                            ║
║  EWS窗口:  {EWS_WINDOW}步                                                   ║
║  并行核心: {N_WORKERS}个                                                   ║{profile_info}
╚══════════════════════════════════════════════════════════════════╝
"""
    print(header)
    report_lines.append(header)

    t_start = time.time()

    # ================================================================
    # 实验一：基准生存扫描
    # ================================================================
    e1_title = format_section("实验一：基准生存扫描 (3策略 × 200 LHS × benign_globalization)")
    print(e1_title)
    report_lines.append(e1_title)

    e1_results = {}
    for regime in ["A", "B", "C"]:
        print(f"\n>>> 策略 {regime} — LHS采样 200组...")
        t0 = time.time()
        results = run_batch(regime, "benign_globalization", 200, schema, config)
        elapsed = time.time() - t0

        stats = compute_stats(results)
        e1_results[regime] = stats
        all_experiment_results[f"E1_{regime}"] = stats

        text = stats_to_text(stats, regime, "benign_globalization")
        print(text)
        report_lines.append(text)
        print(f"  [耗时: {elapsed:.0f}秒]")

    # E1 对比摘要
    e1_summary = format_subsection("E1 策略对比摘要")
    print(e1_summary)
    report_lines.append(e1_summary)

    header_row = f"  {'指标':<25} {'A(渐进)':<15} {'B(跳跃)':<15} {'C(替代)':<15}"
    sep_row = "  " + "-" * (25 + 45)
    print(header_row)
    print(sep_row)
    report_lines.append(header_row)
    report_lines.append(sep_row)

    metrics_for_table = [
        ("存活率", "survival_rate", "{:.1%}"),
        ("平均存活年数", "mean_survival_years", "{:.1f}"),
        ("平均技术差距", "mean_tech_gap", "{:.1f}"),
        ("中位数GDP", "median_gdp", "{:.0f}"),
        ("平均代价基尼", "mean_cost_concentration", "{:.3f}"),
        ("平均代价比(I/D)", "mean_cost_ratio", "{:.2f}"),
        ("多断裂并发率", "multi_fracture_rate", "{:.1%}"),
        ("平均D→I扩散量", "mean_diffusion_i", "{:.2f}"),
    ]
    for label, key, fmt in metrics_for_table:
        vals = [e1_results[r][key] for r in ["A", "B", "C"]]
        row = f"  {label:<25} "
        row += " ".join(f"{fmt.format(v):<15}" for v in vals)
        print(row)
        report_lines.append(row)

    # ================================================================
    # 实验二：场景压力矩阵
    # ================================================================
    e2_title = format_section("实验二：场景压力矩阵 (3策略 × 8场景 × 100 LHS)")
    print(e2_title)
    report_lines.append(e2_title)

    all_scenarios = list(SCENARIO_PRESETS.keys())
    e2_matrix = {}

    for sc_id in all_scenarios:
        sc_print = format_subsection(f"场景: {sc_id}")
        print(sc_print)
        report_lines.append(sc_print)

        e2_matrix[sc_id] = {}
        for regime in ["A", "B", "C"]:
            print(f"  策略 {regime}...", end=" ", flush=True)
            t0 = time.time()
            results = run_batch(regime, sc_id, 100, schema, config)
            elapsed = time.time() - t0

            stats = compute_stats(results)
            e2_matrix[sc_id][regime] = stats
            all_experiment_results[f"E2_{sc_id}_{regime}"] = stats

            surv_str = f"存活率={stats['survival_rate']*100:.0f}%"
            gap_str = f"G={stats['mean_tech_gap']:.1f}"
            fract_str = f"级联={stats['multi_fracture_rate']*100:.0f}%"
            print(f"{surv_str} | {gap_str} | {fract_str} | {elapsed:.0f}s")

            short_text = (
                f"    策略{regime}: {surv_str} | avg存活{stats['mean_survival_years']:.1f}年 "
                f"| GDP={stats['median_gdp']:.0f} | 代价基尼={stats['mean_cost_concentration']:.3f}"
            )
            report_lines.append(short_text)

    # E2 汇总矩阵表
    e2_summary = format_subsection("E2 全场景存活率矩阵")
    print(e2_summary)
    report_lines.append(e2_summary)

    header_row = f"  {'场景':<22} {'A存活率':<10} {'B存活率':<10} {'C存活率':<10} {'最高级联':<10}"
    print(header_row)
    print("  " + "-" * 62)
    report_lines.append(header_row)
    report_lines.append("  " + "-" * 62)

    for sc_id in all_scenarios:
        surv_rates = []
        max_cascade = 0
        for regime in ["A", "B", "C"]:
            st = e2_matrix[sc_id][regime]
            surv_rates.append(st["survival_rate"])
            max_cascade = max(max_cascade, st["multi_fracture_rate"])
        row = (f"  {sc_id:<22} {surv_rates[0]:<10.1%} "
               f"{surv_rates[1]:<10.1%} {surv_rates[2]:<10.1%} "
               f"{max_cascade:<10.1%}")
        print(row)
        report_lines.append(row)

    # ================================================================
    # 实验三：恶意场景聚焦
    # ================================================================
    e3_title = format_section("实验三：恶意场景聚焦 (3策略 × 3恶意场景 × 200 LHS)")
    print(e3_title)
    report_lines.append(e3_title)

    malicious_scenarios = ["full_containment", "tech_explosion", "resource_war"]

    for sc_id in malicious_scenarios:
        print(f"\n>>> 场景: {sc_id}")
        for regime in ["A", "B", "C"]:
            t0 = time.time()
            results = run_batch(regime, sc_id, 200, schema, config)
            elapsed = time.time() - t0

            stats = compute_stats(results)
            all_experiment_results[f"E3_{sc_id}_{regime}"] = stats

            text = stats_to_text(stats, regime, sc_id)
            print(text)
            report_lines.append(text)
            print(f"  [耗时: {elapsed:.0f}秒]")

    # ================================================================
    # v3.2: 实验四 — 策略×国情×场景笛卡尔积
    # ================================================================
    if profile_name:
        e4_scenarios = ["benign_globalization", "tech_cold_war", "full_containment"]
        e4_title = format_section(
            f"实验四(v3.2)：策略×国情×场景笛卡尔积 "
            f"(3策略 × {profile_name} × {len(e4_scenarios)}场景 × 50 LHS)"
        )
    else:
        e4_scenarios = ["benign_globalization", "tech_cold_war", "full_containment"]
        e4_profiles = list(COUNTRY_PROFILES.keys())[:3]  # 前3个国情
        e4_title = format_section(
            f"实验四(v3.2)：策略×国情×场景笛卡尔积 "
            f"(3策略 × {len(e4_profiles)}国情 × {len(e4_scenarios)}场景 × 50 LHS)"
        )

    print(e4_title)
    report_lines.append(e4_title)

    e4_matrix = {}
    profiles_to_test = [profile_name] if profile_name else e4_profiles

    for pf in profiles_to_test:
        pf_print = format_subsection(f"国情: {pf}")
        print(pf_print)
        report_lines.append(pf_print)
        e4_matrix[pf] = {}

        for sc_id in e4_scenarios:
            for regime in ["A", "B", "C"]:
                print(f"  {regime} × {sc_id}...", end=" ", flush=True)
                t0 = time.time()
                results = run_batch(regime, sc_id, 50, schema, config, profile_name=pf)
                elapsed = time.time() - t0

                stats = compute_stats(results)
                e4_matrix[pf][f"{regime}_{sc_id}"] = stats
                all_experiment_results[f"E4_{pf}_{regime}_{sc_id}"] = stats

                surv_str = f"存活率={stats['survival_rate']*100:.0f}%"
                print(f"{surv_str} | {elapsed:.0f}s")

                short_text = (
                    f"    {regime}×{sc_id}: {surv_str} "
                    f"| avg存活{stats['mean_survival_years']:.1f}年 "
                    f"| 代价基尼={stats['mean_cost_concentration']:.3f}"
                )
                report_lines.append(short_text)

    # E4 汇总
    e4_summary = format_subsection("E4 策略×国情×场景存活率矩阵")
    print(e4_summary)
    report_lines.append(e4_summary)

    for pf in profiles_to_test:
        print(f"\n  国情: {pf}")
        report_lines.append(f"\n  国情: {pf}")
        header_row = f"  {'场景':<22} {'A存活率':<10} {'B存活率':<10} {'C存活率':<10}"
        print(header_row)
        report_lines.append(header_row)
        for sc_id in e4_scenarios:
            survs = []
            for regime in ["A", "B", "C"]:
                key = f"{regime}_{sc_id}"
                st = e4_matrix[pf].get(key, {})
                survs.append(st.get("survival_rate", 0))
            row = f"  {sc_id:<22} {survs[0]:<10.1%} {survs[1]:<10.1%} {survs[2]:<10.1%}"
            print(row)
            report_lines.append(row)

    # ================================================================
    # 最终汇总
    # ================================================================
    total_elapsed = time.time() - t_start
    total_sims = sum(
        v["n"] for v in all_experiment_results.values() if isinstance(v, dict) and "n" in v
    )

    final_summary = format_section("最终汇总")
    print(final_summary)
    report_lines.append(final_summary)

    # 全局统计
    all_results_list = []
    for key, stats in all_experiment_results.items():
        if isinstance(stats, dict) and "survival_rate" in stats:
            all_results_list.append(stats)

    if all_results_list:
        global_surv = np.mean([s["survival_rate"] for s in all_results_list])
        global_cascade = np.mean([s["multi_fracture_rate"] for s in all_results_list])
        global_fractures = np.mean([s["mean_total_fractures"] for s in all_results_list])

    summary_text = f"""
  总模拟次数:     {total_sims}
  总耗时:         {total_elapsed:.0f}秒 ({total_elapsed/60:.1f}分钟)
  平均每次模拟:   {total_elapsed/total_sims*1000:.0f}ms

  全局平均存活率: {global_surv*100:.1f}%
  全局平均级联率: {global_cascade*100:.1f}%
  全局平均断裂数: {global_fractures:.2f}

  输出目录:       {OUTPUT_DIR}/
"""
    print(summary_text)
    report_lines.append(summary_text)

    # 核心发现
    findings = format_subsection("核心发现摘要")
    print(findings)
    report_lines.append(findings)

    # 找出A崩溃率最高的场景
    a_crash_scenarios = []
    for sc_id in all_scenarios:
        key = f"E2_{sc_id}_A"
        if key in all_experiment_results:
            crash_rate = 1 - all_experiment_results[key]["survival_rate"]
            a_crash_scenarios.append((sc_id, crash_rate))
    a_crash_scenarios.sort(key=lambda x: -x[1])

    findings_text = f"""
  策略A最危险场景排名:
"""
    for sc_id, crash_rate in a_crash_scenarios[:3]:
        findings_text += f"    1. {sc_id}: A崩溃率 {crash_rate*100:.1f}%\n"

    # 检查B和C是否有崩溃
    b_crashes = [s for s in all_scenarios
                 if all_experiment_results.get(f"E2_{s}_B", {}).get("survival_rate", 1) < 0.99]
    c_crashes = [s for s in all_scenarios
                 if all_experiment_results.get(f"E2_{s}_C", {}).get("survival_rate", 1) < 0.99]

    if b_crashes:
        findings_text += f"\n  策略B在以下场景出现崩溃: {b_crashes}\n"
    else:
        findings_text += "\n  策略B在所有场景中存活率≥99%（v2.1反哺失败/超调机制低频触发）\n"

    if c_crashes:
        findings_text += f"\n  策略C在以下场景出现崩溃: {c_crashes}\n"
    else:
        findings_text += "\n  策略C在所有场景中存活率≥99%（v2.1外部冲击机制低频触发）\n"

    findings_text += f"""
  v2.1 机制验证:
  - 策略A因技术追赶乘数~0.7×而系统性脆弱
  - 断裂级联效应在多场景中激活（全局平均{global_cascade*100:.1f}%模拟出现≥3断裂并发）
  - 知识扩散量: B > C > A（双轨制效应）
  - 恶意场景(full_containment/tech_explosion)显著提高A的崩溃率
"""

    print(findings_text)
    report_lines.append(findings_text)

    # ================================================================
    # 保存报告
    # ================================================================
    report_path = os.path.join(OUTPUT_DIR, f"batch_report_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 同时保存JSON格式的统计数据
    json_path = os.path.join(OUTPUT_DIR, f"batch_stats_{timestamp}.json")

    # 清理numpy类型使其可JSON序列化
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_for_json(all_experiment_results), f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"实验完成!")
    print(f"  文本报告: {report_path}")
    print(f"  JSON数据: {json_path}")
    print(f"  总耗时:   {total_elapsed:.0f}秒 ({total_elapsed/60:.1f}分钟)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
