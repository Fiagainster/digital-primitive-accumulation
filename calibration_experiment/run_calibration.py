#!/usr/bin/env python
"""
定性校准实验 —— 完整运行脚本
============================================================
对论文模型进行定性校准（stylized facts calibration），
验证模型能否复现 3 条核心跨国经验事实。

实验设计:
  - 6 种国情 × 3 种策略 × benign_globalization
  - 每个组合 5 次重复（随机种子不同）
  - T=50 基准运行（共 90 次模拟）
  - T=80 延长运行（对关键组合，共 18 次模拟）

总计: 108 次模拟，预计 20-40 分钟

输出:
  data/           - 原始时间序列和汇总数据
  logs/           - 运行日志
  results/        - 中间分析结果
  figures/        - 可视化图表
"""

import sys
import os
import json
import time
import traceback
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 确保可以导入 src
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np
from src.core.engine import SimulationEngine
from src.core.config import SimulationConfig
from src.params.schema import get_schema
from src.params.presets import get_scenario
from src.params.country_profiles import get_country_profile, get_profile_names

# ============================================================
# 配置
# ============================================================
PROFILES = get_profile_names()  # 6 种国情
REGIMES = ["A", "B", "C"]       # 3 种策略
SCENARIO = "benign_globalization"
N_REPEATS = 5                   # 每种组合重复次数
T_MAX_BASELINE = 50.0           # 基准时间跨度
T_MAX_EXTENDED = 80.0           # 延长运行
SEED_BASE = 42

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "data"
LOGS_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

for d in [DATA_DIR, LOGS_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# 日志系统
# ============================================================
class Logger:
    """同时输出到控制台和文件."""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        try:
            print(line)
        except UnicodeEncodeError:
            # Windows GBK console fallback
            print(line.encode("gbk", errors="replace").decode("gbk", errors="replace"))
        self.file.write(line + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================
# 单次模拟运行
# ============================================================
def run_single_simulation(profile: str, regime: str, scenario: str,
                          T_max: float, seed: int,
                          logger: Optional[Logger] = None) -> dict:
    """运行单次模拟并返回完整结果.

    Returns:
        dict with keys: summary, dataframe, params, profile, regime, seed, success, error
    """
    try:
        schema = get_schema()
        params = schema.build_from_profile(
            profile_name=profile,
            regime=regime,
            scenario=scenario,
        )

        config = SimulationConfig(
            T_max=T_max,
            verbose=False,
            random_seed=seed,
        )

        engine = SimulationEngine(
            params=params, config=config, regime=regime,
        )
        result = engine.run()

        return {
            "summary": result["summary"],
            "dataframe": result["dataframe"],
            "params": params,
            "profile": profile,
            "regime": regime,
            "scenario": scenario,
            "T_max": T_max,
            "seed": seed,
            "success": True,
            "error": None,
        }
    except Exception as e:
        if logger:
            logger.log(f"  ERROR: {type(e).__name__}: {str(e)}")
        return {
            "summary": {"survived": False, "survival_years": 0, "error": str(e)},
            "dataframe": None,
            "params": {},
            "profile": profile,
            "regime": regime,
            "scenario": scenario,
            "T_max": T_max,
            "seed": seed,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def extract_timeseries_summary(df, profile, regime, seed) -> dict:
    """从时间序列 DataFrame 提取关键变量的摘要统计."""
    if df is None or len(df) == 0:
        return {}

    # 计算工业就业占比（注意：工业 = i 部门）
    L_total = df["L_b"] + df["L_i"] + df["L_d_infra"] + df["L_d_app"] + df["L_s_high"] + df["L_s_low"]
    L_industry_share = df["L_i"] / L_total.replace(0, np.nan)

    # 人均 GDP（模型单位，近似）
    # 总劳动力近似为总人口基础
    gdp_pc = df["gdp"] / L_total.replace(0, np.nan)

    summary = {
        "profile": profile,
        "regime": regime,
        "seed": seed,
        # 工业就业占比的时间序列特征
        "L_industry_share_initial": float(L_industry_share.iloc[0]) if len(L_industry_share) > 0 else np.nan,
        "L_industry_share_final": float(L_industry_share.iloc[-1]) if len(L_industry_share) > 0 else np.nan,
        "L_industry_share_max": float(L_industry_share.max()),
        "L_industry_share_max_t": float(df["t"].iloc[L_industry_share.idxmax()]) if not L_industry_share.empty else np.nan,
        "L_industry_share_mean": float(L_industry_share.mean()),
        # GDP 特征
        "gdp_initial": float(df["gdp"].iloc[0]),
        "gdp_final": float(df["gdp"].iloc[-1]),
        "gdp_pc_initial": float(gdp_pc.iloc[0]),
        "gdp_pc_final": float(gdp_pc.iloc[-1]),
        "gdp_growth_rate": float((df["gdp"].iloc[-1] / max(df["gdp"].iloc[0], 0.01)) ** (1.0 / max(df["t"].iloc[-1], 0.1)) - 1),
        # 福利/不平等
        "welfare_initial": float(df["welfare_index"].iloc[0]),
        "welfare_final": float(df["welfare_index"].iloc[-1]),
        "gini_initial": float(df["gini"].iloc[0]),
        "gini_final": float(df["gini"].iloc[-1]),
        # 技术
        "tech_gap_initial": float(df["tech_gap"].iloc[0]),
        "tech_gap_final": float(df["tech_gap"].iloc[-1]),
        # 部门劳动力
        "L_i_initial": float(df["L_i"].iloc[0]),
        "L_i_final": float(df["L_i"].iloc[-1]),
        "L_d_total_initial": float(df["L_d_infra"].iloc[0] + df["L_d_app"].iloc[0]),
        "L_d_total_final": float(df["L_d_infra"].iloc[-1] + df["L_d_app"].iloc[-1]),
        "L_total_initial": float(L_total.iloc[0]),
        "L_total_final": float(L_total.iloc[-1]),
        # 失业率
        "unemployment_peak": float(df["unemployment_rate"].max()),
        "unemployment_final": float(df["unemployment_rate"].iloc[-1]),
    }
    return summary


# ============================================================
# 主运行逻辑
# ============================================================
def main():
    log = Logger(LOGS_DIR / f"calibration_run_{TIMESTAMP}.log")
    log.log("=" * 70)
    log.log("定性校准实验 — 开始运行")
    log.log("=" * 70)
    log.log(f"国情: {', '.join(PROFILES)}")
    log.log(f"策略: {', '.join(REGIMES)}")
    log.log(f"场景: {SCENARIO}")
    log.log(f"重复次数: {N_REPEATS}")
    log.log(f"基准 T_max: {T_MAX_BASELINE}")
    log.log(f"时间戳: {TIMESTAMP}")
    log.log("")

    # ============================================================
    # Phase 1: T=50 基准运行
    # ============================================================
    log.log("-" * 70)
    log.log("Phase 1: T=50 基准运行 (6 profiles × 3 regimes × 5 repeats = 90 runs)")
    log.log("-" * 70)

    all_results = []         # 完整结果（含时间序列）
    all_summaries = []       # 汇总统计
    ts_data_for_fact1 = {}   # 按 (profile, regime) 分组的时间序列

    n_combinations = len(PROFILES) * len(REGIMES)
    total_runs_phase1 = n_combinations * N_REPEATS
    run_count = 0
    start_time = time.time()

    for profile in PROFILES:
        for regime in REGIMES:
            combo_key = f"{profile}_{regime}"
            combo_results = []
            combo_ts_summaries = []

            log.log(f"\n>>> {profile} × 策略{regime}")

            for repeat in range(N_REPEATS):
                run_count += 1
                seed = SEED_BASE + repeat * 100 + hash(profile) % 1000 + hash(regime) % 100

                elapsed = time.time() - start_time
                eta = (elapsed / max(run_count - 1, 1)) * (total_runs_phase1 - run_count + 1) if run_count > 1 else 0

                log.log(f"  [{run_count}/{total_runs_phase1}] repeat={repeat+1}/{N_REPEATS} seed={seed} "
                        f"(elapsed={elapsed:.0f}s, ETA={eta:.0f}s)")

                result = run_single_simulation(
                    profile=profile, regime=regime, scenario=SCENARIO,
                    T_max=T_MAX_BASELINE, seed=seed, logger=log,
                )

                s = result["summary"]
                status = "[OK]" if s.get("survived") else f"[FAIL @{s.get('survival_years', 0):.1f}y]"
                log.log(f"    → {status} | GDP={s.get('final_gdp', 0):.1f} | "
                        f"welfare={s.get('final_welfare', 0):.3f} | "
                        f"G={s.get('final_tech_gap', 0):.2f}")

                # 提取时间序列摘要
                if result["success"] and result["dataframe"] is not None:
                    ts_summ = extract_timeseries_summary(
                        result["dataframe"], profile, regime, seed
                    )
                    combo_ts_summaries.append(ts_summ)

                combo_results.append(result)

            # 保存该组合的汇总
            all_results.extend(combo_results)
            all_summaries.extend(combo_ts_summaries)

            # 保存组合级别的时间序列聚合
            if combo_ts_summaries:
                ts_data_for_fact1[combo_key] = {
                    "profile": profile,
                    "regime": regime,
                    "n_repeats": len(combo_ts_summaries),
                    "summaries": combo_ts_summaries,
                }

            # 存活率统计
            survived = sum(1 for r in combo_results if r["summary"].get("survived", False))
            log.log(f"    ★ {profile}×{regime}: {survived}/{N_REPEATS} 存活 ({100*survived/N_REPEATS:.0f}%)")

    elapsed_phase1 = time.time() - start_time
    log.log(f"\nPhase 1 完成! 耗时: {elapsed_phase1:.0f}s ({elapsed_phase1/60:.1f}min)")

    # ============================================================
    # Phase 2: T=80 延长运行（仅对关键组合）
    # ============================================================
    log.log("\n" + "-" * 70)
    log.log("Phase 2: T=80 延长运行 (6 profiles × 3 regimes = 18 runs, 1 repeat each)")
    log.log("-" * 70)

    extended_results = []
    run_count_ext = 0
    start_time_ext = time.time()

    for profile in PROFILES:
        for regime in REGIMES:
            run_count_ext += 1
            seed = SEED_BASE + 500  # 固定种子用于延长运行

            log.log(f"  [{run_count_ext}/18] {profile} × 策略{regime} T=80")

            result = run_single_simulation(
                profile=profile, regime=regime, scenario=SCENARIO,
                T_max=T_MAX_EXTENDED, seed=seed, logger=log,
            )

            s = result["summary"]
            status = "[OK]" if s.get("survived") else f"[FAIL @{s.get('survival_years', 0):.1f}y]"
            log.log(f"    → {status} | GDP={s.get('final_gdp', 0):.1f} | "
                    f"welfare={s.get('final_welfare', 0):.3f}")

            extended_results.append(result)

    elapsed_phase2 = time.time() - start_time_ext
    log.log(f"\nPhase 2 完成! 耗时: {elapsed_phase2:.0f}s ({elapsed_phase2/60:.1f}min)")

    # ============================================================
    # 保存数据
    # ============================================================
    log.log("\n" + "-" * 70)
    log.log("保存数据...")
    log.log("-" * 70)

    # 1. 保存 Phase 1 完整结果（JSON 序列化摘要，不含 DataFrame）
    phase1_serializable = []
    for r in all_results:
        entry = {
            "profile": r["profile"],
            "regime": r["regime"],
            "scenario": r["scenario"],
            "T_max": r["T_max"],
            "seed": r["seed"],
            "success": r["success"],
            "error": r.get("error"),
            "summary": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in r["summary"].items()
                       if not isinstance(v, (dict, list, np.ndarray))},
        }
        # 添加 evaluation 摘要
        if r["success"] and "evaluation" in r["summary"]:
            ev = r["summary"]["evaluation"]
            if isinstance(ev, dict):
                for k, v in ev.items():
                    if isinstance(v, (int, float, bool, str)):
                        entry[f"eval_{k}"] = v
        phase1_serializable.append(entry)

    with open(DATA_DIR / f"phase1_all_summaries_{TIMESTAMP}.json", "w", encoding="utf-8") as f:
        json.dump(phase1_serializable, f, ensure_ascii=False, indent=2, default=str)
    log.log(f"  [OK] Phase 1 汇总: {DATA_DIR / f'phase1_all_summaries_{TIMESTAMP}.json'}")

    # 2. 保存时间序列摘要（用于 Fact 1 和 Fact 3）
    with open(DATA_DIR / f"timeseries_summaries_{TIMESTAMP}.json", "w", encoding="utf-8") as f:
        json.dump(ts_data_for_fact1, f, ensure_ascii=False, indent=2, default=str)
    log.log(f"  ✓ 时间序列摘要: {DATA_DIR / f'timeseries_summaries_{TIMESTAMP}.json'}")

    # 3. 保存 Phase 2 延长运行结果
    phase2_serializable = []
    for r in extended_results:
        entry = {
            "profile": r["profile"],
            "regime": r["regime"],
            "T_max": r["T_max"],
            "seed": r["seed"],
            "success": r["success"],
            "error": r.get("error"),
            "summary": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in r["summary"].items()
                       if not isinstance(v, (dict, list, np.ndarray))},
        }
        phase2_serializable.append(entry)

    with open(DATA_DIR / f"phase2_extended_{TIMESTAMP}.json", "w", encoding="utf-8") as f:
        json.dump(phase2_serializable, f, ensure_ascii=False, indent=2, default=str)
    log.log(f"  ✓ Phase 2 延长: {DATA_DIR / f'phase2_extended_{TIMESTAMP}.json'}")

    # 4. 保存代表性时间序列（每种组合的第一个 repeat，用于详细分析）
    representative_ts = {}
    saved_keys = set()
    for r in all_results:
        key = f"{r['profile']}_{r['regime']}"
        if key not in saved_keys and r["success"] and r["dataframe"] is not None:
            representative_ts[key] = {
                "profile": r["profile"],
                "regime": r["regime"],
                "seed": r["seed"],
                "timeseries": r["dataframe"].to_dict(orient="list"),
            }
            saved_keys.add(key)

    with open(DATA_DIR / f"representative_timeseries_{TIMESTAMP}.json", "w", encoding="utf-8") as f:
        json.dump(representative_ts, f, ensure_ascii=False, indent=2, default=str)
    log.log(f"  ✓ 代表性时间序列: {DATA_DIR / f'representative_timeseries_{TIMESTAMP}.json'}")

    # 5. 保存完整结果（含 DataFrame）为 pickle（用于后续深度分析）
    # 只保存 Phase 1 的完整数据（太大则压缩）
    try:
        with open(DATA_DIR / f"phase1_full_results_{TIMESTAMP}.pkl", "wb") as f:
            pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.log(f"  ✓ Phase 1 完整结果 (PKL): {DATA_DIR / f'phase1_full_results_{TIMESTAMP}.pkl'}")
    except Exception as e:
        log.log(f"  ⚠ PKL 保存失败（可能文件过大）: {e}")

    # 6. 保存运行元数据
    metadata = {
        "timestamp": TIMESTAMP,
        "profiles": PROFILES,
        "regimes": REGIMES,
        "scenario": SCENARIO,
        "n_repeats": N_REPEATS,
        "T_max_baseline": T_MAX_BASELINE,
        "T_max_extended": T_MAX_EXTENDED,
        "seed_base": SEED_BASE,
        "total_runs_phase1": total_runs_phase1,
        "total_runs_phase2": len(extended_results),
        "elapsed_phase1_seconds": elapsed_phase1,
        "elapsed_phase2_seconds": elapsed_phase2,
        # 存活率汇总
        "survival_summary": {},
    }
    for profile in PROFILES:
        for regime in REGIMES:
            key = f"{profile}_{regime}"
            subset = [r for r in all_results if r["profile"] == profile and r["regime"] == regime]
            survived = sum(1 for r in subset if r["summary"].get("survived", False))
            metadata["survival_summary"][key] = {
                "survived": survived,
                "total": len(subset),
                "rate": survived / max(len(subset), 1),
            }

    with open(DATA_DIR / f"metadata_{TIMESTAMP}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    log.log(f"  ✓ 元数据: {DATA_DIR / f'metadata_{TIMESTAMP}.json'}")

    # ============================================================
    # 生成快速汇总报告
    # ============================================================
    log.log("\n" + "=" * 70)
    log.log("快速汇总报告")
    log.log("=" * 70)

    log.log(f"\n{'国情':<25} {'策略':>6} {'存活率':>8} {'GDP终值':>10} {'福利终值':>10} {'工业就业峰值':>12}")
    log.log("-" * 75)

    for profile in PROFILES:
        profile_label = get_country_profile(profile).label
        for regime in REGIMES:
            subset = [r for r in all_results if r["profile"] == profile and r["regime"] == regime]
            if not subset:
                continue
            survived = sum(1 for r in subset if r["summary"].get("survived", False))
            rate = survived / len(subset)
            avg_gdp = np.mean([r["summary"].get("final_gdp", 0) for r in subset])
            avg_welfare = np.mean([r["summary"].get("final_welfare", 0) for r in subset])

            # 从时间序列摘要中获取工业就业峰值
            ts_key = f"{profile}_{regime}"
            peak_L_i = np.nan
            if ts_key in ts_data_for_fact1:
                peaks = [s.get("L_industry_share_max", np.nan) for s in ts_data_for_fact1[ts_key]["summaries"]]
                peak_L_i = np.nanmean(peaks) if peaks else np.nan

            log.log(f"{profile_label:<25} {regime:>6} {rate:>7.0%} {avg_gdp:>10.1f} {avg_welfare:>10.3f} {peak_L_i:>12.3f}")

    log.log("\n" + "=" * 70)
    log.log(f"实验完成! 总耗时: {elapsed_phase1 + elapsed_phase2:.0f}s ({(elapsed_phase1 + elapsed_phase2)/60:.1f}min)")
    log.log(f"数据已保存至: {DATA_DIR}")
    log.log(f"日志已保存至: {LOGS_DIR}")
    log.log("=" * 70)

    log.close()

    # 返回关键数据引用
    return {
        "timestamp": TIMESTAMP,
        "data_dir": str(DATA_DIR),
        "logs_dir": str(LOGS_DIR),
        "results_dir": str(RESULTS_DIR),
        "figures_dir": str(FIGURES_DIR),
        "n_runs_phase1": total_runs_phase1,
        "n_runs_phase2": len(extended_results),
        "metadata": metadata,
    }


if __name__ == "__main__":
    main()
