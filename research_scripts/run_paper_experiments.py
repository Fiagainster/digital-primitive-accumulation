"""论文实验总驱动 (v3.8)。

一次定义论文所需的全部实验批次，逐批并行执行、逐批落盘。

与旧脚本的差异（均为"让实验本身站得住"的修正）：
  1. **种子真正生效**：每个任务由派生的确定种子驱动，同种子必复现、异种子给分布。
  2. **重复运行**：随机模型的统计量（存活率、断裂构成、成本分布）按多次重复汇总，
     而非把单次运行的轨迹当作该配置的确定结果。
  3. **参数定义域对齐 default.yaml**：
       - sigma_b            [0.3, 1.5]（旧的 σ 扫描与 w_emp×σ 交互跑到 2.0 → 截断）
       - epsilon_welfare    [0.5, 2.5]（exp6 旧值含 3.0 → 截断）
       - welfare_employment_exponent [0.1, 0.5]（exp6 旧值含 0.60 → 截断）
  4. **模块开关统一**：所有批次都用 default.yaml 的默认配置（platform / demography /
     spatial 全开），旧稳健性脚本曾把它们关掉、与普查不一致。
  5. **完整落盘**：五群组累计代价（cost_by_group）、CDI 与五维分解（evaluation）、
     数字依附（dependency_info）全部写入结果，论文表格可逐格复算。

用法::

    python research_scripts/run_paper_experiments.py                 # 跑全部
    python research_scripts/run_paper_experiments.py --only census cost_table
    python research_scripts/run_paper_experiments.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research_scripts.common.runner import build_param_sets          # noqa: E402
from research_scripts.exp_harness import (                           # noqa: E402
    RUN_ROOT, derive_seed, make_task, run_batch,
)
from src.params.presets import SCENARIO_PRESETS                      # noqa: E402
from src.params.country_profiles import get_profile_names            # noqa: E402
from src.params.schema import get_schema                             # noqa: E402

MASTER_SEED = 20261002
T_DEFAULT = 50.0
T_LONG = 80.0

PROFILES = get_profile_names()
SCENARIOS = list(SCENARIO_PRESETS.keys())
REGIMES = ["A", "B", "C"]

# 参数定义域（取自 config/default.yaml 的 schema 边界）
SIGMA_B_LEVELS = [round(0.3 + i * 0.1, 2) for i in range(13)]        # 0.3 .. 1.5
EPS_FINE_LEVELS = [round(0.70 + i * 0.05, 2) for i in range(13)]     # 0.70 .. 1.30
THRESHOLD_LEVELS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
G_WORLD_LEVELS = [0.01, 0.015, 0.02, 0.03, 0.05]
WEMP_LEVELS = [0.25, 0.30, 0.35, 0.40, 0.45]
WEMP_SIGMA_LEVELS = [0.8, 1.0, 1.2, 1.4, 1.5]                        # 替换越界的 2.0
KI_LEVELS = list(range(50, 401, 25))
KI_CONDITIONS = [(0.65, 1.0), (0.40, 1.0), (0.65, 0.7)]

REPS = {
    "census": 5, "threshold": 5, "gworld": 5, "eps_fine": 5, "sigma": 5,
    "wemp_eps": 5, "mixture": 3, "ki": 5, "cost_table": 30,
    "calibration": 5, "exp": 3, "t80": 5, "cost_lhs": 1,
    "extraction": 5, "gspeed": 5,
}

BATCH_ORDER = [
    "census", "cost_table", "calibration", "threshold", "sigma", "eps_fine",
    "wemp_eps", "ki", "gworld", "mixture", "cost_lhs", "extraction", "gspeed",
    "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "t80",
]


def _seed(tag: str, key: str, rep: int) -> int:
    return derive_seed(MASTER_SEED, tag, key, rep)


# --------------------------------------------------------------------------
# 各批次任务构造
# --------------------------------------------------------------------------

def build_census() -> list:
    """全组合普查：6 国情 × 8 场景 × 3 策略，T=50，每格 5 次重复。"""
    tasks = []
    for profile in PROFILES:
        for scenario in SCENARIOS:
            for regime in REGIMES:
                key = f"{profile}|{scenario}|{regime}"
                for rep in range(REPS["census"]):
                    tasks.append(make_task(
                        regime=regime,
                        params=build_param_sets(regime=regime, scenario=scenario,
                                                profile=profile),
                        T_max=T_DEFAULT, seed=_seed("census", key, rep),
                        meta={"batch": "census", "profile": profile,
                              "scenario": scenario, "regime": regime, "rep": rep,
                              "key": key},
                    ))
    return tasks


def build_threshold() -> list:
    """福利崩溃阈值扫描：7 阈值 × 6 国情 × 3 策略，每格 5 次。"""
    tasks = []
    for profile in PROFILES:
        for regime in REGIMES:
            for th in THRESHOLD_LEVELS:
                key = f"{profile}|{regime}|{th}"
                ov = {"welfare_collapse_threshold": th}
                for rep in range(REPS["threshold"]):
                    tasks.append(make_task(
                        regime=regime,
                        params=build_param_sets(regime=regime, profile=profile,
                                                scenario="benign_globalization",
                                                overrides=ov),
                        T_max=T_DEFAULT, seed=_seed("threshold", key, rep),
                        meta={"batch": "threshold", "profile": profile,
                              "regime": regime, "threshold": th, "rep": rep,
                              "key": key},
                    ))
    return tasks


def build_gworld() -> list:
    """g_world × 阈值 交互（A 策略速度陷阱）：2 国情 × 5 g_world × 7 阈值，每格 5 次。"""
    tasks = []
    for profile in ["manufacturing_giant", "post_soviet"]:
        for gw in G_WORLD_LEVELS:
            for th in THRESHOLD_LEVELS:
                key = f"{profile}|{gw}|{th}"
                ov = {"welfare_collapse_threshold": th, "g_world": gw}
                for rep in range(REPS["gworld"]):
                    tasks.append(make_task(
                        regime="A",
                        params=build_param_sets(regime="A", profile=profile,
                                                scenario="benign_globalization",
                                                overrides=ov),
                        T_max=T_DEFAULT, seed=_seed("gworld", key, rep),
                        meta={"batch": "gworld", "profile": profile, "regime": "A",
                              "g_world": gw, "threshold": th, "rep": rep,
                              "key": key},
                    ))
    return tasks


def build_gspeed() -> list:
    """g_world × 制度能力 交互（A 策略速度陷阱边界）：2 国情 × 6 g_world × 4 inst × 5 次。

    对应正文 §5.1 的二维边界表与 §5.2「制度能力是缓冲而非救援」的定量表述。
    """
    tasks = []
    for profile in ["manufacturing_giant", "post_soviet"]:
        for gw in [0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
            for inst in [0.50, 0.65, 0.80, 0.90]:
                key = f"{profile}|{gw}|{inst}"
                ov = {"g_world": gw, "institution_capacity": inst}
                for rep in range(REPS["gspeed"]):
                    tasks.append(make_task(
                        regime="A",
                        params=build_param_sets(regime="A", profile=profile,
                                                scenario="benign_globalization",
                                                overrides=ov),
                        T_max=T_DEFAULT, seed=_seed("gspeed", key, rep),
                        meta={"batch": "gspeed", "profile": profile, "regime": "A",
                              "g_world": gw, "institution_capacity": inst,
                              "rep": rep, "key": key},
                    ))
    return tasks


def build_eps_fine() -> list:
    """σ_b 精细扫描（B 策略）：13 水平 × 2 国情，每格 5 次。

    注：原脚本名为 epsilon_fine 但写入的始终是 sigma_b —— 此处保留 σ_b 语义。
    """
    tasks = []
    for profile in ["manufacturing_giant", "digital_leapfrogger"]:
        for lv in EPS_FINE_LEVELS:
            key = f"{profile}|{lv}"
            for rep in range(REPS["eps_fine"]):
                tasks.append(make_task(
                    regime="B",
                    params=build_param_sets(regime="B", profile=profile,
                                            scenario="benign_globalization",
                                            overrides={"sigma_b": lv}),
                    T_max=T_DEFAULT, seed=_seed("eps_fine", key, rep),
                    meta={"batch": "eps_fine", "profile": profile, "regime": "B",
                          "sigma_b": lv, "rep": rep, "key": key},
                ))
    return tasks


def build_sigma() -> list:
    """σ_b 扫描：13 水平（0.3–1.5，已按定义域截断）× 3 国情 × 3 策略，每格 5 次。"""
    tasks = []
    for profile in ["manufacturing_giant", "resource_rich", "digital_leapfrogger"]:
        for regime in REGIMES:
            for lv in SIGMA_B_LEVELS:
                key = f"{profile}|{regime}|{lv}"
                for rep in range(REPS["sigma"]):
                    tasks.append(make_task(
                        regime=regime,
                        params=build_param_sets(regime=regime, profile=profile,
                                                scenario="benign_globalization",
                                                overrides={"sigma_b": lv}),
                        T_max=T_DEFAULT, seed=_seed("sigma", key, rep),
                        meta={"batch": "sigma", "profile": profile, "regime": regime,
                              "sigma_b": lv, "rep": rep, "key": key},
                    ))
    return tasks


def build_wemp_eps() -> list:
    """w_emp × σ_b 交互（B 策略）：5 × 5 × 4 国情，每格 5 次。"""
    tasks = []
    profiles = ["manufacturing_giant", "resource_rich", "digital_leapfrogger",
                "post_soviet"]
    for profile in profiles:
        for w in WEMP_LEVELS:
            for s in WEMP_SIGMA_LEVELS:
                key = f"{profile}|{w}|{s}"
                ov = {"welfare_employment_exponent": w, "sigma_b": s}
                for rep in range(REPS["wemp_eps"]):
                    tasks.append(make_task(
                        regime="B",
                        params=build_param_sets(regime="B", profile=profile,
                                                scenario="benign_globalization",
                                                overrides=ov),
                        T_max=T_DEFAULT, seed=_seed("wemp_eps", key, rep),
                        meta={"batch": "wemp_eps", "profile": profile, "regime": "B",
                              "w_emp": w, "sigma_b": s, "rep": rep, "key": key},
                    ))
    return tasks


def build_ki() -> list:
    """K_i 敏感性（B 策略）：15 的 K_i × 3 制度条件，每格 5 次。"""
    tasks = []
    for ki in KI_LEVELS:
        for inst, h_i in KI_CONDITIONS:
            key = f"{ki}|{inst}|{h_i}"
            ov = {"IC_K_i": ki, "institution_capacity": inst, "IC_H_i": h_i}
            for rep in range(REPS["ki"]):
                tasks.append(make_task(
                    regime="B",
                    params=build_param_sets(regime="B", profile="manufacturing_giant",
                                            scenario="benign_globalization",
                                            overrides=ov),
                    T_max=T_DEFAULT, seed=_seed("ki", key, rep),
                    meta={"batch": "ki", "profile": "manufacturing_giant",
                          "regime": "B", "K_i": ki, "institution_capacity": inst,
                          "H_i": h_i, "rep": rep, "key": key},
                ))
    return tasks


def build_mixture() -> list:
    """混合策略：4 国情 × 2 场景 × 40 个 Dirichlet(1,1,1) 配比，每个配比 3 次。"""
    profiles = ["manufacturing_giant", "resource_rich", "digital_leapfrogger",
                "post_soviet"]
    scenarios = ["benign_globalization", "tech_cold_war"]
    n_samples = 40

    # 与论文所述一致：Dirichlet(1,1,1) 对称采样（不是 LHS）
    rng = np.random.default_rng(MASTER_SEED)
    lam_by_key = {}
    for profile in profiles:
        for scenario in scenarios:
            lam_by_key[(profile, scenario)] = rng.dirichlet([1.0, 1.0, 1.0],
                                                            size=n_samples)

    tasks = []
    for profile in profiles:
        for scenario in scenarios:
            for i in range(n_samples):
                lam = lam_by_key[(profile, scenario)][i]
                key = f"{profile}|{scenario}|{i}"
                for rep in range(REPS["mixture"]):
                    tasks.append(make_task(
                        regime="M",
                        params=build_param_sets(regime="M", profile=profile,
                                                scenario=scenario,
                                                overrides={
                                                    "mixed_lambda": [float(x) for x in lam],
                                                    "mixed_name": f"D{profile[:4]}_{i:02d}",
                                                }),
                        T_max=T_DEFAULT, seed=_seed("mixture", key, rep),
                        meta={"batch": "mixture", "profile": profile,
                              "scenario": scenario, "idx": i, "rep": rep,
                              "lambda_A": float(lam[0]), "lambda_B": float(lam[1]),
                              "lambda_C": float(lam[2]), "key": key},
                    ))
    return tasks


def build_cost_lhs() -> list:
    """成本分布 LHS：3 国情 × 3 场景 × 50 个 LHS 参数点（C 策略）。

    修正点：**国情真正传入参数集**（旧脚本只把它当标签，三组跑的是同一批参数）。
    """
    schema = get_schema()
    strategy_ids = [pid for pid in schema.all_ids()
                    if schema.get(pid).layer == "strategy"][:15]
    profiles = ["manufacturing_giant", "resource_rich", "digital_leapfrogger"]
    scenarios = ["benign_globalization", "tech_cold_war", "full_containment"]
    n_lhs = 50

    tasks = []
    for profile in profiles:
        for scenario in scenarios:
            rng = np.random.default_rng(_seed("cost_lhs", f"{profile}|{scenario}", 0))
            samples = schema.sample_lhs(n_lhs, strategy_ids, rng)
            for i in range(n_lhs):
                params = build_param_sets(regime="C", profile=profile, scenario=scenario)
                for j, pid in enumerate(strategy_ids):
                    params[pid] = float(samples[i, j])
                key = f"{profile}|{scenario}|{i}"
                tasks.append(make_task(
                    regime="C", params=params, T_max=T_DEFAULT,
                    seed=_seed("cost_lhs", key, 0),
                    meta={"batch": "cost_lhs", "profile": profile,
                          "scenario": scenario, "idx": i, "rep": 0, "key": key,
                          # 修复（2026-10-08）：落盘实际抽样参数向量，使 LHS 点
                          # 即使抽样器行为改变也可回溯（旧批次未保存，无法恢复）。
                          "lhs_param_ids": list(strategy_ids),
                          "lhs_params": {pid: float(samples[i, j])
                                         for j, pid in enumerate(strategy_ids)}},
                ))
    return tasks


def build_cost_table() -> list:
    """成本表：4 国情 × B 策略 × 30 次重复（覆盖随机冲击），T=50。"""
    tasks = []
    profiles = ["manufacturing_giant", "resource_rich", "digital_leapfrogger",
                "post_soviet"]
    for profile in profiles:
        key = profile
        for rep in range(REPS["cost_table"]):
            tasks.append(make_task(
                regime="B",
                params=build_param_sets(regime="B", profile=profile,
                                        scenario="benign_globalization"),
                T_max=T_DEFAULT, seed=_seed("cost_table", key, rep),
                meta={"batch": "cost_table", "profile": profile, "regime": "B",
                      "rep": rep, "key": key},
            ))
    return tasks


def build_calibration(t_max: float, tag: str, reps: int) -> list:
    """定性校准：6 国情 × 3 策略 × reps 次（默认 T=50；另跑一轮 T=80）。"""
    tasks = []
    for profile in PROFILES:
        for regime in REGIMES:
            key = f"{profile}|{regime}"
            for rep in range(reps):
                tasks.append(make_task(
                    regime=regime,
                    params=build_param_sets(regime=regime, profile=profile,
                                            scenario="benign_globalization"),
                    T_max=t_max, seed=_seed(tag, key, rep),
                    meta={"batch": tag, "profile": profile, "regime": regime,
                          "rep": rep, "key": key},
                ))
    return tasks


def build_exp1() -> list:
    """exp1 非线性追赶：2 国情 × 2 场景 × 6 g_world × 3 模式，A 策略，每格 3 次。"""
    tasks = []
    for profile in ["manufacturing_giant", "digital_leapfrogger"]:
        for scenario in ["benign_globalization", "tech_cold_war"]:
            for gw in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
                for mode in ["linear", "logistic", "piecewise"]:
                    key = f"{profile}|{scenario}|{gw}|{mode}"
                    ov = {"g_world": gw, "catchup_mode": mode}
                    for rep in range(REPS["exp"]):
                        tasks.append(make_task(
                            regime="A",
                            params=build_param_sets(regime="A", profile=profile,
                                                    scenario=scenario, overrides=ov),
                            T_max=T_DEFAULT, seed=_seed("exp1", key, rep),
                            meta={"batch": "exp1", "profile": profile,
                                  "scenario": scenario, "g_world": gw,
                                  "catchup_mode": mode, "rep": rep, "key": key},
                        ))
    return tasks


def build_exp2() -> list:
    """exp2 B 策略再分配反事实：4 国情 × 2 变体，每格 3 次。"""
    variants = [("baseline", {"B_phi_d_jump": 0.6, "B_g_target": 0.25}),
                ("redistribute", {"B_phi_d_jump": 0.3, "B_g_target": 0.15})]
    tasks = []
    for profile in ["manufacturing_giant", "resource_rich", "digital_leapfrogger",
                    "post_soviet"]:
        for name, ov in variants:
            key = f"{profile}|{name}"
            for rep in range(REPS["exp"]):
                tasks.append(make_task(
                    regime="B",
                    params=build_param_sets(regime="B", profile=profile,
                                            scenario="benign_globalization",
                                            overrides=ov),
                    T_max=T_DEFAULT, seed=_seed("exp2", key, rep),
                    meta={"batch": "exp2", "profile": profile, "variant": name,
                          "rep": rep, "key": key},
                ))
    return tasks


_MIXTURE_VARIANTS = [
    ("B_dominant_A", (0.2, 0.7, 0.1), "manufacturing_giant"),
    ("B_dominant_A_v2", (0.1, 0.8, 0.1), "manufacturing_giant"),
    ("B_dominant_A_v3", (0.3, 0.6, 0.1), "manufacturing_giant"),
    ("A_dominant_B", (0.7, 0.2, 0.1), "post_soviet"),
    ("A_dominant_B_v2", (0.8, 0.1, 0.1), "post_soviet"),
    ("A_dominant_B_v3", (0.6, 0.3, 0.1), "post_soviet"),
    ("C_dominant_B", (0.1, 0.2, 0.7), "resource_rich"),
    ("C_dominant_B_v2", (0.1, 0.1, 0.8), "resource_rich"),
    ("C_dominant_B_v3", (0.1, 0.3, 0.6), "resource_rich"),
    ("A_B_balanced", (0.7, 0.3, 0.0), "digital_leapfrogger"),
    ("A_B_balanced_v2", (0.8, 0.2, 0.0), "digital_leapfrogger"),
    ("A_B_balanced_v3", (0.6, 0.4, 0.0), "digital_leapfrogger"),
]
_INST_VARIANTS = [
    ("B_dominant_A_inst50", (0.2, 0.7, 0.1), "manufacturing_giant", 0.50),
    ("B_dominant_A_inst75", (0.2, 0.7, 0.1), "manufacturing_giant", 0.75),
    ("A_dominant_B_inst50", (0.7, 0.2, 0.1), "post_soviet", 0.50),
    ("A_dominant_B_inst75", (0.7, 0.2, 0.1), "post_soviet", 0.75),
]


def build_exp3() -> list:
    """exp3 混合策略参数敏感性：12 配比变体 + 4 制度变体，每格 3 次。"""
    tasks = []
    for label, lvec, profile in _MIXTURE_VARIANTS:
        ov = {"mixed_lambda": list(lvec), "mixed_name": label}
        for rep in range(REPS["exp"]):
            tasks.append(make_task(
                regime="M",
                params=build_param_sets(regime="M", profile=profile,
                                        scenario="benign_globalization", overrides=ov),
                T_max=T_DEFAULT, seed=_seed("exp3", label, rep),
                meta={"batch": "exp3", "profile": profile, "variant": label,
                      "kind": "mixture", "rep": rep, "key": label},
            ))
    for label, lvec, profile, inst in _INST_VARIANTS:
        ov = {"mixed_lambda": list(lvec), "mixed_name": label,
              "institution_capacity": inst}
        for rep in range(REPS["exp"]):
            tasks.append(make_task(
                regime="M",
                params=build_param_sets(regime="M", profile=profile,
                                        scenario="benign_globalization", overrides=ov),
                T_max=T_DEFAULT, seed=_seed("exp3", label, rep),
                meta={"batch": "exp3", "profile": profile, "variant": label,
                      "kind": "institution", "institution_capacity": inst,
                      "rep": rep, "key": label},
            ))
    return tasks


def build_exp4() -> list:
    """exp4 T=80：4 个代表组合。"""
    runs = [("A", "manufacturing_giant", "benign_globalization"),
            ("C", "manufacturing_giant", "benign_globalization"),
            ("C", "digital_leapfrogger", "benign_globalization"),
            ("C", "resource_rich", "benign_globalization")]
    tasks = []
    for regime, profile, scenario in runs:
        key = f"{regime}|{profile}"
        for rep in range(REPS["exp"]):
            tasks.append(make_task(
                regime=regime,
                params=build_param_sets(regime=regime, profile=profile,
                                        scenario=scenario),
                T_max=T_LONG, seed=_seed("exp4", key, rep),
                meta={"batch": "exp4", "profile": profile, "regime": regime,
                      "scenario": scenario, "rep": rep, "key": key},
            ))
    return tasks


def build_exp5() -> list:
    """exp5 高制度能力：3 个组合。"""
    runs = [("A", "small_open", "benign_globalization", 0.95),
            ("C", "small_open", "benign_globalization", 0.95),
            ("C", "populous_weak_industry", "benign_globalization", 0.80)]
    tasks = []
    for regime, profile, scenario, inst in runs:
        key = f"{regime}|{profile}|{inst}"
        for rep in range(REPS["exp"]):
            tasks.append(make_task(
                regime=regime,
                params=build_param_sets(regime=regime, profile=profile,
                                        scenario=scenario,
                                        overrides={"institution_capacity": inst}),
                T_max=T_DEFAULT, seed=_seed("exp5", key, rep),
                meta={"batch": "exp5", "profile": profile, "regime": regime,
                      "institution_capacity": inst, "rep": rep, "key": key},
            ))
    return tasks


def build_exp6() -> list:
    """exp6 福利敏感性（B 策略、制造业大国）：ε_welfare 5 级 + 就业权重 5 级。

    修正：两个扫描轴都按 default.yaml 定义域截断（旧值 3.0 / 0.60 越界）。
    """
    tasks = []
    for eps in [0.8, 1.2, 1.5, 2.0, 2.5]:
        key = f"eps|{eps}"
        for rep in range(REPS["exp"]):
            tasks.append(make_task(
                regime="B",
                params=build_param_sets(regime="B", profile="manufacturing_giant",
                                        scenario="benign_globalization",
                                        overrides={"epsilon_welfare": eps}),
                T_max=T_DEFAULT, seed=_seed("exp6", key, rep),
                meta={"batch": "exp6", "axis": "epsilon_welfare", "value": eps,
                      "profile": "manufacturing_giant", "regime": "B",
                      "rep": rep, "key": key},
            ))
    for w in [0.10, 0.20, 0.30, 0.40, 0.50]:
        key = f"wemp|{w}"
        for rep in range(REPS["exp"]):
            tasks.append(make_task(
                regime="B",
                params=build_param_sets(regime="B", profile="manufacturing_giant",
                                        scenario="benign_globalization",
                                        overrides={"welfare_employment_exponent": w}),
                T_max=T_DEFAULT, seed=_seed("exp6", key, rep),
                meta={"batch": "exp6", "axis": "welfare_employment_exponent",
                      "value": w, "profile": "manufacturing_giant", "regime": "B",
                      "rep": rep, "key": key},
            ))
    return tasks


def build_t80_from_census() -> list:
    """T=80 重跑：以普查中"多数重复可存续"的配置为样本，每格 5 次。

    定义：某配置在 census 的 5 次重复中存活 ≥3 次（即存活占比 ≥ 0.5），
    即视为 T=50 可行，再用 T=80 重跑（全新运行，不是从 50 年检查点续跑）。

    注意（2026-10-08）：本批次的 tag 为 "t80"，与 census 的 tag 不同，
    因此 derive_seed 给出的种子与普查同配置的种子**完全不同**——
    两者没有配对关系，不能把 T=80 的结果读作"原普查幸存轨迹的延长"。
    见《新增代码核对与待确认清单》第二项。
    """
    from research_scripts.exp_harness import load_batch
    data = load_batch("census")
    agg = {}
    for r in data["results"]:
        if not r or not r.get("ok"):
            continue
        m = r["meta"]
        k = (m["profile"], m["scenario"], m["regime"])
        a = agg.setdefault(k, [0, 0])
        a[1] += 1
        if r["summary"].get("survived"):
            a[0] += 1

    viable = [k for k, (s, n) in agg.items() if n and s / n >= 0.5]
    tasks = []
    for profile, scenario, regime in sorted(viable):
        key = f"{profile}|{scenario}|{regime}"
        for rep in range(REPS["t80"]):
            tasks.append(make_task(
                regime=regime,
                params=build_param_sets(regime=regime, profile=profile,
                                        scenario=scenario),
                T_max=T_LONG, seed=_seed("t80", key, rep),
                meta={"batch": "t80", "profile": profile, "scenario": scenario,
                      "regime": regime, "rep": rep, "key": key},
            ))
    print(f"  [t80] 普查中多数可存续的配置: {len(viable)} 个 -> {len(tasks)} 次")
    return tasks


def build_extraction() -> list:
    """抽取率悖论：数字化投资目标 B_g_target 7 级 × 2 国情 × 5 次（B 策略，benign）。

    B_g_target 直接决定策略 B 的名义提取率 tau（tau = min(g_target*K_d/fund, tau_max)），
    因此该批次即"提取强度"扫描。原论文此处的 10%/35% 两档结论缺乏可复现脚本，
    本批次把整条曲线补齐。
    """
    tasks = []
    for profile in ["manufacturing_giant", "resource_rich"]:
        for gt in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            key = f"{profile}|{gt}"
            for rep in range(REPS["extraction"]):
                tasks.append(make_task(
                    regime="B",
                    params=build_param_sets(regime="B", profile=profile,
                                            scenario="benign_globalization",
                                            overrides={"B_g_target": gt}),
                    T_max=T_DEFAULT, seed=_seed("extraction", key, rep),
                    meta={"batch": "extraction", "profile": profile,
                          "B_g_target": gt, "rep": rep, "key": key},
                ))
    return tasks


def build_batch(name: str) -> list:
    if name == "census":
        return build_census()
    if name == "threshold":
        return build_threshold()
    if name == "gworld":
        return build_gworld()
    if name == "eps_fine":
        return build_eps_fine()
    if name == "sigma":
        return build_sigma()
    if name == "wemp_eps":
        return build_wemp_eps()
    if name == "ki":
        return build_ki()
    if name == "mixture":
        return build_mixture()
    if name == "cost_lhs":
        return build_cost_lhs()
    if name == "cost_table":
        return build_cost_table()
    if name == "calibration":
        return build_calibration(T_DEFAULT, "calibration", REPS["calibration"])
    if name == "calibration_t80":
        return build_calibration(T_LONG, "calibration_t80", 1)
    if name == "exp1":
        return build_exp1()
    if name == "exp2":
        return build_exp2()
    if name == "exp3":
        return build_exp3()
    if name == "exp4":
        return build_exp4()
    if name == "exp5":
        return build_exp5()
    if name == "exp6":
        return build_exp6()
    if name == "extraction":
        return build_extraction()
    if name == "gspeed":
        return build_gspeed()
    if name == "t80":
        return build_t80_from_census()
    raise KeyError(name)


def main() -> None:
    ap = argparse.ArgumentParser(description="论文实验总驱动 (v3.8)")
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定批次")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", help="已有结果则跳过")
    ap.add_argument("--list", action="store_true", help="列出批次与规模")
    args = ap.parse_args()

    names = args.only or BATCH_ORDER + ["calibration_t80"]

    if args.list:
        print(f"输出目录: {RUN_ROOT}")
        for nm in names:
            try:
                n = len(build_batch(nm))
            except Exception as exc:  # noqa: BLE001
                n = f"(不可用: {exc})"
            print(f"  {nm:16s} {n}")
        return

    print(f"输出目录: {RUN_ROOT}")
    for nm in names:
        tasks = build_batch(nm)
        run_batch(nm, tasks, workers=args.workers,
                  skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
