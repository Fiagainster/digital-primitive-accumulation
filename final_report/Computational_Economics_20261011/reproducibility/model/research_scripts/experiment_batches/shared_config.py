"""
共享配置 —— 各阶段实验的统一参数和工具函数.

设计原则:
    - 每个阶段可独立运行，也可按序推进
    - 前序阶段的输出自动成为后序阶段的输入
    - 所有结果输出到 experiment_batches/output/ 下
"""

import sys
from pathlib import Path

# 确保可以导入 src 和 common
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ---- 输出目录 ----
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
LOG_DIR = OUTPUT_DIR / "logs"
RESULT_DIR = OUTPUT_DIR / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"
PHASE_RECORD = OUTPUT_DIR / "phase_record.json"

for d in [OUTPUT_DIR, LOG_DIR, RESULT_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- 核心实验维度 ----

# 国情预设（6种）
ALL_PROFILES = [
    "manufacturing_giant",      # 制造业大国
    "resource_rich",            # 资源富国
    "digital_leapfrogger",      # 数字跳跃者
    "small_open",               # 小型开放经济体
    "post_soviet",              # 后苏联工业国
    "populous_weak_industry",   # 人口大国弱工业
]

# 积累策略（3种纯策略 + 代表性混合策略）
ALL_REGIMES = ["A", "B", "C"]
ALL_MIXTURES = [
    "pure_A", "pure_B", "pure_C",
    "A_dominant_B", "B_dominant_A",
    "A_B_balanced",
    "uniform",
]

# 国际场景（8种）
ALL_SCENARIOS = [
    "benign_globalization",     # 良性全球化
    "tech_cold_war",            # 技术冷战
    "resource_crash",           # 资源价格崩盘
    "tech_boom",                # 技术繁荣
    "global_recession",         # 全球衰退
    "full_containment",         # 全面封锁
    "tech_explosion",           # 技术爆发
    "resource_war",             # 资源战争
]

# ---- 第一阶段：策略空间普查配置 ----
# 覆盖全部组合：国情×策略×场景
PHASE1_CORE_PROFILES = [
    "manufacturing_giant",
    "resource_rich",
    "digital_leapfrogger",
    "small_open",
    "post_soviet",
    "populous_weak_industry",
]
PHASE1_CORE_SCENARIOS = [
    "benign_globalization",
    "tech_cold_war",
    "resource_crash",
    "tech_boom",
    "global_recession",
    "full_containment",
    "tech_explosion",
    "resource_war",
]
PHASE1_T_MAX = 50

# ---- 第二阶段：关键参数窗口配置 ----
# 基于第一阶段结果，扫描对结果影响最大的参数
PHASE2_KEY_PARAMS = {
    "strategy_params": [
        # 策略A参数
        "A_extraction_max", "A_protection_coefficient", "A_safety_margin",
        # 策略B参数
        "B_g_target", "B_extraction_ceiling", "B_feedback_threshold", "B_feedback_ratio",
        # 策略C参数
        "C_resource_intensity", "C_fdi_openness", "C_tech_import_preference",
    ],
    "structural_params": [
        "xi_infra_to_app",          # infra→app 技术约束
        "institution_capacity",     # 制度能力
        "g_world",                  # 世界经济增长率
        "block_max",                # 最大封锁强度
        "diff_policy_intensity",    # 扩散政策强度
        "capital_flight_sensitivity", # 资本外逃敏感度
    ],
}

PHASE2_DEFAULT_SWEEP_RANGES = {
    "xi_infra_to_app": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
    "block_max": [0.0, 0.20, 0.40, 0.60, 0.80, 0.95],
    "institution_capacity": [0.20, 0.35, 0.50, 0.65, 0.80, 0.90],
    "g_world": [0.01, 0.03, 0.05, 0.07, 0.10, 0.12],
}

# ---- 第三阶段：生存边界测绘配置 ----
PHASE3_N_SAMPLES = 200         # LHS采样数
PHASE3_T_MAX = 100
PHASE3_N_REPEATS = 2           # 每个参数组合重复次数

# ---- 第四阶段：机制深挖配置 ----
PHASE4_N_LHS = 50
PHASE4_T_MAX = 80

# ---- 第五阶段：前沿边界配置 ----
PHASE5_GRID_SIZE = 20          # 断裂边界网格精度（全量用50）
PHASE5_N_LHS = 300             # 技术追赶扫描样本数


# ============================================================
# 工具函数
# ============================================================

import json
import time
from datetime import datetime
from typing import Optional

def phase_started(phase_name: str) -> dict:
    """记录阶段开始."""
    record = load_phase_record()
    record["phases"][phase_name] = {
        "status": "started",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "summary": {},
    }
    save_phase_record(record)
    return record


def phase_completed(phase_name: str, summary: dict = None):
    """记录阶段完成."""
    record = load_phase_record()
    if phase_name in record["phases"]:
        record["phases"][phase_name]["status"] = "completed"
        record["phases"][phase_name]["completed_at"] = datetime.now().isoformat()
        if summary:
            record["phases"][phase_name]["summary"] = summary
    save_phase_record(record)


def phase_status(phase_name: str) -> Optional[str]:
    """查询阶段状态."""
    record = load_phase_record()
    return record["phases"].get(phase_name, {}).get("status")


def load_phase_record() -> dict:
    """加载阶段执行记录."""
    if PHASE_RECORD.exists():
        with open(PHASE_RECORD, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"phases": {}, "created_at": datetime.now().isoformat()}


def save_phase_record(record: dict):
    """保存阶段执行记录."""
    with open(PHASE_RECORD, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_header(title: str):
    """打印格式化的阶段标题."""
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")


def summarize_simulations(results: list) -> dict:
    """对一批模拟结果做基本统计."""
    survived = [r for r in results if r.get("summary", {}).get("survived", False)]
    dead = [r for r in results if not r.get("summary", {}).get("survived", False)]

    survivals = [r["summary"]["survival_years"] for r in results]
    gdps = [r["summary"].get("final_gdp", 0) for r in results]
    cdis = [r["summary"].get("cdi", 0) for r in results if "cdi" in r.get("summary", {})]
    tech_gaps = [r["summary"].get("final_tech_gap", 0) for r in results]
    fractures = [r["summary"].get("total_fractures", 0) for r in results]

    import numpy as np

    return {
        "total": len(results),
        "survived": len(survived),
        "dead": len(dead),
        "survival_rate": len(survived) / max(len(results), 1),
        "survival_years": {
            "mean": float(np.mean(survivals)),
            "median": float(np.median(survivals)),
            "min": float(np.min(survivals)),
            "max": float(np.max(survivals)),
            "std": float(np.std(survivals)),
        },
        "final_gdp": {
            "mean": float(np.mean(gdps)),
            "median": float(np.median(gdps)),
        },
        "final_tech_gap": {
            "mean": float(np.mean(tech_gaps)),
            "median": float(np.median(tech_gaps)),
        },
        "total_fractures": {
            "mean": float(np.mean(fractures)),
            "median": float(np.median(fractures)),
        },
        "cdi": {
            "mean": float(np.mean(cdis)) if cdis else None,
            "median": float(np.median(cdis)) if cdis else None,
        } if cdis else None,
    }


def save_phase_results(phase_name: str, data: dict, filename: str = None):
    """保存阶段结果（自动加时间戳）."""
    ts = get_timestamp()
    fname = filename or f"{phase_name}_{ts}.json"
    path = RESULT_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [保存] 结果已保存到: {path}")
    return path


# ---- 从前期结果中提取线索 ----

def load_phase_summary(phase_name: str) -> dict:
    """加载某个阶段的摘要."""
    record = load_phase_record()
    return record["phases"].get(phase_name, {}).get("summary", {})
