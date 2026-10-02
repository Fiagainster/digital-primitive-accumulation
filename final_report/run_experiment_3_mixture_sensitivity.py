#!/usr/bin/env python
"""实验3：混合策略参数敏感性扫描（P1）

验证混合策略"机制互斥"的结论是否在参数变化下稳健。

方法：
1. 比例扫描：对 4 个关键混合配比各测试 3 个变体（±10%）
2. 制度能力扫描：对部分配比测试 inst 变化

总计: ~16 次单次诊断
"""

import sys
import json
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from research_scripts.common.runner import quick_run

# === 配置 ===
PROFILE = "manufacturing_giant"
SCENARIO = "benign_globalization"
T_MAX = 50
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "supplement" / "exp3_mixture"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 标准配比定义（来自 STANDARD_MIXTURES）
# 格式: (label, (lambda_A, lambda_B, lambda_C), profile)
MIXTURE_VARIANTS = [
    # B_dominant_A 原配比: (0.2, 0.7, 0.1)
    ("B_dominant_A", (0.2, 0.7, 0.1), "manufacturing_giant"),   # 基准
    ("B_dominant_A_v2", (0.1, 0.8, 0.1), "manufacturing_giant"), # B=80% 更激进
    ("B_dominant_A_v3", (0.3, 0.6, 0.1), "manufacturing_giant"), # B=60% 更温和

    # A_dominant_B 原配比: (0.7, 0.2, 0.1)
    ("A_dominant_B", (0.7, 0.2, 0.1), "post_soviet"),            # 基准
    ("A_dominant_B_v2", (0.8, 0.1, 0.1), "post_soviet"),        # A=80%
    ("A_dominant_B_v3", (0.6, 0.3, 0.1), "post_soviet"),        # A=60%

    # C_dominant_B 原配比: (0.1, 0.2, 0.7)
    ("C_dominant_B", (0.1, 0.2, 0.7), "resource_rich"),          # 基准
    ("C_dominant_B_v2", (0.1, 0.1, 0.8), "resource_rich"),      # C=80%
    ("C_dominant_B_v3", (0.1, 0.3, 0.6), "resource_rich"),      # C=60%

    # A_B_balanced 原配比: (0.7, 0.3, 0.0)
    ("A_B_balanced", (0.7, 0.3, 0.0), "digital_leapfrogger"),    # 基准
    ("A_B_balanced_v2", (0.8, 0.2, 0.0), "digital_leapfrogger"),# A=80%
    ("A_B_balanced_v3", (0.6, 0.4, 0.0), "digital_leapfrogger"),# A=60%
]

# 制度能力扫描（对部分基准配比）
INST_VARIANTS = [
    ("B_dominant_A_inst50", (0.2, 0.7, 0.1), "manufacturing_giant", 0.50),
    ("B_dominant_A_inst75", (0.2, 0.7, 0.1), "manufacturing_giant", 0.75),
    ("A_dominant_B_inst50", (0.7, 0.2, 0.1), "post_soviet", 0.50),
    ("A_dominant_B_inst75", (0.7, 0.2, 0.1), "post_soviet", 0.75),
]


def run_one(label, lambda_vec, profile, inst=None):
    """单次混合策略运行."""
    overrides = {}
    if inst is not None:
        overrides["institution_capacity"] = inst

    try:
        result = quick_run(
            regime="M",
            profile=profile,
            scenario=SCENARIO,
            T_max=T_MAX,
            overrides=overrides,
        )
        # 手动设置 lambda（quick_run 使用默认混合）
        # 需要通过参数传递混合 lambda
        s = result["summary"]
        return {
            "label": label,
            "lambda": lambda_vec,
            "profile": profile,
            "inst_cap": inst if inst else "default",
            "survived": s["survived"],
            "survival_years": s["survival_years"],
            "final_gdp": s.get("final_gdp", 0),
            "final_tech_gap": s.get("final_tech_gap", 0),
            "final_welfare": s.get("evaluation", {}).get("livelihood", 0),
            "cdi": s.get("evaluation", {}).get("cdi", 0),
            "total_fractures": s.get("total_fractures", 0),
            "fracture_types": s.get("fracture_types", []),
            "incompatibility": s.get("incompatibility_info", {}),
        }
    except Exception as e:
        return {"label": label, "survived": False, "survival_years": 0, "error": str(e)}


def main():
    total = len(MIXTURE_VARIANTS) + len(INST_VARIANTS)
    print(f"实验3：混合策略参数敏感性扫描")
    print(f"  比例扫描: {len(MIXTURE_VARIANTS)} 次")
    print(f"  制度扫描: {len(INST_VARIANTS)} 次")
    print(f"  总计: {total} 次")
    print(f"  注意：混合策略 lambda 需通过 --mixed-lambda 传递")
    print()

    results = []
    start_time = time.time()

    # === 比例扫描 ===
    for i, (label, lvec, profile) in enumerate(MIXTURE_VARIANTS):
        print(f"[{i+1}/{total}] {label} {lvec} × {profile}")

        # quick_run 通过 overrides 传 mixed_lambda
        overrides = {"mixed_lambda": lvec, "mixed_name": label}
        try:
            result = quick_run(
                regime="M",
                profile=profile,
                scenario=SCENARIO,
                T_max=T_MAX,
                overrides=overrides,
            )
            s = result["summary"]
            r = {
                "label": label, "lambda": lvec, "profile": profile,
                "inst_cap": "default", "survived": s["survived"],
                "survival_years": s["survival_years"],
                "final_gdp": s.get("final_gdp", 0),
                "final_tech_gap": s.get("final_tech_gap", 0),
                "final_welfare": s.get("evaluation", {}).get("livelihood", 0),
                "cdi": s.get("evaluation", {}).get("cdi", 0),
                "total_fractures": s.get("total_fractures", 0),
                "fracture_types": s.get("fracture_types", []),
            }
        except Exception as e:
            r = {"label": label, "survived": False, "survival_years": 0, "error": str(e)}

        results.append(r)
        status = "[OK]" if r["survived"] else f"[FAIL @{r['survival_years']:.1f}y]"
        print(f"  → {status} | welfare={r.get('final_welfare', 0):.3f}")

    # === 制度扫描 ===
    for j, (label, lvec, profile, inst) in enumerate(INST_VARIANTS):
        idx = len(MIXTURE_VARIANTS) + j + 1
        print(f"[{idx}/{total}] {label} inst={inst}")

        overrides = {"mixed_lambda": lvec, "institution_capacity": inst}
        try:
            result = quick_run(
                regime="M",
                profile=profile,
                scenario=SCENARIO,
                T_max=T_MAX,
                overrides=overrides,
            )
            s = result["summary"]
            r = {
                "label": label, "lambda": lvec, "profile": profile,
                "inst_cap": inst, "survived": s["survived"],
                "survival_years": s["survival_years"],
                "final_gdp": s.get("final_gdp", 0),
                "final_tech_gap": s.get("final_tech_gap", 0),
                "final_welfare": s.get("evaluation", {}).get("livelihood", 0),
                "cdi": s.get("evaluation", {}).get("cdi", 0),
                "total_fractures": s.get("total_fractures", 0),
                "fracture_types": s.get("fracture_types", []),
            }
        except Exception as e:
            r = {"label": label, "survived": False, "survival_years": 0, "error": str(e)}

        results.append(r)
        status = "[OK]" if r["survived"] else f"[FAIL @{r['survival_years']:.1f}y]"
        print(f"  → {status} | welfare={r.get('final_welfare', 0):.3f}")

    # === 汇总 ===
    elapsed = time.time() - start_time
    print(f"\n完成！总耗时: {elapsed:.0f}s")

    print("\n=== 死亡率汇总 ===")
    dead = [r for r in results if not r["survived"]]
    print(f"  总体: {len(dead)}/{len(results)} 死亡 ({len(dead)/max(len(results),1):.0%})")

    output_path = OUTPUT_DIR / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至: {output_path}")

    return results


if __name__ == "__main__":
    main()
