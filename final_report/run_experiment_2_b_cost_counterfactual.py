#!/usr/bin/env python
"""实验2：B 策略零代价反事实对照（P0）

验证"数字精英零代价"在多大程度上是 B 策略参数设定的内置结果。

方法：
- 修改 B_phi_d_jump 从默认 0.6 降至 0.3（数字部门利润 50% 再分配）
- 同时降低 B_g_target 从 0.25 至 0.15（降低提取目标）
- 对 4 个主要国情各运行一次诊断
- 提取数字精英累积代价与基准对比

总计: 4 次单次诊断
"""

import sys
import json
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from research_scripts.common.runner import quick_run

# === 配置 ===
PROFILES = [
    "manufacturing_giant",
    "resource_rich",
    "digital_leapfrogger",
    "post_soviet",
]
T_MAX = 50
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "supplement" / "exp2_counterfactual"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one(profile, tag, overrides):
    """单次运行."""
    label = f"{profile}_{tag}"
    try:
        result = quick_run(
            regime="B",
            profile=profile,
            scenario="benign_globalization",
            T_max=T_MAX,
            overrides=overrides,
        )
        s = result["summary"]
        cost_detail = s.get("cost_detail", {})
        return {
            "label": label,
            "profile": profile,
            "tag": tag,
            "B_phi_d_jump": overrides.get("B_phi_d_jump", 0.6),
            "B_g_target": overrides.get("B_g_target", 0.25),
            "survived": s["survived"],
            "survival_years": s["survival_years"],
            "final_gdp": s.get("final_gdp", 0),
            "final_tech_gap": s.get("final_tech_gap", 0),
            "final_welfare": s.get("evaluation", {}).get("livelihood", 0),
            "cdi": s.get("evaluation", {}).get("cdi", 0),
            "peak_unemployment": s.get("peak_unemployment", 0),
            "digital_elite_cost": cost_detail.get("digital_high_skill", 0) if isinstance(cost_detail.get("digital_high_skill"), (int, float)) else cost_detail.get("digital_high_skill", {}).get("cumulative_cost", 0),
            "industrial_worker_cost": cost_detail.get("industrial_workers", 0) if isinstance(cost_detail.get("industrial_workers"), (int, float)) else cost_detail.get("industrial_workers", {}).get("cumulative_cost", 0),
            "cost_ratio_ind_dig": s.get("cost_ratio_industrial_digital", 0),
            "cost_detail": cost_detail,
            "fracture_types": s.get("fracture_types", []),
            "terminal_fracture": s.get("terminal_fracture", None),
        }
    except Exception as e:
        return {
            "label": label, "profile": profile, "tag": tag,
            "survived": False, "survival_years": 0, "error": str(e),
        }


def main():
    # 两组对照
    variants = [
        ("baseline", {"B_phi_d_jump": 0.6, "B_g_target": 0.25}),    # 默认
        ("redistribute", {"B_phi_d_jump": 0.3, "B_g_target": 0.15}), # 50%再分配
    ]

    total = len(PROFILES) * len(variants)
    print(f"实验2：B 策略零代价反事实对照")
    print(f"  {len(PROFILES)} 国情 × {len(variants)} 变体")
    print(f"  总计 {total} 次模拟")
    print()

    results = []
    start_time = time.time()
    count = 0

    for profile in PROFILES:
        for tag, overrides in variants:
            count += 1
            print(f"[{count}/{total}] {profile} × {tag}")
            r = run_one(profile, tag, overrides)
            results.append(r)

            status = "[OK]" if r["survived"] else f"[FAIL @{r['survival_years']:.1f}y]"
            cost_digital = r.get("digital_elite_cost", "?")
            cost_industrial = r.get("industrial_worker_cost", "?")
            print(f"  → {status} | 数字精英代价={cost_digital} | "
                  f"工业代价={cost_industrial} | welfare={r.get('final_welfare', 0):.3f}")

    elapsed = time.time() - start_time
    print(f"\n完成！总耗时: {elapsed:.0f}s")

    # === 关键对比 ===
    print("\n=== 关键对比：数字精英代价 ===")
    for profile in PROFILES:
        baseline = next(r for r in results if r["profile"] == profile and r["tag"] == "baseline")
        redist = next(r for r in results if r["profile"] == profile and r["tag"] == "redistribute")
        print(f"\n  {profile}:")
        print(f"    基准 (phi=0.6):  数字代价={baseline.get('digital_elite_cost', '?')}  "
              f"存活={baseline['survival_years']:.1f}y  welfare={baseline.get('final_welfare', 0):.3f}")
        print(f"    再分配 (phi=0.3): 数字代价={redist.get('digital_elite_cost', '?')}  "
              f"存活={redist['survival_years']:.1f}y  welfare={redist.get('final_welfare', 0):.3f}")
        if redist["survived"] and not baseline["survived"]:
            print(f"    ⚠ 再分配后B策略存活！核心发现需重评")
        elif redist.get("digital_elite_cost", 0) > 0.1:
            print(f"    ✓ 数字精英开始承担代价")

    # 保存
    output_path = OUTPUT_DIR / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至: {output_path}")

    return results


if __name__ == "__main__":
    main()
