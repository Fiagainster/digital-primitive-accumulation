#!/usr/bin/env python
"""实验5：小型开放体/人口大国的高制度能力对照（P1）

验证"积累门槛"是结构性还是制度性的。

方法：
- small_open × A, inst=0.95 (原 0.40)
- small_open × C, inst=0.95
- populous_weak_industry × C, inst=0.80 (原 0.30)

总计: 3 次单次诊断
"""

import sys
import json
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from research_scripts.common.runner import quick_run

# === 配置 ===
RUNS = [
    ("A", "small_open", "benign_globalization", 0.95),
    ("C", "small_open", "benign_globalization", 0.95),
    ("C", "populous_weak_industry", "benign_globalization", 0.80),
]
T_MAX = 50
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "supplement" / "exp5_high_inst"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one(regime, profile, scenario, inst):
    label = f"{regime}_{profile}_inst{inst}"
    try:
        result = quick_run(
            regime=regime,
            profile=profile,
            scenario=scenario,
            T_max=T_MAX,
            overrides={"institution_capacity": inst},
        )
        s = result["summary"]
        return {
            "label": label,
            "regime": regime,
            "profile": profile,
            "inst_cap_original": {"small_open": 0.40, "populous_weak_industry": 0.30}.get(profile, 0.4),
            "inst_cap_test": inst,
            "survived": s["survived"],
            "survival_years": s["survival_years"],
            "final_gdp": s.get("final_gdp", 0),
            "final_tech_gap": s.get("final_tech_gap", 0),
            "final_welfare": s.get("evaluation", {}).get("livelihood", 0),
            "cdi": s.get("evaluation", {}).get("cdi", 0),
            "total_fractures": s.get("total_fractures", 0),
            "fracture_types": s.get("fracture_types", []),
        }
    except Exception as e:
        return {"label": label, "survived": False, "survival_years": 0, "error": str(e)}


def main():
    print(f"实验5：高制度能力小国/弱工业国对照")
    print(f"  {len(RUNS)} 次模拟")
    print()

    results = []
    start_time = time.time()

    for i, (regime, profile, scenario, inst) in enumerate(RUNS):
        print(f"[{i+1}/{len(RUNS)}] {regime} × {profile} × inst={inst}")
        r = run_one(regime, profile, scenario, inst)
        results.append(r)

        status = "[OK]" if r["survived"] else f"[FAIL @{r['survival_years']:.1f}y]"
        print(f"  → {status} | GDP={r.get('final_gdp', 0):.0f} | "
              f"welfare={r.get('final_welfare', 0):.3f}")

    print("\n=== 结论评估 ===")
    for r in results:
        profile = r["profile"]
        original_inst = r["inst_cap_original"]
        test_inst = r["inst_cap_test"]
        if r["survived"]:
            print(f"  {profile}: inst {original_inst}→{test_inst} → ✅ 存活")
            print(f"    → 积累门槛是**制度性**的（提高制度可突破）")
        else:
            print(f"  {profile}: inst {original_inst}→{test_inst} → ❌ 仍死亡")
            print(f"    → 积累门槛是**结构性**的（制度不足以突破）")

    output_path = OUTPUT_DIR / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至: {output_path}")

    return results


if __name__ == "__main__":
    main()
