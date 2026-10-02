#!/usr/bin/env python
"""实验4：T=80 延长运行 — A 策略和 C 策略（P1）

验证"延迟死亡"是否为 B 策略专属现象。

方法：
- A 策略：manufacturing_giant × benign_globalization, T=80
- C 策略：manufacturing_giant, digital_leapfrogger, resource_rich × benign, T=80
- 对比已知的 B 策略 T=80 结果

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
RUNS = [
    ("A", "manufacturing_giant", "benign_globalization"),
    ("C", "manufacturing_giant", "benign_globalization"),
    ("C", "digital_leapfrogger", "benign_globalization"),
    ("C", "resource_rich", "benign_globalization"),
]
T_MAX = 80
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "supplement" / "exp4_t80"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one(regime, profile, scenario):
    """单次运行."""
    label = f"{regime}_{profile}_{scenario}_T80"
    try:
        result = quick_run(
            regime=regime,
            profile=profile,
            scenario=scenario,
            T_max=T_MAX,
        )
        s = result["summary"]
        return {
            "label": label,
            "regime": regime,
            "profile": profile,
            "scenario": scenario,
            "T_max": T_MAX,
            "survived": s["survived"],
            "survival_years": s["survival_years"],
            "final_gdp": s.get("final_gdp", 0),
            "final_tech_gap": s.get("final_tech_gap", 0),
            "final_welfare": s.get("evaluation", {}).get("livelihood", 0),
            "cdi": s.get("evaluation", {}).get("cdi", 0),
            "peak_unemployment": s.get("peak_unemployment", 0),
            "total_fractures": s.get("total_fractures", 0),
            "fracture_types": s.get("fracture_types", []),
            "terminal_fracture": s.get("terminal_fracture", None),
        }
    except Exception as e:
        return {"label": label, "survived": False, "survival_years": 0, "error": str(e)}


def main():
    print(f"实验4：T=80 延长运行")
    print(f"  {len(RUNS)} 次模拟")
    print()

    results = []
    start_time = time.time()

    for i, (regime, profile, scenario) in enumerate(RUNS):
        print(f"[{i+1}/{len(RUNS)}] {regime} × {profile} × {scenario} T=80")
        r = run_one(regime, profile, scenario)
        results.append(r)

        status = "[OK]" if r["survived"] else f"[FAIL @{r['survival_years']:.1f}y]"
        print(f"  → {status} | GDP={r.get('final_gdp', 0):.0f} | "
              f"welfare={r.get('final_welfare', 0):.3f} | "
              f"失业={r.get('peak_unemployment', 0):.1%}")

    # 对比已知的 B 策略 T=80 结果
    print("\n=== 与 B 策略 T=80 对比 ===")
    print("  已知: resource_rich×B T=80 → ❌ @78.8y (延迟死亡)")
    print("  已知: digital_leapfrogger×B T=80 → ✅ @80.0y (唯一幸存者)")
    for r in results:
        regime = r["regime"]
        profile = r["profile"]
        if r["survived"]:
            print(f"  {regime}×{profile}: ✅ T=80 — 无延迟死亡")
        else:
            print(f"  {regime}×{profile}: ❌@{r['survival_years']:.1f}y — "
                  f"{'延迟死亡' if r['survival_years'] > 50 else 'T=50内已死亡'}")

    output_path = OUTPUT_DIR / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至: {output_path}")

    return results


if __name__ == "__main__":
    main()
