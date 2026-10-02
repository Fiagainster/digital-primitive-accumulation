#!/usr/bin/env python
"""实验6：CES 替代弹性 ε 和就业质量权重的敏感性（P1）

量化福利崩溃对关键福利函数参数的敏感程度。

方法：对 manufacturing_giant × B 策略，扫描：
1. epsilon_welfare: 0.8, 1.2, 1.5(默认), 2.0, 3.0
2. welfare_employment_exponent: 0.20, 0.30(默认), 0.40, 0.50, 0.60

记录崩溃时间和最终福利指数。

总计: 10 次单次诊断
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
REGIME = "B"
T_MAX = 50
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "supplement" / "exp6_welfare"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 参数扫描值
EPSILON_VALUES = [0.8, 1.2, 1.5, 2.0, 3.0]
EMPLOYMENT_EXP_VALUES = [0.20, 0.30, 0.40, 0.50, 0.60]


def run_one(label, overrides):
    try:
        result = quick_run(
            regime=REGIME,
            profile=PROFILE,
            scenario=SCENARIO,
            T_max=T_MAX,
            overrides=overrides,
        )
        s = result["summary"]
        cost_detail = s.get("cost_detail", {})
        return {
            "label": label,
            **overrides,
            "survived": s["survived"],
            "survival_years": s["survival_years"],
            "final_gdp": s.get("final_gdp", 0),
            "final_tech_gap": s.get("final_tech_gap", 0),
            "final_welfare": s.get("evaluation", {}).get("livelihood", 0),
            "cdi": s.get("evaluation", {}).get("cdi", 0),
            "peak_unemployment": s.get("peak_unemployment", 0),
            "digital_elite_cost": cost_detail.get("digital_high_skill", 0) if isinstance(cost_detail.get("digital_high_skill"), (int, float)) else cost_detail.get("digital_high_skill", {}).get("cumulative_cost", cost_detail.get("digital_high_skill", 0)),
            "industrial_worker_cost": cost_detail.get("industrial_workers", 0) if isinstance(cost_detail.get("industrial_workers"), (int, float)) else cost_detail.get("industrial_workers", {}).get("cumulative_cost", cost_detail.get("industrial_workers", 0)),
            "total_fractures": s.get("total_fractures", 0),
            "fracture_types": s.get("fracture_types", []),
        }
    except Exception as e:
        return {"label": label, "survived": False, "survival_years": 0, "error": str(e)}


def main():
    total = len(EPSILON_VALUES) + len(EMPLOYMENT_EXP_VALUES)
    print(f"实验6：CES 替代弹性与就业权重敏感性")
    print(f"  ε 扫描: {len(EPSILON_VALUES)} 级别")
    print(f"  就业权重扫描: {len(EMPLOYMENT_EXP_VALUES)} 级别")
    print(f"  总计: {total} 次")
    print()

    results = []
    start_time = time.time()
    count = 0

    # epsilon_welfare 扫描
    for eps in EPSILON_VALUES:
        count += 1
        label = f"eps_{eps}"
        print(f"[{count}/{total}] epsilon_welfare={eps}")

        r = run_one(label, {"epsilon_welfare": eps})
        # 标记为默认
        if eps == 1.5:
            r["is_default_eps"] = True
        results.append(r)

        status = "[OK]" if r["survived"] else f"[FAIL @{r['survival_years']:.1f}y]"
        print(f"  → {status} | welfare={r.get('final_welfare', 0):.4f} | "
              f"GDP={r.get('final_gdp', 0):.0f}")

    # welfare_employment_exponent 扫描
    for wexp in EMPLOYMENT_EXP_VALUES:
        count += 1
        label = f"emp_exp_{wexp}"
        print(f"[{count}/{total}] welfare_employment_exponent={wexp}")

        r = run_one(label, {"welfare_employment_exponent": wexp})
        if wexp == 0.30:
            r["is_default_emp"] = True
        results.append(r)

        status = "[OK]" if r["survived"] else f"[FAIL @{r['survival_years']:.1f}y]"
        print(f"  → {status} | welfare={r.get('final_welfare', 0):.4f} | "
              f"GDP={r.get('final_gdp', 0):.0f}")

    # === 汇总 ===
    elapsed = time.time() - start_time
    print(f"\n完成！总耗时: {elapsed:.0f}s")

    print("\n=== ε 扫描结果 ===")
    for eps in EPSILON_VALUES:
        r = next(x for x in results if x["label"] == f"eps_{eps}")
        default_marker = " ← 默认" if eps == 1.5 else ""
        print(f"  ε={eps:.1f}: {'✅' if r['survived'] else '❌'} "
              f"存活={r['survival_years']:.1f}y | welfare={r.get('final_welfare', 0):.4f}"
              f"{default_marker}")

    print("\n=== 就业权重扫描结果 ===")
    for wexp in EMPLOYMENT_EXP_VALUES:
        r = next(x for x in results if x["label"] == f"emp_exp_{wexp}")
        default_marker = " ← 默认" if wexp == 0.30 else ""
        print(f"  w_emp={wexp:.2f}: {'✅' if r['survived'] else '❌'} "
              f"存活={r['survival_years']:.1f}y | welfare={r.get('final_welfare', 0):.4f}"
              f"{default_marker}")

    output_path = OUTPUT_DIR / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至: {output_path}")

    return results


if __name__ == "__main__":
    main()
