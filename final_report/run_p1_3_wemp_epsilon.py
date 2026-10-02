#!/usr/bin/env python
"""P1-3: 就业质量权重w_emp与CES弹性ε的二维交互扫描.

扫描 welfare_employment_exponent [0.25, 0.30, 0.35, 0.40, 0.45] ×
      sigma_b [0.8, 1.0, 1.2, 1.5, 2.0]，
覆盖4国情×B策略。
找出"死亡区"和"存活区"在二维参数空间中的分布，
评估默认值(w_emp=0.30, ε=1.5->sigma_b=0.8?->不对，默认sigma_b=0.8，默认w_emp=0.30)
在二维空间中的位置.

总计: 5×5×4 = 100次
"""

import sys, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from research_scripts.common.runner import build_param_sets
from src.core.engine import SimulationEngine
from src.core.config import SimulationConfig

# === 配置 ===
W_EMP_VALUES = [0.25, 0.30, 0.35, 0.40, 0.45]
EPSILON_VALUES = [0.8, 1.0, 1.2, 1.5, 2.0]
PROFILES = ["manufacturing_giant", "resource_rich", "digital_leapfrogger", "post_soviet"]
SCENARIO = "benign_globalization"
T_MAX = 50

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "supplement" / "p1_3_wemp_epsilon"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_isolated(kwargs: dict) -> dict:
    try:
        overrides = {
            "welfare_employment_exponent": kwargs["w_emp"],
            "sigma_b": kwargs["epsilon"],
        }
        params = build_param_sets(regime="B", scenario=SCENARIO,
                                  profile=kwargs["profile"], overrides=overrides)
        config = SimulationConfig(T_max=T_MAX, platform_enabled=False,
                                  demography_enabled=False, spatial_enabled=False)
        engine = SimulationEngine(params=params, config=config, regime="B")
        result = engine.run()
        s = result["summary"]
        return {
            "label": kwargs["label"], "profile": kwargs["profile"],
            "w_emp": kwargs["w_emp"], "epsilon": kwargs["epsilon"],
            "survived": s["survived"], "survival_years": s["survival_years"],
            "final_gdp": s.get("final_gdp", 0),
            "final_tech_gap": s.get("final_tech_gap", 0),
            "final_welfare": s.get("final_welfare", s.get("evaluation", {}).get("livelihood", 0)),
            "cdi": s.get("evaluation", {}).get("cdi", 0),
            "total_fractures": s.get("total_fractures", 0),
            "fracture_types": s.get("fracture_types", []),
            "terminal_fracture": s.get("terminal_fracture", None),
            "peak_unemployment": s.get("peak_unemployment", 0),
        }
    except Exception as e:
        return {"label": kwargs["label"], "error": str(e),
                "w_emp": kwargs.get("w_emp"), "epsilon": kwargs.get("epsilon"),
                "survived": False}


def analyze(results):
    print("\n" + "=" * 70)
    print("P1-3 w_emp × ε 二维交互扫描 — 分析报告")
    print("=" * 70)

    for profile in PROFILES:
        print(f"\n=== {profile} × B策略 ===")
        print(f"    死亡区(-) vs 存活区(+) 在 (w_emp x epsilon) 空间中的分布:")
        # 表头
        header = f"    {'w_emp\\ε':<10}"
        for eps in EPSILON_VALUES:
            header += f" ε={eps:<5.1f}"
        print(header)
        print(f"    {'-'*10}{'-'*len(EPSILON_VALUES)*10}")

        for w in W_EMP_VALUES:
            row = f"    {w:<10.2f}"
            for eps in EPSILON_VALUES:
                matches = [r for r in results
                          if r.get("w_emp") == w
                          and r.get("epsilon") == eps
                          and r.get("profile") == profile]
                if matches and matches[0].get("survived"):
                    row += "  +存活  "
                elif matches:
                    row += "  -死亡  "
                else:
                    row += "   ?    "
            print(row)

        # 标注默认值位置
        print(f"    * 默认值: w_emp=0.30, epsilon(sigma_b)=0.8")
        default_match = [r for r in results
                        if r.get("w_emp") == 0.30
                        and r.get("epsilon") == 0.8
                        and r.get("profile") == profile]
        if default_match:
            status = "存活区" if default_match[0].get("survived") else "死亡区"
            print(f"      默认值位于: {status}")

    # 全局汇总
    print(f"\n=== 全局汇总 ===")
    for w in W_EMP_VALUES:
        for eps in EPSILON_VALUES:
            subset = [r for r in results
                     if r.get("w_emp") == w and r.get("epsilon") == eps]
            survived = sum(1 for r in subset if r.get("survived"))
            total = max(len(subset), 1)
            print(f"  w_emp={w:.2f} ε={eps:.1f}: {survived}/{total}存活 ({survived/total:.0%})")


def main():
    tasks = []
    for profile in PROFILES:
        for w in W_EMP_VALUES:
            for eps in EPSILON_VALUES:
                tasks.append({
                    "label": f"{profile}×B×w={w}×ε={eps}",
                    "profile": profile, "w_emp": w, "epsilon": eps,
                })

    total = len(tasks)
    print(f"P1-3: w_emp × ε 二维交互扫描")
    print(f"  {len(W_EMP_VALUES)}w_emp × {len(EPSILON_VALUES)}ε × {len(PROFILES)}国情 = {total}次")
    print(f"  预计: ~12分钟\n")

    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_run_isolated, t): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if len(results) % 20 == 0:
                print(f"  [{len(results)}/{total}] {time.time()-start:.0f}s")

    elapsed = time.time() - start
    print(f"\n完成！{elapsed:.0f}s ({elapsed/60:.1f}min)")

    analyze(results)

    out = OUTPUT_DIR / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至: {out}")

    return results


if __name__ == "__main__":
    main()
