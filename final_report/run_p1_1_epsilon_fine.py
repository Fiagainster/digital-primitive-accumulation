#!/usr/bin/env python
"""P1-1: CES替代弹性ε精细化扫描.

在 ε∈[0.7, 1.3] 区间以0.05为步长精细扫描，精确标定B策略从存活->崩溃的临界ε。
针对 manufacturing_giant×B（关键崩溃案例）和 digital_leapfrogger×B（对照）.

总计: 13×2 = 26次
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
EPSILON_VALUES = [round(0.70 + i * 0.05, 2) for i in range(13)]  # 0.70 .. 1.30
PROFILES = ["manufacturing_giant", "digital_leapfrogger"]
SCENARIO = "benign_globalization"
T_MAX = 50

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "supplement" / "p1_1_epsilon_fine"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_isolated(kwargs: dict) -> dict:
    try:
        overrides = {"sigma_b": kwargs["epsilon"]}
        params = build_param_sets(regime="B", scenario=SCENARIO,
                                  profile=kwargs["profile"], overrides=overrides)
        config = SimulationConfig(T_max=T_MAX, platform_enabled=False,
                                  demography_enabled=False, spatial_enabled=False)
        engine = SimulationEngine(params=params, config=config, regime="B")
        result = engine.run()
        s = result["summary"]
        return {
            "label": kwargs["label"], "profile": kwargs["profile"],
            "epsilon": kwargs["epsilon"],
            "survived": s["survived"], "survival_years": s["survival_years"],
            "final_gdp": s.get("final_gdp", 0),
            "final_tech_gap": s.get("final_tech_gap", 0),
            "final_welfare": s.get("final_welfare", s.get("evaluation", {}).get("livelihood", 0)),
            "cdi": s.get("evaluation", {}).get("cdi", 0),
            "total_fractures": s.get("total_fractures", 0),
            "fracture_types": s.get("fracture_types", []),
            "terminal_fracture": s.get("terminal_fracture", None),
            "peak_unemployment": s.get("peak_unemployment", 0),
            "cost_ratio_ind_dig": s.get("cost_ratio_industrial_digital", 0),
        }
    except Exception as e:
        return {"label": kwargs["label"], "error": str(e),
                "epsilon": kwargs.get("epsilon"), "survived": False}


def analyze(results):
    print("\n" + "=" * 70)
    print("P1-1 CES替代弹性ε精细化扫描 — 分析报告")
    print("=" * 70)

    for profile in PROFILES:
        print(f"\n=== {profile} × B策略 ===")
        print(f"  {'ε':<8} {'结局':<18} {'GDP':>10} {'welfare':>10} {'tech_gap':>10} {'失业率':>8}")
        print(f"  {'-'*8} {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

        prev_survived = None
        critical_eps = None
        for eps in EPSILON_VALUES:
            matches = [r for r in results
                      if r.get("epsilon") == eps and r.get("profile") == profile]
            for r in matches:
                status = "存活" if r.get("survived") else f"死亡@{r.get('survival_years',0):.1f}y"
                print(f"  {eps:<8.2f} {status:<18} {r.get('final_gdp',0):>10.0f} "
                      f"{r.get('final_welfare',0):>10.3f} {r.get('final_tech_gap',0):>10.1f} "
                      f"{r.get('peak_unemployment',0):>8.1%}")

                # 检测临界值
                if prev_survived is True and not r.get("survived") and critical_eps is None:
                    critical_eps = eps
                    print(f"  >>> 临界ε ≈ {eps:.2f}（存活->死亡切换点）<<<")
                prev_survived = r.get("survived")

        if critical_eps is None:
            print(f"  >>> 在扫描范围内未检测到临界切换 <<<")

    # 与Atkinson范围比较
    print(f"\n=== 临界值与Atkinson(1970)标准范围[0.5, 2.0]比较 ===")
    for profile in PROFILES:
        survivals = [r.get("epsilon") for r in results
                    if r.get("profile") == profile and r.get("survived")]
        deaths = [r.get("epsilon") for r in results
                 if r.get("profile") == profile and not r.get("survived")]
        if survivals and deaths:
            critical = (max(survivals) + min(deaths)) / 2
        elif survivals:
            critical = max(survivals) + 0.05
        else:
            critical = min(deaths) - 0.05
        print(f"  {profile}: 临界ε ≈ {critical:.2f}")
        if 0.9 < critical < 1.1:
            print(f"    -> 临界值在Atkinson范围中间，默认ε=1.5的崩溃结论稳健")
        elif 1.4 < critical < 1.6:
            print(f"    -> 临界值接近默认值1.5，需标注为'边界取值'")
        elif critical < 0.8:
            print(f"    -> 临界值远低于默认值，结论高度稳健")


def main():
    tasks = []
    for profile in PROFILES:
        for eps in EPSILON_VALUES:
            tasks.append({
                "label": f"{profile}×B×ε={eps}", "profile": profile, "epsilon": eps,
            })

    total = len(tasks)
    print(f"P1-1: CES替代弹性ε精细化扫描")
    print(f"  {len(EPSILON_VALUES)}ε水平 × {len(PROFILES)}国情 = {total}次")
    print(f"  预计: ~5分钟\n")

    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_run_isolated, t): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            status = "+" if r.get("survived") else "-"
            print(f"  [{len(results)}/{total}] ε={r.get('epsilon', '?'):.2f} "
                  f"{r.get('profile', '?')} {status}")

    elapsed = time.time() - start
    print(f"\n完成！{elapsed:.0f}s")

    analyze(results)

    out = OUTPUT_DIR / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至: {out}")

    return results


if __name__ == "__main__":
    main()
