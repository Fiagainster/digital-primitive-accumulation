#!/usr/bin/env python
"""P1-2: 替代弹性σ系统性扫描.

扫描 sigma_b [0.3, 2.0] step 0.1，覆盖3国情×3策略。
关注σ对以下发现的影响:
- A策略速度陷阱临界g_world位置
- B策略制造业巨人死亡年限
- C策略资源富国DL分化程度

总计: 18×3×3 = 162次
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
SIGMA_VALUES = [round(0.3 + i * 0.1, 1) for i in range(18)]  # 0.3 .. 2.0
PROFILES = ["manufacturing_giant", "resource_rich", "digital_leapfrogger"]
REGIMES = ["A", "B", "C"]
SCENARIO = "benign_globalization"
T_MAX = 50

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "supplement" / "p1_2_sigma_scan"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_isolated(kwargs: dict) -> dict:
    try:
        overrides = {"sigma_b": kwargs["sigma"]}
        params = build_param_sets(regime=kwargs["regime"], scenario=SCENARIO,
                                  profile=kwargs["profile"], overrides=overrides)
        config = SimulationConfig(T_max=T_MAX, platform_enabled=False,
                                  demography_enabled=False, spatial_enabled=False)
        engine = SimulationEngine(params=params, config=config, regime=kwargs["regime"])
        result = engine.run()
        s = result["summary"]
        return {
            "label": kwargs["label"], "profile": kwargs["profile"],
            "regime": kwargs["regime"], "sigma": kwargs["sigma"],
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
                "sigma": kwargs.get("sigma"), "survived": False}


def analyze(results):
    print("\n" + "=" * 70)
    print("P1-2 替代弹性σ系统性扫描 — 分析报告")
    print("=" * 70)

    # 1. 总体存活率（按σ）
    print("\n=== 总体存活率（按σ）===")
    for sigma in SIGMA_VALUES:
        subset = [r for r in results if r.get("sigma") == sigma]
        survived = sum(1 for r in subset if r.get("survived"))
        rate = survived / max(len(subset), 1)
        bar = "#" * int(rate * 40)
        print(f"  σ={sigma:.1f}: {survived}/{len(subset)} ({rate:.1%}) {bar}")

    # 2. 分策略存活率（按σ）
    print("\n=== 分策略存活率（按σ）===")
    for regime in REGIMES:
        print(f"\n  策略{regime}:")
        for sigma in SIGMA_VALUES:
            subset = [r for r in results
                     if r.get("sigma") == sigma and r.get("regime") == regime]
            survived = sum(1 for r in subset if r.get("survived"))
            rate = survived / max(len(subset), 1)
            if rate == 0:
                print(f"    σ={sigma:.1f}: 全灭")
            elif rate == 1.0:
                print(f"    σ={sigma:.1f}: 全活")
            else:
                # 列出存活的具体国情
                alive = [r["profile"] for r in subset if r.get("survived")]
                print(f"    σ={sigma:.1f}: {survived}/{len(subset)} {alive}")

    # 3. B策略制造业巨人死亡年限
    print("\n=== B策略 manufacturing_giant 死亡年限（按σ）===")
    for sigma in SIGMA_VALUES:
        matches = [r for r in results
                   if r.get("sigma") == sigma
                   and r.get("regime") == "B"
                   and r.get("profile") == "manufacturing_giant"]
        for r in matches:
            status = "存活" if r.get("survived") else f"死亡@{r.get('survival_years', 0):.1f}y"
            print(f"    σ={sigma:.1f}: {status} | GDP={r.get('final_gdp', 0):.0f}")

    # 4. C策略resource_rich的DL
    print("\n=== C策略 resource_rich DL（按σ）===")
    for sigma in SIGMA_VALUES:
        matches = [r for r in results
                   if r.get("sigma") == sigma
                   and r.get("regime") == "C"
                   and r.get("profile") == "resource_rich"]
        for r in matches:
            print(f"    σ={sigma:.1f}: survived={r.get('survived')} "
                  f"years={r.get('survival_years', 0):.1f} "
                  f"GDP={r.get('final_gdp', 0):.0f} "
                  f"tech_gap={r.get('final_tech_gap', 0):.1f}")

    # 5. 关键边界
    print("\n=== 关键边界识别 ===")
    for regime in REGIMES:
        b_survive = []
        for sigma in SIGMA_VALUES:
            subset = [r for r in results
                     if r.get("sigma") == sigma and r.get("regime") == regime]
            survived = sum(1 for r in subset if r.get("survived"))
            if survived > 0 and len(b_survive) == 0:
                b_survive.append(sigma)
            if survived == 0 and len(b_survive) == 1:
                b_survive.append(sigma)
                break
        if len(b_survive) >= 2:
            print(f"  策略{regime}: 从sigma={b_survive[0]:.1f}(有存活) -> sigma={b_survive[1]:.1f}(全灭)")


def main():
    tasks = []
    for profile in PROFILES:
        for regime in REGIMES:
            for sigma in SIGMA_VALUES:
                tasks.append({
                    "label": f"{profile}×{regime}×σ={sigma}",
                    "profile": profile, "regime": regime, "sigma": sigma,
                })

    total = len(tasks)
    print(f"P1-2: 替代弹性σ系统性扫描")
    print(f"  {len(SIGMA_VALUES)}σ水平 × {len(PROFILES)}国情 × {len(REGIMES)}策略 = {total}次")
    print(f"  预计: ~20分钟\n")

    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_run_isolated, t): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if len(results) % 30 == 0:
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
