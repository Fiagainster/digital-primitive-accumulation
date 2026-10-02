#!/usr/bin/env python
"""P0-2: 社会福利崩溃阈值系统性扫描.

对 welfare_collapse_threshold 进行7水平扫描，覆盖全部6国情×3策略。
关键输出: (a) 存活率热力图; (b) B策略死亡年限漂移; (c) A策略速度陷阱临界g_world漂移.

总计: 7×6×3 + 5×2×7(g_world子扫描) = 126 + 70 = ~196次
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
THRESHOLD_VALUES = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
ALL_REGIMES = ["A", "B", "C"]
ALL_PROFILES = [
    "manufacturing_giant", "resource_rich", "digital_leapfrogger",
    "small_open", "post_soviet", "populous_weak_industry",
]
SCENARIO = "benign_globalization"
T_MAX = 50

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "supplement" / "p0_2_welfare_threshold"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_isolated(kwargs: dict) -> dict:
    """隔离运行单次模拟（供多进程调用）."""
    try:
        engine = SimulationEngine(
            params=kwargs["params"], config=kwargs.get("config", SimulationConfig()),
            regime=kwargs["regime"],
        )
        result = engine.run()
        s = result["summary"]
        return {
            "label": kwargs["label"],
            "profile": kwargs["profile"],
            "regime": kwargs["regime"],
            "threshold": kwargs["threshold"],
            "survived": s["survived"],
            "survival_years": s["survival_years"],
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
                "survived": False, "survival_years": 0,
                "profile": kwargs.get("profile"), "regime": kwargs.get("regime"),
                "threshold": kwargs.get("threshold")}


def _run_gworld_isolated(kwargs: dict) -> dict:
    """隔离运行g_world子扫描."""
    try:
        overrides = {"welfare_collapse_threshold": kwargs["threshold"],
                     "g_world": kwargs["g_world"]}
        params = build_param_sets(regime="A", scenario=SCENARIO,
                                  profile=kwargs["profile"], overrides=overrides)
        engine = SimulationEngine(
            params=params,
            config=SimulationConfig(T_max=T_MAX, platform_enabled=False,
                                    demography_enabled=False, spatial_enabled=False),
            regime="A",
        )
        result = engine.run()
        s = result["summary"]
        return {
            "label": kwargs["label"], "profile": kwargs["profile"], "regime": "A",
            "threshold": kwargs["threshold"], "g_world": kwargs["g_world"],
            "survived": s["survived"], "survival_years": s["survival_years"],
            "final_gdp": s.get("final_gdp", 0),
            "final_tech_gap": s.get("final_tech_gap", 0),
            "terminal_fracture": s.get("terminal_fracture", None),
        }
    except Exception as e:
        return {"label": kwargs["label"], "error": str(e), "survived": False}


def analyze(results, sub_results):
    """汇总分析."""
    print("\n" + "=" * 70)
    print("P0-2 福利崩溃阈值扫描 — 分析报告")
    print("=" * 70)

    # 1. 总体存活率（按阈值）
    print("\n=== 总体存活率（按阈值）===")
    for th in THRESHOLD_VALUES:
        subset = [r for r in results if r.get("threshold") == th]
        survived = sum(1 for r in subset if r.get("survived"))
        rate = survived / max(len(subset), 1)
        print(f"  th={th:.2f}: {survived}/{len(subset)} ({rate:.1%})")

    # 2. 存活率矩阵（阈值×策略）
    print("\n=== 存活率矩阵（阈值×策略）===")
    for regime in ALL_REGIMES:
        print(f"\n  策略{regime}:")
        header = "    " + " ".join(f"th={t:.2f}" for t in THRESHOLD_VALUES)
        print(header)
        for profile in ALL_PROFILES:
            row = f"    {profile:<25}"
            for th in THRESHOLD_VALUES:
                matches = [r for r in results
                          if r.get("threshold") == th
                          and r.get("regime") == regime
                          and r.get("profile") == profile]
                survived = matches[0].get("survived") if matches else False
                row += "  +" if survived else "  -"
            print(row)

    # 3. B策略制造业巨人死亡年限
    print("\n=== B策略 manufacturing_giant 死亡年限（按阈值）===")
    for th in THRESHOLD_VALUES:
        matches = [r for r in results
                   if r.get("threshold") == th
                   and r.get("regime") == "B"
                   and r.get("profile") == "manufacturing_giant"]
        for r in matches:
            status = "存活" if r.get("survived") else f"死亡@{r.get('survival_years', 0):.1f}y"
            print(f"    th={th:.2f}: {status} | GDP={r.get('final_gdp', 0):.0f}")

    # 4. g_world子扫描
    if sub_results:
        print("\n=== g_world × threshold 交互（A策略速度陷阱漂移）===")
        for profile in ["manufacturing_giant", "post_soviet"]:
            print(f"\n  {profile}:")
            print(f"    {'g_world':<8}", end="")
            for th in THRESHOLD_VALUES:
                print(f" th={th:.2f}", end="")
            print()
            for g_w in [0.01, 0.015, 0.02, 0.03, 0.05]:
                print(f"    {g_w:<8}", end="")
                for th in THRESHOLD_VALUES:
                    matches = [r for r in sub_results
                              if r.get("profile") == profile
                              and r.get("g_world") == g_w
                              and r.get("threshold") == th]
                    status = "+" if (matches and matches[0].get("survived")) else "-"
                    print(f"     {status}", end="")
                print()


def main():
    # === 构建主扫描任务 ===
    tasks = []
    for profile in ALL_PROFILES:
        for regime in ALL_REGIMES:
            for th in THRESHOLD_VALUES:
                params = build_param_sets(regime=regime, scenario=SCENARIO,
                                          profile=profile,
                                          overrides={"welfare_collapse_threshold": th})
                config = SimulationConfig(T_max=T_MAX, platform_enabled=False,
                                          demography_enabled=False, spatial_enabled=False)
                tasks.append({
                    "params": params, "regime": regime, "config": config,
                    "label": f"{profile}×{regime}×th={th}",
                    "profile": profile, "threshold": th,
                })

    total = len(tasks)
    print(f"P0-2: 福利崩溃阈值系统性扫描")
    print(f"  {len(THRESHOLD_VALUES)}阈值 × {len(ALL_PROFILES)}国情 × {len(ALL_REGIMES)}策略 = {total}次")
    print(f"  预计: ~15分钟\n")

    # === 并行执行 ===
    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_run_isolated, t): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if len(results) % 20 == 0:
                elapsed = time.time() - start
                print(f"  [{len(results)}/{total}] {elapsed:.0f}s elapsed")

    elapsed = time.time() - start
    print(f"\n主扫描完成！{elapsed:.0f}s ({elapsed/60:.1f}min)")

    # === 追加g_world子扫描 ===
    print("\n--- 追加: g_world × threshold 子扫描 (A策略) ---")
    g_world_values = [0.01, 0.015, 0.02, 0.03, 0.05]
    g_profiles = ["manufacturing_giant", "post_soviet"]

    sub_tasks = []
    for profile in g_profiles:
        for g_w in g_world_values:
            for th in THRESHOLD_VALUES:
                sub_tasks.append({
                    "label": f"gworld_{profile}×A×g={g_w}×th={th}",
                    "profile": profile, "threshold": th, "g_world": g_w,
                })

    sub_results = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_run_gworld_isolated, t): i for i, t in enumerate(sub_tasks)}
        for future in as_completed(futures):
            sub_results.append(future.result())
    print(f"g_world子扫描完成！{len(sub_results)}次")

    # === 分析 ===
    analyze(results, sub_results)

    # === 保存 ===
    out = OUTPUT_DIR / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"main": results, "gworld_subscan": sub_results}, f,
                  ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至: {out}")

    return results, sub_results


if __name__ == "__main__":
    main()
