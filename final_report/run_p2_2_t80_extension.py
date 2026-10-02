#!/usr/bin/env python
"""P2-2: T=80对所有国情×策略组合的延长运行.

从Phase 1普查结果中提取所有T=50存活的组合，逐一以T=80重跑。
报告: (a) 有多少T=50存活在T=80转为死亡; (b) 延迟死亡是否系统性集中.

总计: ~50次（取决于Phase 1存活组合数）
"""

import sys, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from research_scripts.common.runner import build_param_sets
from src.core.engine import SimulationEngine
from src.core.config import SimulationConfig

# === Phase 1 普查数据 ===
PHASE1_DATA = Path(__file__).resolve().parent / "final_report" / "data" / "phase1_20260726_114855_results.json"

T_MAX = 80
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "supplement" / "p2_2_t80_extension"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_phase1_survivors() -> list:
    """从Phase 1结果中提取所有T=50存活的组合."""
    if not PHASE1_DATA.exists():
        print(f"[!] Phase 1数据文件不存在: {PHASE1_DATA}")
        print("  使用手动列出的已知存活组合...")
        # 从报告文档可知的存活率矩阵回退
        return _fallback_survivors()

    with open(PHASE1_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

    survivors = []
    for s in data.get("summaries", []):
        if s.get("survived"):
            survivors.append({
                "profile": s["profile"],
                "regime": s["regime"],
                "scenario": s["scenario"],
            })

    print(f"从Phase 1数据加载了 {len(survivors)} 个T=50存活组合")
    return survivors


def _fallback_survivors() -> list:
    """Phase 1存活率回退列表（基于报告中的6×3×8存活率矩阵）."""
    # 存活率矩阵见 final_report/01_实验全貌.md
    survivors = []
    # A: manuf_giant 50%存活 -> benign, tech_cold_war, global_recession, resource_crash
    # B: resource_rich 100%, digital_leapfrogger 100%, manuf_giant 12.5% -> tech_cold_war
    # C: manuf_giant 100%, resource_rich 62.5%(5 scenarios), digital_leap 100%, post_soviet 100%

    # 简化：对所有"有可能存活"的概要进行扫描
    # 至少包含: A×manuf_giant, B×resource_rich, B×digital_leap, C×manuf_giant,
    #          C×resource_rich, C×digital_leap, C×post_soviet
    # 加上各场景的相应组合

    profiles_regimes_for_t80 = [
        # (profile, regime, scenarios)
        ("manufacturing_giant", "A", ["benign_globalization", "tech_cold_war",
                                       "global_recession", "resource_crash"]),
        ("manufacturing_giant", "C", ["benign_globalization", "tech_cold_war",
                                       "global_recession", "resource_crash",
                                       "tech_boom", "resource_war",
                                       "full_containment", "tech_explosion"]),
        ("resource_rich", "B", ["benign_globalization", "tech_cold_war",
                                 "global_recession", "resource_crash",
                                 "tech_boom", "resource_war",
                                 "full_containment", "tech_explosion"]),
        ("resource_rich", "C", ["benign_globalization", "tech_cold_war",
                                 "global_recession", "resource_war"]),
        ("digital_leapfrogger", "B", ["benign_globalization", "tech_cold_war",
                                       "global_recession", "resource_crash",
                                       "tech_boom", "resource_war",
                                       "full_containment", "tech_explosion"]),
        ("digital_leapfrogger", "C", ["benign_globalization", "tech_cold_war",
                                       "global_recession", "resource_crash",
                                       "tech_boom", "resource_war",
                                       "full_containment", "tech_explosion"]),
        ("post_soviet", "C", ["benign_globalization", "tech_cold_war",
                               "global_recession", "resource_crash",
                               "tech_boom", "resource_war",
                               "full_containment", "tech_explosion"]),
    ]

    for profile, regime, scenarios in profiles_regimes_for_t80:
        for scenario in scenarios:
            survivors.append({
                "profile": profile, "regime": regime, "scenario": scenario,
            })

    print(f"使用回退列表: {len(survivors)} 个T=50幸存组合")
    return survivors


def _run_isolated(kwargs: dict) -> dict:
    try:
        params = build_param_sets(regime=kwargs["regime"], scenario=kwargs["scenario"],
                                  profile=kwargs["profile"])
        config = SimulationConfig(T_max=T_MAX, platform_enabled=False,
                                  demography_enabled=False, spatial_enabled=False)
        engine = SimulationEngine(params=params, config=config, regime=kwargs["regime"])
        result = engine.run()
        s = result["summary"]
        return {
            "label": kwargs["label"], "profile": kwargs["profile"],
            "regime": kwargs["regime"], "scenario": kwargs["scenario"],
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
                "profile": kwargs.get("profile"), "regime": kwargs.get("regime"),
                "scenario": kwargs.get("scenario"), "survived": False}


def analyze(results):
    print("\n" + "=" * 70)
    print("P2-2 T=80延长运行 — 分析报告")
    print("=" * 70)

    # 按策略分组
    for regime in ["A", "B", "C"]:
        subset = [r for r in results if r.get("regime") == regime]
        if not subset:
            continue
        survived = sum(1 for r in subset if r.get("survived"))
        dead = sum(1 for r in subset if not r.get("survived"))
        print(f"\n=== 策略{regime} ===")
        print(f"  T=80存活: {survived}/{len(subset)} ({survived/max(len(subset),1):.1%})")

        # 列出死亡案例
        deaths = [r for r in subset if not r.get("survived")]
        if deaths:
            print(f"\n  T=50存活->T=80死亡的案例 ({len(deaths)}个):")
            for r in sorted(deaths, key=lambda x: x.get("survival_years", 0)):
                print(f"    {r['profile']}×{r['regime']}×{r['scenario']}: "
                      f"死亡@{r.get('survival_years',0):.1f}y | "
                      f"GDP={r.get('final_gdp',0):.0f} | "
                      f"断裂={r.get('terminal_fracture','?')}")

        # 列出存活案例
        alive = [r for r in subset if r.get("survived")]
        if alive:
            print(f"\n  真正的长期幸存者 ({len(alive)}个):")
            for r in sorted(alive, key=lambda x: -(x.get("cdi", 0) or 0)):
                print(f"    {r['profile']}×{r['regime']}×{r['scenario']}: "
                      f"GDP={r.get('final_gdp',0):.0f} | "
                      f"welfare={r.get('final_welfare',0):.3f} | "
                      f"CDI={r.get('cdi',0):.4f}")

    # 总体延迟死亡统计
    total_dead_at_80 = sum(1 for r in results if not r.get("survived"))
    total = max(len(results), 1)
    print(f"\n=== 总体 ===")
    print(f"  T=50存活 -> T=80死亡: {total_dead_at_80}/{total} ({total_dead_at_80/total:.1%})")
    if total_dead_at_80 / total > 0.3:
        print(f"  [!] 超过30%的T=50存活在T=80转为死亡 -- '中期窗口'定位需修正")
    elif total_dead_at_80 / total > 0.1:
        print(f"  '延迟死亡'是一个显著但有限的现象")
    else:
        print(f"  '延迟死亡'是罕见事件，T=50窗口可信")

    # 按国情统计
    print(f"\n=== 按国情统计（T=50存活->T=80死亡比例）===")
    profiles_seen = sorted(set(r.get("profile", "?") for r in results))
    for profile in profiles_seen:
        subset = [r for r in results if r.get("profile") == profile]
        dead = sum(1 for r in subset if not r.get("survived"))
        print(f"  {profile}: {dead}/{len(subset)}转为死亡 ({dead/max(len(subset),1):.0%})")


def main():
    survivors = load_phase1_survivors()

    tasks = []
    for s in survivors:
        tasks.append({
            "label": f"{s['profile']}×{s['regime']}×{s['scenario']}_T80",
            "profile": s["profile"], "regime": s["regime"], "scenario": s["scenario"],
        })

    total = len(tasks)
    print(f"P2-2: T=80延长运行")
    print(f"  {total}个T=50存活组合 -> T=80")
    print(f"  预计: ~8分钟\n")

    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_run_isolated, t): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            status = "[OK]" if r.get("survived") else f"[DIED @{r.get('survival_years',0):.1f}y]"
            print(f"  [{len(results)}/{total}] {r.get('label','?')} {status}")

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
