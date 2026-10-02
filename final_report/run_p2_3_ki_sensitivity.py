#!/usr/bin/env python
"""P2-3: B策略K_i敏感性形式化检验.

检验§6.5假说: "B策略的敌人不是数字基础太弱，而是工业基础太大"。
在 manufacturing_giant 和 digital_leapfrogger 之间创建 K_i 连续变化的虚拟国情，
标定 K_i 与B策略存活年限的函数关系。

扫描: IC_K_i [50, 400] step 25 (15水平) ×
       institution_capacity [0.40, 0.65] (2水平) ×
       IC_H_i [0.7, 1.0] (2水平)

总计: 15×(1基准+2调节) ≈ 45次
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
K_I_VALUES = list(range(50, 401, 25))  # 50 .. 400 step 25
SCENARIO = "benign_globalization"
T_MAX = 50

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "supplement" / "p2_3_ki_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_isolated(kwargs: dict) -> dict:
    try:
        overrides = {
            "IC_K_i": kwargs["K_i"],
            "institution_capacity": kwargs["inst"],
            "IC_H_i": kwargs["H_i"],
        }
        params = build_param_sets(regime="B", scenario=SCENARIO,
                                  profile="manufacturing_giant",
                                  overrides=overrides)
        config = SimulationConfig(T_max=T_MAX, platform_enabled=False,
                                  demography_enabled=False, spatial_enabled=False)
        engine = SimulationEngine(params=params, config=config, regime="B")
        result = engine.run()
        s = result["summary"]
        return {
            "label": kwargs["label"], "K_i": kwargs["K_i"],
            "inst": kwargs["inst"], "H_i": kwargs["H_i"],
            "survived": s["survived"], "survival_years": s["survival_years"],
            "final_gdp": s.get("final_gdp", 0),
            "final_tech_gap": s.get("final_tech_gap", 0),
            "final_welfare": s.get("final_welfare", s.get("evaluation", {}).get("livelihood", 0)),
            "cdi": s.get("evaluation", {}).get("cdi", 0),
            "total_fractures": s.get("total_fractures", 0),
            "fracture_types": s.get("fracture_types", []),
            "terminal_fracture": s.get("terminal_fracture", None),
            "peak_unemployment": s.get("peak_unemployment", 0),
            "final_K_i": s.get("sector_K_i", 0),
        }
    except Exception as e:
        return {"label": kwargs["label"], "error": str(e),
                "K_i": kwargs.get("K_i"), "survived": False}


def analyze(results):
    print("\n" + "=" * 70)
    print("P2-3 B策略K_i敏感性形式化检验 — 分析报告")
    print("=" * 70)

    # 按调节变量分组
    conditions = [
        ("基准 (inst=0.65, H_i=1.0)", 0.65, 1.0),
        ("弱制度 (inst=0.40, H_i=1.0)", 0.40, 1.0),
        ("低人力资本 (inst=0.65, H_i=0.7)", 0.65, 0.7),
    ]

    for label, inst, H_i in conditions:
        print(f"\n=== {label} ===")
        print(f"  {'K_i':>6} {'结局':<16} {'存活年':>8} {'GDP':>10} {'welfare':>10} "
              f"{'tech_gap':>10} {'失业峰值':>8}")
        print(f"  {'-'*6} {'-'*16} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

        for K_i in K_I_VALUES:
            matches = [r for r in results
                      if r.get("K_i") == K_i
                      and abs(r.get("inst", 0) - inst) < 0.01
                      and abs(r.get("H_i", 0) - H_i) < 0.01]
            for r in matches:
                status = "存活" if r.get("survived") else f"死亡@{r.get('survival_years',0):.1f}y"
                print(f"  {K_i:>6.0f} {status:<16} {r.get('survival_years',0):>8.1f} "
                      f"{r.get('final_gdp',0):>10.0f} {r.get('final_welfare',0):>10.3f} "
                      f"{r.get('final_tech_gap',0):>10.1f} {r.get('peak_unemployment',0):>8.1%}")

    # 安全上限识别
    print(f"\n=== 安全上限识别（基准条件）===")
    safe_ki = []
    for K_i in K_I_VALUES:
        matches = [r for r in results
                  if r.get("K_i") == K_i
                  and abs(r.get("inst", 0) - 0.65) < 0.01
                  and abs(r.get("H_i", 0) - 1.0) < 0.01]
        if matches and matches[0].get("survived"):
            safe_ki.append(K_i)
    if safe_ki:
        print(f"  B策略安全 K_i 范围: {min(safe_ki):.0f} ~ {max(safe_ki):.0f}")
        print(f"  K_i 安全上限 ≈ {max(safe_ki):.0f}")
    else:
        print(f"  B策略在所有K_i水平均死亡")

    # 假说检验
    print(f"\n=== 假说检验 ===")
    if safe_ki:
        max_safe = max(safe_ki)
        print(f"  '工业基础太大'假说: K_i > {max_safe} 时B策略系统性地死亡")
        print(f"  manufacturing_giant: K_i=400 -> {'超过' if 400 > max_safe else '未超过'}安全上限")
        print(f"  digital_leapfrogger: K_i=150 -> {'超过' if 150 > max_safe else '未超过'}安全上限")


def main():
    tasks = []
    conditions = [
        (0.65, 1.0),   # 基准
        (0.40, 1.0),   # 弱制度
        (0.65, 0.7),   # 低人力资本
    ]

    for K_i in K_I_VALUES:
        for inst, H_i in conditions:
            label_cond = "base" if (inst == 0.65 and H_i == 1.0) else \
                        "low_inst" if inst == 0.40 else "low_H"
            tasks.append({
                "label": f"K_i={K_i}_inst={inst}_H={H_i}",
                "K_i": K_i, "inst": inst, "H_i": H_i,
            })

    total = len(tasks)
    print(f"P2-3: B策略K_i敏感性形式化检验")
    print(f"  {len(K_I_VALUES)}K_i水平 × {len(conditions)}条件 = {total}次")
    print(f"  预计: ~5分钟\n")

    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_run_isolated, t): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            status = "+" if r.get("survived") else "-"
            print(f"  [{len(results)}/{total}] K_i={r.get('K_i','?'):.0f} "
                  f"inst={r.get('inst','?'):.2f} H={r.get('H_i','?'):.1f} {status}")

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
