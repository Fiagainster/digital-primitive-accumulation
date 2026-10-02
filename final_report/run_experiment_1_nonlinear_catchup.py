#!/usr/bin/env python
"""实验1：非线性追赶函数稳健性检验（P0 ★最重要）

验证 A 策略的"速度陷阱"在非线性追赶假设下是否仍然成立。

方法：
- 在 logistic 和 piecewise 两种非线性追赶模式下
- 扫描 g_world 参数：0.01, 0.02, 0.03, 0.05, 0.08, 0.10
- 2 国情 × 2 场景 = 4 种组合
- 每个组合运行 3 种追赶模式：linear(基准), logistic, piecewise

总计: 2×6×2×3 = 72 次（含基准线性对照）
"""

import sys
import json
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from research_scripts.common.runner import quick_run

# === 配置 ===
PROFILES = ["manufacturing_giant", "digital_leapfrogger"]
SCENARIOS = ["benign_globalization", "tech_cold_war"]
G_WORLD_VALUES = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
CATCHUP_MODES = ["linear", "logistic", "piecewise"]
T_MAX = 50

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "supplement" / "exp1_nonlinear"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one(profile, scenario, g_world, catchup_mode):
    """单次运行."""
    label = f"{profile}_{scenario}_g{g_world}_{catchup_mode}"
    try:
        result = quick_run(
            regime="A",
            profile=profile,
            scenario=scenario,
            T_max=T_MAX,
            overrides={
                "g_world": g_world,
                "catchup_mode": catchup_mode,
            },
        )
        s = result["summary"]
        return {
            "label": label,
            "profile": profile,
            "scenario": scenario,
            "g_world": g_world,
            "catchup_mode": catchup_mode,
            "survived": s["survived"],
            "survival_years": s["survival_years"],
            "final_gdp": s.get("final_gdp", 0),
            "final_tech_gap": s.get("final_tech_gap", 0),
            "final_welfare": s.get("evaluation", {}).get("livelihood", 0),
            "cdi": s.get("evaluation", {}).get("cdi", 0),
            "total_fractures": s.get("total_fractures", 0),
            "fracture_types": s.get("fracture_types", []),
            "terminal_fracture": s.get("terminal_fracture", None),
        }
    except Exception as e:
        return {
            "label": label,
            "profile": profile,
            "scenario": scenario,
            "g_world": g_world,
            "catchup_mode": catchup_mode,
            "survived": False,
            "survival_years": 0,
            "error": str(e),
        }


def main():
    total = len(PROFILES) * len(SCENARIOS) * len(G_WORLD_VALUES) * len(CATCHUP_MODES)
    print(f"实验1：非线性追赶稳健性检验")
    print(f"  {len(PROFILES)} 国情 × {len(SCENARIOS)} 场景 × {len(G_WORLD_VALUES)} g_world × {len(CATCHUP_MODES)} 模式")
    print(f"  总计 {total} 次模拟")
    print(f"  预计时间: ~15-20 分钟")
    print()

    results = []
    start_time = time.time()
    count = 0

    for profile in PROFILES:
        for scenario in SCENARIOS:
            for g_world in G_WORLD_VALUES:
                for mode in CATCHUP_MODES:
                    count += 1
                    elapsed = time.time() - start_time
                    eta = (elapsed / max(count - 1, 1)) * (total - count + 1)
                    print(f"[{count}/{total}] {profile} × {scenario} × g={g_world} × {mode} "
                          f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

                    r = run_one(profile, scenario, g_world, mode)
                    results.append(r)

                    # 简要状态
                    status = "[OK]" if r["survived"] else f"[FAIL @{r['survival_years']:.1f}y]"
                    print(f"  → {status} | GDP={r.get('final_gdp', 0):.0f} | "
                          f"welfare={r.get('final_welfare', 0):.3f}")

    # === 汇总分析 ===
    elapsed = time.time() - start_time
    print(f"\n完成！总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # 按模式汇总存活率
    print("\n=== 存活率汇总（按模式×g_world）===")
    for mode in CATCHUP_MODES:
        print(f"\n  {mode}:")
        for g in G_WORLD_VALUES:
            subset = [r for r in results if r["catchup_mode"] == mode and r["g_world"] == g]
            survived = sum(1 for r in subset if r["survived"])
            rate = survived / max(len(subset), 1)
            avg_years = sum(r.get("survival_years", 0) for r in subset) / max(len(subset), 1)
            print(f"    g={g:.2f}: {survived}/{len(subset)}存活 ({rate:.0%}) | "
                  f"avg存活={avg_years:.1f}y")

    # 保存完整结果
    output_path = OUTPUT_DIR / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至: {output_path}")

    # 保存摘要 CSV
    csv_path = OUTPUT_DIR / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("profile,scenario,g_world,catchup_mode,survived,survival_years,"
                "final_gdp,final_tech_gap,final_welfare,cdi,total_fractures\n")
        for r in results:
            f.write(f"{r['profile']},{r['scenario']},{r['g_world']},{r['catchup_mode']},"
                    f"{r['survived']},{r['survival_years']},{r.get('final_gdp', 0)},"
                    f"{r.get('final_tech_gap', 0)},{r.get('final_welfare', 0)},"
                    f"{r.get('cdi', 0)},{r.get('total_fractures', 0)}\n")
    print(f"CSV 已保存至: {csv_path}")

    return results


if __name__ == "__main__":
    main()
