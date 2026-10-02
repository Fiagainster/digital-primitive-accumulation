#!/usr/bin/env python
"""P2-1: 混合策略空间大规模系统性探索.

在二维单纯形(A:B:C配比)上进行拉丁超立方采样，覆盖4国情×2场景。
关键输出: (a) 可行配比区域分布; (b) 不可行区域特征; (c) 是否存在"混合甜点".

采用LHS采样，每组合约40个样本，总计约320次。
"""

import sys, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from research_scripts.common.runner import build_param_sets
from src.core.engine import SimulationEngine
from src.core.config import SimulationConfig

# === 配置 ===
PROFILES = ["manufacturing_giant", "resource_rich", "digital_leapfrogger", "post_soviet"]
SCENARIOS = ["benign_globalization", "tech_cold_war"]
N_SAMPLES_PER = 40  # 每组合LHS样本数
T_MAX = 50

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "supplement" / "p2_1_mixture_lhs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sample_simplex(n: int, rng: np.random.Generator) -> np.ndarray:
    """在2D单纯形 (A+B+C=1) 上均匀采样."""
    # Dirichlet(1,1,1) = 均匀分布
    samples = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    return samples  # shape: (n, 3), each row sums to 1


def _run_isolated(kwargs: dict) -> dict:
    try:
        la, lb, lc = kwargs["lambda_vec"]
        # 使用mixed_strategy框架：regime="M"，通过mixed_lambda覆盖配比
        overrides = {
            "mixed_lambda": [la, lb, lc],
            "mixed_name": f"LHS_{la:.2f}_{lb:.2f}_{lc:.2f}",
        }
        params = build_param_sets(regime="M", scenario=kwargs["scenario"],
                                  profile=kwargs["profile"], overrides=overrides)
        config = SimulationConfig(T_max=T_MAX, platform_enabled=False,
                                  demography_enabled=False, spatial_enabled=False)
        engine = SimulationEngine(params=params, config=config, regime="M")
        result = engine.run()
        s = result["summary"]
        return {
            "label": kwargs["label"], "profile": kwargs["profile"],
            "scenario": kwargs["scenario"],
            "lambda_A": la, "lambda_B": lb, "lambda_C": lc,
            "survived": s["survived"], "survival_years": s["survival_years"],
            "final_gdp": s.get("final_gdp", 0),
            "final_tech_gap": s.get("final_tech_gap", 0),
            "final_welfare": s.get("final_welfare", s.get("evaluation", {}).get("livelihood", 0)),
            "cdi": s.get("evaluation", {}).get("cdi", 0),
            "total_fractures": s.get("total_fractures", 0),
            "fracture_types": s.get("fracture_types", []),
            "terminal_fracture": s.get("terminal_fracture", None),
            "cost_ratio_ind_dig": s.get("cost_ratio_industrial_digital", 0),
        }
    except Exception as e:
        return {"label": kwargs["label"], "error": str(e),
                "profile": kwargs.get("profile"), "scenario": kwargs.get("scenario"),
                "lambda_A": kwargs["lambda_vec"][0],
                "lambda_B": kwargs["lambda_vec"][1],
                "lambda_C": kwargs["lambda_vec"][2],
                "survived": False}


def analyze(results):
    print("\n" + "=" * 70)
    print("P2-1 混合策略LHS探索 — 分析报告")
    print("=" * 70)

    for profile in PROFILES:
        for scenario in SCENARIOS:
            subset = [r for r in results
                     if r.get("profile") == profile and r.get("scenario") == scenario]
            survived = [r for r in subset if r.get("survived")]
            dead = [r for r in subset if not r.get("survived")]
            rate = len(survived) / max(len(subset), 1)
            print(f"\n  {profile} × {scenario}: "
                  f"{len(survived)}/{len(subset)}存活 ({rate:.1%})")

            if survived:
                # 存活样本的平均配比
                avg_la = np.mean([r["lambda_A"] for r in survived])
                avg_lb = np.mean([r["lambda_B"] for r in survived])
                avg_lc = np.mean([r["lambda_C"] for r in survived])
                print(f"    存活区平均配比: A={avg_la:.2f} B={avg_lb:.2f} C={avg_lc:.2f}")

                # 最佳CDI的配比
                best = max(survived, key=lambda r: r.get("cdi", 0))
                print(f"    最佳CDI配比: A={best['lambda_A']:.2f} B={best['lambda_B']:.2f} "
                      f"C={best['lambda_C']:.2f} CDI={best.get('cdi', 0):.4f}")

    # 全局统计
    print(f"\n=== 全局统计 ===")
    total_survived = sum(1 for r in results if r.get("survived"))
    total = len(results)
    print(f"  混合策略总体存活率: {total_survived}/{total} ({total_survived/total:.1%})")

    # 对比纯策略（从现有数据可知纯策略总体存活率约34.7%）
    print(f"  纯策略总体存活率(Phase 1): 34.7%")
    diff = total_survived/total - 0.347
    if diff > 0.05:
        print(f"  -> 混合策略存活率显著高于纯策略 (+{diff:.1%})")
    elif diff < -0.05:
        print(f"  -> 混合策略存活率显著低于纯策略 ({diff:.1%})")
    else:
        print(f"  -> 混合策略与纯策略存活率无显著差异")

    # 纯策略顶点检查
    print(f"\n=== 纯策略顶点 vs 混合区域 ===")
    for profile in PROFILES:
        for scenario in SCENARIOS:
            subset = [r for r in results
                     if r.get("profile") == profile and r.get("scenario") == scenario]
            # 接近纯策略顶点的样本（某个λ>0.8）
            near_pure = [r for r in subset
                        if r.get("lambda_A", 0) > 0.8
                        or r.get("lambda_B", 0) > 0.8
                        or r.get("lambda_C", 0) > 0.8]
            mid_mix = [r for r in subset
                      if max(r.get("lambda_A", 0), r.get("lambda_B", 0), r.get("lambda_C", 0)) <= 0.8]
            near_rate = sum(1 for r in near_pure if r.get("survived")) / max(len(near_pure), 1)
            mid_rate = sum(1 for r in mid_mix if r.get("survived")) / max(len(mid_mix), 1)
            if near_pure or mid_mix:
                print(f"  {profile}×{scenario}: 近纯策略={near_rate:.0%} "
                      f"({len(near_pure)}样本) | 中间区域={mid_rate:.0%} ({len(mid_mix)}样本)")


def main():
    rng = np.random.default_rng(42)
    all_lambdas = {}  # (profile, scenario) -> samples

    tasks = []
    for profile in PROFILES:
        for scenario in SCENARIOS:
            samples = sample_simplex(N_SAMPLES_PER, rng)
            all_lambdas[(profile, scenario)] = samples
            for i in range(N_SAMPLES_PER):
                la, lb, lc = samples[i]
                tasks.append({
                    "label": f"{profile}×{scenario}×LHS_{i}",
                    "profile": profile, "scenario": scenario,
                    "lambda_vec": (float(la), float(lb), float(lc)),
                })

    total = len(tasks)
    print(f"P2-1: 混合策略LHS探索")
    print(f"  {len(PROFILES)}国情 × {len(SCENARIOS)}场景 × {N_SAMPLES_PER}样本 = {total}次")
    print(f"  预计: ~35分钟\n")

    start = time.time()
    results = []

    # 分批并行避免内存压力
    BATCH = 80
    for batch_start in range(0, total, BATCH):
        batch = tasks[batch_start:batch_start + BATCH]
        with ProcessPoolExecutor(max_workers=12) as ex:
            futures = {ex.submit(_run_isolated, t): i for i, t in enumerate(batch)}
            for future in as_completed(futures):
                r = future.result()
                results.append(r)

        elapsed = time.time() - start
        n_done = len(results)
        print(f"  [{n_done}/{total}] {elapsed:.0f}s elapsed "
              f"({sum(1 for r in results if r.get('survived'))}存活)")

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
