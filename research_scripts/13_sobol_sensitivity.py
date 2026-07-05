#!/usr/bin/env python
"""13_Sobol全局敏感性分析 —— 计算每个参数的一阶和总效应Sobol指数 (v3.5: 含新参数).

研究问题:
    在参数空间中，哪些参数对模型输出的方差贡献最大？
    参数间的交互效应有多强（总效应 - 一阶效应）？
    不同策略的敏感参数是否不同？

复杂度: ★★★★★ (极高)
预计时间: 数小时至数十小时（取决于N值和核心数）
并行度: 多进程（最大利用所有核心）
用途: 方法论严谨性——为论文提供敏感性分析表

方法论:
    Saltelli方法: N × (2D + 2) 次模拟
    N=512（基准样本量），D=有效维度数
    = 512 × (2×15 + 2) = 16384次/组合

运行方式:
    python research_scripts/13_sobol_sensitivity.py
    python research_scripts/13_sobol_sensitivity.py --regime B --N 256
    python research_scripts/13_sobol_sensitivity.py --N 128 --regimes A  # 快速版
"""

import sys
import os
import argparse
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.core.config import SimulationConfig
from src.params.schema import get_schema
from src.params.presets import get_scenario
from common.logger import ResearchLogger
from common.result_store import ResultStore
from common.runner import (
    _run_single_isolated, build_param_sets, get_optimal_workers,
)

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "13_sobol_sensitivity"

KEY_OUTPUTS_SOBOL = [
    "survived",
    "survival_years",
    "final_tech_gap",
    "final_gdp",
    "total_fractures",
    "cost_concentration",
]


def saltelli_sample(N: int, D: int, bounds: np.ndarray, seed: int = 42) -> dict:
    """Saltelli采样矩阵.

    A: N×D 矩阵
    B: N×D 矩阵
    AB_i: N×D 矩阵（B的第i列替换为A的第i列）

    Returns:
        {"A": A, "B": B} 用于构建AB_i
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 1, (N, D))
    B = rng.uniform(0, 1, (N, D))
    return {"A": A, "B": B, "N": N, "D": D, "bounds": bounds}


def compute_sobol_indices(Y_A, Y_B, Y_AB_list) -> dict:
    """从模型输出计算Sobol指数.

    Args:
        Y_A: A矩阵的输出 (N,)
        Y_B: B矩阵的输出 (N,)
        Y_AB_list: AB_i矩阵的输出列表 [(N,) × D]

    Returns:
        {"S1": [...], "ST": [...], "S2": [[...]]}  # 可选二阶
    """
    N = len(Y_A)
    D = len(Y_AB_list)

    # 总方差
    f0_sq = np.mean(Y_A) ** 2
    var_Y = np.mean(Y_A * Y_B) - f0_sq

    S1 = np.zeros(D)
    ST = np.zeros(D)

    for i in range(D):
        Y_AB_i = Y_AB_list[i]

        # 一阶效应
        S1[i] = (np.mean(Y_B * Y_AB_i) - f0_sq) / var_Y

        # 总效应
        ST[i] = 1.0 - (np.mean(Y_A * Y_AB_i) - f0_sq) / var_Y

    # 截断到 [0, 1]
    S1 = np.clip(S1, 0, 1)
    ST = np.clip(ST, 0, 1)

    return {"S1": S1, "ST": ST, "total_variance": var_Y}


def main():
    parser = argparse.ArgumentParser(description="Sobol全局敏感性分析")
    parser.add_argument("--regimes", type=str, nargs="+", default=["A"],
                       help="策略列表 (默认: A; 可指定多个如 A B C)")
    parser.add_argument("--scenario", default="benign_globalization",
                       help="国际场景")
    parser.add_argument("--profile", default="manufacturing_giant",
                       help="国情预设")
    parser.add_argument("--N", type=int, default=256,
                       help="基准样本量N (默认: 256; Saltelli需要N×(2D+2)次)")
    parser.add_argument("--T-max", type=float, default=60.0,
                       help="模拟年限 (默认: 60)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数")
    parser.add_argument("--outputs", type=str, default=None,
                       help="分析的关键输出变量，逗号分隔")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                       help="中间结果保存间隔 (默认: 每1000次)")
    args = parser.parse_args()

    regimes = args.regimes
    if args.outputs:
        key_outputs = args.outputs.split(",")
    else:
        key_outputs = KEY_OUTPUTS_SOBOL

    # 获取参数维度
    schema = get_schema()
    param_ids = [pid for pid in schema.all_ids()
                 if schema.get(pid).layer == "strategy"]
    param_ids = param_ids[:15]  # 限制维度
    D = len(param_ids)

    # 获取参数边界
    bounds = np.zeros((D, 2))
    for i, pid in enumerate(param_ids):
        p = schema.get(pid)
        bounds[i, 0] = p.domain[0] if p.domain else 0.0
        bounds[i, 1] = p.domain[1] if p.domain else 1.0

    total_runs_per_regime = args.N * (2 * D + 2)

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"Sobol: N={args.N}, D={D}, 策略={regimes} | "
        f"每策略{total_runs_per_regime}次模拟",
        N=args.N, D=D, regimes=regimes, param_ids=param_ids,
    )
    logger.logger.info(f"总模拟次数: {total_runs_per_regime * len(regimes)}")
    logger.logger.info(f"参数维度: {param_ids}")
    logger.logger.info(f"关键输出: {key_outputs}")

    store.set_config(vars(args))
    n_workers = get_optimal_workers(args.workers)

    all_sobol_results = {}

    for regime in regimes:
        logger.phase_start(f"策略 {regime}", regimes.index(regime) + 1, len(regimes))
        logger.logger.info(f"Saltelli采样 N={args.N}, D={D}...")

        # Saltelli采样
        sampling = saltelli_sample(args.N, D, bounds, seed=42)

        # 构建所有任务
        tasks = []

        # A矩阵: N 次
        for i in range(args.N):
            params = build_param_sets(regime=regime, scenario=args.scenario,
                                     profile=args.profile)
            for j, pid in enumerate(param_ids):
                lo, hi = bounds[j]
                params[pid] = lo + sampling["A"][i, j] * (hi - lo)
            tasks.append({
                "params": params, "regime": regime,
                "config": SimulationConfig(T_max=args.T_max, random_seed=42+i, verbose=False),
                "sim_id": len(tasks), "_type": "A", "_idx": i,
            })

        # B矩阵: N 次
        for i in range(args.N):
            params = build_param_sets(regime=regime, scenario=args.scenario,
                                     profile=args.profile)
            for j, pid in enumerate(param_ids):
                lo, hi = bounds[j]
                params[pid] = lo + sampling["B"][i, j] * (hi - lo)
            tasks.append({
                "params": params, "regime": regime,
                "config": SimulationConfig(T_max=args.T_max, random_seed=1000+i, verbose=False),
                "sim_id": len(tasks), "_type": "B", "_idx": i,
            })

        # AB_i矩阵: N×D 次
        for dim in range(D):
            for i in range(args.N):
                params = build_param_sets(regime=regime, scenario=args.scenario,
                                         profile=args.profile)
                for j, pid in enumerate(param_ids):
                    lo, hi = bounds[j]
                    if j == dim:
                        params[pid] = lo + sampling["A"][i, j] * (hi - lo)
                    else:
                        params[pid] = lo + sampling["B"][i, j] * (hi - lo)
                tasks.append({
                    "params": params, "regime": regime,
                    "config": SimulationConfig(T_max=args.T_max, random_seed=2000+i, verbose=False),
                    "sim_id": len(tasks), "_type": "AB", "_idx": i, "_dim": dim,
                })

        logger.logger.info(f"总任务数: {len(tasks)}")

        # 运行
        results = [None] * len(tasks)
        completed = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_single_isolated, task): i
                for i, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"summary": {"survived": False, "error": str(e)},
                                    "_error": str(e)}

                completed += 1
                if completed % max(1, len(tasks) // 10) == 0:
                    logger.progress(completed, len(tasks))

        logger.progress(len(tasks), len(tasks), f"策略{regime} 全部完成")

        # 提取输出
        def extract_output(r, output_name):
            s = r["summary"]
            if output_name == "survived":
                return 1.0 if s.get("survived", False) else 0.0
            elif output_name == "survival_years":
                return s.get("survival_years", 0)
            elif output_name == "final_tech_gap":
                return s.get("final_tech_gap", 999)
            elif output_name == "final_gdp":
                return s.get("final_gdp", 0)
            elif output_name == "total_fractures":
                return s.get("total_fractures", 0)
            elif output_name == "cost_concentration":
                return s.get("cost_concentration", 0)
            return 0.0

        # 按类型分组
        A_results = [r for r in results if r.get("_type") == "A"]
        B_results = [r for r in results if r.get("_type") == "B"]
        AB_results = [r for r in results if r.get("_type") == "AB"]

        # 对每个输出计算Sobol指数
        sobol_outputs = {}
        for output_name in key_outputs:
            Y_A = np.array([extract_output(r, output_name) for r in A_results])
            Y_B = np.array([extract_output(r, output_name) for r in B_results])

            # 构建 Y_AB_list by dimension
            Y_AB_list = []
            for dim in range(D):
                Y_AB_dim = [extract_output(r, output_name)
                           for r in AB_results if r.get("_dim") == dim]
                Y_AB_list.append(np.array(Y_AB_dim))

            try:
                sobol = compute_sobol_indices(Y_A, Y_B, Y_AB_list)
                sobol_outputs[output_name] = sobol
            except Exception as e:
                logger.logger.error(f"Sobol计算失败 ({output_name}): {e}")
                sobol_outputs[output_name] = {"S1": np.zeros(D), "ST": np.zeros(D)}

        # 输出排名
        logger.logger.info(f"\n{'─' * 80}")
        logger.logger.info(f"  策略{regime} — Sobol敏感性排名")
        logger.logger.info(f"{'─' * 80}")

        for output_name in key_outputs:
            if output_name not in sobol_outputs:
                continue
            sobol = sobol_outputs[output_name]
            logger.logger.info(f"\n  输出: {output_name} (总方差={sobol.get('total_variance',0):.2e})")

            # 排名
            ranked = sorted(enumerate(sobol["ST"]),
                          key=lambda x: -x[1])
            logger.table(f"Sobol总效应指数 (Top 10)",
                        ["排名", "参数", "S1(一阶)", "ST(总效应)", "交互(ST-S1)"],
                        [[str(i+1),
                          param_ids[idx][:30],
                          f"{sobol['S1'][idx]:.4f}",
                          f"{sobol['ST'][idx]:.4f}",
                          f"{sobol['ST'][idx] - sobol['S1'][idx]:.4f}"]
                         for i, (idx, _) in enumerate(ranked[:10])])

            # 最重要的参数
            top_idx = ranked[0][0]
            logger.finding(
                f"{output_name}: 最重要参数 = {param_ids[top_idx]} "
                f"(ST={sobol['ST'][top_idx]:.4f}, S1={sobol['S1'][top_idx]:.4f})",
                level="KEY"
            )

            # 交互效应
            high_interaction = [(idx, sobol["ST"][idx] - sobol["S1"][idx])
                              for idx in range(D)
                              if sobol["ST"][idx] - sobol["S1"][idx] > 0.1]
            if high_interaction:
                logger.finding(
                    f"{output_name}: {len(high_interaction)}个参数有显著交互效应(ST-S1>0.1)"
                )

        all_sobol_results[regime] = {
            "param_ids": param_ids,
            "outputs": sobol_outputs,
        }

    # ================================================================
    # 跨策略比较
    # ================================================================
    if len(regimes) > 1:
        logger.logger.info(f"\n{'─' * 80}")
        logger.logger.info("  跨策略敏感参数比较")
        logger.logger.info(f"{'─' * 80}")

        for output_name in key_outputs:
            logger.logger.info(f"\n  输出: {output_name}")
            for regime in regimes:
                if regime not in all_sobol_results:
                    continue
                sobol = all_sobol_results[regime]["outputs"].get(output_name, {})
                if not sobol.get("ST") is not None:
                    continue
                top3 = sorted(enumerate(sobol["ST"]), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"{param_ids[idx][:20]}={sobol['ST'][idx]:.3f}"
                                   for idx, _ in top3)
                logger.logger.info(f"    策略{regime} Top3: {top_str}")

    # 保存
    sobol_path = store.result_dir / "sobol_results.json"
    with open(sobol_path, "w", encoding="utf-8") as f:
        json.dump(store._make_serializable(all_sobol_results), f,
                  ensure_ascii=False, indent=2)

    store.set_parameters({"analysis": "sobol", "results_file": str(sobol_path)})
    store.save_all()

    logger.experiment_end(success=True)

    print(f"\n✓ Sobol敏感性分析完成")
    print(f"  Sobol结果: {sobol_path}")


if __name__ == "__main__":
    main()
