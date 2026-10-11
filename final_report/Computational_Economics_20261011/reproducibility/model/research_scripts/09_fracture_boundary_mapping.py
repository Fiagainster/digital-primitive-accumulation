#!/usr/bin/env python
"""09_断裂临界边界绘制 —— EX-F-01: 在参数空间中精确绘制断裂的"活/死"边界 (v3.5).

研究问题:
    各断裂模式的临界参数值是多少？参数空间中是否存在"安全走廊"？
    哪些参数组合会导致系统从"存活"突变为"崩溃"（临界点）？

复杂度: ★★★★☆ (高)
预计时间: 30分钟-2小时
并行度: 多进程
用途: 识别不可逆点、为政策制定提供临界阈值参考

方法:
    1. 从粗扫结果中识别最敏感的参数对
    2. 在参数平面上进行密集网格扫描（50×50 = 2500点）
    3. 绘制断裂临界曲面的2D相图

运行方式:
    python research_scripts/09_fracture_boundary_mapping.py
    python research_scripts/09_fracture_boundary_mapping.py --param1 extraction_rate --param2 feedback_ratio --grid-size 40
"""

import sys
import os
import argparse
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.core.config import SimulationConfig
from common.logger import ResearchLogger
from common.result_store import ResultStore
from common.runner import (
    _run_single_isolated, build_param_sets, get_optimal_workers,
)

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "09_fracture_boundary_mapping"

# 预定义的敏感参数对（基于先验知识和EX-S-01结果）
PREDEFINED_PARAM_PAIRS = {
    "B_extraction_feedback": {
        "param1": "extraction_rate",
        "param2": "feedback_ratio",
        "range1": (0.1, 0.7),
        "range2": (0.01, 0.40),
        "regime": "B",
        "label": "策略B: 提取率 vs 反哺比例",
    },
    "B_tolerance_timing": {
        "param1": "industry_tolerance",
        "param2": "feedback_trigger",
        "range1": (0.1, 0.9),
        "range2": (0.2, 0.9),
        "regime": "B",
        "label": "策略B: 工业容忍度 vs 反哺触发阈值",
    },
    "A_protection_diffusion": {
        "param1": "protection_intensity",
        "param2": "diffusion_policy_strength",
        "range1": (0.0, 0.8),
        "range2": (0.0, 0.6),
        "regime": "A",
        "label": "策略A: 保护强度 vs 扩散政策力度",
    },
    "C_dependency_resource": {
        "param1": "external_dependency",
        "param2": "resource_export_share",
        "range1": (0.1, 0.9),
        "range2": (0.1, 0.9),
        "regime": "C",
        "label": "策略C: 外部依赖 vs 资源出口占比",
    },
}

PARAM_ID_MAP = {
    # 将可读名称映射到参数schema中的实际ID
    "extraction_rate": "tau",
    "feedback_ratio": "feedback_ratio",
    "industry_tolerance": "industry_tolerance",
    "feedback_trigger": "feedback_trigger",
    "protection_intensity": "protection_intensity",
    "diffusion_policy_strength": "diffusion_policy_strength",
    "external_dependency": "ext_dep_base",
    "resource_export_share": "resource_export_share",
}


def resolve_param_id(name: str) -> str:
    """解析参数名到实际ID."""
    return PARAM_ID_MAP.get(name, name)


def main():
    parser = argparse.ArgumentParser(description="断裂临界边界绘制 (EX-F-01)")
    parser.add_argument("--pair", type=str, default="B_extraction_feedback",
                       help=f"预定义参数对: {list(PREDEFINED_PARAM_PAIRS.keys())} "
                            "或 'all'")
    parser.add_argument("--param1", type=str, default=None,
                       help="自定义参数1 (覆盖预定义)")
    parser.add_argument("--param2", type=str, default=None,
                       help="自定义参数2 (覆盖预定义)")
    parser.add_argument("--range1", type=str, default=None,
                       help="参数1范围，如 '0.1,0.7'")
    parser.add_argument("--range2", type=str, default=None,
                       help="参数2范围，如 '0.01,0.4'")
    parser.add_argument("--grid-size", type=int, default=50,
                       help="每维度的网格点数 (默认: 50, 总计2500次模拟)")
    parser.add_argument("--regime", default=None,
                       help="策略 (覆盖预定义)")
    parser.add_argument("--T-max", type=float, default=60.0,
                       help="模拟年限 (默认: 60)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数 (默认: 自动)")
    parser.add_argument("--list-pairs", action="store_true",
                       help="列出所有预定义参数对")
    args = parser.parse_args()

    if args.list_pairs:
        print("\n预定义参数对:")
        for key, info in PREDEFINED_PARAM_PAIRS.items():
            print(f"  [{key}]")
            print(f"    {info['label']}")
            print(f"    策略={info['regime']}, "
                  f"{info['param1']} ∈ {info['range1']}, "
                  f"{info['param2']} ∈ {info['range2']}")
        return

    # 确定要分析的参数对
    if args.pair == "all":
        pairs = list(PREDEFINED_PARAM_PAIRS.items())
    else:
        pair_info = PREDEFINED_PARAM_PAIRS.get(args.pair)
        if pair_info is None:
            print(f"未知参数对: {args.pair}")
            print(f"可选: {list(PREDEFINED_PARAM_PAIRS.keys())}")
            return
        pairs = [(args.pair, pair_info)]

    total_runs = len(pairs) * args.grid_size * args.grid_size

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"参数对={len(pairs)}个 × 网格={args.grid_size}×{args.grid_size}",
        pairs=[p[0] for p in pairs], grid_size=args.grid_size,
    )
    logger.logger.info(f"总模拟次数: {total_runs}")

    store.set_config(vars(args))
    n_workers = get_optimal_workers(args.workers)

    all_results = []

    # 逐参数对分析
    for pair_idx, (pair_name, pair_info) in enumerate(pairs):
        param1_name = args.param1 or pair_info["param1"]
        param2_name = args.param2 or pair_info["param2"]
        param1_id = resolve_param_id(param1_name)
        param2_id = resolve_param_id(param2_name)

        if args.range1:
            range1 = tuple(float(x) for x in args.range1.split(","))
        else:
            range1 = pair_info["range1"]
        if args.range2:
            range2 = tuple(float(x) for x in args.range2.split(","))
        else:
            range2 = pair_info["range2"]

        regime = args.regime or pair_info["regime"]

        logger.phase_start(
            f"{pair_info['label']} ({pair_idx + 1}/{len(pairs)})",
            pair_idx + 1, len(pairs)
        )
        logger.logger.info(f"  策略: {regime}")
        logger.logger.info(f"  {param1_name}({param1_id}) ∈ {range1}")
        logger.logger.info(f"  {param2_name}({param2_id}) ∈ {range2}")
        logger.logger.info(f"  网格: {args.grid_size}×{args.grid_size} = {args.grid_size**2} 次模拟")

        # 构建网格
        values1 = np.linspace(range1[0], range1[1], args.grid_size)
        values2 = np.linspace(range2[0], range2[1], args.grid_size)

        tasks = []
        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                params = build_param_sets(regime=regime)
                params[param1_id] = v1
                params[param2_id] = v2

                task = {
                    "params": params,
                    "regime": regime,
                    "config": SimulationConfig(
                        T_max=args.T_max,
                        random_seed=42,
                        verbose=False,
                    ),
                    "sim_id": len(tasks),
                    "_v1": v1,
                    "_v2": v2,
                    "_param1": param1_name,
                    "_param2": param2_name,
                    "_pair": pair_name,
                }
                tasks.append(task)

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

        logger.progress(len(tasks), len(tasks), f"{pair_name} 完成")

        # ================================================================
        # 分析相图
        # ================================================================
        # 重构2D网格
        survived_grid = np.full((args.grid_size, args.grid_size), np.nan)
        cdi_grid = np.full((args.grid_size, args.grid_size), np.nan)
        fracture_grid = np.full((args.grid_size, args.grid_size), np.nan)

        for idx, r in enumerate(results):
            i = idx // args.grid_size
            j = idx % args.grid_size
            if r.get("_success", True):
                survived_grid[i, j] = 1.0 if r["summary"]["survived"] else 0.0
                cdi_grid[i, j] = r["summary"].get("evaluation", {}).get("cdi", float("nan"))
                fracture_grid[i, j] = r["summary"].get("total_fractures", 0)

        # 存活率
        overall_surv = np.nanmean(survived_grid)
        logger.stat("存活率", f"{overall_surv:.1%}")

        # 找到临界边界：存活→崩溃的过渡带
        survival_gradient = np.abs(np.gradient(survived_grid.astype(float)))
        critical_zone_size = np.sum(survival_gradient > 0.1) / np.sum(~np.isnan(survived_grid))
        logger.stat("临界过渡带占比", f"{critical_zone_size:.1%}")

        # 安全走廊：存活率 > 80% 的连续区域
        safe_zone = survived_grid > 0.8
        safe_fraction = np.nansum(safe_zone) / np.sum(~np.isnan(survived_grid))
        logger.stat("安全区域占比(存活率>80%)", f"{safe_fraction:.1%}")

        # 最优区域：CDI最高的区域
        best_cdi_idx = np.nanargmax(cdi_grid)
        best_i, best_j = np.unravel_index(best_cdi_idx, cdi_grid.shape)
        logger.stat("最优参数组合",
                   f"{param1_name}={values1[best_i]:.3f}, "
                   f"{param2_name}={values2[best_j]:.3f} → CDI={cdi_grid[best_i, best_j]:.3f}")

        # 保存网格数据
        np.savez(
            store.result_dir / f"boundary_{pair_name}.npz",
            param1_name=param1_name, param2_name=param2_name,
            param1_id=param1_id, param2_id=param2_id,
            values1=values1, values2=values2,
            survived=survived_grid, cdi=cdi_grid,
            fractures=fracture_grid,
        )

        # 生成相图
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # 存活/崩溃
            im0 = axes[0].pcolormesh(values2, values1, survived_grid,
                                     cmap="RdYlGn", vmin=0, vmax=1)
            axes[0].set_xlabel(param2_name)
            axes[0].set_ylabel(param1_name)
            axes[0].set_title(f"Survival ({overall_surv:.0%})")
            plt.colorbar(im0, ax=axes[0])

            # CDI
            im1 = axes[1].pcolormesh(values2, values1, cdi_grid,
                                     cmap="viridis")
            axes[1].set_xlabel(param2_name)
            axes[1].set_ylabel(param1_name)
            axes[1].set_title("CDI")
            plt.colorbar(im1, ax=axes[1])

            # 断裂数
            im2 = axes[2].pcolormesh(values2, values1, fracture_grid,
                                     cmap="YlOrRd", vmin=0)
            axes[2].set_xlabel(param2_name)
            axes[2].set_ylabel(param1_name)
            axes[2].set_title("Fracture Count")
            plt.colorbar(im2, ax=axes[2])

            fig.suptitle(pair_info["label"], fontsize=14)
            plt.tight_layout()

            fig_path = store.figure_path(f"phase_{pair_name}.png")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.logger.info(f"  ✓ 相图: {fig_path}")
        except Exception as e:
            logger.logger.error(f"  相图生成失败: {e}")

        all_results.extend(results)

    # 保存
    store.add_batch(all_results)
    store.save_all()

    logger.experiment_end(success=True)


if __name__ == "__main__":
    main()
