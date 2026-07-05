#!/usr/bin/env python
"""02_参数敏感性分析 —— 局部参数扰动 (±10%) 对关键输出的影响 (v3.5).

研究问题:
    在基准参数点附近，哪些参数对模型输出的影响最大？
    v3.5新增参数: incompat_AB_resource, credit_bias_digital, platform_empowerment,
                  xi_infra_to_app, eta_network_app, capital_flight_sensitivity 等

复杂度: ★★☆☆☆ (简单-中等)
预计时间: 3-10分钟
并行度: 多进程 (默认使用所有可用核心)
用途: 识别关键参数、验证参数校准、发现非线性响应

运行方式:
    python research_scripts/02_parameter_sensitivity.py
    python research_scripts/02_parameter_sensitivity.py --regime B --profile digital_leapfrogger
    python research_scripts/02_parameter_sensitivity.py --n-perturbations 5 --range 0.2
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
from common.runner import build_param_sets, _run_single_isolated, get_optimal_workers

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "02_parameter_sensitivity"

# 关键输出变量（v3.5: 新增分层技术差距和就业质量）
KEY_OUTPUTS = [
    "survived",            # 是否存活
    "survival_years",      # 存活时间
    "final_tech_gap",      # 最终技术差距
    "final_tech_gap_infra",  # v3.5: 基础设施层技术差距
    "final_tech_gap_app",   # v3.5: 应用层技术差距
    "final_gdp",           # 最终GDP
    "peak_unemployment",   # 峰值失业率
    "cost_concentration",  # 代价基尼
    "total_fractures",     # 断裂数量
]


def compute_elasticity(base_val, perturbed_val, param_base, param_perturbed):
    """计算弹性: (ΔY/Y) / (ΔX/X)."""
    if base_val == 0:
        return float("inf") if perturbed_val != 0 else 0.0
    delta_y = (perturbed_val - base_val) / base_val
    delta_x = (param_perturbed - param_base) / param_base
    if delta_x == 0:
        return 0.0
    return delta_y / delta_x


def main():
    parser = argparse.ArgumentParser(description="参数敏感性分析 —— 局部扰动法")
    parser.add_argument("--regime", default="A", choices=["A", "B", "C"],
                       help="积累策略 (默认: A)")
    parser.add_argument("--scenario", default="benign_globalization",
                       help="国际场景 (默认: benign_globalization)")
    parser.add_argument("--profile", default="manufacturing_giant",
                       help="国情预设 (默认: manufacturing_giant)")
    parser.add_argument("--T-max", type=float, default=40.0,
                       help="模拟年限 (默认: 40, 减少计算量)")
    parser.add_argument("--n-perturbations", type=int, default=3,
                       help="每个参数的扰动水平数 (默认: 3, 水平为 ±10%, ±20%, ±30%)")
    parser.add_argument("--range", type=float, default=0.3,
                       help="最大扰动幅度 (默认: 0.3 = ±30%%)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数 (默认: 自动)")
    parser.add_argument("--param-filter", type=str, default=None,
                       help="仅分析匹配此字符串的参数名 (如 'tau' 或 'extraction')")
    args = parser.parse_args()

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)
    n_workers = get_optimal_workers(args.workers)

    logger.experiment_start(
        f"策略={args.regime} | 场景={args.scenario} | 扰动范围=±{args.range:.0%}",
        regime=args.regime, scenario=args.scenario, profile=args.profile,
    )

    store.set_config(vars(args))

    # 构建基准参数
    base_params = build_param_sets(
        regime=args.regime,
        scenario=args.scenario,
        profile=args.profile,
    )

    # 运行基准模拟
    logger.phase_start("基准模拟", 1, 3)
    t0 = logger.get_elapsed()

    config = SimulationConfig(T_max=args.T_max, verbose=False)
    base_task = {"params": base_params, "regime": args.regime, "config": config, "sim_id": -1}
    base_result = _run_single_isolated(base_task)
    base_summary = base_result["summary"]

    logger.sim_end(-1, base_summary)
    logger.stat("基准: 存活", base_summary["survived"])
    logger.stat("基准: 存活年数", base_summary["survival_years"], "年")
    logger.stat("基准: GDP", base_summary["final_gdp"], "")
    logger.stat("基准: 技术差距", base_summary["final_tech_gap"], "")

    # 识别要分析的战略参数
    logger.phase_start("参数扰动", 2, 3)

    strategy_param_ids = [
        pid for pid in base_params.keys()
        if isinstance(base_params[pid], (int, float))
        and pid not in ("mixed_lambda", "mixed_name", "seed")
    ]

    # 参数过滤
    if args.param_filter:
        strategy_param_ids = [
            pid for pid in strategy_param_ids
            if args.param_filter.lower() in pid.lower()
        ]
        logger.logger.info(f"参数过滤 '{args.param_filter}': {len(strategy_param_ids)} 个参数匹配")

    logger.logger.info(f"待分析参数: {len(strategy_param_ids)} 个")
    logger.logger.info(f"扰动水平: {args.n_perturbations} (最大 ±{args.range:.0%})")

    # 构建所有扰动任务
    perturbation_factors = np.linspace(-args.range, args.range, args.n_perturbations)
    perturbation_factors = [f for f in perturbation_factors if abs(f) > 0.001]  # 排除0

    tasks = []
    for pid in strategy_param_ids:
        base_val = base_params[pid]
        if base_val == 0:
            continue
        for factor in perturbation_factors:
            perturbed_params = base_params.copy()
            perturbed_params[pid] = base_val * (1.0 + factor)
            tasks.append({
                "params": perturbed_params,
                "regime": args.regime,
                "config": config,
                "sim_id": len(tasks),
                "_param_id": pid,
                "_factor": factor,
                "_base_val": base_val,
            })

    total_tasks = len(tasks)
    logger.logger.info(f"总模拟次数: {total_tasks}")
    logger.logger.info(f"使用 {n_workers} 个并行核心")

    # 运行所有扰动
    results = [None] * total_tasks
    completed = 0

    if total_tasks <= 3:
        # 串行
        for i, task in enumerate(tasks):
            results[i] = _run_single_isolated(task)
            completed += 1
            if completed % max(1, total_tasks // 10) == 0:
                logger.progress(completed, total_tasks)
    else:
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
                if completed % max(1, total_tasks // 20) == 0:
                    logger.progress(completed, total_tasks)

    # ================================================================
    # 分析结果
    # ================================================================
    logger.phase_start("敏感性分析", 3, 3)

    # 按参数聚合弹性
    sensitivity_scores = {}  # pid -> {output: mean_abs_elasticity}
    all_param_scores = {}

    for i, task in enumerate(tasks):
        if results[i] is None:
            continue

        pid = task["_param_id"]
        perturbed_summary = results[i]["summary"]
        factor = task["_factor"]
        base_val = task["_base_val"]

        if pid not in sensitivity_scores:
            sensitivity_scores[pid] = {out: [] for out in KEY_OUTPUTS}

        for output in KEY_OUTPUTS:
            base_out = base_summary.get(output, 0)
            pert_out = perturbed_summary.get(output, 0)
            if isinstance(base_out, bool):
                base_out = 1.0 if base_out else 0.0
                pert_out = 1.0 if pert_out else 0.0
            elas = compute_elasticity(base_out, pert_out, base_val, base_val * (1 + factor))
            sensitivity_scores[pid][output].append(abs(elas) if not np.isinf(abs(elas)) else 100)

    # 计算每个参数的平均绝对弹性
    for pid in sensitivity_scores:
        all_param_scores[pid] = {}
        for output in KEY_OUTPUTS:
            vals = sensitivity_scores[pid][output]
            all_param_scores[pid][output] = np.mean(vals) if vals else 0.0

    # 计算综合敏感度
    for pid in all_param_scores:
        all_param_scores[pid]["_composite"] = np.mean([
            all_param_scores[pid][out] for out in KEY_OUTPUTS
        ])

    # 排序输出
    sorted_params = sorted(all_param_scores.items(),
                          key=lambda x: x[1]["_composite"], reverse=True)

    # ================================================================
    # 输出报告
    # ================================================================
    logger.logger.info(f"\n{'─' * 80}")
    logger.logger.info("  参数敏感性排名 (综合敏感度 = 各输出弹性的均值)")
    logger.logger.info(f"{'─' * 80}")

    # 表格头部
    headers = ["排名", "参数", "综合敏感度"] + [out[:12] for out in KEY_OUTPUTS[:4]]
    logger.table("参数弹性排名 (Top 20)", headers,
                 [[str(i + 1),
                   pid[:30],
                   f"{scores['_composite']:.3f}"]
                  + [f"{scores.get(out, 0):.3f}" for out in KEY_OUTPUTS[:4]]
                  for i, (pid, scores) in enumerate(sorted_params[:20])])

    # 关键发现
    logger.finding(f"最敏感参数: {sorted_params[0][0]} (综合敏感度: {sorted_params[0][1]['_composite']:.3f})",
                   level="KEY")
    if len(sorted_params) > 1:
        logger.finding(f"第二敏感参数: {sorted_params[1][0]} (综合敏感度: {sorted_params[1][1]['_composite']:.3f})")

    # 检查非线性
    nonlinear_params = []
    for pid in all_param_scores:
        vals = sensitivity_scores[pid].get("survived", [])
        if len(vals) >= 3:
            # 检查是否弹性随扰动幅度变化（非线性迹象）
            if max(vals) - min(vals) > 0.5:
                nonlinear_params.append(pid)
    if nonlinear_params:
        logger.finding(f"检测到 {len(nonlinear_params)} 个参数可能具有非线性响应", level="WARN")

    # ================================================================
    # 保存结果
    # ================================================================
    sensitivity_data = {
        "base_summary": base_summary,
        "param_scores": {pid: scores for pid, scores in sorted_params},
        "nonlinear_params": nonlinear_params,
        "config": vars(args),
    }

    import json
    sens_path = store.result_dir / "sensitivity_results.json"
    with open(sens_path, "w", encoding="utf-8") as f:
        json.dump(store._make_serializable(sensitivity_data), f,
                  ensure_ascii=False, indent=2)

    # 添加简要结果到存储
    store.add_result({"summary": base_summary, "params": base_params})
    store.set_parameters({"analysis_type": "parameter_sensitivity"})
    store.save_all()

    logger.experiment_end(success=True)

    print(f"\n✓ 敏感性分析完成")
    print(f"  结果: {sens_path}")
    print(f"  日志: {logger.log_file}")
    print(f"  最敏感参数: {sorted_params[0][0]}")


if __name__ == "__main__":
    main()
