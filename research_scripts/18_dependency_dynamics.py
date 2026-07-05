#!/usr/bin/env python
"""18_数字依附动态分析 —— 依附自我强化与逆转路径 (v3.5).

研究问题:
    - 数字依附一旦形成是否可逆？自我强化强度有多大？
    - 依附自我强化循环的触发阈值与速度
    - 从依附到自主的逆转条件（制度驱动/危机驱动/市场驱动）
    - 锁定效应的突破时间窗口与隐性追赶天花板

复杂度: ★★☆☆☆ (中等)
预计时间: 10-60秒
并行度: 单线程
用途: 回答"技术依赖是否注定永久落后"的核心问题

运行方式:
    python research_scripts/18_dependency_dynamics.py
    python research_scripts/18_dependency_dynamics.py --sweep institution_capacity --values 0.3,0.5,0.7,0.9
    python research_scripts/18_dependency_dynamics.py --profile resource_rich --regime C
    python research_scripts/18_dependency_dynamics.py --sweep block_max --values 0.2,0.4,0.6,0.8
"""

import sys
import os
import argparse
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.core.engine import SimulationEngine
from src.core.config import SimulationConfig
from src.core.state import ALL_SECTORS, DIGITAL_SECTORS
from src.params.country_profiles import get_profile_names
from common.logger import ResearchLogger
from common.result_store import ResultStore
from common.runner import build_param_sets, quick_run, get_optimal_workers

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "18_dependency_dynamics"

AVAILABLE_SWEEPS = {
    "institution_capacity": (0.2, 0.9, "制度能力"),
    "block_max": (0.1, 0.9, "技术封锁最大强度"),
    "C_indigenous_rd_ratio": (0.1, 0.7, "自主创新投入比"),
    "xi_infra_to_app": (0.1, 0.5, "infra→app技术溢出"),
}


def extract_dependency_metrics(result: dict) -> dict:
    """从模拟结果中提取依附动态指标."""
    s = result["summary"]
    v35 = s.get("v35_metrics", {})
    dep_info = s.get("dependency_info", {})

    return {
        "chip_dep": dep_info.get("chip_dependency",
                   v35.get("dependency_chip", 0)),
        "capital_dep": dep_info.get("capital_dependency",
                       v35.get("dependency_capital", 0)),
        "talent_dep": dep_info.get("talent_dependency",
                      v35.get("dependency_talent", 0)),
        "ecosystem_dep": dep_info.get("ecosystem_dependency",
                         v35.get("dependency_ecosystem", 0)),
        "infra_dep": dep_info.get("infrastructure_dependency",
                     v35.get("dependency_infrastructure", 0)),
        "composite_dep": dep_info.get("composite", 0),
        "tech_gap_infra": v35.get("tech_gap_infra",
                           s.get("final_tech_gap_infra",
                           s.get("final_tech_gap", 0))),
        "tech_gap_app": v35.get("tech_gap_app",
                         s.get("final_tech_gap_app",
                         s.get("final_tech_gap", 0))),
        "d_infra_import": v35.get("d_infra_import_dep", 0),
        "d_app_import": v35.get("d_app_import_dep", 0),
        "survived": s.get("survived", False),
        "survival_years": s.get("survival_years", 0),
        "emp_quality": s.get("employment_quality_index",
                        v35.get("employment_quality_index", 1.0)),
    }


def main():
    parser = argparse.ArgumentParser(description="数字依附自我强化与逆转分析 (v3.5)")
    parser.add_argument("--regime", default="A", choices=["A", "B", "C", "M"],
                       help="积累策略 (默认: A)")
    parser.add_argument("--profile", default="manufacturing_giant",
                       help="国情预设 (默认: manufacturing_giant)")
    parser.add_argument("--scenario", default="tech_cold_war",
                       help="国际场景 (默认: tech_cold_war)")
    parser.add_argument("--T-max", type=float, default=60.0,
                       help="模拟年限 (默认: 60)")
    parser.add_argument("--sweep", default=None,
                       choices=list(AVAILABLE_SWEEPS.keys()),
                       help="参数扫描: institution_capacity | block_max | C_indigenous_rd_ratio | xi_infra_to_app")
    parser.add_argument("--values", type=str, default="0.3,0.5,0.7,0.9",
                       help="扫描参数值，逗号分隔")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行核心数 (默认: 自动)")
    args = parser.parse_args()

    # 初始化
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"策略={args.regime} | 场景={args.scenario} | 国情={args.profile}",
        regime=args.regime, scenario=args.scenario, profile=args.profile,
        sweep=args.sweep,
    )

    store.set_config(vars(args))

    # ================================================================
    # Phase 1: 运行模拟
    # ================================================================
    logger.phase_start("Phase 1: 模拟运行", 1, 4)

    if args.sweep:
        sweep_param = args.sweep
        sweep_values = [float(v) for v in args.values.split(",")]
        sweep_info = AVAILABLE_SWEEPS.get(sweep_param, (0, 1, sweep_param))
        logger.logger.info(f"扫描参数: {sweep_param} ({sweep_info[2]}) = {sweep_values}")

        results = []
        for i, val in enumerate(sweep_values):
            overrides = {sweep_param: val}
            try:
                result = quick_run(
                    regime=args.regime, scenario=args.scenario,
                    profile=args.profile, T_max=args.T_max, overrides=overrides
                )
                dep = extract_dependency_metrics(result)
                dep["param_value"] = val
                results.append(dep)
                logger.logger.debug(f"  [{i+1}/{len(sweep_values)}] {sweep_param}={val:.3f} 完成")
            except Exception as e:
                logger.logger.error(f"  [{i+1}/{len(sweep_values)}] 失败: {e}")
    else:
        logger.logger.info("单次详细模拟模式")
        result = quick_run(
            regime=args.regime, scenario=args.scenario,
            profile=args.profile, T_max=args.T_max
        )
        dep = extract_dependency_metrics(result)
        dep["param_value"] = None
        results = [dep]

    logger.phase_end("Phase 1", logger.get_elapsed())

    # ================================================================
    # Phase 2: 依附演化轨迹分析
    # ================================================================
    logger.phase_start("Phase 2: 依附演化轨迹", 2, 4)
    t0 = time.time()

    if args.sweep:
        # 依附指标 vs 扫参值 表格
        param_name = args.sweep
        headers = [param_name, "chip_dep", "capital_dep", "talent_dep",
                   "eco_dep", "comp_dep", "G_infra", "G_app", "survived"]
        rows = []
        for d in results:
            rows.append([
                f"{d['param_value']:.3f}",
                f"{d['chip_dep']:.3f}",
                f"{d['capital_dep']:.3f}",
                f"{d['talent_dep']:.3f}",
                f"{d['ecosystem_dep']:.3f}",
                f"{d['composite_dep']:.3f}",
                f"{d['tech_gap_infra']:.2f}",
                f"{d['tech_gap_app']:.2f}",
                "Y" if d["survived"] else "N",
            ])

        logger.table(f"依附指标 vs {param_name}", headers, rows)

        # 依附自我强化检测
        dep_trend = [d["composite_dep"] for d in results]
        param_vals = [d["param_value"] for d in results]

        # 检测是否存在阈值效应（陡增）
        if len(dep_trend) >= 3:
            diffs = np.diff(dep_trend)
            max_diff_idx = np.argmax(np.abs(diffs))
            if abs(diffs[max_diff_idx]) > 0.15:
                threshold_val = (param_vals[max_diff_idx] + param_vals[max_diff_idx + 1]) / 2
                logger.finding(
                    f"检测到依附陡变: {param_name}={threshold_val:.2f}附近，"
                    f"综合依附指数变化={diffs[max_diff_idx]:.3f}——可能存在阈值效应",
                    level="KEY"
                )

        # 逆转条件分析
        reversal_candidates = [d for d in results
                             if d["chip_dep"] < 0.5 and d["survived"]]
        if reversal_candidates:
            rev_vals = [d["param_value"] for d in reversal_candidates]
            logger.finding(
                f"在 {param_name}={rev_vals} 时出现依附逆转 "
                f"(chip_dep<0.5)——制度/创新投入可以有效降低依附",
                level="KEY"
            )
        else:
            logger.finding(
                "当前参数范围内未观察到依附逆转——数字依附具有强锁定效应",
                level="WARN"
            )

        # 隐性天花板检测
        high_dep = [d for d in results if d["chip_dep"] > 0.7]
        if high_dep:
            avg_gap_infra = np.mean([d["tech_gap_infra"] for d in high_dep])
            avg_gap_app = np.mean([d["tech_gap_app"] for d in high_dep])
            logger.finding(
                f"高依附组(chip_dep>0.7)平均infra技术差距={avg_gap_infra:.1f}，"
                f"app差距={avg_gap_app:.1f}——"
                "可能存在隐性追赶天花板：依附导致技术永远无法收敛",
                level="WARN"
            )

    else:
        # 单次运行的依附状态
        d = results[0]
        logger.logger.info("\n  六维依附状态 (v3.5):")
        dep_items = [
            ("芯片依附", d["chip_dep"]),
            ("资本依附", d["capital_dep"]),
            ("人才依附", d["talent_dep"]),
            ("生态依附", d["ecosystem_dep"]),
            ("基础设施综合依附", d["infra_dep"]),
            ("综合依附指数", d["composite_dep"]),
            ("infra进口依赖", d["d_infra_import"]),
            ("app进口依赖", d["d_app_import"]),
        ]
        for label, val in dep_items:
            if val > 0.7:
                level_tag = "[高危]"
            elif val > 0.4:
                level_tag = "[中等]"
            else:
                level_tag = "[安全]"
            logger.logger.info(f"    {label:<20} {val:.3f} {level_tag}")

        logger.logger.info("\n  分层技术差距:")
        logger.stat("infra技术差距", d["tech_gap_infra"])
        logger.stat("app技术差距", d["tech_gap_app"])
        logger.stat("就业质量指数", d["emp_quality"], "[0,1]")

        if d["tech_gap_infra"] > d["tech_gap_app"] * 1.5:
            logger.finding(
                "infra技术差距显著大于app——应用层繁荣但基础受制于人",
                level="WARN"
            )
        elif d["tech_gap_app"] > d["tech_gap_infra"] * 1.5:
            logger.finding(
                "app技术差距显著大于infra——基础设施尚可但应用生态薄弱"
            )
        else:
            logger.finding("infra与app技术差距相对均衡")

    logger.phase_end("Phase 2", time.time() - t0)

    # ================================================================
    # Phase 3: 逆转路径深度分析
    # ================================================================
    logger.phase_start("Phase 3: 逆转路径分析", 3, 4)
    t0 = time.time()

    if args.sweep:
        # 分析依附-追赶的关系
        dep_gap_corr_infra = np.corrcoef(
            [d["composite_dep"] for d in results],
            [d["tech_gap_infra"] for d in results]
        )[0, 1] if len(results) >= 3 else 0
        dep_gap_corr_app = np.corrcoef(
            [d["composite_dep"] for d in results],
            [d["tech_gap_app"] for d in results]
        )[0, 1] if len(results) >= 3 else 0

        logger.stat("依附-技术差距相关性(infra)", dep_gap_corr_infra)
        logger.stat("依附-技术差距相关性(app)", dep_gap_corr_app)

        if abs(dep_gap_corr_infra) > 0.6:
            logger.finding(
                f"infra技术差距与依附高度相关(r={dep_gap_corr_infra:.3f})——"
                "基础设施层追赶受依附影响更深",
                level="KEY"
            )

        # 制度/自主创新的边际效益
        if len(results) >= 2:
            first = results[0]
            last = results[-1]
            dep_change = last["composite_dep"] - first["composite_dep"]
            gap_change = last["tech_gap_infra"] - first["tech_gap_infra"]
            logger.stat(f"{args.sweep}变化导致的依附变化", dep_change)
            logger.stat(f"{args.sweep}变化导致的infra差距变化", gap_change)

            if dep_change < -0.1:
                logger.finding(
                    f"提高{args.sweep}可有效降低数字依附 "
                    f"(依附变化={dep_change:.3f})"
                )
    else:
        # 单次运行的路径诊断
        d = results[0]
        logger.logger.info("\n  依附路径诊断:")
        if d["composite_dep"] > 0.7:
            logger.logger.warning("  深度依附状态——需要大幅度制度变革才能逆转")
        elif d["composite_dep"] > 0.4:
            logger.logger.info("  中度依附——存在逆转窗口但需要持续努力")
        else:
            logger.logger.info("  低依附——自主可控状态")

        if d["tech_gap_infra"] < 2.0 and d["composite_dep"] < 0.5:
            logger.finding("技术接近前沿且依附低——良性发展路径")
        elif d["tech_gap_infra"] > 5.0 and d["composite_dep"] > 0.7:
            logger.finding("技术锁死+高度依附——恶性循环已形成", level="WARN")

    logger.phase_end("Phase 3", time.time() - t0)

    # ================================================================
    # Phase 4: 保存结果
    # ================================================================
    logger.phase_start("Phase 4: 保存结果", 4, 4)
    t0 = time.time()

    # 将提取的指标保存到store
    store.set_parameters({
        "dependency_metrics": results if args.sweep else [results[0]],
        "sweep_param": args.sweep,
    })
    store.save_all()

    logger.phase_end("Phase 4", time.time() - t0)

    logger.experiment_end(success=True, summary=(
        f"策略={args.regime} | 扫参={args.sweep or '单次'} | "
        f"模拟次数={len(results)}"
    ))

    print(f"\n结果已保存至: {store.result_dir}")
    print(f"  日志文件: {logger.log_file}")


if __name__ == "__main__":
    main()
