#!/usr/bin/env python
"""17_数字分层分析 —— infra vs app 追赶路径差异与技术约束瓶颈 (v3.5).

研究问题:
    - 基础设施层(infra)与应用层(app)的追赶路径如何分化？
    - infra受封锁影响多大？app能否在infra落后的情况下繁荣？
    - infra→app技术约束(xi_infra_to_app)在什么条件下成为瓶颈？
    - 两种数字发展模式的分岔条件是什么？

复杂度: ★★☆☆☆ (中等)
预计时间: 10-60秒
并行度: 单线程
用途: 理解六部门模型中数字分层的最核心机制

运行方式:
    python research_scripts/17_digital_layer_analysis.py
    python research_scripts/17_digital_layer_analysis.py --regime B --profile digital_leapfrogger
    python research_scripts/17_digital_layer_analysis.py --sweep xi_infra_to_app --values 0.1,0.2,0.3,0.5
    python research_scripts/17_digital_layer_analysis.py --sweep block_max --values 0.2,0.4,0.6,0.8
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
from src.core.state import ALL_SECTORS, DIGITAL_SECTORS, SERVICE_SECTORS
from src.params.country_profiles import get_profile_names
from common.logger import ResearchLogger
from common.result_store import ResultStore
from common.runner import build_param_sets, quick_run, get_optimal_workers

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "17_digital_layer_analysis"

AVAILABLE_SWEEPS = {
    "xi_infra_to_app": (0.1, 0.5, "infra→app技术溢出强度"),
    "block_max": (0.1, 0.9, "技术封锁最大强度"),
    "alpha_d_infra": (0.15, 0.45, "基础设施层产出弹性"),
}


def main():
    parser = argparse.ArgumentParser(description="数字infra/app分层动态分析 (v3.5)")
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
                       help="参数扫描模式: xi_infra_to_app | block_max | alpha_d_infra")
    parser.add_argument("--values", type=str, default="0.1,0.2,0.3,0.4,0.5",
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
    # Phase 1: 参数扫描或单次运行
    # ================================================================
    logger.phase_start("Phase 1: 模拟运行", 1, 3)

    if args.sweep:
        # 参数扫描模式
        sweep_param = args.sweep
        sweep_values = [float(v) for v in args.values.split(",")]
        sweep_info = AVAILABLE_SWEEPS.get(sweep_param, (0, 1, sweep_param))
        logger.logger.info(f"扫描参数: {sweep_param} ({sweep_info[2]}) = {sweep_values}")
        logger.logger.info(f"扫描点数: {len(sweep_values)}")

        results = []
        for i, val in enumerate(sweep_values):
            overrides = {sweep_param: val}
            try:
                result = quick_run(
                    regime=args.regime, scenario=args.scenario,
                    profile=args.profile, T_max=args.T_max, overrides=overrides
                )
                results.append({"value": val, "result": result})
                logger.logger.debug(f"  [{i+1}/{len(sweep_values)}] {sweep_param}={val:.3f} 完成")
            except Exception as e:
                logger.logger.error(f"  [{i+1}/{len(sweep_values)}] {sweep_param}={val:.3f} 失败: {e}")
    else:
        # 单次详细运行模式
        logger.logger.info("单次详细模拟模式")
        result = quick_run(
            regime=args.regime, scenario=args.scenario,
            profile=args.profile, T_max=args.T_max
        )
        results = [{"value": None, "result": result}]

    logger.phase_end("Phase 1", logger.get_elapsed())

    # ================================================================
    # Phase 2: 分层分析
    # ================================================================
    logger.phase_start("Phase 2: infra/app 分层分析", 2, 3)
    t0 = time.time()

    if args.sweep:
        # 输出 infra vs app 分化表
        logger.logger.info(f"\n  infra vs app 技术差距分化 (扫描 {sweep_param}):")
        headers = ["参数值", "infra_G", "app_G", "infra_GDP%", "app_GDP%",
                   "infra_DL", "app_DL", "emp_quality"]
        rows = []
        for r in results:
            val = r["value"]
            s = r["result"]["summary"]
            v35 = s.get("v35_metrics", {})

            g_infra = v35.get("tech_gap_infra", s.get("final_tech_gap_infra",
                             s.get("final_tech_gap", 0)))
            g_app = v35.get("tech_gap_app", s.get("final_tech_gap_app",
                         s.get("final_tech_gap", 0)))
            gdp = max(s.get("final_gdp", 1), 1)
            k_infra = s.get("sector_K_d_infra", 0)
            k_app = s.get("sector_K_d_app", 0)
            dl_infra = s.get("sector_DL_d_infra", 0)
            dl_app = s.get("sector_DL_d_app", 0)
            emp_q = s.get("employment_quality_index",
                         v35.get("employment_quality_index", 1.0))

            rows.append([
                f"{val:.3f}",
                f"{g_infra:.2f}",
                f"{g_app:.2f}",
                f"{100*k_infra/gdp:.1f}",
                f"{100*k_app/gdp:.1f}",
                f"{dl_infra:.3f}",
                f"{dl_app:.3f}",
                f"{emp_q:.3f}",
            ])

        logger.table("infra vs app 技术差距分化", headers, rows)

        # 瓶颈检测
        infra_constrained = []
        for r in results:
            s = r["result"]["summary"]
            dl_infra = s.get("sector_DL_d_infra", 0)
            dl_app = s.get("sector_DL_d_app", 0)
            if dl_app > 0.3 and dl_infra < 0.1:
                infra_constrained.append(r["value"])

        if infra_constrained:
            logger.finding(
                f"infra→app约束在 {sweep_param}={infra_constrained} 时成为瓶颈——"
                "应用层有发展潜力但基础设施层严重拖后腿",
                level="KEY"
            )
        else:
            logger.finding("当前参数范围内infra未对app形成显著约束瓶颈")

        # 分化检测
        gaps_infra = []
        gaps_app = []
        for r in results:
            s = r["result"]["summary"]
            v35 = s.get("v35_metrics", {})
            gaps_infra.append(v35.get("tech_gap_infra", s.get("final_tech_gap_infra",
                               s.get("final_tech_gap", 0))))
            gaps_app.append(v35.get("tech_gap_app", s.get("final_tech_gap_app",
                             s.get("final_tech_gap", 0))))

        avg_infra = np.mean(gaps_infra)
        avg_app = np.mean(gaps_app)
        if abs(avg_infra - avg_app) > 1.0:
            which = "infra" if avg_infra > avg_app else "app"
            logger.finding(
                f"infra与app技术差距显著分化: infra差距={avg_infra:.1f}, "
                f"app差距={avg_app:.1f}——{which}层追赶明显滞后",
                level="WARN"
            )
    else:
        # 单次运行的详细输出
        s = results[0]["result"]["summary"]
        v35 = s.get("v35_metrics", {})

        logger.logger.info(f"\n  六部门结构 (v3.5):")
        logger.logger.info(f"  全部六部门: {', '.join(ALL_SECTORS)}")
        logger.logger.info(f"  数字部门: {', '.join(DIGITAL_SECTORS)}")
        logger.logger.info(f"  服务部门: {', '.join(SERVICE_SECTORS)}")

        # 部门资本和数字化水平
        logger.logger.info("\n  部门资本存量与数字化水平:")
        for sector_key in ALL_SECTORS:
            k_field = f"sector_K_{sector_key}"
            dl_field = f"sector_DL_{sector_key}"
            k_val = s.get(k_field, 0)
            dl_val = s.get(dl_field, 0)
            logger.logger.info(f"    {sector_key:<10} K={k_val:.1f}  DL={dl_val:.3f}")

        # infra vs app 技术差距
        logger.logger.info("\n  infra vs app 技术追赶:")
        g_infra = v35.get("tech_gap_infra", s.get("final_tech_gap_infra",
                         s.get("final_tech_gap", 0)))
        g_app = v35.get("tech_gap_app", s.get("final_tech_gap_app",
                       s.get("final_tech_gap", 0)))
        logger.stat("infra技术差距", g_infra)
        logger.stat("app技术差距", g_app)
        logger.stat("infra进口依赖", v35.get("d_infra_import_dep", "N/A"))
        logger.stat("app进口依赖", v35.get("d_app_import_dep", "N/A"))

        # 就业质量
        emp_q = s.get("employment_quality_index",
                     v35.get("employment_quality_index", 1.0))
        logger.stat("就业质量指数", emp_q, "[0,1]")

    logger.phase_end("Phase 2", time.time() - t0)

    # ================================================================
    # Phase 3: 发现与总结
    # ================================================================
    logger.phase_start("Phase 3: 发现汇总", 3, 3)
    t0 = time.time()

    if args.sweep:
        # 参数扫描模式下的综合发现
        survived_count = sum(1 for r in results
                          if r["result"]["summary"].get("survived", False))
        logger.stat("整体存活率", f"{survived_count}/{len(results)}",
                   f"({survived_count/len(results)*100:.1f}%)")

        # infra/app 资本比分析
        k_infra_vals = [r["result"]["summary"].get("sector_K_d_infra", 0)
                      for r in results]
        k_app_vals = [r["result"]["summary"].get("sector_K_d_app", 0)
                    for r in results]
        avg_ratio = np.mean([ki / max(ka, 0.01) for ki, ka in zip(k_infra_vals, k_app_vals)])
        logger.stat("infra/app资本比(均值)", avg_ratio)

        if avg_ratio < 0.5:
            logger.finding("infra资本平均不足app的一半——基础设施层严重投资不足")

        # v3.5: 六维依附状态
        dep_data = {}
        for r in results:
            dep_info = r["result"]["summary"].get("dependency_info", {})
            if dep_info:
                for dim in ["chip_dependency", "capital_dependency",
                          "talent_dependency", "ecosystem_dependency",
                          "infrastructure_dependency"]:
                    val = dep_info.get(dim, 0)
                    if dim not in dep_data:
                        dep_data[dim] = []
                    dep_data[dim].append(val)

        if dep_data:
            logger.logger.info("\n  扫描平均六维依附状态:")
            for dim, vals in dep_data.items():
                logger.stat(f"    平均{dim}", np.mean(vals))
    else:
        # 单次运行模式的综合诊断
        s = results[0]["result"]["summary"]
        ev = s.get("evaluation", {})
        if ev.get("enabled"):
            logger.logger.info("\n  五维评价 (v3.5):")
            logger.stat("CDI综合发展指数", ev.get("cdi", float("nan")))
            logger.stat("  经济活力", ev.get("economy", float("nan")))
            logger.stat("  民生福祉", ev.get("livelihood", float("nan")))
            logger.stat("  发展韧性", ev.get("resilience", float("nan")))
            logger.stat("  结构健康", ev.get("structure", float("nan")))
            logger.stat("  主权自主", ev.get("sovereignty", float("nan")))

        # 六维依附
        dep_info = s.get("dependency_info", {})
        if dep_info:
            logger.logger.info("\n  六维数字依附:")
            for dim_label, dim_key in [
                ("基础设施综合依附", "infrastructure_dependency"),
                ("芯片依附", "chip_dependency"),
                ("云服务依附", "cloud_dependency"),
                ("操作系统依附", "os_dependency"),
                ("数据依附", "data_dependency"),
                ("标准自主权", "standards_autonomy"),
                ("资本依附", "capital_dependency"),
                ("人才依附", "talent_dependency"),
                ("生态依附", "ecosystem_dependency"),
                ("综合依附指数", "composite"),
            ]:
                val = dep_info.get(dim_key, "N/A")
                logger.stat(f"    {dim_label}", val)

    logger.phase_end("Phase 3", time.time() - t0)

    # ================================================================
    # Phase 4: 保存结果
    # ================================================================
    logger.phase_start("Phase 4: 保存结果", 4, 4)
    t0 = time.time()

    for r in results:
        store.add_result(r["result"])
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
