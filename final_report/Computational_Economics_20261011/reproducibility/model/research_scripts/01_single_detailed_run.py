#!/usr/bin/env python
"""01_单次详细模拟 —— 1次模拟的完整诊断报告 (v3.5: 六部门 + 混合策略).

研究问题:
    给定特定的国情 × 策略 × 场景组合，系统会产生什么样的完整发展轨迹？

复杂度: ★☆☆☆☆ (简单)
预计时间: 5-30秒/次
并行度: 单线程
用途: 快速诊断、调试、理解单点行为

运行方式:
    python research_scripts/01_single_detailed_run.py
    python research_scripts/01_single_detailed_run.py --regime B --profile digital_leapfrogger
    python research_scripts/01_single_detailed_run.py --regime A --scenario tech_cold_war --profile manufacturing_giant
    python research_scripts/01_single_detailed_run.py --regime M --mixture A_dominant_B
    python research_scripts/01_single_detailed_run.py --list-profiles
    python research_scripts/01_single_detailed_run.py --list-scenarios
"""

import sys
import os
import argparse
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np

from src.core.engine import SimulationEngine
from src.core.config import SimulationConfig
from src.core.state import ALL_SECTORS, DIGITAL_SECTORS, SERVICE_SECTORS
from src.params.presets import SCENARIO_PRESETS
from src.params.country_profiles import list_country_profiles, get_profile_names
from src.models.mixed_strategy import STANDARD_MIXTURES
from src.analysis.visualize import Visualizer
from common.logger import ResearchLogger
from common.result_store import ResultStore
from common.runner import build_param_sets

# ============================================================
# 配置
# ============================================================
EXPERIMENT_NAME = "01_single_detailed_run"


def main():
    parser = argparse.ArgumentParser(description="单次详细模拟 —— 完整诊断报告")
    parser.add_argument("--regime", default="A", choices=["A", "B", "C", "M"],
                       help="积累策略 (默认: A)")
    parser.add_argument("--scenario", default="benign_globalization",
                       help="国际场景 (默认: benign_globalization)")
    parser.add_argument("--profile", default="manufacturing_giant",
                       help="国情预设 (默认: manufacturing_giant)")
    parser.add_argument("--T-max", type=float, default=60.0,
                       help="模拟年限 (默认: 60)")
    parser.add_argument("--mixed-lambda", type=str, default=None,
                       help="混合策略λ向量，如 '0.5,0.3,0.2' (仅regime=M时有效)")
    parser.add_argument("--mixture", type=str, default=None,
                       help="预设混合策略名称，如 'A_dominant_B' (仅regime=M时有效)")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    parser.add_argument("--list-profiles", action="store_true",
                       help="列出所有国情预设后退出")
    parser.add_argument("--list-scenarios", action="store_true",
                       help="列出所有国际场景后退出")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    parser.add_argument("--config", type=str, default=None,
                       help="v3.7: YAML配置文件路径 (如 config/default.yaml)")
    args = parser.parse_args()

    # 列表模式
    if args.list_profiles:
        print("\n可用的国情预设:")
        for p in list_country_profiles():
            print(f"  [{p['name']:<25}] {p['label']} — {p['description'][:60]}")
        return

    if args.list_scenarios:
        print("\n可用的国际场景:")
        for name, s in SCENARIO_PRESETS.items():
            print(f"  [{name:<25}] {s.get('description', '?')[:60]}")
        return

    # 初始化日志和存储
    logger = ResearchLogger(EXPERIMENT_NAME)
    store = ResultStore(EXPERIMENT_NAME)

    logger.experiment_start(
        f"国情={args.profile} | 策略={args.regime} | 场景={args.scenario} | T_max={args.T_max}",
        regime=args.regime, scenario=args.scenario, profile=args.profile,
    )

    # 构建参数 (v3.7: 支持YAML配置)
    params = build_param_sets(
        regime=args.regime,
        scenario=args.scenario,
        profile=args.profile,
        config_path=args.config,
    )

    # 混合策略 (v3.5: 支持预设混合名称)
    mixed_lambda = None
    mixture_name = None
    if args.regime == "M" or args.mixed_lambda or args.mixture:
        # 预设混合名称优先
        if args.mixture and args.mixture in STANDARD_MIXTURES:
            mixture_name = args.mixture
            params["mixed_lambda"] = STANDARD_MIXTURES[args.mixture]
            params["mixed_name"] = args.mixture
            mixed_lambda = params["mixed_lambda"]
        elif args.mixed_lambda:
            parts = [float(x) for x in args.mixed_lambda.split(",")]
            if len(parts) == 3 and abs(sum(parts) - 1.0) < 0.01:
                params["mixed_lambda"] = tuple(parts)
                mixed_lambda = tuple(parts)
        if not mixed_lambda:
            params["mixed_lambda"] = (0.5, 0.3, 0.2)
            mixed_lambda = (0.5, 0.3, 0.2)

    config = SimulationConfig(
        T_max=args.T_max,
        verbose=args.verbose,
        random_seed=args.seed,
    )

    # 记录配置
    store.set_config({
        "regime": args.regime,
        "scenario": args.scenario,
        "profile": args.profile,
        "T_max": args.T_max,
        "mixed_lambda": mixed_lambda,
        "mixture_name": mixture_name,
        "seed": args.seed,
    })
    store.set_parameters(params)

    # ================================================================
    # 运行模拟
    # ================================================================
    logger.logger.info("开始模拟...")

    engine = SimulationEngine(
        params=params, config=config, regime=args.regime,
        config_path=args.config,
    )
    result = engine.run()
    summary = result["summary"]

    # ================================================================
    # 输出详细诊断
    # ================================================================
    logger.logger.info(f"\n{'─' * 60}")
    logger.logger.info("  模拟结果诊断报告")
    logger.logger.info(f"{'─' * 60}")

    # 1. 基本结果
    survival_status = "[OK] survived" if summary["survived"] else f"[FAIL] collapsed [{summary.get('terminal_fracture', '?')}]"
    logger.logger.info(f"\n  [生存状态]  {survival_status}")
    logger.stat("存活年数", summary["survival_years"], "年")
    logger.stat("期末GDP", summary["final_gdp"], "")
    logger.stat("期末技术差距", summary["final_tech_gap"], "")
    logger.stat("峰值失业率", summary.get("peak_unemployment", 0), "")
    logger.stat("代价集中度(基尼)", summary.get("cost_concentration", 0), "")

    # 2. 断裂诊断
    fractures = summary.get("fracture_types", [])
    n_fractures = summary.get("total_fractures", 0)
    logger.logger.info(f"\n  [断裂诊断]  共触发 {n_fractures} 种断裂")
    for ft in fractures:
        logger.logger.info(f"    [!] {ft}")

    # 3. 五维评价 (v3.5)
    ev = summary.get("evaluation", {})
    if ev.get("enabled"):
        logger.logger.info(f"\n  [五维评价 v3.5]")
        logger.stat("CDI 综合发展指数", ev.get("cdi", float("nan")), "")
        logger.stat("  经济活力", ev.get("economy", float("nan")), "")
        logger.stat("  民生福祉", ev.get("livelihood", float("nan")), "")
        logger.stat("  发展韧性", ev.get("resilience", float("nan")), "")
        logger.stat("  结构健康", ev.get("structure", float("nan")), "")
        logger.stat("  主权自主", ev.get("sovereignty", float("nan")), "")
        logger.stat("红线等级", ev.get("red_line_level", "NONE"), "")
        logger.stat("可接受", ev.get("is_acceptable", True), "")

        if ev.get("red_line_failures"):
            logger.logger.warning(f"    [FAIL] red line failures: {', '.join(ev['red_line_failures'])}")
        if ev.get("red_line_warnings"):
            logger.logger.warning(f"    [WARN] red line warnings: {', '.join(ev['red_line_warnings'])}")

    # 4. 代价分布
    cost_detail = summary.get("cost_detail", {})
    if cost_detail:
        logger.logger.info(f"\n  [代价分布 (按群体)]")
        for group, cost in cost_detail.items():
            if isinstance(cost, dict):
                logger.logger.info(f"    {group}: 失业={cost.get('unemployment_years',0):.1f}年 "
                                   f"薪资损失={cost.get('wage_loss',0):.2f} "
                                   f"累计代价={cost.get('cumulative_cost',0):.2f}")

    # 5. EWS预警
    ews_info = summary.get("ews_info", {})
    if ews_info:
        logger.logger.info(f"\n  [EWS 早期预警]")
        logger.stat("最大EWS(K_i)", ews_info.get("max_K_i", 0), "")
        logger.stat("最大EWS(G)", ews_info.get("max_tech_gap", 0), "")
        logger.stat("韧性指数均值", ews_info.get("mean_resilience", 0), "")

    # 6. v3.5 六部门指标
    logger.logger.info(f"\n  [v3.5 六部门指标]")
    logger.logger.info(f"  所有部门: {', '.join(ALL_SECTORS)}")
    logger.logger.info(f"  数字部门 (infra+app): {', '.join(DIGITAL_SECTORS)}")
    logger.logger.info(f"  服务部门 (high+low): {', '.join(SERVICE_SECTORS)}")

    # 部门资本存量
    sector_info = summary.get("sector_info", {})
    if sector_info:
        logger.logger.info("  期末部门资本存量:")
        for sector_key in ALL_SECTORS:
            k_val = sector_info.get(f"K_{sector_key}", None)
            if k_val is not None:
                logger.logger.info(f"    {sector_key:<10} K={k_val:.1f}")

    # v3.5: 分层次技术差距
    tech_gap_infra = summary.get("final_tech_gap_infra", None)
    tech_gap_app = summary.get("final_tech_gap_app", None)
    if tech_gap_infra is not None:
        logger.stat("技术差距(infra)", tech_gap_infra, "")
    if tech_gap_app is not None:
        logger.stat("技术差距(app)", tech_gap_app, "")

    # v3.5: 就业质量指数
    emp_quality = summary.get("employment_quality_index", None)
    if emp_quality is not None:
        logger.stat("就业质量指数", emp_quality, "[0,1]")

    # v3.5: 六维数字依附
    dep_info = summary.get("dependency_info", {})
    if dep_info:
        logger.logger.info("  六维数字依附:")
        logger.stat("  基础设施综合依附", dep_info.get("infrastructure_dependency", 0), "")
        logger.stat("  芯片依附", dep_info.get("chip_dependency", 0), "")
        logger.stat("  云服务依附", dep_info.get("cloud_dependency", 0), "")
        logger.stat("  操作系统依附", dep_info.get("os_dependency", 0), "")
        logger.stat("  数据依附", dep_info.get("data_dependency", 0), "")
        logger.stat("  标准自主权", dep_info.get("standards_autonomy", 0), "")
        logger.stat("  资本依附", dep_info.get("capital_dependency", 0), "")
        logger.stat("  人才依附", dep_info.get("talent_dependency", 0), "")
        logger.stat("  生态依附", dep_info.get("ecosystem_dependency", 0), "")
        logger.stat("  综合依附指数", dep_info.get("composite", 0), "")

    # v3.5: 混合策略不兼容信息
    incompat_info = summary.get("incompatibility_info", {})
    if incompat_info:
        logger.logger.info("  混合策略不兼容:")
        logger.stat("  不兼容惩罚", incompat_info.get("penalty", 0), "")
        logger.stat("  AB冲突", incompat_info.get("AB_conflict", 0), "")
        logger.stat("  BC冲突", incompat_info.get("BC_conflict", 0), "")
        logger.stat("  AC冲突", incompat_info.get("AC_conflict", 0), "")

    # ================================================================
    # 可视化
    # ================================================================
    logger.logger.info(f"\n{'─' * 60}")
    logger.logger.info("  生成图表...")

    viz = Visualizer()
    fig_name = f"single_{args.regime}_{args.profile}_{args.scenario}.png"

    try:
        viz.plot_single_simulation(result, f"策略{args.regime} × {args.profile} × {args.scenario}",
                                   filename=str(store.figure_path(fig_name)))
        logger.logger.info(f"  ✓ 时间序列图: {store.figure_path(fig_name)}")
    except Exception as e:
        logger.logger.error(f"  图表生成失败: {e}")

    # ================================================================
    # 保存结果
    # ================================================================
    store.add_result(result)
    store.save_all()

    logger.experiment_end(success=True, summary=(
        f"{survival_status} | CDI={ev.get('cdi', float('nan')):.3f} | "
        f"断裂={n_fractures}"
    ))

    print(f"\n✓ 详细报告已保存至: {store.result_dir}")
    print(f"  日志文件: {logger.log_file}")


if __name__ == "__main__":
    main()
