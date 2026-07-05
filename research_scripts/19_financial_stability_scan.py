#!/usr/bin/env python
"""19_金融稳定性与就业质量扫描 —— 资产泡沫/资本外逃/信贷错配联动分析 (v3.5).

研究问题:
    - 数字资产泡沫在什么条件下破裂？破裂对infra/app的不同冲击？
    - 资本外逃的触发阈值——财富集中度+政治不稳定如何联动？
    - 信贷过度流向数字部门是否挤出工业融资？
    - 就业质量指数如何随策略选择演化？

复杂度: ★★★☆☆ (中等偏高)
预计时间: 1-10分钟
并行度: 多进程
用途: 揭示数字经济发展中的金融脆弱性与社会成本

运行方式:
    python research_scripts/19_financial_stability_scan.py
    python research_scripts/19_financial_stability_scan.py --regime B
    python research_scripts/19_financial_stability_scan.py --sweep capital_flight_sensitivity
    python research_scripts/19_financial_stability_scan.py --sweep credit_bias_digital --values 0.5,1.0,1.5,2.0
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
EXPERIMENT_NAME = "19_financial_stability_scan"

AVAILABLE_SWEEPS = {
    "capital_flight_sensitivity": (0.01, 0.50, "资本外逃敏感度"),
    "credit_bias_digital": (0.5, 3.0, "信贷数字偏好倍数"),
    "institution_capacity": (0.2, 0.9, "制度能力"),
    "financial_openness": (0.1, 0.9, "金融开放度"),
}


def extract_financial_metrics(result: dict) -> dict:
    """提取金融稳定性指标."""
    s = result["summary"]
    v35 = s.get("v35_metrics", {})

    return {
        "survived": s.get("survived", False),
        "survival_years": s.get("survival_years", 0),
        "final_gdp": s.get("final_gdp", 0),
        "employment_quality": s.get("employment_quality_index",
                              v35.get("employment_quality_index", 1.0)),
        "credit_rationing": s.get("credit_rationing_ratio", 1.0),
        "bubble_pressure": s.get("bubble_pressure", 0),
        "capital_flight": s.get("capital_flight_rate", 0),
        "cost_concentration": s.get("cost_concentration", 0),
        "wealth_gini": s.get("final_gini", s.get("cost_concentration", 0)),
        "fracture_count": s.get("total_fractures", 0),
        "fracture_types": s.get("fracture_types", []),
        "political_support": s.get("political_support", 0.7),
        # v3.5: 六部门资本
        "d_infra_K": s.get("sector_K_d_infra", 0),
        "d_app_K": s.get("sector_K_d_app", 0),
        "i_K": s.get("sector_K_i", 0),
        "b_K": s.get("sector_K_b", 0),
        # v3.5: 分层技术差距
        "tech_gap_infra": v35.get("tech_gap_infra",
                           s.get("final_tech_gap_infra",
                           s.get("final_tech_gap", 0))),
        "tech_gap_app": v35.get("tech_gap_app",
                         s.get("final_tech_gap_app",
                         s.get("final_tech_gap", 0))),
        # 评价
        "cdi": s.get("evaluation", {}).get("cdi", float("nan")),
    }


def main():
    parser = argparse.ArgumentParser(description="金融稳定性与就业质量分析 (v3.5)")
    parser.add_argument("--regime", default="B", choices=["A", "B", "C", "M"],
                       help="积累策略 (默认: B——跳跃策略数字泡沫风险最高)")
    parser.add_argument("--profile", default="manufacturing_giant",
                       help="国情预设 (默认: manufacturing_giant)")
    parser.add_argument("--scenario", default="benign_globalization",
                       help="国际场景 (默认: benign_globalization)")
    parser.add_argument("--T-max", type=float, default=60.0,
                       help="模拟年限 (默认: 60)")
    parser.add_argument("--sweep", default=None,
                       choices=list(AVAILABLE_SWEEPS.keys()),
                       help="参数扫描: capital_flight_sensitivity | credit_bias_digital | institution_capacity | financial_openness")
    parser.add_argument("--values", type=str, default="0.05,0.15,0.25,0.35,0.45",
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
        logger.logger.info(f"扫描点数: {len(sweep_values)}")

        results = []
        for i, val in enumerate(sweep_values):
            overrides = {sweep_param: val}
            try:
                result = quick_run(
                    regime=args.regime, scenario=args.scenario,
                    profile=args.profile, T_max=args.T_max, overrides=overrides
                )
                m = extract_financial_metrics(result)
                m["param_value"] = val
                results.append(m)
                logger.logger.debug(f"  [{i+1}/{len(sweep_values)}] {sweep_param}={val:.3f} 完成")
            except Exception as e:
                logger.logger.error(f"  [{i+1}/{len(sweep_values)}] 失败: {e}")
    else:
        logger.logger.info("单次详细模拟模式")
        result = quick_run(
            regime=args.regime, scenario=args.scenario,
            profile=args.profile, T_max=args.T_max
        )
        m = extract_financial_metrics(result)
        m["param_value"] = None
        results = [m]

    logger.phase_end("Phase 1", logger.get_elapsed())

    # ================================================================
    # Phase 2: 金融稳定性扫描分析
    # ================================================================
    logger.phase_start("Phase 2: 金融稳定性扫描", 2, 4)
    t0 = time.time()

    if args.sweep:
        param_name = args.sweep

        # 输出扫描结果表
        headers = [param_name, "emp_q", "credit", "bubble", "flight",
                   "cost_conc", "fractures", "pol_sup", "Kd_app/Ki"]
        rows = []
        for m in results:
            k_ratio = m["d_app_K"] / max(m["i_K"], 0.01)
            rows.append([
                f"{m['param_value']:.3f}",
                f"{m['employment_quality']:.3f}",
                f"{m['credit_rationing']:.3f}",
                f"{m['bubble_pressure']:.3f}",
                f"{m['capital_flight']:.3f}",
                f"{m['cost_concentration']:.3f}",
                str(m['fracture_count']),
                f"{m['political_support']:.3f}",
                f"{k_ratio:.2f}",
            ])

        logger.table(f"金融稳定性扫描 ({param_name})", headers, rows)

        # 资本外逃阈值分析
        if "capital_flight" in param_name.lower():
            high_flight = [m for m in results if m["param_value"] > 0.25]
            low_flight = [m for m in results if m["param_value"] < 0.1]
            if high_flight and low_flight:
                hq = np.mean([m["employment_quality"] for m in high_flight])
                lq = np.mean([m["employment_quality"] for m in low_flight])
                hb = np.mean([m["bubble_pressure"] for m in high_flight])
                lb = np.mean([m["bubble_pressure"] for m in low_flight])
                logger.finding(
                    f"高资本外逃敏感度下: 就业质量={hq:.3f} vs 低敏感度={lq:.3f} "
                    f"(差={abs(hq-lq):.3f}) | 泡沫压力={hb:.3f} vs {lb:.3f}",
                    level="KEY"
                )

            # 寻找临界阈值
            crashed = [m for m in results if not m["survived"]]
            if crashed:
                crash_threshold = min(m["param_value"] for m in crashed)
                logger.finding(
                    f"资本外逃崩溃阈值: {param_name}={crash_threshold:.3f}——"
                    "超过此值可能导致系统崩溃",
                    level="WARN"
                )

        # 信贷错配分析
        if "credit_bias" in param_name.lower():
            high_bias = [m for m in results if m["param_value"] > 1.5]
            low_bias = [m for m in results if m["param_value"] < 1.0]
            if high_bias and low_bias:
                hf = np.mean([m["fracture_count"] for m in high_bias])
                lf = np.mean([m["fracture_count"] for m in low_bias])
                hb = np.mean([m["bubble_pressure"] for m in high_bias])
                lb = np.mean([m["bubble_pressure"] for m in low_bias])
                logger.finding(
                    f"高信贷数字偏好下: 平均断裂数={hf:.1f} vs 低偏好={lf:.1f} "
                    f"| 泡沫压力={hb:.3f} vs {lb:.3f}",
                    level="KEY"
                )

                # 挤出效应
                h_ki = np.mean([m["d_app_K"] / max(m["i_K"], 0.01)
                             for m in high_bias])
                l_ki = np.mean([m["d_app_K"] / max(m["i_K"], 0.01)
                             for m in low_bias])
                if h_ki > l_ki * 2:
                    logger.finding(
                        f"信贷数字偏好导致工业挤出: d_app/i比值 "
                        f"高偏好={h_ki:.1f} vs 低偏好={l_ki:.1f}",
                        level="WARN"
                    )

        # 制度能力的缓冲效应
        if "institution" in param_name.lower():
            high_inst = [m for m in results if m["param_value"] > 0.6]
            low_inst = [m for m in results if m["param_value"] < 0.3]
            if high_inst and low_inst:
                h_surv = np.mean([m["survived"] for m in high_inst])
                l_surv = np.mean([m["survived"] for m in low_inst])
                logger.finding(
                    f"制度能力缓冲效应: 高制度存活率={h_surv:.1%} vs "
                    f"低制度={l_surv:.1%}——好制度可作为金融风险缓冲器",
                    level="KEY"
                )

        # 泡沫破裂风险检测
        burst_candidates = [m for m in results if m["bubble_pressure"] > 1.0]
        if burst_candidates:
            logger.finding(
                f"{len(burst_candidates)}/{len(results)} 组参数泡沫压力>1.0——"
                "存在破裂风险，需关注数字资产过热",
                level="WARN"
            )

        # 金融开放度分析
        if "financial_openness" in param_name.lower():
            high_open = [m for m in results if m["param_value"] > 0.6]
            low_open = [m for m in results if m["param_value"] < 0.3]
            if high_open and low_open:
                h_flight = np.mean([m["capital_flight"] for m in high_open])
                l_flight = np.mean([m["capital_flight"] for m in low_open])
                logger.finding(
                    f"金融开放度与资本外逃: 高开放外逃率={h_flight:.3f} vs "
                    f"低开放={l_flight:.3f}——高金融开放度增加资本外逃风险"
                )

    else:
        # 单次运行的金融健康检查
        m = results[0]
        logger.logger.info("\n  金融健康检查 (v3.5):")
        logger.stat("就业质量指数", m["employment_quality"], "[0,1]")
        logger.stat("信贷可得性", m["credit_rationing"])
        logger.stat("泡沫压力", m["bubble_pressure"])
        logger.stat("资本外逃率", m["capital_flight"])
        logger.stat("代价集中度", m["cost_concentration"])
        logger.stat("财富基尼", m["wealth_gini"])
        logger.stat("政治支持度", m["political_support"])
        logger.stat("断裂数", m["fracture_count"])
        if m["fracture_types"]:
            logger.logger.info(f"    断裂类型: {', '.join(m['fracture_types'])}")

        logger.logger.info("\n  部门资本分布:")
        k_total = m["d_infra_K"] + m["d_app_K"] + m["i_K"] + m["b_K"]
        if k_total > 0:
            logger.logger.info(f"    d_infra: {m['d_infra_K']:.0f} "
                             f"({100*m['d_infra_K']/k_total:.1f}%)")
            logger.logger.info(f"    d_app:   {m['d_app_K']:.0f} "
                             f"({100*m['d_app_K']/k_total:.1f}%)")
            logger.logger.info(f"    i:       {m['i_K']:.0f} "
                             f"({100*m['i_K']/k_total:.1f}%)")
            logger.logger.info(f"    b:       {m['b_K']:.0f} "
                             f"({100*m['b_K']/k_total:.1f}%)")

        logger.logger.info("\n  分层技术差距:")
        logger.stat("infra技术差距", m["tech_gap_infra"])
        logger.stat("app技术差距", m["tech_gap_app"])
        logger.stat("CDI", m["cdi"])

    logger.phase_end("Phase 2", time.time() - t0)

    # ================================================================
    # Phase 3: 就业质量与金融稳定联动分析
    # ================================================================
    logger.phase_start("Phase 3: 就业质量与金融联动", 3, 4)
    t0 = time.time()

    if args.sweep and len(results) >= 3:
        # 就业质量 vs 泡沫压力
        emp_bubble_corr = np.corrcoef(
            [m["employment_quality"] for m in results],
            [m["bubble_pressure"] for m in results]
        )[0, 1] if len(results) >= 3 else 0

        # 就业质量 vs 资本外逃
        emp_flight_corr = np.corrcoef(
            [m["employment_quality"] for m in results],
            [m["capital_flight"] for m in results]
        )[0, 1] if len(results) >= 3 else 0

        # 泡沫 vs 断裂
        bubble_fract_corr = np.corrcoef(
            [m["bubble_pressure"] for m in results],
            [m["fracture_count"] for m in results]
        )[0, 1] if len(results) >= 3 else 0

        logger.stat("就业质量-泡沫压力相关性", emp_bubble_corr)
        logger.stat("就业质量-资本外逃相关性", emp_flight_corr)
        logger.stat("泡沫压力-断裂数相关性", bubble_fract_corr)

        if abs(emp_bubble_corr) > 0.5:
            direction = "恶化" if emp_bubble_corr < 0 else "改善（反常）"
            logger.finding(
                f"泡沫压力与就业质量显著相关(r={emp_bubble_corr:.3f})——"
                f"泡沫上升导致就业质量{direction}",
                level="KEY" if emp_bubble_corr < -0.5 else "INFO"
            )

        if abs(bubble_fract_corr) > 0.5:
            logger.finding(
                f"泡沫压力与断裂数强相关(r={bubble_fract_corr:.3f})——"
                "数字泡沫破裂可能是系统断裂的重要诱因",
                level="WARN" if bubble_fract_corr > 0.5 else "INFO"
            )
    else:
        # 单次诊断
        m = results[0]
        logger.logger.info("\n  风险评估 (v3.5):")
        risks = []
        if m["bubble_pressure"] > 1.0:
            risks.append("数字资产泡沫压力偏高——需关注破裂风险")
            logger.logger.warning(f"    [高危] 数字资产泡沫压力偏高")
        if m["employment_quality"] < 0.6:
            risks.append("就业质量偏低——零工经济占比过高")
            logger.logger.warning(f"    [警告] 就业质量偏低({m['employment_quality']:.3f})")
        if m["cost_concentration"] > 0.5:
            risks.append("代价高度集中——社会撕裂风险")
            logger.logger.warning(f"    [警告] 代价高度集中({m['cost_concentration']:.3f})")
        if m["political_support"] < 0.4:
            risks.append("政治支持度危险——政策空间严重受限")
            logger.logger.warning(f"    [高危] 政治支持度危险({m['political_support']:.3f})")
        if m["fracture_count"] >= 2:
            risks.append(f"多断裂并发({m['fracture_count']}种)——系统性风险")
            logger.logger.warning(f"    [高危] 多断裂并发({m['fracture_count']}种)")
        if not risks:
            logger.logger.info("    当前参数下金融稳定性良好")

        if risks:
            for r_text in risks:
                logger.finding(r_text, level="WARN")

    logger.phase_end("Phase 3", time.time() - t0)

    # ================================================================
    # Phase 4: 保存结果
    # ================================================================
    logger.phase_start("Phase 4: 保存结果", 4, 4)
    t0 = time.time()

    store.set_parameters({
        "financial_metrics": results if args.sweep else [results[0]],
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
