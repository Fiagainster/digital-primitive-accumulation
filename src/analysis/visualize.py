"""可视化分析 —— 时间序列、相图、代价分布、策略比较 (v3.0 升级)."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# v3.0: 导入评价系统的雷达图生成器
from ..evaluation.radar_chart import RadarChartGenerator

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


class Visualizer:
    """可视化工具集."""

    def __init__(self, output_dir: str = "experiments/results/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_single_simulation(self, result: dict, title: str = None,
                               save: bool = True, filename: str = None):
        """绘制单次模拟的完整面板."""
        df = result["dataframe"]
        summary = result["summary"]
        t = df["t"].values

        fig, axes = plt.subplots(3, 3, figsize=(16, 12))

        # [0,0] GDP与部门产出
        ax = axes[0, 0]
        ax.plot(t, df["Y_b"], label="基础部门", linewidth=1.2)
        ax.plot(t, df["Y_i"], label="工业部门", linewidth=1.5)
        ax.plot(t, df["Y_d"], label="数字部门", linewidth=1.5)
        ax.plot(t, df["gdp"], "k--", label="GDP", linewidth=1.0, alpha=0.6)
        ax.set_title("部门产出")
        ax.set_ylabel("亿元/年")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # [0,1] 资本存量
        ax = axes[0, 1]
        ax.plot(t, df["K_b"], label="K_b", linewidth=1.2)
        ax.plot(t, df["K_i"], label="K_i", linewidth=1.5)
        ax.plot(t, df["K_d"], label="K_d", linewidth=1.5)
        ax.set_title("资本存量")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [0,2] 数字化水平
        ax = axes[0, 2]
        ax.plot(t, df["DL_b"], label="DL_b", linewidth=1.2)
        ax.plot(t, df["DL_i"], label="DL_i", linewidth=1.5)
        ax.plot(t, df["DL_d"], label="DL_d", linewidth=1.5)
        ax.set_title("数字化水平")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [1,0] 技术差距与竞争力
        ax = axes[1, 0]
        ax.plot(t, df["tech_gap"], "r-", label="技术差距G", linewidth=1.5)
        ax.set_title("技术差距")
        ax.set_ylabel("A_world / A_d")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [1,1] 福利与不平等
        ax = axes[1, 1]
        ax.plot(t, df["welfare_index"], "g-", label="福利指数", linewidth=1.5)
        ax2 = ax.twinx()
        ax2.plot(t, df["gini"], "orange", linestyle="--", label="基尼系数", linewidth=1.0)
        ax.set_title("福利与不平等")
        ax.legend(fontsize=8, loc="upper left")
        ax2.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

        # [1,2] 失业率
        ax = axes[1, 2]
        ax.fill_between(t, 0, df["unemployment_rate"], alpha=0.3, color="red")
        ax.plot(t, df["unemployment_rate"], "r-", linewidth=1.5)
        ax.set_title("失业率")
        ax.set_ylabel("比例")
        ax.grid(True, alpha=0.3)

        # [2,0] EWS信号
        ax = axes[2, 0]
        if "ews_K_i_composite" in df.columns:
            ax.plot(t, df["ews_K_i_composite"], label="EWS-K_i", linewidth=1.2)
        if "ews_tech_gap_composite" in df.columns:
            ax.plot(t, df["ews_tech_gap_composite"], label="EWS-G", linewidth=1.2)
        ax.axhline(0.4, color="orange", linestyle="--", alpha=0.5, label="Warning")
        ax.axhline(0.7, color="red", linestyle="--", alpha=0.5, label="Critical")
        ax.set_title("早期预警信号")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [2,1] 代价分布（按群组）
        ax = axes[2, 1]
        groups = ["数字高技能", "工业劳动者", "基础劳动者", "服务业", "失业者"]
        costs = [
            df["hh_digital_cost"].iloc[-1],
            df["hh_industrial_cost"].iloc[-1],
            df["hh_base_cost"].iloc[-1],
            df["hh_service_cost"].iloc[-1] if "hh_service_cost" in df.columns else 0,
            df["hh_unemployed_cost"].iloc[-1],
        ]
        colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9E9E9E"]
        bars = ax.bar(groups, costs, color=colors, alpha=0.8)
        ax.set_title("累计代价（按群组）")
        ax.set_ylabel("累计代价")
        ax.tick_params(axis="x", rotation=15)
        # 添加数值标签
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(costs)*0.02,
                    f"{cost:.0f}", ha="center", fontsize=8)

        # [2,2] 数据资产
        ax = axes[2, 2]
        ax.plot(t, df["data_producer"], label="生产者数据", linewidth=1.2)
        ax.plot(t, df["data_consumer"], label="消费者数据", linewidth=1.2)
        ax.plot(t, df["data_public"], label="公共数据", linewidth=1.2)
        ax.set_title("数据资产")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        suptitle = title or f"策略{summary.get('regime','?')} — "
        if summary.get("survived"):
            suptitle += f"存活 {summary['survival_years']:.0f}年"
        else:
            suptitle += f"崩溃于{summary['survival_years']:.0f}年 [{summary.get('terminal_fracture','?')}]"
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            fname = filename or f"simulation_{summary.get('regime','X')}.png"
            filepath = os.path.join(self.output_dir, fname)
            fig.savefig(filepath)
            plt.close(fig)
            print(f"  Figure saved: {filepath}")
        else:
            plt.show()

    def plot_strategy_comparison(self, results: list, save: bool = True):
        """绘制策略比较面板."""
        regimes = ["A", "B", "C"]
        regime_results = {r: [res for res in results
                              if res["summary"].get("regime") == r]
                          for r in regimes}

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        metrics = [
            ("survived", "存活率", lambda rs: np.mean([r["summary"]["survived"] for r in rs])),
            ("survival_years", "平均存活年数", lambda rs: np.mean([r["summary"]["survival_years"] for r in rs])),
            ("final_tech_gap", "最终技术差距", lambda rs: np.mean([r["summary"]["final_tech_gap"] for r in rs])),
            ("cost_concentration", "代价基尼系数", lambda rs: np.mean([r["summary"]["cost_concentration"] for r in rs])),
            ("cost_ratio", "工业/数字代价比", lambda rs: np.mean([r["summary"]["cost_ratio_industrial_digital"] for r in rs])),
            ("final_welfare", "最终福利指数", lambda rs: np.mean([r["summary"]["final_welfare"] for r in rs])),
        ]

        for idx, (metric, title, fn) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            values = [fn(regime_results[r]) for r in regimes]
            colors = ["#2196F3", "#FF5722", "#4CAF50"]
            bars = ax.bar(["渐进A", "跳跃B", "替代C"], values, color=colors, alpha=0.8)
            ax.set_title(title)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                        f"{val:.3f}", ha="center", fontsize=9)

        fig.suptitle("策略比较", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, "strategy_comparison.png")
            fig.savefig(filepath)
            plt.close(fig)
            print(f"  Figure saved: {filepath}")

    def plot_phase_diagram(self, results: list, x_metric: str = "final_tech_gap",
                           y_metric: str = "survival_years",
                           color_by: str = "regime", save: bool = True):
        """绘制参数空间相图."""
        fig, ax = plt.subplots(figsize=(10, 7))

        regime_colors = {"A": "#2196F3", "B": "#FF5722", "C": "#4CAF50"}
        regime_markers = {"A": "o", "B": "s", "C": "^"}

        for regime in ["A", "B", "C"]:
            subset = [r for r in results if r["summary"].get("regime") == regime]
            if not subset:
                continue
            x = [r["summary"].get(x_metric, 0) for r in subset]
            y = [r["summary"].get(y_metric, 0) for r in subset]
            survived = [r["summary"]["survived"] for r in subset]

            # 存活/崩溃用不同透明度
            x_surv = [x[i] for i, s in enumerate(survived) if s]
            y_surv = [y[i] for i, s in enumerate(survived) if s]
            x_dead = [x[i] for i, s in enumerate(survived) if not s]
            y_dead = [y[i] for i, s in enumerate(survived) if not s]

            ax.scatter(x_surv, y_surv, c=regime_colors[regime], marker=regime_markers[regime],
                      label=f"策略{regime}-存活", alpha=0.7, s=30)
            ax.scatter(x_dead, y_dead, c=regime_colors[regime], marker="x",
                      label=f"策略{regime}-崩溃", alpha=0.5, s=20)

        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(f"参数空间相图: {y_metric} vs {x_metric}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        if save:
            filepath = os.path.join(self.output_dir, f"phase_{y_metric}_vs_{x_metric}.png")
            fig.savefig(filepath)
            plt.close(fig)
            print(f"  Figure saved: {filepath}")

    def plot_five_dimension_radar(self, results: list, save: bool = True):
        """v3.0: 绘制五维策略雷达图 (并排比较).

        从每个结果中提取五维评价得分，生成雷达图。
        """
        radar = RadarChartGenerator(self.output_dir)

        # 按策略聚合评价结果
        regime_evals = {}
        for r in results:
            summary = r["summary"]
            regime = summary.get("regime", "?")
            ev = summary.get("evaluation", {})
            if not ev.get("enabled", False):
                continue
            if regime not in regime_evals:
                regime_evals[regime] = {
                    "economy": [], "livelihood": [], "resilience": [],
                    "structure": [], "sovereignty": [], "cdi": [],
                    "acceptable": 0, "total": 0,
                }
            acc = regime_evals[regime]
            acc["economy"].append(ev.get("economy", 0))
            acc["livelihood"].append(ev.get("livelihood", 0))
            acc["resilience"].append(ev.get("resilience", 0))
            acc["structure"].append(ev.get("structure", 0))
            acc["sovereignty"].append(ev.get("sovereignty", 0))
            acc["cdi"].append(ev.get("cdi", 0))
            acc["total"] += 1
            if ev.get("is_acceptable", True):
                acc["acceptable"] += 1

        # 取均值
        scores_dict = {}
        for regime, acc in regime_evals.items():
            if acc["total"] == 0:
                continue
            label = f"{regime} (n={acc['total']}, 可接受={acc['acceptable']}/{acc['total']})"
            scores_dict[label] = {
                "economy": np.mean(acc["economy"]),
                "livelihood": np.mean(acc["livelihood"]),
                "resilience": np.mean(acc["resilience"]),
                "structure": np.mean(acc["structure"]),
                "sovereignty": np.mean(acc["sovereignty"]),
            }

        if scores_dict:
            radar.plot_strategy_radar(
                scores_dict,
                title="积累策略五维评价画像",
                filename="five_dimension_radar.png"
            )

            # 同时生成对比矩阵文本
            matrix_text = radar.generate_comparison_matrix([
                {
                    "label": k,
                    "scores": v,
                    "cdi": {"cdi_variants": {"default": np.mean(regime_evals[k.split()[0]]["cdi"])}},
                }
                for k, v in scores_dict.items()
            ])
            matrix_path = os.path.join(self.output_dir, "comparison_matrix.txt")
            with open(matrix_path, "w", encoding="utf-8") as f:
                f.write(matrix_text)
            print(f"  Comparison matrix saved: {matrix_path}")

    def plot_cost_waterfall(self, results: list, save: bool = True):
        """v3.0: 绘制代价瀑布图 —— 代价在各家庭群组间的传递路径."""
        # 按策略聚合各群组的代价
        regime_costs = {"A": {}, "B": {}, "C": {}}
        for r in results:
            summary = r["summary"]
            regime = summary.get("regime", "?")
            if regime not in regime_costs:
                continue
            # 使用已有指标
            cost_ratio = summary.get("cost_ratio_industrial_digital", 1.0)
            cost_conc = summary.get("cost_concentration", 0.0)
            deind_type = summary.get("deind_type_final", "unknown")

            if "cost_ratio" not in regime_costs[regime]:
                regime_costs[regime] = {
                    "cost_ratios": [], "cost_concs": [],
                    "deind_types": {},
                    "n": 0,
                }
            rc = regime_costs[regime]
            rc["cost_ratios"].append(cost_ratio)
            rc["cost_concs"].append(cost_conc)
            rc["deind_types"][deind_type] = rc["deind_types"].get(deind_type, 0) + 1
            rc["n"] += 1

        # 绘制瀑布图：工业/数字代价比 + 去工业化类型分布
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        regimes = ["A", "B", "C"]
        colors = ["#2196F3", "#FF5722", "#4CAF50"]
        labels = ["渐进A", "跳跃B", "替代C"]

        for idx, (regime, color, label) in enumerate(zip(regimes, colors, labels)):
            ax = axes[idx]
            rc = regime_costs.get(regime, {"cost_ratios": [1.0], "cost_concs": [0.0],
                                            "deind_types": {}, "n": 0})
            if rc["n"] == 0:
                ax.text(0.5, 0.5, "无数据", transform=ax.transAxes, ha="center")
                ax.set_title(label)
                continue

            # 左侧：代价比分布（箱线图简化）
            ratios = rc["cost_ratios"]
            parts = ax.violinplot(ratios, positions=[0.5], showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            ax.set_ylabel("工业/数字 代价比")
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="均等线")
            ax.axhline(2.5, color="orange", linestyle="--", alpha=0.5, label="撕裂线(2.5)")
            ax.set_xticks([])
            ax.legend(fontsize=7)
            ax.set_ylim(0, max(max(ratios) * 1.1, 5.0))

            # 标注均值
            mean_r = np.mean(ratios)
            ax.annotate(f"均值={mean_r:.1f}",
                       xy=(0.6, mean_r), fontsize=8, color="red", fontweight="bold")

            # 标题含去工业化类型分布
            deind_str = ", ".join(f"{k}:{v}" for k, v in rc["deind_types"].items())
            ax.set_title(f"{label} (n={rc['n']})\n去工业化: {deind_str}", fontsize=10)

        fig.suptitle("代价分布与去工业化模式（按策略）", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, "cost_waterfall.png")
            fig.savefig(filepath)
            plt.close(fig)
            print(f"  Figure saved: {filepath}")


def generate_academic_report(results: list, output_dir: str = "experiments/results"):
    """生成学术分析报告."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "academic_report.txt")

    summaries = [r["summary"] for r in results]
    n = len(summaries)

    lines = []
    lines.append("=" * 70)
    lines.append("原始积累模型 —— 模拟实验学术分析报告")
    lines.append("=" * 70)
    lines.append(f"\n总模拟次数: {n}")
    lines.append(f"生成时间: {pd.Timestamp.now()}")

    # 总体统计
    survived = [s["survived"] for s in summaries]
    lines.append(f"\n--- 总体存活率 ---")
    lines.append(f"总体存活率: {np.mean(survived)*100:.1f}%")

    # 分策略
    lines.append(f"\n--- 分策略统计 ---")
    for regime in ["A", "B", "C"]:
        rs = [s for s in summaries if s.get("regime") == regime]
        if not rs:
            continue
        surv_rate = np.mean([s["survived"] for s in rs])
        avg_years = np.mean([s["survival_years"] for s in rs])
        avg_gap = np.mean([s["final_tech_gap"] for s in rs])
        avg_cost_conc = np.mean([s["cost_concentration"] for s in rs])
        avg_cost_ratio = np.mean([s["cost_ratio_industrial_digital"] for s in rs])
        lines.append(f"\n策略{regime} ({len(rs)}次模拟):")
        lines.append(f"  存活率: {surv_rate*100:.1f}%")
        lines.append(f"  平均存活年数: {avg_years:.1f}")
        lines.append(f"  平均最终技术差距: {avg_gap:.2f}")
        lines.append(f"  平均代价集中度: {avg_cost_conc:.3f}")
        lines.append(f"  平均工业/数字代价比: {avg_cost_ratio:.1f}")

    # 断裂模式
    lines.append(f"\n--- 断裂模式频率 ---")
    fracture_counts = {}
    for s in summaries:
        for ft in s.get("fracture_types", []):
            fracture_counts[ft] = fracture_counts.get(ft, 0) + 1
    for ft, count in sorted(fracture_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {ft}: {count}次 ({100*count/n:.1f}%)")

    # 分场景
    lines.append(f"\n--- 分场景统计 ---")
    scenarios = set(s.get("scenario", "unknown") for s in summaries)
    for sc in sorted(scenarios):
        rs = [s for s in summaries if s.get("scenario") == sc]
        if not rs:
            continue
        surv_rate = np.mean([s["survived"] for s in rs])
        lines.append(f"  {sc}: 存活率 {surv_rate*100:.1f}% ({len(rs)}次)")

    # EWS效能
    lines.append(f"\n--- EWS预警效能 ---")
    avg_ews_K = np.mean([s.get("ews_max_K_i", 0) for s in summaries])
    avg_ews_G = np.mean([s.get("ews_max_tech_gap", 0) for s in summaries])
    lines.append(f"  平均最大EWS(K_i): {avg_ews_K:.3f}")
    lines.append(f"  平均最大EWS(tech_gap): {avg_ews_G:.3f}")

    # v2.1 知识扩散
    lines.append(f"\n--- 知识扩散与部门差距 (v2.1) ---")
    for regime in ["A", "B", "C"]:
        rs = [s for s in summaries if s.get("regime") == regime]
        if not rs:
            continue
        avg_diff_i = np.mean([s.get("diffusion_cumulative_i", 0) for s in rs])
        avg_diff_b = np.mean([s.get("diffusion_cumulative_b", 0) for s in rs])
        avg_gap_i = np.mean([s.get("sector_gap_i", 0) for s in rs])
        avg_gap_b = np.mean([s.get("sector_gap_b", 0) for s in rs])
        lines.append(f"  策略{regime}: D→I扩散={avg_diff_i:.2f} D→B扩散={avg_diff_b:.2f}")
        lines.append(f"         部门内差距 I={avg_gap_i:.3f} B={avg_gap_b:.3f}")

    # v2.1 断裂级联
    lines.append(f"\n--- 断裂级联统计 (v2.1) ---")
    multi_collapse = sum(1 for s in summaries if s.get("total_fractures", 0) >= 3)
    lines.append(f"  多断裂并发(≥3种): {multi_collapse}/{n} ({100*multi_collapse/n:.1f}%)")

    # v3.0: 五维评价统计
    lines.append(f"\n--- v3.0 五维评价 ---")
    eval_enabled = any(s.get("evaluation", {}).get("enabled", False) for s in summaries)
    if eval_enabled:
        lines.append(f"\n{'维度':<20} {'A-渐进':>8} {'B-跳跃':>8} {'C-替代':>8}")
        lines.append("-" * 45)
        dims = [
            ("经济活力", "economy"),
            ("民生福祉", "livelihood"),
            ("发展韧性", "resilience"),
            ("结构健康", "structure"),
            ("主权自主", "sovereignty"),
            ("综合CDI", "cdi"),
        ]
        for dim_name, dim_key in dims:
            row = f"{dim_name:<20}"
            for regime in ["A", "B", "C"]:
                vals = [s["evaluation"].get(dim_key, 0)
                        for s in summaries
                        if s.get("regime") == regime
                        and s.get("evaluation", {}).get("enabled", False)]
                avg_val = np.mean(vals) if vals else 0.0
                row += f" {avg_val:>8.3f}"
            lines.append(row)

        # 红线统计
        lines.append(f"\n--- 人道主义红线 (v3.0) ---")
        for regime in ["A", "B", "C"]:
            evals = [s.get("evaluation", {})
                     for s in summaries
                     if s.get("regime") == regime
                     and s.get("evaluation", {}).get("enabled", False)]
            if not evals:
                continue
            n_accept = sum(1 for e in evals if e.get("is_acceptable", True))
            n_total = len(evals)
            failures = {}
            warnings = {}
            for e in evals:
                for f in e.get("red_line_failures", []):
                    failures[f] = failures.get(f, 0) + 1
                for w in e.get("red_line_warnings", []):
                    warnings[w] = warnings.get(w, 0) + 1
            lines.append(f"  策略{regime}: 可接受率 {n_accept}/{n_total} "
                         f"({100*n_accept/max(n_total,1):.0f}%)")
            if failures:
                for f, c in failures.items():
                    lines.append(f"    ✗ {f}: {c}次")
            if warnings:
                for w, c in warnings.items():
                    lines.append(f"    ⚡ {w}: {c}次")

        # CDI不同权重下策略排序
        lines.append(f"\n--- CDI不同权重下的策略排序 ---")
        presets = ["default", "growth_first", "livelihood_first",
                   "security_first", "balanced"]
        preset_labels = ["民生优先(默认)", "增长优先", "民生极致",
                         "安全优先", "均衡"]
        for preset, plabel in zip(presets, preset_labels):
            regime_cdis = {}
            for regime in ["A", "B", "C"]:
                vals = []
                for s in summaries:
                    if (s.get("regime") == regime
                            and s.get("evaluation", {}).get("enabled", False)):
                        variants = s["evaluation"].get("cdi_variants", {})
                        vals.append(variants.get(preset, 0))
                regime_cdis[regime] = np.mean(vals) if vals else 0.0
            ranking = sorted(regime_cdis.items(), key=lambda x: -x[1])
            ranking_str = " > ".join(f"{r}:{v:.3f}" for r, v in ranking)
            lines.append(f"  {plabel}: {ranking_str}")

    # v3.2: 学术验证统计
    lines.append(f"\n--- v3.2 学术验证 ---")

    # SFC可信度
    sfc_trust = [s.get("validation", {}).get("sfc_trustworthiness", "N/A")
                 for s in summaries if s.get("validation", {}).get("sfc_trustworthiness")]
    if sfc_trust:
        high = sum(1 for t in sfc_trust if t == "HIGH")
        med = sum(1 for t in sfc_trust if t == "MEDIUM")
        low = sum(1 for t in sfc_trust if t == "LOW")
        lines.append(f"  SFC可信度: HIGH={high} MEDIUM={med} LOW={low}")

    # SFC残差统计
    residuals = [s.get("validation", {}).get("sfc_max_residual", 0)
                 for s in summaries if s.get("validation", {}).get("sfc_max_residual", 0) > 0]
    if residuals:
        lines.append(f"  SFC最大残差: 中位数={np.median(residuals):.4f} "
                     f"95分位={np.percentile(residuals, 95):.4f}")

    # 劳动力技能侵蚀
    erosions = [s.get("validation", {}).get("skill_erosion_industrial", 1.0)
                for s in summaries if s.get("validation", {})]
    if erosions:
        min_erosion = min(erosions)
        lines.append(f"  工业技能侵蚀: 最严重={min_erosion:.3f} "
                     f"(<1.0=技能退化, >1.0=技能提升)")

    # 通用异常检测 vs 预定义断裂
    uncovered = [s.get("validation", {}).get("uncovered_anomalies", 0)
                 for s in summaries if s.get("validation", {})]
    if uncovered:
        total_uncovered = sum(uncovered)
        lines.append(f"  未覆盖异常: {total_uncovered}个 "
                     f"({'存在预定义断裂之外的异常模式' if total_uncovered > 0 else '所有异常均被预定义断裂覆盖'})")

    # 意外发现
    all_findings = []
    for s in summaries:
        findings = s.get("validation", {}).get("unexpected_findings", [])
        all_findings.extend(findings)
    if all_findings:
        lines.append(f"  意外发现: {len(all_findings)}条（详见各模拟报告）")

    # v3.2: 五维相关性矩阵
    lines.append(f"\n--- v3.2 五维评价相关性矩阵 ---")
    from ..monitors.academic_validator import AcademicValidator
    corr_matrix = AcademicValidator.compute_dimension_correlations(results)
    if corr_matrix:
        dims = ["economy", "livelihood", "resilience", "structure", "sovereignty"]
        dim_labels = {"economy": "经济", "livelihood": "民生", "resilience": "韧性",
                      "structure": "结构", "sovereignty": "主权"}
        header = f"{'':<8}"
        for d in dims:
            header += f" {dim_labels[d]:>6}"
        lines.append(header)
        for d1 in dims:
            row = f"{dim_labels[d1]:<8}"
            for d2 in dims:
                r = corr_matrix.get(d1, {}).get(d2, 0)
                row += f" {r:>6.2f}"
            lines.append(row)
        # 解读
        interpretations = AcademicValidator.interpret_correlations(corr_matrix)
        if interpretations:
            lines.append(f"\n  相关性解读:")
            for interp in interpretations[:5]:
                lines.append(f"  · {interp}")

    # v3.2: 断裂触发矩阵
    lines.append(f"\n--- v3.2 断裂触发矩阵 ---")
    from ..monitors.academic_validator import (
        build_fracture_trigger_matrix, identify_dormant_fractures
    )
    fracture_matrix = build_fracture_trigger_matrix(results)
    if fracture_matrix:
        for ft_name, regime_counts in sorted(fracture_matrix.items()):
            total = sum(regime_counts.values())
            regime_str = ", ".join(f"{r}:{c}" for r, c in sorted(regime_counts.items()))
            lines.append(f"  {ft_name}: {total}次 [{regime_str}]")
        # 休眠断裂
        dormant = identify_dormant_fractures(fracture_matrix, n)
        if dormant:
            lines.append(f"\n  休眠断裂（触发率<1%）:")
            for d in dormant:
                lines.append(f"  · {d}")

    lines.append(f"\n{'=' * 70}")
    lines.append("报告结束")
    lines.append("=" * 70)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Academic report saved to {filepath}")
    return "\n".join(lines)
