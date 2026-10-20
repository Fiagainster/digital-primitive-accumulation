"""雷达图生成器 —— 五维策略画像可视化."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
from typing import Dict, List, Optional


class RadarChartGenerator:
    """策略五维雷达图生成器.

    为每种策略（或混合配比）生成五维雷达图。
    支持多策略并排比较、时间序列雷达图和对比矩阵。
    """

    # 五维标签（中文）
    DIM_LABELS = ["经济活力", "民生福祉", "发展韧性", "结构健康", "主权自主"]
    DIM_KEYS = ["economy", "livelihood", "resilience", "structure", "sovereignty"]

    # 配色方案
    DEFAULT_COLORS = {
        "A": "#2196F3",      # 蓝 — 渐进
        "B": "#FF5722",      # 橙 — 跳跃
        "C": "#4CAF50",      # 绿 — 替代
        "mixed": "#9C27B0",  # 紫 — 混合
    }

    def __init__(self, output_dir: str = "experiments/results/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 中文字体设置
        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        })

    def plot_strategy_radar(self, scores_dict: Dict[str, dict],
                            title: str = "策略五维画像",
                            save: bool = True,
                            filename: str = None) -> str:
        """绘制多策略雷达图（并排比较）.

        Args:
            scores_dict: {策略名: {"economy": 0.6, "livelihood": 0.7, ...}}
            title: 图表标题
            save: 是否保存
            filename: 文件名

        Returns:
            str: 保存的文件路径
        """
        n_strategies = len(scores_dict)
        if n_strategies == 0:
            return ""

        n_dims = len(self.DIM_KEYS)
        angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        # 子图布局
        n_cols = min(n_strategies, 4)
        n_rows = (n_strategies + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                  subplot_kw=dict(polar=True),
                                  figsize=(5 * n_cols, 5 * n_rows))
        if n_strategies == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).flatten()

        for idx, (label, scores) in enumerate(scores_dict.items()):
            ax = axes[idx]
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # 提取值
            values = [scores.get(k, 0.0) for k in self.DIM_KEYS]
            values += values[:1]

            # 填充
            color = self.DEFAULT_COLORS.get(label,
                     self.DEFAULT_COLORS.get(label[0], "#607D8B"))
            ax.fill(angles, values, color=color, alpha=0.25)
            ax.plot(angles, values, color=color, linewidth=2, label=label)

            # 添加数值标注
            for angle, val, dim_key in zip(angles[:-1], values[:-1], self.DIM_KEYS):
                ax.annotate(f"{val:.2f}",
                           xy=(angle, val),
                           xytext=(5, 5),
                           textcoords="offset points",
                           fontsize=8,
                           color=color)

            # 网格
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(self.DIM_LABELS, fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
            ax.set_title(label, fontsize=12, fontweight="bold", pad=20)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(n_strategies, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save:
            fname = filename or "strategy_radar.png"
            filepath = os.path.join(self.output_dir, fname)
            fig.savefig(filepath)
            plt.close(fig)
            return filepath
        else:
            plt.show()
            return ""

    def plot_single_radar(self, regime: str, scores: dict,
                          cdi: float = None,
                          red_line_status: str = "PASS",
                          save: bool = True) -> str:
        """绘制单个策略的雷达图（带CDI和红线标注）.

        Args:
            regime: 策略名称
            scores: 五维得分字典
            cdi: CDI值
            red_line_status: 红线状态
            save: 是否保存

        Returns:
            str: 保存路径
        """
        n_dims = len(self.DIM_KEYS)
        angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        values = [scores.get(k, 0.0) for k in self.DIM_KEYS]
        values += values[:1]

        color = self.DEFAULT_COLORS.get(regime,
                 self.DEFAULT_COLORS.get(regime[0], "#607D8B"))

        # 主填充
        ax.fill(angles, values, color=color, alpha=0.3)
        ax.plot(angles, values, color=color, linewidth=2.5)

        # 参考圆（0.5）
        ref = [0.5] * (n_dims + 1)
        ax.plot(angles, ref, "k--", linewidth=0.8, alpha=0.4, label="参考线(0.5)")

        # 数值标注
        for angle, val, dim_label in zip(angles[:-1], values[:-1], self.DIM_LABELS):
            ax.annotate(f"{val:.2f}",
                       xy=(angle, val),
                       xytext=(8, 8),
                       textcoords="offset points",
                       fontsize=9,
                       fontweight="bold",
                       color=color,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                 alpha=0.7))

        # 标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.DIM_LABELS, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
        ax.grid(True, alpha=0.3)

        # CDI和红线状态
        status_text = f"策略{regime}"
        if cdi is not None:
            status_text += f"  |  CDI={cdi:.3f}"
        if red_line_status == "FAILURE":
            status_text += "  |  ⚠ 人道主义失败"
        elif red_line_status == "WARNING":
            status_text += "  |  ⚡ 人道主义警告"
        else:
            status_text += "  |  ✓ 红线通过"

        ax.set_title(status_text, fontsize=13, fontweight="bold", pad=25,
                     color="red" if red_line_status == "FAILURE" else "black")

        if save:
            filepath = os.path.join(self.output_dir,
                                    f"radar_{regime}.png")
            fig.savefig(filepath)
            plt.close(fig)
            return filepath
        else:
            plt.show()
            return ""

    def generate_comparison_matrix(self, results: List[dict],
                                   include_red_lines: bool = True) -> str:
        """生成策略对比矩阵的文本表格.

        Args:
            results: 每条记录包含 {"label": str, "scores": dict, "cdi": dict, "red_line": RedLineResult}
            include_red_lines: 是否包含红线信息

        Returns:
            str: 格式化的对比矩阵文本
        """
        if not results:
            return ""

        labels = [r.get("label", f"策略{i}") for i, r in enumerate(results)]

        lines = []
        lines.append("策略对比矩阵")
        lines.append("=" * 75)

        # 表头
        header = f"{'维度':<18}"
        for label in labels:
            header += f" {label:>12}"
        lines.append(header)
        lines.append("-" * 75)

        # 五维得分
        dim_labels = [
            ("经济活力(Eco)", "economy"),
            ("民生福祉(Liv)", "livelihood"),
            ("发展韧性(Res)", "resilience"),
            ("结构健康(Str)", "structure"),
            ("主权自主(Sov)", "sovereignty"),
        ]

        for dim_name, dim_key in dim_labels:
            row = f"{dim_name:<18}"
            for r in results:
                scores = r.get("scores", {})
                val = scores.get(dim_key, 0.0)
                row += f" {val:>12.2f}"
            lines.append(row)

        lines.append("-" * 75)

        # CDI行（各种预设）
        cdi_presets = ["default", "growth_first", "livelihood_first",
                       "security_first", "balanced"]
        cdi_labels = ["CDI(民生优先)", "CDI(增长优先)", "CDI(民生极致)",
                      "CDI(安全优先)", "CDI(均衡)"]

        for cdi_label, preset in zip(cdi_labels, cdi_presets):
            row = f"{cdi_label:<18}"
            for r in results:
                cdi_dict = r.get("cdi", {})
                variants = cdi_dict.get("cdi_variants", {}) if isinstance(cdi_dict, dict) else {}
                val = variants.get(preset, cdi_dict.get("cdi", 0.0) if isinstance(cdi_dict, dict) else 0.0)
                row += f" {val:>12.3f}"
            lines.append(row)

        if include_red_lines:
            lines.append("-" * 75)
            row = f"{'红线状态':<18}"
            for r in results:
                rl = r.get("red_line", None)
                if rl:
                    if hasattr(rl, 'level'):
                        status = rl.level
                    else:
                        status = rl.get("level", "PASS")
                else:
                    status = "N/A"
                row += f" {status:>12}"
            lines.append(row)

        lines.append("=" * 75)
        return "\n".join(lines)

    def plot_cdi_heatmap_by_time(self,
                                  time_series: Dict[str, List[Dict[str, float]]],
                                  save: bool = True,
                                  filename: str = None) -> str:
        """绘制时间维度的CDI热力图.

        Args:
            time_series: {策略名: [{t: ..., economy: ..., livelihood: ..., ...}, ...]}
            save: 是否保存
            filename: 文件名

        Returns:
            str: 保存路径
        """
        if not time_series:
            return ""

        n_strategies = len(time_series)
        strategy_labels = list(time_series.keys())

        fig, axes = plt.subplots(n_strategies, 1, figsize=(14, 3 * n_strategies))
        if n_strategies == 1:
            axes = [axes]

        for idx, (label, ts_list) in enumerate(time_series.items()):
            ax = axes[idx]
            if not ts_list:
                continue

            # 提取数据
            times = [d.get("t", i) for i, d in enumerate(ts_list)]
            data = {}
            for dim in self.DIM_KEYS:
                data[dim] = [d.get(dim, 0.0) for d in ts_list]

            # 堆叠热力图（每个维度一行）
            n_dims = len(self.DIM_KEYS)
            img_data = np.zeros((n_dims, len(times)))
            for j, dim in enumerate(self.DIM_KEYS):
                img_data[j, :] = np.array(data[dim])

            im = ax.imshow(img_data, aspect="auto", cmap="RdYlGn",
                          vmin=0.0, vmax=1.0, interpolation="bilinear")

            # 标签
            ax.set_yticks(range(n_dims))
            ax.set_yticklabels(self.DIM_LABELS, fontsize=9)
            ax.set_title(f"策略{label}", fontsize=12, fontweight="bold")
            ax.set_xlabel("时间步")

            # 颜色条
            plt.colorbar(im, ax=ax, label="得分", shrink=0.8)

        fig.suptitle("五维得分时间热力图", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            fname = filename or "cdi_heatmap.png"
            filepath = os.path.join(self.output_dir, fname)
            fig.savefig(filepath)
            plt.close(fig)
            return filepath
        else:
            plt.show()
            return ""
