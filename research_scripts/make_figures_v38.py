"""论文图件生成 (v3.8) —— 全部英文标注、constrained layout 防遮挡、PDF+PNG 双导出。

图件清单
  fig_survival_heatmap     存活率热力图（6 国情 × 3 策略，8 场景汇总）
  fig_fracture_fingerprint 断裂类型指纹（按策略分组条形）
  fig_b_cost_distribution  B 策略五群组累计代价分布（4 国情 × 30 次重复箱线）
  fig_calib_fact1_curves   校准事实 1：制造业就业占比—人均产出轨迹
  fig_calib_fact2_inst     校准事实 2：初始制度能力与期末 GDP
  fig_calib_fact3_deind    校准事实 3：去工业化幅度与后续增长

运行::

    python research_scripts/make_figures_v38.py --out D:/Project/latex论文协作/output
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RUN_ROOT = _PROJECT_ROOT / "experiments_runs" / "v38"

# --------------------------------------------------------------------------
# 统一风格：衬线字体、与 Springer sn-jnl 接近；尺寸按单栏/双栏
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "legend.frameon": False,
    "figure.dpi": 160,
    "savefig.dpi": 320,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

STRATEGY_COLORS = {"A": "#B4436C", "B": "#D98E04", "C": "#2E6F8E"}
GROUP_COLORS = ["#8C8C8C", "#2E6F8E", "#4E9A51", "#D98E04", "#B4436C"]

SHORT_PROFILE = {
    "manufacturing_giant": "Manufacturing\ngiant",
    "resource_rich": "Resource-rich\neconomy",
    "digital_leapfrogger": "Digital\nleapfrogger",
    "small_open": "Small open\neconomy",
    "post_soviet": "Post-Soviet\neconomy",
    "populous_weak_industry": "Populous weakly\nindustrialized",
}
PROFILE_ORDER = ["manufacturing_giant", "resource_rich", "digital_leapfrogger",
                 "small_open", "post_soviet", "populous_weak_industry"]

FRACTURE_LABEL = {
    "industrial_withering": "Industrial\nwithering",
    "social_collapse": "Welfare\ncollapse",
    "tech_lock_in": "Technology\nlock-in",
    "jump_induced_deindustrialization": "Abrupt\ndeindustrialization",
    "human_capital_erosion": "Human-capital\nerosion",
}
FRACTURE_ORDER = ["industrial_withering", "social_collapse", "tech_lock_in",
                  "jump_induced_deindustrialization", "human_capital_erosion"]

GROUP_LABEL = {
    "digital_high_skill": "Digital-sector\nworkers",
    "industrial_workers": "Industrial\nworkers",
    "base_workers": "Primary\nsector",
    "service_workers": "Service\nsector",
    "unemployed": "Unemployed",
}

# 箱线图专用单行短标签（多面板并排时避免 x 轴刻度相互遮挡）
GROUP_LABEL_SHORT = {
    "digital_high_skill": "Digital",
    "industrial_workers": "Industrial",
    "base_workers": "Primary",
    "service_workers": "Service",
    "unemployed": "Unemployed",
}
GROUP_ORDER = ["digital_high_skill", "industrial_workers", "base_workers",
               "service_workers", "unemployed"]


def load(tag):
    p = RUN_ROOT / f"{tag}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ok(data):
    return [r for r in data["results"] if r and r.get("ok")]


def ftypes(s):
    ft = s.get("fracture_types")
    if isinstance(ft, dict):
        return list(ft.keys())
    return list(ft or [])


def save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  [ok] {stem}.pdf / .png")


# --------------------------------------------------------------------------
# 1. 存活率热力图
# --------------------------------------------------------------------------

def fig_survival_heatmap(out_dir: Path):
    data = load("census")
    if not data:
        print("  [skip] fig_survival_heatmap: 缺少 census")
        return
    rs = ok(data)
    cell = defaultdict(lambda: [0, 0])
    for r in rs:
        m = r["meta"]
        c = cell[(m["profile"], m["regime"])]
        c[1] += 1
        if r["summary"].get("survived"):
            c[0] += 1

    mat = np.full((len(PROFILE_ORDER), 3), np.nan)
    cnt = np.zeros((len(PROFILE_ORDER), 3), dtype=int)
    tot = np.zeros((len(PROFILE_ORDER), 3), dtype=int)
    for i, p in enumerate(PROFILE_ORDER):
        for j, g in enumerate(["A", "B", "C"]):
            s, n = cell.get((p, g), [0, 0])
            if n:
                mat[i, j] = s / n
                cnt[i, j] = s
                tot[i, j] = n

    fig, ax = plt.subplots(figsize=(3.5, 4.3), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#C0392B", "#E8A33D", "#F2E394", "#7FBF6A", "#2E7D32"])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(3), [f"Strategy {g}" for g in "ABC"])
    ax.set_yticks(range(len(PROFILE_ORDER)),
                  [SHORT_PROFILE[p] for p in PROFILE_ORDER])
    ax.tick_params(length=0)

    for i in range(len(PROFILE_ORDER)):
        for j in range(3):
            if tot[i, j] == 0:
                continue
            v = mat[i, j]
            txt = f"{100*v:.1f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    fontweight="bold",
                    color="white" if (v < 0.22 or v > 0.80) else "#1A1A1A")

    for i in range(len(PROFILE_ORDER) + 1):
        ax.axhline(i - 0.5, color="white", lw=1.6)
    for j in range(4):
        ax.axvline(j - 0.5, color="white", lw=1.6)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("Viability share", fontsize=8.5)
    cb.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    cb.outline.set_linewidth(0.5)
    save(fig, out_dir, "fig_survival_heatmap")


# --------------------------------------------------------------------------
# 2. 断裂指纹
# --------------------------------------------------------------------------

def fig_fracture_fingerprint(out_dir: Path):
    data = load("census")
    if not data:
        print("  [skip] fig_fracture_fingerprint: 缺少 census")
        return
    rs = ok(data)
    cnt = defaultdict(lambda: defaultdict(int))
    n_reg = defaultdict(int)
    for r in rs:
        g = r["meta"]["regime"]
        n_reg[g] += 1
        for t in set(ftypes(r["summary"])):
            cnt[t][g] += 1

    x = np.arange(len(FRACTURE_ORDER))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 3.1), constrained_layout=True)
    for k, g in enumerate("ABC"):
        vals = [100 * cnt[t][g] / max(n_reg[g], 1) for t in FRACTURE_ORDER]
        bars = ax.bar(x + (k - 1) * width, vals, width,
                      label=f"Strategy {g}", color=STRATEGY_COLORS[g],
                      edgecolor="white", linewidth=0.5, zorder=3)
        ax.bar_label(bars, fmt="%.0f", padding=1.5, fontsize=7, color="#333333")

    ax.set_xticks(x, [FRACTURE_LABEL[t] for t in FRACTURE_ORDER])
    ax.set_ylabel("Share of runs (%)")
    ax.set_ylim(0, max(60, ax.get_ylim()[1] * 1.18))
    ax.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    save(fig, out_dir, "fig_fracture_fingerprint")


# --------------------------------------------------------------------------
# 3. B 策略成本分布
# --------------------------------------------------------------------------

def fig_b_cost_distribution(out_dir: Path):
    data = load("cost_table")
    if not data:
        print("  [skip] fig_b_cost_distribution: 缺少 cost_table")
        return
    rs = ok(data)
    profiles = ["resource_rich", "digital_leapfrogger", "manufacturing_giant",
                "post_soviet"]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.15), constrained_layout=True,
                             sharey=True)
    for ax, p in zip(axes, profiles):
        sub = [r for r in rs if r["meta"]["profile"] == p]
        data_g = []
        for g in GROUP_ORDER:
            vals = [float(r["summary"]["cost_by_group"][g]) for r in sub
                    if (r["summary"].get("cost_by_group") or {}).get(g) is not None]
            data_g.append(np.array(vals) if vals else np.array([0.0]))

        bp = ax.boxplot(data_g, widths=0.62, patch_artist=True, showfliers=False,
                        medianprops=dict(color="#1A1A1A", lw=1.0),
                        whiskerprops=dict(color="#666666", lw=0.7),
                        capprops=dict(color="#666666", lw=0.7))
        for patch, c in zip(bp["boxes"], GROUP_COLORS):
            patch.set_facecolor(c)
            patch.set_edgecolor("#555555")
            patch.set_linewidth(0.5)
            patch.set_alpha(0.9)

        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xticks(range(1, len(GROUP_ORDER) + 1),
                      [GROUP_LABEL_SHORT[g] for g in GROUP_ORDER], rotation=90,
                      ha="center", va="top", fontsize=7.0)
        ax.tick_params(axis="x", pad=2)
        ax.set_title(SHORT_PROFILE[p].replace("\n", " "), fontsize=8.2)
        ax.grid(axis="y", lw=0.35, color="#E4E4E4", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("Cumulative welfare cost")
    axes[0].set_ylim(bottom=-0.02)
    fig.supxlabel("Cost-bearing group (symlog scale)", fontsize=8)
    save(fig, out_dir, "fig_b_cost_distribution")


# --------------------------------------------------------------------------
# 4-6. 校准图（需要时间序列）
# --------------------------------------------------------------------------

def _ts_data():
    """优先使用普查存活轨迹（ts_viable），回退到固定组合批次（ts_calibration）。"""
    for tag in ("ts_viable", "ts_calibration"):
        data = load(tag)
        if not data:
            continue
        rs = [r for r in data["results"] if r and r.get("ok")]
        if rs:
            return rs
    return None


def _emp_share(ts):
    """制造业就业占比 = L_i / Σ L（六部门）."""
    keys = ["L_i", "L_b", "L_d_infra", "L_d_app", "L_s_high", "L_s_low"]
    tot = np.zeros_like(np.array(ts["L_i"], dtype=float))
    for k in keys:
        tot = tot + np.array(ts[k], dtype=float)
    tot[tot <= 0] = np.nan
    return np.array(ts["L_i"], dtype=float) / tot


def _pc_gdp(ts):
    keys = ["L_i", "L_b", "L_d_infra", "L_d_app", "L_s_high", "L_s_low"]
    emp = np.zeros_like(np.array(ts["gdp"], dtype=float))
    for k in keys:
        emp = emp + np.array(ts[k], dtype=float)
    emp[emp <= 0] = np.nan
    return np.array(ts["gdp"], dtype=float) / emp


def _smooth(a, w):
    """居中滑动平均，抑制高频抖动（保留原时间顺序）。"""
    a = np.asarray(a, dtype=float)
    n = len(a)
    w = max(5, int(w))
    if n < w:
        return a
    pad = w // 2
    ext = np.concatenate([np.full(pad, a[0]), a, np.full(pad, a[-1])])
    ker = np.ones(w) / w
    return np.convolve(ext, ker, mode="valid")[:n]


def fig_calib_fact1(out_dir: Path):
    rs = _ts_data()
    if not rs:
        print("  [skip] fig_calib_fact1_curves: 缺少 ts_calibration")
        return
    combos = sorted({(r["meta"]["profile"], r["meta"]["regime"]) for r in rs})
    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)

    cmap = plt.get_cmap("tab10")
    handles, labels = [], []
    for i, (p, g) in enumerate(combos):
        sub = [r for r in rs
               if r["meta"]["profile"] == p and r["meta"]["regime"] == g
               and r["summary"].get("survived")]
        for j, r in enumerate(sub):
            y = _emp_share(r["timeseries"])
            x = _pc_gdp(r["timeseries"])
            ok_mask = np.isfinite(y) & np.isfinite(x)
            if ok_mask.sum() < 5:
                continue
            xv, yv = x[ok_mask], y[ok_mask]
            w = max(5, len(xv) // 20)
            xv, yv = _smooth(xv, w), _smooth(yv, w)
            ln, = ax.plot(xv, 100 * yv,
                          color=cmap(i % 10), lw=1.0, alpha=0.75)
            if j == 0:
                handles.append(ln)
                labels.append(f"{p.replace('_', ' ')} · {g}")

    ax.set_xscale("log")
    ax.set_xlabel("GDP per worker (log scale, model units)")
    ax.set_ylabel("Manufacturing employment share (%)")
    ax.grid(lw=0.4, color="#E6E6E6")
    ax.set_axisbelow(True)
    ax.legend(handles, labels, ncol=2, fontsize=6.4, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), columnspacing=1.0, handlelength=1.8)
    save(fig, out_dir, "fig_calib_fact1_curves")


def fig_calib_fact2(out_dir: Path):
    data = load("calibration")
    if not data:
        print("  [skip] fig_calib_fact2_inst: 缺少 calibration")
        return
    rs = ok(data)
    from src.params.country_profiles import get_country_profile

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True,
                            sharey=True)
    for ax, g in zip(axes, "ABC"):
        xs, ys = [], []
        for r in rs:
            if r["meta"]["regime"] != g:
                continue
            p = r["meta"]["profile"]
            try:
                cp = get_country_profile(p)
                inst = float(getattr(cp, "institution_capacity", None)
                             if not isinstance(cp, dict)
                             else cp.get("institution_capacity"))
            except Exception:  # noqa: BLE001
                continue
            if inst is None:
                continue
            xs.append(inst)
            ys.append(float(r["summary"]["final_gdp"]))
        if not xs:
            continue
        ax.scatter(xs, ys, s=16, color=STRATEGY_COLORS[g], alpha=0.55,
                   edgecolor="white", linewidth=0.4, zorder=3)
        if len(set(xs)) > 1:
            coef = np.polyfit(xs, np.log10(np.maximum(ys, 1)), 1)
            xx = np.linspace(min(xs), max(xs), 50)
            ax.plot(xx, 10 ** np.polyval(coef, xx), color="#333333", lw=1.0,
                    ls="--", zorder=2)
        ax.set_yscale("log")
        ax.set_title(f"Strategy {g}", fontsize=8.5)
        ax.set_xlabel("Initial institutional capacity")
        ax.grid(lw=0.35, color="#EAEAEA", zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Terminal GDP (log scale)")
    save(fig, out_dir, "fig_calib_fact2_inst")


def fig_calib_fact3(out_dir: Path):
    # 事实 3 需要横跨「快速 / 渐进去工业化」的充分样本，故使用完整 18 组合×5 次批次
    data = load("ts_calibration")
    rs = [r for r in data["results"] if r and r.get("ok")] if data else []
    if not rs:
        print("  [skip] fig_calib_fact3_deind: 缺少 ts_calibration")
        return
    xs, ys, cols = [], [], []
    for r in rs:
        y = _emp_share(r["timeseries"])
        if not np.isfinite(y).any():
            continue
        peak = np.nanmax(y)
        last = y[np.isfinite(y)][-1]
        if peak <= 0:
            continue
        deind = (peak - last) / peak * 100.0
        g = np.array(r["timeseries"]["gdp"], dtype=float)
        yrs = np.array(r["timeseries"]["t"], dtype=float)
        ok = np.isfinite(g) & (g > 0) & np.isfinite(yrs)
        if ok.sum() < 5:
            continue
        g0, g1 = g[ok][0], g[ok][-1]
        span = yrs[ok][-1] - yrs[ok][0]
        # 年均复合增长率：使不同长度（含提前终止）的轨迹可比
        growth = ((g1 / g0) ** (1.0 / span) - 1) * 100 if span > 0 and g0 > 0 else np.nan
        xs.append(deind)
        ys.append(growth)
        cols.append(STRATEGY_COLORS[r["meta"]["regime"]])

    fig, ax = plt.subplots(figsize=(3.9, 3.1), constrained_layout=True)
    ax.scatter(xs, ys, s=16, c=cols, alpha=0.55, edgecolor="white", linewidth=0.35,
               zorder=3)
    ax.axhline(0, color="#888888", lw=0.7, ls=":")
    handles = [plt.Line2D([], [], marker="o", ls="", color=STRATEGY_COLORS[g],
                          label=f"Strategy {g}") for g in "ABC"]
    ax.legend(handles=handles, fontsize=7, loc="lower left")
    ax.set_xlabel("Deindustrialization extent (%)")
    ax.set_ylabel("Annualized GDP growth (%)")
    ax.grid(lw=0.35, color="#EAEAEA", zorder=0)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=STRATEGY_COLORS[g],
                          label=f"Strategy {g}", markersize=5) for g in "ABC"]
    ax.legend(handles=handles, loc="best", fontsize=7.5)
    save(fig, out_dir, "fig_calib_fact3_deind")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_PROJECT_ROOT / "figures_v38"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    print(f"输出目录: {out_dir}")
    fig_survival_heatmap(out_dir)
    fig_fracture_fingerprint(out_dir)
    fig_b_cost_distribution(out_dir)
    fig_calib_fact1(out_dir)
    fig_calib_fact2(out_dir)
    fig_calib_fact3(out_dir)


if __name__ == "__main__":
    main()
