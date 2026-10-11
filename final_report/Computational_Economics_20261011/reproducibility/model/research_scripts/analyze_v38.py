"""从 v3.8 实验批次重算论文的全部数据表 (v3.8)。

读取 ``experiments_runs/v38/*.json``，输出：
  * ``analysis_report.md``  —— 人读的逐表重算结果
  * ``tables.json``         —— 机读的结构化结果

第一阶段严格重算：缺失批次、关键字段或失败记录导致非零退出。
450条LHS保留探索归档身份，不生成正式参数稳健性结论。
"""

from __future__ import annotations

import json
import math
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT.parent / "runtime_packages"))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RUN_ROOT = _PROJECT_ROOT / "experiments_runs" / "v38"

GROUPS = ["digital_high_skill", "industrial_workers", "base_workers",
          "service_workers", "unemployed"]
REGIMES = ["A", "B", "C"]
PROFILES_ORDER = ["manufacturing_giant", "resource_rich", "digital_leapfrogger",
                  "small_open", "post_soviet", "populous_weak_industry"]


# --------------------------------------------------------------------------

def load(tag: str):
    p = RUN_ROOT / f"{tag}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ok_results(data):
    rs=data["results"]
    if not rs or any(not r or not r["ok"] for r in rs):
        raise ValueError("Missing or failed execution record")
    for r in rs:
        s=r["summary"]
        for k in ("final_gdp","final_welfare","survival_years"):
            if not np.isfinite(s[k]): raise ValueError(k)
        ev=s["evaluation"]
        if set(ev["five_dimensions"])!={"economy","livelihood","resilience","structure","sovereignty"}:
            raise ValueError("Invalid five_dimensions fields")
        if not np.isfinite([ev["cdi"],*ev["five_dimensions"].values()]).all():
            raise ValueError("Invalid evaluation values")
        if set(s["cost_by_group"])!=set(GROUPS): raise ValueError("Invalid cost fields")
    return rs


def frac_ftypes(summary) -> list:
    ft = summary.get("fracture_types")
    if isinstance(ft, dict):
        return list(ft.keys())
    return list(ft or [])


def mean_sd(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def pct(x, nd=1):
    if x is None:
        return "n/a"
    return f"{100 * x:.{nd}f}%"


L = []          # report lines
T = {}          # structured tables


def w(line=""):
    L.append(line)
    print(line)


def section(title):
    w()
    w(f"## {title}")
    w()


# --------------------------------------------------------------------------
# 1. 普查：存活率矩阵 + 断裂构成
# --------------------------------------------------------------------------

def do_census():
    data = load("census")
    if not data:
        return
    rs = ok_results(data)
    section("1. 普查（6 国情 × 8 场景 × 3 策略 × 5 次重复 = 720 次）")
    w(f"- 成功运行: {len(rs)}")

    # 每格存活率
    cell = defaultdict(lambda: [0, 0])
    for r in rs:
        m = r["meta"]
        k = (m["profile"], m["regime"])
        c = cell[k]
        c[1] += 1
        if r["summary"].get("survived"):
            c[0] += 1

    profiles = sorted({k[0] for k in cell})
    w()
    w("### 1.1 存活率矩阵（每格 40 次重复：8 场景 × 5 次）")
    w()
    w("| Country type | A | B | C |")
    w("|---|---|---|---|")
    mat = {}
    for p in profiles:
        row = []
        for g in REGIMES:
            s, n = cell.get((p, g), [0, 0])
            row.append(f"{pct(s / n) if n else 'n/a'} ({s}/{n})")
            mat[f"{p}|{g}"] = s / n if n else None
        w(f"| {p} | " + " | ".join(row) + " |")
    tot_s = sum(c[0] for c in cell.values())
    tot_n = sum(c[1] for c in cell.values())
    w()
    w(f"- 总体存活率: **{pct(tot_s / tot_n)} ({tot_s}/{tot_n})**")
    T["census_matrix"] = mat
    T["census_overall"] = [tot_s, tot_n]

    # 每策略的存活率
    w()
    w("### 1.2 按策略汇总的存活率")
    w()
    w("| Strategy | survived | share |")
    w("|---|---|---|")
    by_reg = {}
    for g in REGIMES:
        s = sum(cell[(p, g)][0] for p in profiles if (p, g) in cell)
        n = sum(cell[(p, g)][1] for p in profiles if (p, g) in cell)
        w(f"| {g} | {s}/{n} | {pct(s / n)} |")
        by_reg[g] = [s, n]
    T["census_by_regime"] = by_reg

    # 断裂构成
    w()
    w("### 1.3 断裂类型构成")
    w()
    w("口径：以各策略的「运行—断裂类型」出现次数为分母；单次运行可计入多种类型。")
    w()
    cnt = defaultdict(Counter)
    n_by_reg = Counter()
    for r in rs:
        g = r["meta"]["regime"]
        n_by_reg[g] += 1
        for t in set(frac_ftypes(r["summary"])):
            cnt[g][t] += 1
    all_types = sorted({t for g in REGIMES for t in cnt[g]})
    w("| Breakdown mode | " + " | ".join(f"{g} (share / per-run)" for g in REGIMES) + " |")
    w("|---|---|---|---|")
    comp = {}
    for t in all_types:
        cells = []
        for g in REGIMES:
            tot = sum(cnt[g].values())
            c = cnt[g][t]
            cells.append(f"{pct(c / tot) if tot else 'n/a'} / {pct(c / n_by_reg[g])}")
            comp[f"{t}|{g}"] = {"count": c, "share_of_events": c / tot if tot else None,
                                "per_run": c / n_by_reg[g] if n_by_reg[g] else None}
        w(f"| {t} | " + " | ".join(cells) + " |")
    w()
    w(f"- 各策略分母（事件总数）: " + ", ".join(
        f"{g}={sum(cnt[g].values())}" for g in REGIMES))
    T["fracture_composition"] = comp

    # 致命性：出现在存活运行中的断裂类型
    w()
    w("### 1.4 断裂类型的致命性（是否与存活共存）")
    w()
    surv = [r for r in rs if r["summary"].get("survived")]
    coex = Counter()
    for r in surv:
        for t in set(frac_ftypes(r["summary"])):
            coex[t] += 1
    w(f"- 存活运行总数: {len(surv)}")
    w()
    w("| Breakdown mode | 出现在存活运行的次数 |")
    w("|---|---|")
    for t in all_types:
        w(f"| {t} | {coex[t]} |")
    T["fracture_coexist"] = dict(coex)

    # 死亡年数
    w()
    w("### 1.5 终止年数分布（未存活运行）")
    w()
    for g in REGIMES:
        ys = [r["summary"]["survival_years"] for r in rs
              if r["meta"]["regime"] == g and not r["summary"].get("survived")]
        if ys:
            w(f"- 策略 {g}: n={len(ys)}, 中位数 {np.median(ys):.1f}y, "
              f"范围 {min(ys):.1f}–{max(ys):.1f}y")


# --------------------------------------------------------------------------
# 2. T=80 延长
# --------------------------------------------------------------------------

def do_t80():
    data = load("t80")
    if not data:
        return
    rs = ok_results(data)
    section("2. T=80 新种子配置重检（非配对延长）")
    agg = defaultdict(lambda: [0, 0])
    for r in rs:
        m = r["meta"]
        a = agg[m["regime"]]
        a[1] += 1
        if r["summary"].get("survived"):
            a[0] += 1
    w("| Strategy | viable at T=80 | share |")
    w("|---|---|---|")
    tot = [0, 0]
    res = {}
    for g in REGIMES:
        s, n = agg.get(g, [0, 0])
        w(f"| {g} | {s}/{n} | {pct(s / n) if n else 'n/a'} |")
        res[g] = [s, n]
        tot[0] += s
        tot[1] += n
    w(f"| **Total** | {tot[0]}/{tot[1]} | {pct(tot[0] / tot[1]) if tot[1] else 'n/a'} |")
    T["t80"] = res
    T["t80_total"] = tot
    failures = [r for r in rs if not r["summary"]["survived"]]
    T["t80_failure_timing"] = {}
    for g in REGIMES:
        sub = [r for r in failures if r["meta"]["regime"] == g]
        early = sum(r["summary"]["survival_years"] < 50 for r in sub)
        late = len(sub) - early
        T["t80_failure_timing"][g] = {"early_before_50": early, "late_50_to_80": late}
        w(f"- {g}: {early} 次早期失败，{late} 次后期失败")
    w("筛选单位为普查中至少 3/5 次存续的 51 个配置；每配置五个新种子从初始状态运行。")


# --------------------------------------------------------------------------
# 3. 阈值扫描
# --------------------------------------------------------------------------

def do_threshold():
    data = load("threshold")
    if not data:
        return
    rs = ok_results(data)
    section("3. 福利崩溃阈值扫描（7 阈值 × 6 国情 × 3 策略 × 5 次）")
    by_th = defaultdict(lambda: [0, 0])
    for r in rs:
        th = r["meta"]["threshold"]
        a = by_th[th]
        a[1] += 1
        if r["summary"].get("survived"):
            a[0] += 1
    w("| threshold | survived | share |")
    w("|---|---|---|")
    res = {}
    for th in sorted(by_th):
        s, n = by_th[th]
        w(f"| {th:.2f} | {s}/{n} | {pct(s / n)} |")
        res[th] = [s, n]
    T["threshold"] = res

    # C 策略失败国情（复核附录 B.1 的关键）
    w()
    w("### 3.1 各阈值下 C 策略仍失败的国情")
    w()
    cprof = defaultdict(set)
    for r in rs:
        m = r["meta"]
        if m["regime"] == "C" and not r["summary"].get("survived"):
            cprof[m["threshold"]].add(m["profile"])
    w("| threshold | C 策略失败国情 |")
    w("|---|---|")
    for th in sorted(cprof):
        w(f"| {th:.2f} | {', '.join(sorted(cprof[th]))} |")
    T["threshold_C_fail"] = {str(k): sorted(v) for k, v in cprof.items()}


def do_gworld():
    data = load("gworld")
    if not data:
        return
    rs = ok_results(data)
    section("4. g_world × 阈值（A 策略）")
    agg = defaultdict(lambda: [0, 0])
    for r in rs:
        m = r["meta"]
        a = agg[(m["profile"], m["g_world"], m["threshold"])]
        a[1] += 1
        if r["summary"].get("survived"):
            a[0] += 1
    res = {}
    for profile in ["manufacturing_giant", "post_soviet"]:
        w()
        w(f"**{profile}**（每格 5 次，显示存活次数）")
        w()
        ths = sorted({k[2] for k in agg if k[0] == profile})
        gws = sorted({k[1] for k in agg if k[0] == profile})
        w("| g_world | " + " | ".join(f"{t:.2f}" for t in ths) + " |")
        w("|---" * (len(ths) + 1) + "|")
        for gw in gws:
            row = []
            for th in ths:
                s, n = agg.get((profile, gw, th), [0, 0])
                row.append(f"{s}/{n}")
                res[f"{profile}|{gw}|{th}"] = [s, n]
            w(f"| {gw} | " + " | ".join(row) + " |")
    T["gworld"] = res


# --------------------------------------------------------------------------
# 5. 参数扫描（σ_b / ε / w_emp×σ）
# --------------------------------------------------------------------------

def _scan_by(batch, axis, key_filter=None):
    data = load(batch)
    if not data:
        return None
    rs = ok_results(data)
    agg = defaultdict(lambda: [0, 0])
    for r in rs:
        m = r["meta"]
        if key_filter and not key_filter(m):
            continue
        a = agg[m[axis]]
        a[1] += 1
        if r["summary"].get("survived"):
            a[0] += 1
    return agg


def do_sigma():
    agg = _scan_by("sigma", "sigma_b")
    if not agg:
        return
    section("5. σ_b 扫描（13 水平 0.3–1.5，3 国情 × 3 策略 × 5 次）")
    data = load("sigma")
    rs = ok_results(data)
    by_lv = defaultdict(lambda: [0, 0])
    by_lv_reg = defaultdict(lambda: [0, 0])
    for r in rs:
        lv = r["meta"]["sigma_b"]
        g = r["meta"]["regime"]
        a = by_lv[lv]
        a[1] += 1
        if r["summary"].get("survived"):
            a[0] += 1
        b = by_lv_reg[(lv, g)]
        b[1] += 1
        if r["summary"].get("survived"):
            b[0] += 1
    w("| σ_b | overall | A | B | C |")
    w("|---|---|---|---|---|")
    res = {}
    for lv in sorted(by_lv):
        s, n = by_lv[lv]
        cells = []
        for g in REGIMES:
            s2, n2 = by_lv_reg.get((lv, g), [0, 0])
            cells.append(f"{s2}/{n2}")
            res[f"{lv}|{g}"] = [s2, n2]
        w(f"| {lv:.2f} | {s}/{n} | " + " | ".join(cells) + " |")
        res[str(lv)] = [s, n]
    T["sigma"] = res


def do_eps_fine():
    data = load("eps_fine")
    if not data:
        return
    rs = ok_results(data)
    section("6. σ_b 精细扫描（B 策略，13 水平 0.70–1.30，2 国情 × 5 次）")
    out = {}
    for profile in ["manufacturing_giant", "digital_leapfrogger"]:
        w()
        w(f"**{profile}**")
        w()
        w("| σ_b | survived | share | median years | median GDP |")
        w("|---|---|---|---|---|")
        for lv in sorted({r["meta"]["sigma_b"] for r in rs}):
            sub = [r for r in rs if r["meta"]["sigma_b"] == lv
                   and r["meta"]["profile"] == profile]
            if not sub:
                continue
            s = sum(1 for r in sub if r["summary"].get("survived"))
            yrs = [r["summary"]["survival_years"] for r in sub]
            gdps = [r["summary"]["final_gdp"] for r in sub]
            w(f"| {lv:.2f} | {s}/{len(sub)} | {pct(s/len(sub))} | "
              f"{np.median(yrs):.1f} | {np.median(gdps):,.0f} |")
            out[f"{profile}|{lv}"] = {"survived": s, "n": len(sub),
                                      "median_years": float(np.median(yrs)),
                                      "median_gdp": float(np.median(gdps))}
    T["eps_fine"] = out


def do_wemp_eps():
    data = load("wemp_eps")
    if not data:
        return
    rs = ok_results(data)
    section("7. w_emp × σ_b 交互（B 策略，5×5×4 国情 × 5 次）")
    res = {}
    for profile in ["manufacturing_giant", "resource_rich", "digital_leapfrogger",
                    "post_soviet"]:
        s = sum(1 for r in rs if r["meta"]["profile"] == profile
                and r["summary"].get("survived"))
        n = sum(1 for r in rs if r["meta"]["profile"] == profile)
        w(f"- {profile}: {s}/{n} = {pct(s/n)}")
        res[profile] = [s, n]
    T["wemp_eps"] = res


def do_ki():
    data = load("ki")
    if not data:
        return
    rs = ok_results(data)
    section("8. K_i 敏感性（B 策略，15 的 K_i × 3 制度条件 × 5 次）")
    agg = defaultdict(lambda: [0, 0])
    for r in rs:
        m = r["meta"]
        a = agg[(m["institution_capacity"], m["H_i"])]
        a[1] += 1
        if r["summary"].get("survived"):
            a[0] += 1
    w("| inst | H_i | survived | share |")
    w("|---|---|---|---|")
    res = {}
    for k in sorted(agg):
        s, n = agg[k]
        w(f"| {k[0]:.2f} | {k[1]:.1f} | {s}/{n} | {pct(s/n)} |")
        res[f"{k[0]}|{k[1]}"] = [s, n]
    T["ki"] = res


# --------------------------------------------------------------------------
# 9. 混合策略
# --------------------------------------------------------------------------

def do_mixture():
    data = load("mixture")
    if not data:
        return
    rs = ok_results(data)
    section("9. 混合策略（Dirichlet 采样，4 国情 × 2 场景 × 40 配比 × 3 次）")
    s = sum(1 for r in rs if r["summary"].get("survived"))
    w(f"- 全局存活率: **{pct(s/len(rs))} ({s}/{len(rs)})**")
    T["mixture_global"] = [s, len(rs)]

    w()
    w("### 9.1 按国情 × 场景")
    w()
    agg = defaultdict(lambda: [0, 0])
    for r in rs:
        m = r["meta"]
        a = agg[(m["profile"], m["scenario"])]
        a[1] += 1
        if r["summary"].get("survived"):
            a[0] += 1
    res = {}
    for k in sorted(agg):
        s2, n2 = agg[k]
        w(f"- {k[0]} × {k[1]}: {s2}/{n2} = {pct(s2/n2)}")
        res[f"{k[0]}|{k[1]}"] = [s2, n2]
    T["mixture_cells"] = res

    w()
    w("### 9.2 λ_B 分层（分界 0.30）")
    w()
    for lo, hi, lab in [(0.0, 0.30, "λ_B ≤ 0.30"), (0.30, 1.01, "λ_B > 0.30")]:
        sub = [r for r in rs if lo <= r["meta"]["lambda_B"] < hi]
        s3 = sum(1 for r in sub if r["summary"].get("survived"))
        w(f"- {lab}: {s3}/{len(sub)} = {pct(s3/len(sub)) if sub else 'n/a'}")
        T[f"mixture_lambdaB_{lab}"] = [s3, len(sub)]

    # 纯策略对照（取普查里 4 国情 × 2 场景，即混合样本的同构子集）
    census = load("census")
    if census:
        crs = ok_results(census)
        keep_p = {"manufacturing_giant", "resource_rich", "digital_leapfrogger",
                  "post_soviet"}
        keep_s = {"benign_globalization", "tech_cold_war"}
        sub = [r for r in crs if r["meta"]["profile"] in keep_p
               and r["meta"]["scenario"] in keep_s]
        s4 = sum(1 for r in sub if r["summary"].get("survived"))
        w()
        w(f"### 9.3 同口径纯策略对照（同一 4 国情 × 2 场景）")
        w()
        w(f"- 纯策略: **{pct(s4/len(sub))} ({s4}/{len(sub)})**")
        w(f"- 混合策略: {pct(s/len(rs))} ({s}/{len(rs)})")
        T["pure_matched"] = [s4, len(sub)]


# --------------------------------------------------------------------------
# 10. 成本
# --------------------------------------------------------------------------

def do_costs():
    data = load("cost_table")
    if not data:
        return
    rs = ok_results(data)
    section("10. 成本表（4 国情 × B 策略 × 30 次重复）")
    w("五群组累计代价（末值），单位同模型内部度量；括号内为 30 次重复的 [min, max]。")
    w()
    w("| Country type | Digital | Industrial | Base | Service | Unemployed | cost Gini |")
    w("|---|---|---|---|---|---|---|")
    res = {}
    for profile in ["manufacturing_giant", "resource_rich", "digital_leapfrogger",
                    "post_soviet"]:
        sub = [r for r in rs if r["meta"]["profile"] == profile]
        if not sub:
            continue
        cells = []
        gvals = defaultdict(list)
        for r in sub:
            cb = r["summary"].get("cost_by_group") or {}
            for g in GROUPS:
                if g in cb and cb[g] is not None:
                    gvals[g].append(float(cb[g]))
        rec = {}
        for g in GROUPS:
            m, sd = mean_sd(gvals[g])
            lo = min(gvals[g]) if gvals[g] else None
            hi = max(gvals[g]) if gvals[g] else None
            nd=3 if g=="digital_high_skill" else 1
            cells.append(f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]")
            rec[g] = {"mean": m, "sd": sd, "min": lo, "max": hi,
                      "nonzero_count": sum(v!=0 for v in gvals[g]),"n":len(gvals[g])}
        gin = [float(r["summary"]["cost_concentration"]) for r in sub
               if r["summary"].get("cost_concentration") is not None]
        gm, gsd = mean_sd(gin)
        cells.append(f"{gm:.3f}")
        rec["cost_gini"] = {"mean": gm, "sd": gsd}
        surv = sum(1 for r in sub if r["summary"].get("survived"))
        rec["survival"] = [surv, len(sub)]
        w(f"| {profile} | " + " | ".join(cells) + " |")
        res[profile] = rec
    T["cost_table"] = res

    w()
    w("### 10.1 数字部门群组零代价的适用范围")
    w()
    for profile in res:
        g = res[profile]["digital_high_skill"]
        w(f"- {profile}: mean={g['mean']}, max={g['max']}, "
          f"非零记录 {g['nonzero_count']}/{g['n']}")


def do_cost_lhs():
    data = load("cost_lhs")
    if not data:
        return
    rs = ok_results(data)
    section("11. 成本分布 LHS（3 国情 × 3 场景 × 50 参数点，C 策略）")
    w(f"- 运行数: {len(rs)}")
    w("实际历史参数矩阵未保存；保留探索归档，不用于正式 C 参数稳健性结论，不重跑。")
    T["cost_lhs"] = {"n": len(rs), "status": "exploratory_only",
                     "actual_parameter_matrix_saved": False,
                     "formal_C_parameter_robustness_eligible": False}


# --------------------------------------------------------------------------
# 12. 校准
# --------------------------------------------------------------------------

def _spearman(x, y):
    """Spearman 秩相关.

    修复（2026-10-08）：旧实现用 `np.argsort(np.argsort(...))` 排名，未处理
    并列值——相同制度能力的多个国情会被强制排出先后，使相关系数产生偏差
    （18 组合均值下 ρ(inst,GDP) 由正确的 0.739476 被算成 0.762642）。
    现改用平均秩（scipy.stats.rankdata 默认 method="average"）。
    另注意：p 值为渐近 t 近似，n=6 时不可靠，小样本应改用精确置换检验。
    """
    from research_scripts.stage1_statistics import spearman_average
    return spearman_average(x, y)


def do_calibration():
    data = load("calibration")
    if not data:
        return
    rs = ok_results(data)
    section("12. 定性校准（6 国情 × 3 策略 × 5 次，T=50）")
    # 国情 × 策略 均值（与仲裁结论一致：用 18 个组合的均值）
    comb = defaultdict(list)
    for r in rs:
        comb[(r["meta"]["profile"], r["meta"]["regime"])].append(r["summary"])

    x_inst = []
    for (p, g), sums in comb.items():
        from src.params.country_profiles import get_country_profile
        try:
            cp = get_country_profile(p)
            # CountryProfile 是 dataclass（非 dict），需用属性访问
            inst = (getattr(cp, "institution_capacity", None)
                    if not isinstance(cp, dict) else cp.get("institution_capacity"))
        except Exception:  # noqa: BLE001
            inst = None
        if inst is not None:
            x_inst.append((p, g, float(inst), sums))

    if x_inst:
        ins = [v[2] for v in x_inst]
        gdp = [float(np.mean([s["final_gdp"] for s in v[3]])) for v in x_inst]
        wf = [float(np.mean([s["final_welfare"] for s in v[3]])) for v in x_inst]
        yr = [float(np.mean([s["survival_years"] for s in v[3]])) for v in x_inst]
        for label, yy in [("final GDP", gdp), ("final welfare", wf),
                          ("survival years", yr)]:
            r_, p_ = _spearman(ins, yy)
            w(f"- ρ(institutional capacity, {label}) = {r_:.3f} (p={p_:.4f}, n={len(ins)}), "
              f"基于 18 组合均值")
        T["calib_spearman"] = {}
        for label, yy in [("gdp", gdp), ("welfare", wf), ("years", yr)]:
            r_, p_ = _spearman(ins, yy)
            T["calib_spearman"][label] = [r_, p_]

    # 按策略分组
    w()
    w("### 12.1 按策略的体制能力相关性")
    w()
    for g in REGIMES:
        sub = [v for v in x_inst if v[1] == g]
        if len(sub) < 3:
            continue
        ins = [v[2] for v in sub]
        wf = [float(np.mean([s.get("final_welfare", 0) for s in v[3]])) for v in sub]
        from research_scripts.stage1_statistics import exact_spearman
        for field in ("final_welfare", "final_gdp", "survival_years"):
            yy = [float(np.mean([s[field] for s in v[3]])) for v in sub]
            result = exact_spearman(ins, yy)
            result["profile_order"] = [v[0] for v in sub]
            T.setdefault("calib_exact_per_strategy", {})[f"{g}|{field}"] = result
            w(f"- 策略 {g}: ρ(inst, {field}) = {result['rho']:.6f}; "
              f"双侧精确 p={result['p_exact_two_sided']:.6f}, "
              f"n={len(sub)}，平均秩，720 个索引置换，容差 1e-12")

    w()
    w("### 12.2 整体存活率")
    w()
    s = sum(1 for r in rs if r["summary"].get("survived"))
    w(f"- 存活 {s}/{len(rs)} = {pct(s/len(rs))}")
    T["calib_survival"] = [s, len(rs)]


# --------------------------------------------------------------------------
# 11. 补充实验 exp1–exp6 / 抽取率 / 附录诊断案例
# --------------------------------------------------------------------------

def _dl_d(s):
    """数字部门发展水平 DL_d：按数字基础设施/应用资本加权。"""
    ki = float(s.get("sector_K_d_infra") or 0.0)
    ka = float(s.get("sector_K_d_app") or 0.0)
    di = float(s.get("sector_DL_d_infra") or 0.0)
    da = float(s.get("sector_DL_d_app") or 0.0)
    tot = ki + ka
    if tot <= 0:
        return max(di, da)
    return (ki * di + ka * da) / tot


def _mean_key(rs, fn):
    return mean_sd([fn(r["summary"]) for r in rs])


def do_exp1():
    data = load("exp1")
    if not data:
        return
    rs = ok_results(data)
    section("11. 非线性追赶稳健性（exp1：2 国情 × 2 场景 × 6 g × 3 模式 × 3 次）")

    for profile in ["manufacturing_giant", "digital_leapfrogger"]:
        for scenario in ["benign_globalization", "tech_cold_war"]:
            agg = defaultdict(lambda: [0, 0])
            for r in rs:
                m = r["meta"]
                if m["profile"] != profile or m["scenario"] != scenario:
                    continue
                a = agg[(m["g_world"], m["catchup_mode"])]
                a[1] += 1
                if r["summary"].get("survived"):
                    a[0] += 1
            if not agg:
                continue
            w()
            w(f"**{profile} × 策略 A × {scenario}**（存活次数/总次）")
            w()
            modes = ["linear", "logistic", "piecewise"]
            gws = sorted({k[0] for k in agg})
            w("| g_world | " + " | ".join(modes) + " |")
            w("|---" * (len(modes) + 1) + "|")
            for gw in gws:
                row = []
                for md in modes:
                    s, n = agg.get((gw, md), [0, 0])
                    row.append(f"{s}/{n}" if n else "-")
                w(f"| {gw} | " + " | ".join(row) + " |")
            T[f"exp1|{profile}|{scenario}"] = {
                f"{k[0]}|{k[1]}": v for k, v in agg.items()}

    # g=0.10 时的存活年数对比（linear vs logistic）
    w()
    w("### 11.1 $g_{world}=0.10$ 时的存续年数（manuf_giant×A，benign）")
    w()
    w("| 模式 | 平均存续年数 | GDP(期末) |")
    w("|---|---|---|")
    for md in ["linear", "logistic", "piecewise"]:
        sub = [r for r in rs if r["meta"]["profile"] == "manufacturing_giant"
               and r["meta"]["scenario"] == "benign_globalization"
               and r["meta"]["g_world"] == 0.10
               and r["meta"]["catchup_mode"] == md]
        yrs = [r["summary"]["survival_years"] for r in sub]
        gdp = [r["summary"]["final_gdp"] for r in sub]
        m1, _ = mean_sd(yrs)
        m2, _ = mean_sd(gdp)
        w(f"| {md} | {m1:.1f} | {m2:,.0f} |")
        T[f"exp1_g10|{md}"] = {"survival_years": m1, "final_gdp": m2}


def do_exp2():
    data = load("exp2")
    if not data:
        return
    rs = ok_results(data)
    section("12. B 策略联合调整（exp2：B_phi_d_jump 0.6→0.3 与 B_g_target 0.25→0.15）")
    w("历史变体键 redistribute 仅为归档标识；该实验不实施利润税，不识别单参数效应。")
    w("| 国情 | 变体 | 存活 | 期末 GDP | 期末福利 | 代价集中度 |")
    w("|---|---|---|---|---|---|")
    for profile in ["manufacturing_giant", "resource_rich",
                    "digital_leapfrogger", "post_soviet"]:
        for variant in ["baseline", "redistribute"]:
            sub = [r for r in rs if r["meta"]["profile"] == profile
                   and r["meta"]["variant"] == variant]
            if not sub:
                continue
            ns = sum(1 for r in sub if r["summary"].get("survived"))
            g, _ = _mean_key(sub, lambda s: s.get("final_gdp"))
            wl, _ = _mean_key(sub, lambda s: s.get("final_welfare"))
            cc, _ = _mean_key(sub, lambda s: s.get("cost_concentration"))
            w(f"| {profile} | {variant} | {ns}/{len(sub)} | {g:,.0f} | "
              f"{wl:.3f} | {cc:.3f} |")
            T[f"exp2|{profile}|{variant}"] = {
                "survived": ns, "n": len(sub), "final_gdp": g,
                "final_welfare": wl, "cost_concentration": cc,
                "digital_cost_mean": _mean_key(sub, lambda s: s["cost_by_group"]["digital_high_skill"])[0],
                "years": _mean_key(sub, lambda s: s["survival_years"])[0],
                "joint_adjustment": True}


def do_exp3():
    data = load("exp3")
    if not data:
        return
    rs = ok_results(data)
    section("13. 混合策略敏感性（exp3：16 配比/制度变体 × 3 次）")
    w("| 变体 | 国情 | 类型 | 存活 | 平均存续年数 |")
    w("|---|---|---|---|---|")
    seen = []
    for r in rs:
        k = r["meta"]["variant"]
        if k not in seen:
            seen.append(k)
    for k in seen:
        sub = [r for r in rs if r["meta"]["variant"] == k]
        m0 = sub[0]["meta"]
        ns = sum(1 for r in sub if r["summary"].get("survived"))
        yr, _ = _mean_key(sub, lambda s: s.get("survival_years"))
        w(f"| {k} | {m0['profile']} | {m0.get('kind', 'mixture')} | "
          f"{ns}/{len(sub)} | {yr:.1f} |")
        T[f"exp3|{k}"] = {"profile": m0["profile"], "kind": m0.get("kind"),
                          "survived": ns, "n": len(sub), "years": yr}


def do_exp4():
    data = load("exp4")
    if not data:
        return
    rs = ok_results(data)
    section("14. T=80 代表组合（exp4：4 组合 × 3 次）")
    w("| 策略 | 国情 | 存活 | 平均存续年数 | 期末 GDP | 期末福利 | 工业资本 | DL_d |")
    w("|---|---|---|---|---|---|---|---|")
    for regime in REGIMES:
        for profile in ["manufacturing_giant", "digital_leapfrogger", "resource_rich"]:
            sub = [r for r in rs if r["meta"]["regime"] == regime
                   and r["meta"]["profile"] == profile]
            if not sub:
                continue
            ns = sum(1 for r in sub if r["summary"].get("survived"))
            yr, _ = _mean_key(sub, lambda s: s.get("survival_years"))
            g, _ = _mean_key(sub, lambda s: s.get("final_gdp"))
            wl, _ = _mean_key(sub, lambda s: s.get("final_welfare"))
            ki, _ = _mean_key(sub, lambda s: s.get("sector_K_i"))
            dl, _ = _mean_key(sub, _dl_d)
            w(f"| {regime} | {profile} | {ns}/{len(sub)} | {yr:.1f} | {g:,.0f} | "
              f"{wl:.3f} | {ki:,.0f} | {dl:.3f} |")
            T[f"exp4|{regime}|{profile}"] = {
                "survived": ns, "n": len(sub), "years": yr, "final_gdp": g,
                "final_welfare": wl, "K_i": ki, "DL_d": dl}


def do_exp5():
    data = load("exp5")
    if not data:
        return
    rs = ok_results(data)
    section("15. 高制度能力反事实（exp5：3 组合 × 3 次）")
    w("| 策略 | 国情 | 制度能力 | 存活 | 平均存续年数 |")
    w("|---|---|---|---|---|")
    for r in rs:
        pass
    keys = []
    for r in rs:
        k = (r["meta"]["regime"], r["meta"]["profile"],
             r["meta"]["institution_capacity"])
        if k not in keys:
            keys.append(k)
    for k in keys:
        sub = [r for r in rs if (r["meta"]["regime"], r["meta"]["profile"],
                                 r["meta"]["institution_capacity"]) == k]
        ns = sum(1 for r in sub if r["summary"].get("survived"))
        yr, _ = _mean_key(sub, lambda s: s.get("survival_years"))
        w(f"| {k[0]} | {k[1]} | {k[2]:.2f} | {ns}/{len(sub)} | {yr:.1f} |")
        T[f"exp5|{k[0]}|{k[1]}|{k[2]}"] = {
            "survived": ns, "n": len(sub), "years": yr}


def do_exp6():
    data = load("exp6")
    if not data:
        return
    rs = ok_results(data)
    section("16. 福利参数敏感性（exp6：ε 5 级 + 就业权重 5 级，manuf_giant×B）")
    for axis, label in [("epsilon_welfare", r"$\varepsilon_{welfare}$"),
                        ("welfare_employment_exponent", r"$w_{emp}$")]:
        w()
        w(f"**{label} 扫描**（每格 3 次）")
        w()
        w(f"| {axis} | 存活 | 平均存续年数 | 期末福利 |")
        w("|---|---|---|---|")
        vals = sorted({r["meta"]["value"] for r in rs
                       if r["meta"]["axis"] == axis})
        for v in vals:
            sub = [r for r in rs if r["meta"]["axis"] == axis
                   and r["meta"]["value"] == v]
            ns = sum(1 for r in sub if r["summary"].get("survived"))
            yr, _ = _mean_key(sub, lambda s: s.get("survival_years"))
            wl, _ = _mean_key(sub, lambda s: s.get("final_welfare"))
            w(f"| {v} | {ns}/{len(sub)} | {yr:.1f} | {wl:.3f} |")
            T[f"exp6|{axis}|{v}"] = {"survived": ns, "n": len(sub),
                                     "years": yr, "final_welfare": wl}


def do_extraction():
    data = load("extraction")
    if not data:
        return
    rs = ok_results(data)
    section("17. 数字增长目标 B_g_target 扫描（非实际提取率设定；2 国情 × 7 级 × 5 次）")
    w("| $B\\_g\\_target$ | " + " | ".join(
        ["manufacturing_giant", "resource_rich"]) + " |")
    w("|---|---|---|")
    for gt in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        cells = []
        for profile in ["manufacturing_giant", "resource_rich"]:
            sub = [r for r in rs if r["meta"]["B_g_target"] == gt
                   and r["meta"]["profile"] == profile]
            ns = sum(1 for r in sub if r["summary"].get("survived"))
            cells.append(f"{ns}/{len(sub)}" if sub else "-")
            T[f"extraction|{profile}|{gt}"] = {"survived": ns, "n": len(sub)}
        w(f"| {gt:.2f} | " + " | ".join(cells) + " |")


def do_diag():
    """主文/附录案例数字：以 census 中 benign 场景的 5 次重复为准。"""
    data = load("census")
    if not data:
        return
    rs = [r for r in ok_results(data)
          if r["meta"]["scenario"] == "benign_globalization"]
    section("18. 诊断案例关键量（census，benign_globalization，5 次重复均值）")
    fields = [("期末 GDP", lambda s: s.get("final_gdp"), "{:,.0f}"),
              ("期末福利", lambda s: s.get("final_welfare"), "{:.3f}"),
              ("期末基尼", lambda s: s.get("final_gini"), "{:.3f}"),
              ("技术差距", lambda s: s.get("final_tech_gap"), "{:.1f}"),
              ("工业资本", lambda s: s.get("sector_K_i"), "{:,.0f}"),
              ("DL_d", _dl_d, "{:.3f}"),
              ("峰值失业率", lambda s: s.get("peak_unemployment"), "{:.4f}"),
              ("代价集中度", lambda s: s.get("cost_concentration"), "{:.3f}")]
    for regime in REGIMES:
        w()
        w(f"**策略 {regime}**")
        w()
        w("| 国情 | 存活 | 存续年数 | " +
          " | ".join(f[0] for f in fields) + " |")
        w("|---" * (len(fields) + 3) + "|")
        for profile in PROFILES_ORDER:
            sub = [r for r in rs if r["meta"]["profile"] == profile
                   and r["meta"]["regime"] == regime]
            if not sub:
                continue
            ns = sum(1 for r in sub if r["summary"].get("survived"))
            yr, _ = _mean_key(sub, lambda s: s.get("survival_years"))
            cells = []
            rec = {"survived": ns, "n": len(sub), "years": yr}
            for _, fn, fmt in fields:
                v, _ = _mean_key(sub, fn)
                cells.append("n/a" if v is None else fmt.format(v))
                rec[fn.__name__ if hasattr(fn, "__name__") else "x"] = v
            w(f"| {profile} | {ns}/{len(sub)} | {yr:.1f} | " +
              " | ".join(cells) + " |")
            T[f"diag|{regime}|{profile}"] = {
                "survived": ns, "n": len(sub), "years": yr,
                "final_gdp": _mean_key(sub, lambda s: s.get("final_gdp"))[0],
                "final_welfare": _mean_key(sub, lambda s: s.get("final_welfare"))[0],
                "final_gini": _mean_key(sub, lambda s: s.get("final_gini"))[0],
                "tech_gap": _mean_key(sub, lambda s: s.get("final_tech_gap"))[0],
                "K_i": _mean_key(sub, lambda s: s.get("sector_K_i"))[0],
                "DL_d": _mean_key(sub, _dl_d)[0],
                "peak_unemp": _mean_key(sub, lambda s: s.get("peak_unemployment"))[0],
                "cost_conc": _mean_key(sub, lambda s: s.get("cost_concentration"))[0],
            }

    # 五维得分（最有利可行条件下）
    w()
    w("### 18.1 五维评价（可用时给出各策略的可行运行区间）")
    w()
    w("| 策略 | 维度 | min | 均值 | max |")
    w("|---|---|---|---|---|")
    dims = None
    for r in rs:
        ev = (r["summary"].get("evaluation") or {})
        fd = ev.get("five_dimensions")
        if fd:
            dims = sorted(fd.keys())
            break
    if dims:
        for regime in REGIMES:
            for d in dims:
                vals = []
                for r in rs:
                    if r["meta"]["regime"] != regime:
                        continue
                    if not r["summary"]["survived"]:
                        continue
                    ev = (r["summary"].get("evaluation") or {})
                    fd = ev.get("five_dimensions") or {}
                    if isinstance(fd.get(d), (int, float)):
                        vals.append(float(fd[d]))
                if not vals:
                    continue
                w(f"| {regime} | {d} | {min(vals):.3f} | "
                  f"{np.mean(vals):.3f} | {max(vals):.3f} |")
                T[f"five_dim|{regime}|{d}"] = {
                    "min": min(vals), "mean": float(np.mean(vals)),
                    "max": max(vals), "n": len(vals)}


def do_gspeed():
    data = load("gspeed")
    if not data:
        return
    rs = ok_results(data)
    section("19. g_world × 制度能力（A 策略速度陷阱边界，每格 5 次）")
    for profile in ["manufacturing_giant", "post_soviet"]:
        agg = defaultdict(lambda: [0, 0])
        for r in rs:
            if r["meta"]["profile"] != profile:
                continue
            a = agg[(r["meta"]["g_world"], r["meta"]["institution_capacity"])]
            a[1] += 1
            if r["summary"].get("survived"):
                a[0] += 1
        if not agg:
            continue
        w()
        w(f"**{profile} × 策略 A**（存活次数/总次；每格 5 次）")
        w()
        insts = sorted({k[1] for k in agg})
        gws = sorted({k[0] for k in agg})
        w("| $g_{world}$ | " + " | ".join(f"inst={i:.2f}" for i in insts) + " |")
        w("|---" * (len(insts) + 1) + "|")
        for gw in gws:
            row = []
            for i in insts:
                s, n = agg.get((gw, i), [0, 0])
                row.append(f"{s}/{n}" if n else "-")
                T[f"gspeed|{profile}|{gw}|{i}"] = {"survived": s, "n": n}
            w(f"| {gw} | " + " | ".join(row) + " |")


# --------------------------------------------------------------------------

def main():
    global RUN_ROOT
    ap = argparse.ArgumentParser(description="Strict stage-1 postprocessing; received records remain unchanged")
    ap.add_argument("--runs", type=Path, default=RUN_ROOT)
    ap.add_argument("--out", type=Path, default=_PROJECT_ROOT.parent / "results")
    args = ap.parse_args()
    RUN_ROOT = args.runs
    L.clear()
    T.clear()
    for fn in [do_census, do_t80, do_threshold, do_gworld, do_sigma, do_eps_fine,
               do_wemp_eps, do_ki, do_mixture, do_costs, do_cost_lhs,
               do_calibration, do_exp1, do_exp2, do_exp3, do_exp4, do_exp5,
               do_exp6, do_extraction, do_gspeed, do_diag]:
        fn()

    args.out.mkdir(parents=True, exist_ok=True)
    out_md = args.out / "analysis_report.md"
    out_json = args.out / "tables.json"
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("# v3.8 实验结果重算报告\n\n")
        fh.write("\n".join(L) + "\n")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(T, fh, ensure_ascii=False, indent=2)
    print(f"\n\n写出: {out_md}")
    print(f"写出: {out_json}")


if __name__ == "__main__":
    main()
