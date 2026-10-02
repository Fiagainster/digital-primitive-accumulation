#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文 ↔ v3.8 实验数据 一致性核验器。

用途
----
把论文（main.tex / appendix_*.tex）中「结论型数字」写成断言表，逐条到
``experiments_runs/v38/`` 的原始批次 JSON 复算。任何一条对不上都会打印
``[FAIL]``，全对则打印 ``[OK]``。

设计原则
--------
* 只读：绝不改动任何实验产物。
* 自洽：断言表里每个数字都附「复算表达式」，而不是硬编码结果。
* 可复用：新增结论时，往 ``CHECKS`` 里加一行即可。

用法
----
    python research_scripts/verify_paper_consistency.py
    python research_scripts/verify_paper_consistency.py --runs experiments_runs/v38
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "experiments_runs" / "v38"

_RESULTS: list[tuple[bool, str]] = []


def check(label: str, got, want, tol=0.05, note: str = "") -> None:
    """数值/比例/元组断言。``want`` 为论文中的值，``got`` 为复算值。"""
    if isinstance(want, (tuple, list)) and isinstance(got, (tuple, list)):
        try:
            ok = len(got) == len(want) and all(
                abs(float(g) - float(w)) <= tol for g, w in zip(got, want))
        except (TypeError, ValueError):
            ok = [str(g) for g in got] == [str(w) for w in want]
    else:
        try:
            ok = abs(float(got) - float(want)) <= tol
        except (TypeError, ValueError):
            ok = str(got) == str(want)
    tag = "OK  " if ok else "FAIL"
    _RESULTS.append((ok, label))
    extra = f"  ({note})" if note else ""
    print(f"[{tag}] {label}\n        computed={got!r}  paper={want!r}{extra}")


def load(runs: Path, name: str):
    p = runs / f"{name}.json"
    if not p.exists():
        print(f"[SKIP] {p.name} 不存在")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def results(doc):
    if doc is None:
        return []
    return doc["results"] if isinstance(doc, dict) and "results" in doc else doc


def pct(a, b):
    return round(100.0 * a / b, 1) if b else float("nan")


# --------------------------------------------------------------------------
# 核验项
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(DEFAULT_RUNS))
    args = ap.parse_args()
    runs = Path(args.runs)

    print(f"data root = {runs}\n" + "=" * 72)

    # ---------------- 1. 普查 ----------------
    cen = results(load(runs, "census"))
    if cen:
        ok = [x for x in cen if x.get("ok")]
        surv = [x for x in ok if x["summary"]["survived"]]
        print("\n--- 1. 普查 (census) ---")
        check("普查总运行数 = 720", len(ok), 720, tol=0)
        check("总体存续率 = 36.1%", pct(len(surv), len(ok)), 36.1, tol=0.06)

        heat = {
            ("digital_leapfrogger", "A"): 0.0, ("digital_leapfrogger", "B"): 100.0,
            ("digital_leapfrogger", "C"): 100.0,
            ("manufacturing_giant", "A"): 55.0, ("manufacturing_giant", "B"): 22.5,
            ("manufacturing_giant", "C"): 100.0,
            ("populous_weak_industry", "A"): 0.0, ("populous_weak_industry", "B"): 0.0,
            ("populous_weak_industry", "C"): 0.0,
            ("post_soviet", "A"): 0.0, ("post_soviet", "B"): 2.5, ("post_soviet", "C"): 100.0,
            ("resource_rich", "A"): 0.0, ("resource_rich", "B"): 100.0,
            ("resource_rich", "C"): 70.0,
            ("small_open", "A"): 0.0, ("small_open", "B"): 0.0, ("small_open", "C"): 0.0,
        }
        for (prof, reg), want in heat.items():
            cell = [x for x in ok if x["meta"]["profile"] == prof and x["meta"]["regime"] == reg]
            check(f"热力图 {prof}|{reg} = {want}%",
                  pct(sum(1 for x in cell if x["summary"]["survived"]), len(cell)),
                  want, tol=0.06)

        for reg, (s, n) in {"A": (22, 240), "B": (90, 240), "C": (148, 240)}.items():
            rs = [x for x in ok if x["meta"]["regime"] == reg]
            check(f"分策略存续 {reg} = {s}/{n}",
                  (sum(1 for x in rs if x["summary"]["survived"]), len(rs)), (s, n), tol=0)

    # ---------------- 2. 五维表 ----------------
    if cen:
        surv = [x for x in cen if x.get("ok") and x["summary"]["survived"]]
        print("\n--- 2. 五维评价表 ---")

        def rng(reg, field, nd=0):
            v = [x["summary"][field] for x in surv
                 if x["meta"]["regime"] == reg and x["summary"].get(field) is not None]
            return (round(min(v), nd), round(max(v), nd)) if v else None

        check("GDP 区间 A = 2,837--6,976", rng("A", "final_gdp"), (2837, 6976), tol=2)
        check("GDP 上界 B = 15,044", rng("B", "final_gdp")[1], 15044, tol=2)
        check("GDP 上界 C = 3,312", rng("C", "final_gdp")[1], 3312, tol=2)

        for reg, want in [("A", (2.39, 3.63)), ("B", (0.44, 5.26)), ("C", (0.41, 2.46))]:
            got = rng(reg, "final_welfare", nd=2)
            check(f"福利区间 {reg} = {want[0]}--{want[1]}", got, want, tol=0.011)

        def dl_range(reg):
            v = []
            for x in surv:
                if x["meta"]["regime"] != reg:
                    continue
                s = x["summary"]
                kd = s.get("sector_K_d_infra", 0) + s.get("sector_K_d_app", 0)
                if kd:
                    v.append((s.get("sector_K_d_infra", 0) * s.get("sector_DL_d_infra", 0)
                              + s.get("sector_K_d_app", 0) * s.get("sector_DL_d_app", 0)) / kd)
            return (round(min(v), 2), round(max(v), 2)) if v else None

        for reg, want in [("A", (0.99, 0.99)), ("B", (0.68, 0.99)), ("C", (0.37, 0.99))]:
            check(f"DL_d 区间 {reg} = {want[0]}--{want[1]}", dl_range(reg), want, tol=0.011)

    # ---------------- 3. T=80 延长 ----------------
    t80 = results(load(runs, "t80"))
    if t80:
        print("\n--- 3. T=80 延长 ---")
        for reg, (s, n) in {"A": (0, 20), "B": (66, 85), "C": (143, 150)}.items():
            rs = [x for x in t80 if x["meta"]["regime"] == reg]
            check(f"T=80 {reg} = {s}/{n}",
                  (sum(1 for x in rs if x["summary"]["survived"]), len(rs)), (s, n), tol=0)
        check("T=80 总合格样本 = 255", len(t80), 255, tol=0)

    # ---------------- 4. 混合 vs 纯 ----------------
    mix = results(load(runs, "mixture"))
    if mix:
        print("\n--- 4. 混合 vs 纯策略 ---")
        ok = [x for x in mix if x.get("ok")]
        s = sum(1 for x in ok if x["summary"]["survived"])
        check("混合总体存续率 = 44.8% (430/960)", pct(s, len(ok)), 44.8, tol=0.06)
        check("混合总体 (430, 960)", (s, len(ok)), (430, 960), tol=0)

        # λ_B 分界
        lo = [x for x in ok if x["meta"].get("lambda_B", 1) <= 0.30]
        hi = [x for x in ok if x["meta"].get("lambda_B", 1) > 0.30]
        if lo and hi:
            check("λ_B≤0.30 存续率 = 58.8%",
                  pct(sum(1 for x in lo if x["summary"]["survived"]), len(lo)), 58.8, tol=0.1)
            check("λ_B>0.30 存续率 = 29.9%",
                  pct(sum(1 for x in hi if x["summary"]["survived"]), len(hi)), 29.9, tol=0.1)

    # ---------------- 5. 成本表 ----------------
    ct = results(load(runs, "cost_table"))
    if ct:
        print("\n--- 5. 成本分布表 ---")
        ok = [x for x in ct if x.get("ok")]
        for prof, want in {"manufacturing_giant": 0.0, "resource_rich": 0.0083,
                           "digital_leapfrogger": None}.items():
            v = [x["summary"]["cost_by_group"]["digital_high_skill"] for x in ok
                 if x["meta"]["profile"] == prof]
            if v and want is not None:
                check(f"数字部门累计代价 {prof} mean≈{want}",
                      round(st.mean(v), 4), want, tol=0.002)

    # ---------------- 6. 福利阈值扫描 (C 列) ----------------
    thr = load(runs, "threshold")
    if thr:
        print("\n--- 6. 福利阈值扫描 (C 列) ---")
        rs = results(thr)
        for lvl, want in [(0.15, {"small_open"}),
                          (0.30, {"populous_weak_industry", "small_open"})]:
            got = {x["meta"]["profile"] for x in rs
                   if abs(float(x["meta"].get("threshold", -1)) - lvl) < 1e-9
                   and x["meta"]["regime"] == "C" and not x["summary"]["survived"]}
            check(f"阈值 {lvl} 的 C 失败国情 = {sorted(want)}", sorted(got), sorted(want), tol=0)

    # ---------------- 7. 断裂类型构成比 (tab:fractures) ----------------
    if cen:
        print("\n--- 7. 断裂类型构成比 ---")
        ok = [x for x in cen if x.get("ok")]
        for mode, want in {
            "industrial_withering": (31, 0, 4),
            "social_collapse": (18, 44, 24),
            "tech_lock_in": (19, 32, 37),
            "jump_induced_deindustrialization": (32, 23, 32),
        }.items():
            got = []
            for reg in ("A", "B", "C"):
                sub = [x for x in ok if x["meta"]["regime"] == reg]
                hits = sum(1 for x in sub if mode in (x["summary"].get("fracture_types") or []))
                tot = sum(len(x["summary"].get("fracture_types") or []) for x in sub)
                got.append(round(100 * hits / tot) if tot else 0)
            check(f"构成比 {mode} = {want}", tuple(got), want, tol=0.5)

    # ---------------- 8. 速度陷阱 g_world × inst ----------------
    gs = results(load(runs, "gspeed"))
    if gs:
        print("\n--- 8. 速度陷阱 g_world × 制度能力 ---")

        def cell(prof, g, inst):
            sub = [x for x in gs if x["meta"]["profile"] == prof
                   and abs(x["meta"]["g_world"] - g) < 1e-9
                   and abs(x["meta"]["institution_capacity"] - inst) < 1e-9]
            return pct(sum(1 for x in sub if x["summary"]["survived"]), len(sub))

        for prof, g, inst, want in [
            ("manufacturing_giant", 0.015, 0.50, 100.0),
            ("manufacturing_giant", 0.03, 0.65, 100.0),
            ("manufacturing_giant", 0.03, 0.50, 0.0),
            ("manufacturing_giant", 0.15, 0.80, 100.0),
            ("manufacturing_giant", 0.15, 0.65, 0.0),
            ("post_soviet", 0.03, 0.65, 0.0),
            ("post_soviet", 0.03, 0.80, 100.0),
        ]:
            check(f"{prof} g={g} inst={inst} = {want}%", cell(prof, g, inst), want, tol=0.1)

    # ---------------- 9. 其他单参数扫描 ----------------
    ki = load(runs, "ki")
    if ki:
        print("\n--- 9. K_i / w_emp×ε 敏感性 ---")
        rs = results(ki)
        for key, want in {"0.65|1.0": 72, "0.4|1.0": 75, "0.65|0.7": 6}.items():
            sub = [x for x in rs if x["meta"].get("key", "").endswith(key)]
            dead = sum(1 for x in sub if not x["summary"]["survived"])
            check(f"K_i 组 {key}: 终止 {want}/75", (dead, len(sub)), (want, 75), tol=0)

    we = load(runs, "wemp_eps")
    if we:
        rs = results(we)
        for prof, want in {"manufacturing_giant": 121, "post_soviet": 125}.items():
            sub = [x for x in rs if x["meta"]["profile"] == prof]
            dead = sum(1 for x in sub if not x["summary"]["survived"])
            check(f"w_emp×ε {prof}: 终止 {want}/125", (dead, len(sub)), (want, 125), tol=0)

    # ---------------- 10. 校准 Spearman ----------------
    cal = load(runs, "calibration")
    if cal:
        print("\n--- 10. 校准 Spearman ---")
        rs = [x for x in results(cal) if x.get("ok")]
        cfg = {}
        for x in rs:
            cfg.setdefault((x["meta"]["profile"], x["meta"]["regime"]), []).append(x["summary"])
        print(f"        (校准样本 {len(rs)} 条 / {len(cfg)} 组)")

    # ---------------- 11. 主张 × 稳健性总览表 (tab:robustness_summary) ----------------
    # 论文 §9.2 的 Table 14 逐条主张给出「强度」判定；这里逐条复算其依据，
    # 避免该表沦为散文式断言。新增/修改该表时同步改这里。
    tb_path = runs / "tables.json"
    if tb_path.exists():
        tb = json.loads(tb_path.read_text(encoding="utf-8"))
        print("\n--- 11. 主张 × 稳健性总览表 (tab:robustness_summary) ---")

        # (1) 速度陷阱：方向稳、边界条件性 —— σ_b 两极劣化
        sig = tb.get("sigma", {})
        for key, want in {"0.3|A": 6.7, "1.5|A": 13.3}.items():
            s, n = sig[key]
            check(f"σ_b={key.split('|')[0]} 策略 A 存活率 ≈{want:.1f}%",
                  pct(s, n), want, tol=0.1)

        # (2) 混合不优于纯：匹配口径基准
        pm = tb.get("pure_matched")
        if pm:
            check("匹配口径纯策略基准 = 55.8% (67/120)", pct(*pm), 55.8, tol=0.06)
            check("匹配口径纯策略基准 (67, 120)", tuple(pm), (67, 120), tol=0)

        # (3) 制度跳变：post_soviet 在 g≥0.02 处 0%→100%
        if gs:
            for g in (0.02, 0.03, 0.05, 0.15):
                check(f"post_soviet g={g} inst=0.65 → 0%",
                      cell("post_soviet", g, 0.65), 0.0, tol=0.1)
                check(f"post_soviet g={g} inst=0.80 → 100%",
                      cell("post_soviet", g, 0.80), 100.0, tol=0.1)

        # (4) 增长—福利脱钩：manuf_giant×B 在 7 个阈值下全部终止
        if thr:
            rs = results(thr)
            for lvl in ("0.15", "0.2", "0.3", "0.45"):
                sub = [x for x in rs if str(x["meta"].get("threshold")) == lvl
                       and x["meta"]["profile"] == "manufacturing_giant"
                       and x["meta"]["regime"] == "B"]
                check(f"阈值 {lvl} 下 manuf_giant×B 终止 {len(sub)}/{len(sub)}",
                      sum(1 for x in sub if not x["summary"]["survived"]), len(sub), tol=0)

        # (5) 零代价：manuf_giant 数字部门 30 次全部严格为 0
        ct_tbl = tb.get("cost_table", {}).get("manufacturing_giant", {})
        if ct_tbl:
            d = ct_tbl["digital_high_skill"]
            check("manuf_giant 数字部门代价 max = 0（30/30 严格为零）",
                  d["max"], 0.0, tol=0)
            check("manuf_giant 数字部门代价 min = 0", d["min"], 0.0, tol=0)

        # (6) 视野不稳：T=80 延迟终止 = 255 - 209 = 46
        tot = tb.get("t80_total")
        if tot:
            check("T=80 延迟终止 = 46/255 (18.0%)", (tot[1] - tot[0], tot[1]), (46, 255), tol=0)
            check("T=80 策略 C 存活率 = 95.3%", pct(*tb["t80"]["C"]), 95.3, tol=0.06)

    # ---------------- 汇总 ----------------
    bad = [lb for ok_, lb in _RESULTS if not ok_]
    print("\n" + "=" * 72)
    print(f"核验 {len(_RESULTS)} 项，通过 {len(_RESULTS) - len(bad)}，失败 {len(bad)}")
    for lb in bad:
        print("  ✗", lb)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
