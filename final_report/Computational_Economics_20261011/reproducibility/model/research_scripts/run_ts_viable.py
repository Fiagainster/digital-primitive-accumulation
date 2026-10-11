"""存活轨迹批次（供校准类图件使用）。

与 ``run_timeseries_batch.py``（固定 18 组合）不同，本脚本从普查结果
``census.json`` 中挑出**真正存活**的配置，用与普查相同的种子重跑，
从而得到与正文存活率矩阵口径一致的 "viable trajectories"。

产出：``experiments_runs/v38/ts_viable.json``
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RUN_ROOT = _PROJECT_ROOT / "experiments_runs" / "v38"

# 绘图所需列（其余全部丢弃，避免体积爆炸）
COLS = ["t", "gdp", "welfare_index", "gini", "unemployment_rate",
        "L_i", "L_b", "L_d_infra", "L_d_app", "L_s_high", "L_s_low",
        "Y_i", "Y_b", "Y_d_infra", "Y_d_app", "Y_s_high", "Y_s_low",
        "eval_cdi", "eval_livelihood", "eval_economy",
        "hh_digital_cost", "hh_industrial_cost", "hh_base_cost",
        "hh_service_cost", "hh_unemployed_cost",
        "active_fractures_count", "employment_quality_index",
        "dep_composite", "tech_gap", "tech_gap_ratio"]


def _run(args: dict) -> dict:
    from src.core.engine import SimulationEngine
    from src.core.config import SimulationConfig
    from src.params.schema import get_schema

    s = get_schema()
    p = s.build_from_profile(profile_name=args["profile"], regime=args["regime"],
                             scenario=args["scenario"])
    if args.get("overrides"):
        p.update(args["overrides"])
    e = SimulationEngine(params=p,
                         config=SimulationConfig(T_max=50.0, verbose=False,
                                                 random_seed=args["seed"]),
                         regime=args["regime"])
    r = e.run()
    df = r["dataframe"]
    ts = {}
    for c in COLS:
        if c in df.columns:
            vals = df[c].to_numpy(dtype=float)
            ts[c] = [None if not np.isfinite(v) else round(float(v), 6) for v in vals]
    return {"meta": {k: args[k] for k in ("profile", "regime", "scenario", "rep")},
            "seed": args["seed"], "ok": True,
            "summary": {
                "survived": bool(r["summary"]["survived"]),
                "survival_years": float(r["summary"]["survival_years"]),
                "final_gdp": float(r["summary"]["final_gdp"]),
                "final_welfare": float(r["summary"].get("final_welfare") or 0.0),
                "cost_by_group": {k: float(v) for k, v in
                                  (r["summary"].get("cost_by_group") or {}).items()},
            },
            "timeseries": ts}


def _pick_viable():
    """从普查中为每个 (profile, regime) 挑一条存活的代表轨迹。

    优先 ``benign_globalization``（与校准节口径一致）；同格多条时取 rep 最小者。
    """
    with open(RUN_ROOT / "census.json", encoding="utf-8") as fh:
        census = json.load(fh)
    rs = [x for x in census["results"] if x and x.get("ok")
          and x["summary"].get("survived")]
    picked = {}
    for r in rs:
        m = r["meta"]
        key = (m["profile"], m["regime"])
        rank = (0 if m.get("scenario") == "benign_globalization" else 1,
                m.get("rep", 0))
        if key not in picked or rank < picked[key][0]:
            picked[key] = (rank, m, r.get("seed"))
    tasks = []
    for key, (_rank, m, seed) in sorted(picked.items()):
        tasks.append({"profile": m["profile"], "regime": m["regime"],
                      "scenario": m.get("scenario", "benign_globalization"),
                      "rep": m.get("rep", 0),
                      "seed": seed,
                      "overrides": m.get("overrides")})
    return tasks


def main():
    tasks = _pick_viable()
    out = RUN_ROOT / "ts_viable.json"
    print(f"[run ] ts_viable: {len(tasks)} 条存活轨迹 ...", flush=True)
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_run, t): i for i, t in enumerate(tasks)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()

    n_ok = sum(1 for r in results if r and r["summary"]["survived"])
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"tag": "ts_viable", "results": results}, fh, ensure_ascii=False)
    print(f"[done] -> {out}  ({out.stat().st_size/1e6:.1f} MB)  存活 {n_ok}/{len(tasks)}")


if __name__ == "__main__":
    main()
