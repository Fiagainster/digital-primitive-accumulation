"""时间序列批次（供校准类图件使用）。

只抽取绘图真正需要的列，避免完整 dataframe（百余列）把落盘体积放大数十倍。

产出：``experiments_runs/v38/ts_calibration.json``
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

MASTER_SEED = 20261002
RUN_ROOT = _PROJECT_ROOT / "experiments_runs" / "v38"

PROFILES = ["manufacturing_giant", "resource_rich", "digital_leapfrogger",
            "small_open", "post_soviet", "populous_weak_industry"]
REGIMES = ["A", "B", "C"]
REPS = 5

# 绘图所需列（其余全部丢弃）
COLS = ["t", "gdp", "welfare_index", "gini", "unemployment_rate",
        "L_i", "L_b", "L_d_infra", "L_d_app", "L_s_high", "L_s_low",
        "Y_i", "Y_b", "Y_d_infra", "Y_d_app", "Y_s_high", "Y_s_low",
        "eval_cdi", "eval_livelihood", "eval_economy",
        "hh_digital_cost", "hh_industrial_cost", "hh_base_cost",
        "hh_service_cost", "hh_unemployed_cost",
        "active_fractures_count", "employment_quality_index",
        "dep_composite", "tech_gap"]


def _seed(*parts):
    import hashlib
    d = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(d[:4], "big")


def _run(args: dict) -> dict:
    from src.core.engine import SimulationEngine
    from src.core.config import SimulationConfig
    from src.params.schema import get_schema

    s = get_schema()
    p = s.build_from_profile(profile_name=args["profile"], regime=args["regime"],
                             scenario="benign_globalization")
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
    return {"meta": {k: args[k] for k in ("profile", "regime", "rep")},
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


def main():
    tasks = []
    for profile in PROFILES:
        for regime in REGIMES:
            for rep in range(REPS):
                tasks.append({"profile": profile, "regime": regime, "rep": rep,
                              "seed": _seed("calib_ts", profile, regime, rep)})

    out = RUN_ROOT / "ts_calibration.json"
    print(f"[run ] ts_calibration: {len(tasks)} 次 ...", flush=True)
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_run, t): i for i, t in enumerate(tasks)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()

    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"tag": "ts_calibration", "results": results}, fh,
                  ensure_ascii=False)
    print(f"[done] -> {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
