"""Recompute stage-4 figure and table inputs from the frozen v38 records."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
STAGE = Path(__file__).resolve().parents[1]
FIRST = STAGE.parent / "阶段01"
RAW = FIRST / "model/experiments_runs/v38"
OUT = STAGE / "tables"
PROFILES = ["manufacturing_giant", "resource_rich", "digital_leapfrogger",
            "small_open", "post_soviet", "populous_weak_industry"]
EMP = ["L_b", "L_i", "L_d_infra", "L_d_app", "L_s_high", "L_s_low"]
GROUPS = ["digital_high_skill", "industrial_workers", "base_workers", "service_workers", "unemployed"]
MODES = ["industrial_withering", "social_collapse", "fiscal_collapse", "tech_lock_in",
         "dependency_trap", "data_colonialism", "jump_induced_deindustrialization",
         "human_capital_erosion", "premature_deindustrialization"]
METRICS = ["survival_years", "final_gdp", "final_welfare", "final_tech_gap", "cost_concentration"]


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_out(name, rows):
    assert rows
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def profile(r):
    return r["meta"].get("profile", "")


def regime(r):
    return r["summary"].get("regime", r["meta"].get("regime", ""))


def key(r):
    m = r["meta"]
    return m.get("key") or "|".join(str(m.get(k, "")) for k in ("profile", "scenario", "regime"))


def stats(rs):
    result = {"n_runs": len(rs), "n_survived": sum(bool(r["summary"]["survived"]) for r in rs)}
    result["viability_share"] = result["n_survived"] / len(rs)
    for metric in METRICS:
        values = [float(r["summary"][metric]) for r in rs
                  if r["summary"].get(metric) is not None and np.isfinite(r["summary"][metric])]
        if values:
            result[metric + "_mean"] = float(np.mean(values))
            result[metric + "_min"] = min(values)
            result[metric + "_max"] = max(values)
    dl = []
    for r in rs:
        s = r["summary"]
        names = ("sector_K_d_infra", "sector_K_d_app", "sector_DL_d_infra", "sector_DL_d_app")
        if all(k in s for k in names):
            ki, ka, di, da = [s[k] for k in names]
            dl.append((ki * di + ka * da) / max(ki + ka, 0.01))
    if dl:
        result["capital_weighted_DL_d_mean"] = float(np.mean(dl))
    result["source_records"] = ";".join(r["_source"] for r in rs)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    ledger = json.loads((FIRST / "results/unified_results.json").read_text(encoding="utf-8"))
    accepted = json.loads((FIRST / "results/tables.json").read_text(encoding="utf-8"))
    batches = {}
    provenance = {}
    all_rows, config_rows = [], []
    for item in ledger["inventory"]:
        name = item["batch"]
        path = RAW / (name + ".json")
        records = json.loads(path.read_text(encoding="utf-8"))["results"]
        assert len(records) == item["n_executions"] and all(r["ok"] for r in records)
        provenance[name] = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "n_runs": len(records)}
        grouped = defaultdict(list)
        for i, r in enumerate(records):
            r["_source"] = f"{name}.json#/results/{i}"
            r["_index"] = i
            row = {"batch": name, "source": r["_source"], "seed": r["seed"],
                   "configuration_key": key(r), "profile": profile(r), "regime": regime(r),
                   "metadata_json": json.dumps(r["meta"], sort_keys=True, ensure_ascii=False)}
            row.update({k: v for k, v in r["summary"].items() if not isinstance(v, (dict, list))})
            row.update({"cost_" + g: r["summary"].get("cost_by_group", {}).get(g) for g in GROUPS})
            all_rows.append(row)
            grouped[key(r)].append(r)
        if name != "cost_lhs":
            for k, rs in sorted(grouped.items()):
                m = rs[0]["meta"]
                config_rows.append({"batch": name, "configuration_key": k, "profile": profile(rs[0]),
                                    "regime": regime(rs[0]), "scenario": m.get("scenario", "benign_globalization"),
                                    "variant": m.get("variant", ""), **stats(rs)})
        batches[name] = records
    assert len(batches) == 23 and len(all_rows) == 5901
    csv_out("run_inventory.csv", all_rows)
    csv_out("configuration_endpoints.csv", config_rows)
    dump("source_provenance.json", provenance)

    cen = batches["census"]
    census_rows = []
    for p in PROFILES:
        for g in "ABC":
            rs = [r for r in cen if profile(r) == p and regime(r) == g]
            assert len(rs) == 40
            d = {"profile": p, "regime": g, **stats(rs)}
            assert d["viability_share"] == accepted["census_matrix"][p + "|" + g]
            census_rows.append(d)
    csv_out("census_profile_strategy.csv", census_rows)
    mode_rows = []
    for g in "ABC":
        rs = [r for r in cen if regime(r) == g]
        assert len(rs) == 240
        assert all(len(r["summary"]["fracture_types"]) == len(set(r["summary"]["fracture_types"])) for r in rs)
        events = sum(len(r["summary"]["fracture_types"]) for r in rs)
        for mode in MODES:
            found = [r for r in rs if mode in r["summary"]["fracture_types"]]
            coexist = sum(bool(r["summary"]["survived"]) for r in found)
            mode_rows.append({"regime": g, "mode": mode, "n_runs": 240, "runs_with_mode": len(found),
                              "run_prevalence": len(found) / 240, "event_instances_all_modes": events,
                              "event_composition_share": len(found) / events,
                              "viable_runs_with_mode": coexist,
                              "source_records": ";".join(r["_source"] for r in found)})
    csv_out("failure_prevalence.csv", mode_rows)

    selected = defaultdict(list)
    for r in cen:
        selected[(profile(r), r["meta"]["scenario"], regime(r))].append(r)
    eligible = {k for k, rs in selected.items() if sum(bool(r["summary"]["survived"]) for r in rs) >= 3}
    assert len(eligible) == 51
    t80_rows, t80_runs = [], []
    for g in "ABC":
        rs = [r for r in batches["t80"] if regime(r) == g]
        early = sum(not r["summary"]["survived"] and r["summary"]["survival_years"] < 50 for r in rs)
        late = sum(not r["summary"]["survived"] and r["summary"]["survival_years"] >= 50 for r in rs)
        t80_rows.append({"regime": g, "n_runs": len(rs), "viable_at_80": sum(bool(r["summary"]["survived"]) for r in rs),
                         "fail_before_50": early, "fail_at_50_to_80": late})
        for r in rs:
            k = (profile(r), r["meta"]["scenario"], regime(r))
            assert k in eligible and r["seed"] not in {c["seed"] for c in selected[k]}
            t80_runs.append({"source": r["_source"], "seed": r["seed"], "profile": profile(r),
                             "scenario": r["meta"]["scenario"], "regime": g, "survived": r["summary"]["survived"],
                             "recorded_end": r["summary"]["survival_years"],
                             "window": "viable at 80" if r["summary"]["survived"] else ("before 50" if r["summary"]["survival_years"] < 50 else "50 to 80")})
    assert sum(r["viable_at_80"] for r in t80_rows) == 209
    assert sum(r["fail_before_50"] for r in t80_rows) == 8 and sum(r["fail_at_50_to_80"] for r in t80_rows) == 38
    csv_out("t80_failure_windows.csv", t80_rows)
    csv_out("t80_run_windows.csv", t80_runs)

    speed_rows = []
    for p in ("manufacturing_giant", "post_soviet"):
        rs = [r for r in batches["gspeed"] if profile(r) == p]
        for growth in sorted({r["meta"]["g_world"] for r in rs}):
            for inst in sorted({r["meta"]["institution_capacity"] for r in rs}):
                cell = [r for r in rs if r["meta"]["g_world"] == growth and r["meta"]["institution_capacity"] == inst]
                assert len(cell) == 5
                speed_rows.append({"profile": p, "g_world": growth, "institution_capacity": inst, **stats(cell)})
    csv_out("A_growth_capacity_grid.csv", speed_rows)
    c_cases = []
    for p, sc in (("manufacturing_giant", "benign_globalization"), ("resource_rich", "benign_globalization"), ("resource_rich", "full_containment")):
        rs = [r for r in cen if profile(r) == p and regime(r) == "C" and r["meta"]["scenario"] == sc]
        d = {"profile": p, "scenario": sc, **stats(rs)}
        expected = ledger["diagnostic_census_means"][p + "|C|" + sc]
        for k in ("final_gdp", "final_welfare", "final_tech_gap"):
            assert np.isclose(d[k + "_mean"], expected[k]["mean"], rtol=1e-12, atol=1e-12)
        assert np.isclose(d["capital_weighted_DL_d_mean"], expected["capital_weighted_DL_d"])
        c_cases.append(d)
    csv_out("C_selected_cases.csv", c_cases)

    rs_b = [r for r in batches["ts_calibration"] if profile(r) == "manufacturing_giant" and regime(r) == "B"]
    assert len(rs_b) == 5 and [r["_index"] for r in rs_b] == list(range(5, 10))
    b_rows, b_endpoints = [], []
    for r in rs_b:
        ts = r["timeseries"]
        n = len(ts["t"])
        assert all(len(v) == n for v in ts.values())
        for j in range(n):
            b_rows.append({"source": r["_source"] + "/timeseries", "seed": r["seed"], "rep": r["meta"]["rep"],
                           "snapshot": j, **{k: v[j] for k, v in ts.items()}})
        b_endpoints.append({"source": r["_source"], "seed": r["seed"], "rep": r["meta"]["rep"],
                            "recorded_rows": n, **r["summary"],
                            **{k + "_first": ts[k][0] for k in ("t", "gdp", "welfare_index", "L_i", "unemployment_rate")},
                            **{k + "_last": ts[k][-1] for k in ("t", "gdp", "welfare_index", "L_i", "unemployment_rate")}})
    for row in b_endpoints:
        row.pop("cost_by_group", None)
    csv_out("B_diagnostic_timeseries.csv", b_rows)
    csv_out("B_diagnostic_endpoints.csv", b_endpoints)
    b_summary = stats(rs_b)
    b_summary["snapshot_means"] = {k: {"first": float(np.mean([r["timeseries"][k][0] for r in rs_b])),
                                      "last": float(np.mean([r["timeseries"][k][-1] for r in rs_b]))}
                                   for k in ("t", "gdp", "welfare_index", "L_i", "unemployment_rate")}
    cost_rows = []
    for r in batches["cost_table"]:
        for group in GROUPS:
            cost_rows.append({"source": r["_source"], "profile": profile(r), "seed": r["seed"], "group": group,
                              "modeled_cost": r["summary"]["cost_by_group"][group],
                              "recorded_end": r["summary"]["survival_years"], "survived": r["summary"]["survived"]})
    csv_out("B_group_costs.csv", cost_rows)

    with (STAGE.parent / "阶段03/tables/mixture_drawn_and_effective_weights.csv").open(encoding="utf-8-sig") as f:
        weights = list(csv.DictReader(f))
    mixture = batches["mixture"]
    for row in weights:
        rs = [r for r in mixture if profile(r) == row["profile"] and r["meta"]["scenario"] == row["scenario"] and r["meta"]["idx"] == int(row["draw_index"])]
        assert len(rs) == 3
        assert all(np.allclose([float(row["drawn_lambda_" + k]) for k in "ABC"], [r["meta"]["lambda_" + k] for k in "ABC"], rtol=0, atol=1e-12) for r in rs)
        row.update(stats(rs))
    csv_out("mixture_inputs_and_outcomes.csv", weights)
    frame_profiles = ["manufacturing_giant", "resource_rich", "digital_leapfrogger", "post_soviet"]
    frame_scenarios = ["benign_globalization", "tech_cold_war"]
    matched_rows = []
    for p in frame_profiles:
        pure = [r for r in cen if profile(r) == p and r["meta"]["scenario"] in frame_scenarios]
        mixed = [r for r in mixture if profile(r) == p]
        assert len(pure) == 30 and len(mixed) == 240
        for design, rs in (("pure", pure), ("mixture", mixed)):
            matched_rows.append({"profile": p, "design": design, **stats(rs)})
    assert sum(r["n_survived"] for r in matched_rows if r["design"] == "pure") == 67
    assert sum(r["n_survived"] for r in matched_rows if r["design"] == "mixture") == 430
    csv_out("matched_frame_viability.csv", matched_rows)

    initial = json.loads((STAGE.parent / "阶段03/tables/profile_initial_conditions.json").read_text(encoding="utf-8"))
    cal_rows = []
    for p in PROFILES:
        for g in "ABC":
            rs = [r for r in batches["calibration"] if profile(r) == p and regime(r) == g]
            assert len(rs) == 5
            cal_rows.append({"profile": p, "regime": g, "institution_capacity": initial[p]["institution_capacity"], **stats(rs)})
    csv_out("directional_configuration_means.csv", cal_rows)
    growth_rows, representative_rows = [], []
    for tag, rs in (("ts_calibration", batches["ts_calibration"]), ("ts_viable", batches["ts_viable"])):
        for r in rs:
            ts = r["timeseries"]
            total = sum(np.asarray(ts[k], dtype=float) for k in EMP)
            share = np.asarray(ts["L_i"], dtype=float) / total
            gdps = np.asarray(ts["gdp"], dtype=float)
            times = np.asarray(ts["t"], dtype=float)
            mask = np.isfinite(gdps) & (gdps > 0)
            n = int(mask.sum())
            span = float(times[mask][-1] - times[mask][0]) if n >= 2 else 0.0
            eligible = n >= 5 and span > 0
            peak = float(np.nanmax(share))
            deind = float((peak - share[-1]) / peak * 100)
            growth = float(((gdps[mask][-1] / gdps[mask][0]) ** (1 / span) - 1) * 100) if eligible else None
            if tag == "ts_calibration":
                growth_rows.append({"source": r["_source"], "profile": profile(r), "regime": regime(r), "seed": r["seed"],
                                    "valid_points": n, "recorded_span": span, "eligible": eligible,
                                    "deindustrialization_extent_pct": deind, "full_trajectory_gdp_growth_pct": growth})
            else:
                for j in range(len(times)):
                    representative_rows.append({"source": r["_source"], "profile": profile(r), "regime": regime(r),
                                                "seed": r["seed"], "scenario": r["meta"].get("scenario", "benign_globalization"),
                                                "snapshot": j, "t": times[j],
                                                "gdp_per_worker": gdps[j] / total[j],
                                                "manufacturing_share_pct": share[j] * 100})
    eligible_rows = [r for r in growth_rows if r["eligible"]]
    assert len(eligible_rows) == 60
    rho = float(np.corrcoef([r["deindustrialization_extent_pct"] for r in eligible_rows],
                            [r["full_trajectory_gdp_growth_pct"] for r in eligible_rows])[0, 1])
    assert np.isclose(rho, ledger["growth_directional_check"]["pearson_r"], rtol=1e-12)
    csv_out("directional_trajectory_metrics.csv", growth_rows)
    csv_out("selected_representative_paths.csv", representative_rows)
    report = {"archive_batches": 23, "archive_executions": 5901, "census_survived": 260, "census_n": 720,
              "census_by_strategy": {g: [sum(r["n_survived"] for r in census_rows if r["regime"] == g), 240] for g in "ABC"},
              "t80": t80_rows, "B_diagnostic": b_summary, "C_cases": c_cases,
              "mixture_original_draws": len(weights),
              "mixture_corrected_draws": sum(r["changed_by_constraint"] == "True" for r in weights),
              "mixture_unique_accumulation_weights_12dp": len({tuple(round(float(r["accumulation_lambda_" + k]), 12) for k in "ABC") for r in weights}),
              "matched_frame": matched_rows, "growth_directional_n": 60, "growth_directional_r": rho,
              "interpretation": "Descriptive archived outcomes and code-defined diagnostics; no new simulation runs, pooled independent-N claim, or isolated causal effects."}
    dump("evidence_summary.json", report)
    print(json.dumps({k: report[k] for k in ("archive_executions", "census_by_strategy", "t80", "mixture_corrected_draws", "growth_directional_r")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
