"""Build a traceable results ledger directly from the received v38 JSON records.

Does not run a simulation or overwrite the received archive. Record indices are
zero-based JSON pointers; every source file has a SHA256. Run with --help.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE / "runtime_packages"))
sys.path.insert(0, str(STAGE / "model"))
import numpy as np
from research_scripts.stage1_statistics import spearman_average, exact_spearman
from src.params.country_profiles import get_country_profile

GROUPS = ["digital_high_skill", "industrial_workers", "base_workers", "service_workers", "unemployed"]
FIELDS = ["final_gdp", "final_welfare", "survival_years", "final_tech_gap",
          "sector_K_i", "sector_K_d_infra", "sector_K_d_app", "cost_concentration",
          "peak_unemployment"]
LONG_BATCHES = {"t80", "calibration_t80", "exp4"}


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty table: {path}")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list,dict)) else v
                        for k,v in r.items()})


def refs(rs):
    return [r["_ref"] for r in rs]


def count(rs):
    return {"survived": sum(bool(r["summary"]["survived"]) for r in rs),
            "n_runs": len(rs), "sources": refs(rs)}


def means(rs):
    if not rs:
        raise ValueError("Empty aggregation")
    result = count(rs)
    for k in FIELDS:
        if all(k in r["summary"] for r in rs):
            v = np.array([r["summary"][k] for r in rs],float)
            if not np.isfinite(v).all():
                raise ValueError(f"Invalid {k}")
            result[k] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                         "sd": float(v.std(ddof=1)) if len(v)>1 else 0.0}
    if all("sector_DL_d_infra" in r["summary"] for r in rs):
        vals = []
        for r in rs:
            s = r["summary"]
            ki, ka = s["sector_K_d_infra"], s["sector_K_d_app"]
            vals.append((ki*s["sector_DL_d_infra"]+ka*s["sector_DL_d_app"])/(ki+ka))
        result["capital_weighted_DL_d"] = float(np.mean(vals))
    if all("cost_by_group" in r["summary"] for r in rs):
        result["cost_by_group"] = {}
        for g in GROUPS:
            v = np.array([r["summary"]["cost_by_group"][g] for r in rs],float)
            result["cost_by_group"][g] = {"mean": float(v.mean()), "min": float(v.min()),
                                        "max": float(v.max()), "nonzero_count": int((v!=0).sum())}
    return result


def config_key(r):
    m = r["meta"]
    if "key" in m:
        return m["key"]
    return "|".join(str(m.get(k,"benign_globalization" if k=="scenario" else ""))
                    for k in ("profile","scenario","regime"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=STAGE/"model/experiments_runs/v38")
    ap.add_argument("--out", type=Path, default=STAGE/"results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    batches, provenance = {}, {}
    for path in sorted(args.runs.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "results" not in data:
            continue
        rs = data["results"]
        if any(not r or not r.get("ok") for r in rs):
            raise ValueError(f"Failed/missing execution in {path}")
        for i,r in enumerate(rs):
            r["_ref"] = f"{path.name}#/results/{i}"
            for k in ("survived","survival_years","final_gdp","final_welfare"):
                if k not in r["summary"]:
                    raise KeyError(f"{r['_ref']}: summary.{k}")
        batches[path.stem] = rs
        provenance[path.name] = {"sha256": sha(path), "bytes": path.stat().st_size,
                                 "path_relative_to_stage": path.relative_to(STAGE).as_posix()}
    if len(batches)!=23:
        raise ValueError(f"Expected 23 batch files, got {len(batches)}")

    census = batches["census"]
    census_by_config = defaultdict(list)
    for r in census:
        census_by_config[config_key(r)].append(r)
    selected = {k for k,rs in census_by_config.items()
                if sum(bool(r["summary"]["survived"]) for r in rs)>=3}
    t80 = batches["t80"]
    if set(config_key(r) for r in t80)!=selected:
        raise ValueError("T80 selection is inconsistent with the census")
    for r in t80:
        old_seeds={x["seed"] for x in census_by_config[config_key(r)]}
        if r["seed"] in old_seeds:
            raise ValueError("T80 seed is not fresh")
    failures=[r for r in t80 if not r["summary"]["survived"]]
    failure_rows=[]
    for r in failures:
        y=r["summary"]["survival_years"]
        failure_rows.append({"source":r["_ref"],**r["meta"],"seed":r["seed"],
                             "termination_year":y,"phase":"early_before_50" if y<50 else "late_50_to_80",
                             "terminal_fracture":r["summary"]["terminal_fracture"]})
    write_csv(args.out/"t80_failures.csv",failure_rows)
    t80_info={"selection":"at least 3 of 5 census repetitions survived through T=50",
              "n_selected_configurations":len(selected),"fresh_seeds_checked":True,
              "paired_with_census":False,"total":count(t80),"by_strategy":{},"by_country_scenario":{}}
    for g in "ABC":
        rs=[r for r in t80 if r["meta"]["regime"]==g]
        fs=[r for r in rs if not r["summary"]["survived"]]
        t80_info["by_strategy"][g]={**count(rs),"early_failures":sum(r["summary"]["survival_years"]<50 for r in fs),
                                    "late_failures":sum(r["summary"]["survival_years"]>=50 for r in fs),
                                    "failure_year_min":min((r["summary"]["survival_years"] for r in fs),default=None),
                                    "failure_year_max":max((r["summary"]["survival_years"] for r in fs),default=None)}
    t80_info["early_failures"]=sum(r["summary"]["survival_years"]<50 for r in failures)
    t80_info["late_failures"]=len(failures)-t80_info["early_failures"]
    for k in sorted({config_key(r) for r in t80}):
        rs=[r for r in t80 if config_key(r)==k]
        t80_info["by_country_scenario"][k]=means(rs)

    duplicate_refs={}
    for r in batches["ts_viable"]:
        matches=[x for x in census if x["seed"]==r["seed"] and config_key(x)==config_key(r)]
        if len(matches)!=1:
            raise ValueError("Representative trajectory must match one census execution")
        x=matches[0]
        for f in ("final_gdp","final_welfare","survival_years"):
            if not np.isclose(x["summary"][f],r["summary"][f],rtol=1e-9,atol=1e-9):
                raise ValueError(f"Duplicate execution differs: {r['_ref']} {f}")
        duplicate_refs[r["_ref"]]=x["_ref"]

    rows,inventory=[],[]
    for tag,rs in batches.items():
        keys=Counter(config_key(r) for r in rs)
        inventory.append({"batch":tag,"n_executions":len(rs),"n_configuration_keys":len(keys),
                          "repetitions_per_key":sorted(set(keys.values())),
                          "horizon":80 if tag in LONG_BATCHES else 50,
                          "status":"exploratory_unrecoverable_parameters" if tag=="cost_lhs" else
                                   "reexecution_of_census_representatives" if tag=="ts_viable" else "traceable_archive",
                          "source_sha256":provenance[tag+".json"]["sha256"]})
        for r in rs:
            s=r["summary"]
            row={"batch":tag,"source":r["_ref"],"config_key":config_key(r),"seed":r["seed"],
                 "source_sha256":provenance[tag+".json"]["sha256"],
                 "formal_parameter_robustness_eligible":tag!="cost_lhs",
                 "duplicate_of":duplicate_refs.get(r["_ref"],""),**r["meta"],
                 **{k:s[k] for k in FIELDS if k in s},"survived":bool(s["survived"])}
            if "evaluation" in s:
                row["cdi"]=s["evaluation"]["cdi"]
                for k,v in s["evaluation"]["five_dimensions"].items():
                    row["evaluation_"+k]=v
            if "cost_by_group" in s:
                row.update({"cost_"+k:s["cost_by_group"][k] for k in GROUPS})
            rows.append(row)
    write_csv(args.out/"all_execution_records.csv",rows)
    write_csv(args.out/"batch_inventory.csv",inventory)

    calibration=batches["calibration"]
    cal_rows=[]
    for p,g in sorted({(r["meta"]["profile"],r["meta"]["regime"]) for r in calibration}):
        rs=[r for r in calibration if r["meta"]["profile"]==p and r["meta"]["regime"]==g]
        if len(rs)!=5:
            raise ValueError("Calibration requires five repetitions per configuration")
        cp=get_country_profile(p)
        row={"profile":p,"regime":g,"institution_capacity":cp.institution_capacity,
             "n_seed_repetitions":len(rs),"sources":refs(rs)}
        for k in ("final_gdp","final_welfare","survival_years"):
            row[k]=float(np.mean([r["summary"][k] for r in rs]))
        cal_rows.append(row)
    write_csv(args.out/"calibration_configuration_means.csv",cal_rows)
    pooled, exact={},{}
    for k in ("final_gdp","final_welfare","survival_years"):
        x=[r["institution_capacity"] for r in cal_rows]
        y=[r[k] for r in cal_rows]
        rho,p=spearman_average(x,y)
        pooled[k]={"rho":rho,"p_asymptotic_t_approximation":p,"n_configuration_means":18,
                   "unit":"configuration mean","rank_method":"average","sources":refs(calibration)}
        for g in "ABC":
            rs=[r for r in cal_rows if r["regime"]==g]
            obj=exact_spearman([r["institution_capacity"] for r in rs],[r[k] for r in rs])
            obj["profile_order"]=[r["profile"] for r in rs]
            obj["sources"]=[v for r in rs for v in r["sources"]]
            exact[g+"|"+k]=obj
    dump(args.out/"exact_permutation_tests.json",exact)

    sigma={}
    for tag in ("sigma","eps_fine"):
        rs=batches[tag]
        sigma[tag]={}
        for p,g,lv in sorted({(r["meta"]["profile"],r["meta"]["regime"],r["meta"]["sigma_b"]) for r in rs}):
            sub=[r for r in rs if (r["meta"]["profile"],r["meta"]["regime"],r["meta"]["sigma_b"])==(p,g,lv)]
            sigma[tag][f"{p}|{g}|{lv}"]=count(sub)
    sigma_survivors=[{"source":r["_ref"],"seed":r["seed"],**r["meta"]}
                     for tag in ("sigma","eps_fine") for r in batches[tag]
                     if r["meta"]["profile"]=="manufacturing_giant" and r["meta"]["regime"]=="B"
                     and r["summary"]["survived"]]
    write_csv(args.out/"sigma_B_manufacturing_exceptions.csv",sigma_survivors)

    diag={}
    for p,g,sc in sorted({(r["meta"]["profile"],r["meta"]["regime"],r["meta"]["scenario"]) for r in census}):
        sub=[r for r in census if (r["meta"]["profile"],r["meta"]["regime"],r["meta"]["scenario"])==(p,g,sc)]
        diag[f"{p}|{g}|{sc}"]=means(sub)
    cost={p:means([r for r in batches["cost_table"] if r["meta"]["profile"]==p])
          for p in sorted({r["meta"]["profile"] for r in batches["cost_table"]})}
    exp2={}
    for p,variant in sorted({(r["meta"]["profile"],r["meta"]["variant"]) for r in batches["exp2"]}):
        rs=[r for r in batches["exp2"] if r["meta"]["profile"]==p and r["meta"]["variant"]==variant]
        exp2[f"{p}|{variant}"]=means(rs)

    empkeys=["L_i","L_b","L_d_infra","L_d_app","L_s_high","L_s_low"]
    tsrows=[]
    for r in batches["ts_calibration"]:
        ts=r["timeseries"]
        den=sum(np.array(ts[k],float) for k in empkeys)
        share=np.array(ts["L_i"],float)/np.maximum(den,1e-300)
        g=np.array(ts["gdp"],float);t=np.array(ts["t"],float)
        mask=np.isfinite(g)&(g>0)&np.isfinite(t)
        valid=int(mask.sum())
        span=float(t[mask][-1]-t[mask][0]) if valid else 0.0
        eligible=valid>=5 and span>0
        peak=float(np.nanmax(share))
        deind=float((peak-share[-1])/peak*100)
        growth=float(((g[mask][-1]/g[mask][0])**(1/span)-1)*100) if eligible else None
        tsrows.append({"source":r["_ref"],**r["meta"],"seed":r["seed"],"valid_gdp_points":valid,
                       "recorded_span_years":span,"eligible_for_fig3":eligible,"deindustrialization_extent_pct":deind,
                       "full_recorded_trajectory_gdp_CAGR_pct":growth,
                       "growth_window":"first to last positive-GDP recorded observation; no event-conditioned window"})
    write_csv(args.out/"calibration_trajectory_metrics.csv",tsrows)
    eligible=[r for r in tsrows if r["eligible_for_fig3"]]
    xs=np.array([r["deindustrialization_extent_pct"] for r in eligible])
    ys=np.array([r["full_recorded_trajectory_gdp_CAGR_pct"] for r in eligible])
    median=float(np.median(xs))
    growth_info={"source_batch_executions":len(tsrows),"n_eligible_trajectories":len(eligible),
                 "n_excluded_short_trajectories":len(tsrows)-len(eligible),
                 "pearson_r":float(np.corrcoef(xs,ys)[0,1]),"median_deindustrialization_pct":median,
                 "above_median_growth_mean_pct":float(ys[xs>median].mean()),
                 "at_or_below_median_growth_mean_pct":float(ys[xs<=median].mean()),
                 "above_median_n":int((xs>median).sum()),"at_or_below_median_n":int((xs<=median).sum()),
                 "post_event_growth_computed":False,"sources":[r["source"] for r in eligible]}
    reps=[]
    for r in batches["ts_viable"]:
        ts=r["timeseries"]
        share=np.array(ts["L_i"],float)/sum(np.array(ts[k],float) for k in empkeys)
        peak=float(share.max())
        reps.append({"source":r["_ref"],**r["meta"],"seed":r["seed"],
                     "duplicate_census_execution":duplicate_refs[r["_ref"]],
                     "initial_recorded_share":float(share[0]),"peak_share":peak,
                     "terminal_share":float(share[-1]),"deindustrialization_extent_pct":float((peak-share[-1])/peak*100),
                     "monotone_nondecreasing_violations":int((np.diff(share)>1e-10).sum()),
                     "peak_time":float(np.array(ts["t"])[int(np.argmax(share))])})
    write_csv(args.out/"representative_trajectory_selection.csv",reps)

    tsdiag={}
    for p,g in sorted({(r["meta"]["profile"],r["meta"]["regime"]) for r in batches["ts_calibration"]}):
        rs=[r for r in batches["ts_calibration"] if r["meta"]["profile"]==p and r["meta"]["regime"]==g]
        item=means(rs)
        item["first_recorded_snapshot"]={}
        item["terminal_snapshot_mean"]={}
        for k in ("t","gdp","welfare_index","L_i","unemployment_rate"):
            item["first_recorded_snapshot"][k]=float(np.mean([r["timeseries"][k][0] for r in rs]))
            item["terminal_snapshot_mean"][k]=float(np.mean([r["timeseries"][k][-1] for r in rs]))
        item["endpoint_interpretation"]="mean of each run's own terminal snapshot, not a common-year trajectory mean"
        tsdiag[f"{p}|{g}"]=item

    aggregate={}
    for tag,rs in batches.items():
        if tag=="cost_lhs":
            aggregate[tag]={"status":"exploratory_only","n":len(rs),"actual_parameters_saved":False,
                            "formal_C_parameter_robustness_claim":False}
            continue
        aggregate[tag]={k:means([r for r in rs if config_key(r)==k]) for k in sorted({config_key(r) for r in rs})}
    n=sum(len(rs) for rs in batches.values())
    ledger={"version":"stage1_technical_revision","source_files":provenance,"inventory":inventory,
            "execution_counts":{"all":n,"exploratory_LHS":len(batches["cost_lhs"]),
                                "excluding_exploratory_LHS":n-len(batches["cost_lhs"]),
                                "census_representative_reexecutions":len(duplicate_refs),
                                "excluding_LHS_and_known_reexecutions":n-len(batches["cost_lhs"])-len(duplicate_refs),
                                "independent_sample_size_claim":False},
            "duplicate_execution_map":duplicate_refs,"census":count(census),"t80":t80_info,
            "pooled_calibration":pooled,"exact_permutation_file":"exact_permutation_tests.json",
            "sigma_cells":sigma,"sigma_B_manufacturing_exceptions":sigma_survivors,
            "diagnostic_census_means":diag,"cost_table":cost,"exp2_joint_adjustment":exp2,
            "growth_directional_check":growth_info,"representative_trajectories":reps,
            "diagnostic_timeseries_means":tsdiag,
            "configuration_results":aggregate,
            "verification_environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
                                        "is_original_experiment_environment":False}}
    dump(args.out/"unified_results.json",ledger)
    dump(args.out/"source_provenance.json",provenance)
    print(json.dumps({"counts":ledger["execution_counts"],"t80_early_late":[t80_info["early_failures"],t80_info["late_failures"]],
                      "pooled_rho":{k:v["rho"] for k,v in pooled.items()},"growth":growth_info},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
