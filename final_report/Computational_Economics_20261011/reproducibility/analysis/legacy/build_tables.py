"""Write paper tables from audited archived-record aggregations."""
from pathlib import Path
from collections import defaultdict
import csv
import json

STAGE = Path(__file__).resolve().parents[1]
OUT = STAGE / "manuscript"
TABLES = STAGE / "tables"
PN = {"manufacturing_giant":"Manufacturing giant", "resource_rich":"Resource-rich",
      "digital_leapfrogger":"Digital leapfrogger", "small_open":"Small open",
      "post_soviet":"Post-Soviet", "populous_weak_industry":"Populous weakly industrialized"}


def read(name):
    with (TABLES / name).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def esc(value):
    return str(value).replace("_",r"\_").replace("%",r"\%").replace("&",r"\&")


def num(v, digits=1):
    return f"{float(v):,.{digits}f}"


def ratio(rows):
    return f"{sum(r['survived'] in ('1','True') for r in rows)}/{len(rows)}"


def table(stem, caption, label, spec, headers, rows, note):
    lines = [r"\begin{table}[tb]",r"\centering",r"\caption{"+caption+"}",r"\label{"+label+"}",r"\small",
             r"\begin{tabularx}{\textwidth}{@{}"+spec+r"@{}}",r"\toprule",
             " & ".join(headers)+r" \\",r"\midrule"]
    lines += [" & ".join(str(v) for v in row)+r" \\" for row in rows]
    lines += [r"\bottomrule",r"\end{tabularx}",r"\par\vspace{3pt}{\footnotesize "+note+"}",r"\end{table}",""]
    (OUT / (stem+".tex")).write_text("\n".join(lines),encoding="utf-8")


def main():
    evidence = json.loads((TABLES/"evidence_summary.json").read_text(encoding="utf-8"))
    runs = read("run_inventory.csv")
    configs = read("configuration_endpoints.csv")
    cen = read("census_profile_strategy.csv")
    rows = []
    for p in PN:
        cells = [next(r for r in cen if r["profile"]==p and r["regime"]==g) for g in "ABC"]
        rows.append([PN[p]]+[f"{float(r['viability_share'])*100:.1f}\\% ({r['n_survived']}/40)" for r in cells])
    table("stage4_census_table","Census numerical viability shares","tab:survival","Xrrr",
          ["Profile","A","B","C"],rows,"Each cell: eight scenarios times five seeds, nominal T=50. Overall: 260/720 (36.1\\%). The strategy totals are 22/240, 90/240 and 148/240. These are enumeration shares.")
    modes = read("failure_prevalence.csv")
    labels = {"industrial_withering":"Industrial withering","social_collapse":"Welfare collapse","fiscal_collapse":"Fiscal collapse",
              "tech_lock_in":"Technology lock-in","dependency_trap":"Dependency trap","data_colonialism":"Data colonialism",
              "jump_induced_deindustrialization":"Abrupt deindustrialization","human_capital_erosion":"Human-capital erosion",
              "premature_deindustrialization":"Premature deindustrialization"}
    rows = []
    for mode, label in labels.items():
        cells = [next(r for r in modes if r["mode"]==mode and r["regime"]==g) for g in "ABC"]
        rows.append([label]+[f"{r['runs_with_mode']} ({float(r['run_prevalence'])*100:.1f}\\%)" for r in cells]+["Yes" if mode in ("industrial_withering","social_collapse","fiscal_collapse") else "No"])
    table("stage4_modes_table","Recorded-mode prevalence: count and percentage of executions","tab:fractures","Xrrrc",
          ["Mode","A","B","C","Terminal rule"],rows,"Denominator: 240 executions per strategy; multiple modes may occur in a run. All nine monitored modes are retained, including zero-count modes. Event composition has a different denominator.")
    table("stage4_t80_table","Selected-configurations re-examination with fresh seeds","tab:t80_comparison","Xrrr",
          ["Strategy","Viable at 80","Failure before 50","Failure at 50--80"],
          [[r["regime"],f"{r['viable_at_80']}/{r['n_runs']}",r["fail_before_50"],r["fail_at_50_to_80"]] for r in evidence["t80"]],
          "Selection: at least 3/5 census survivors; five new seeds per configuration, initialized from the starting state. Total: 209/255 viable, eight early and 38 later failures. No paired continuation.")
    b = evidence["B_diagnostic"]["snapshot_means"]
    table("stage4_B_endpoint_table","Five diagnostic B trajectories: first and own terminal snapshots","tab:manuf_giant_b","Xrr",
          ["Variable","Mean first recorded","Mean own terminal"],
          [["Recorded time",num(b["t"]["first"],2),num(b["t"]["last"],2)],
           ["GDP (model units)",num(b["gdp"]["first"]),num(b["gdp"]["last"])],
           ["Aggregate welfare",num(b["welfare_index"]["first"],3),num(b["welfare_index"]["last"],3)],
           ["Industrial labor stock",num(b["L_i"]["first"],2),num(b["L_i"]["last"],2)],
           ["Recorded unemployment (\\%)",num(b["unemployment_rate"]["first"]*100,3),num(b["unemployment_rate"]["last"]*100,3)]],
          "Five manufacturing-giant B seeds from ts\\_calibration under benign globalization; 0/5 survive. First states already contain an update. Terminal times range 26.5--44.4; means are not evaluated at a common year.")
    cases = evidence["C_cases"]
    table("stage4_C_case_table","Three illustrative C settings at their own endpoints","tab:c_three_faces","Xrrrrr",
          ["Setting","Viable","GDP","Welfare","Digital DL","Tech. gap"],
          [[PN[r["profile"]]+(" / benign" if r["scenario"]=="benign_globalization" else " / containment"),
            f"{r['n_survived']}/{r['n_runs']}",num(r["final_gdp_mean"]),num(r["final_welfare_mean"],3),
            num(r["capital_weighted_DL_d_mean"],3),num(r["final_tech_gap_mean"])] for r in cases],
          "Five seeds per setting. All endpoint means include failed runs; mean recorded times are 49.97, 49.98 and 48.50 respectively. Digital DL is capital-weighted digitalization, not an autonomy or import-independence measure.")
    nonlinear = [r for r in configs if r["batch"]=="exp1" and r["profile"]=="manufacturing_giant" and r["scenario"]=="benign_globalization"]
    rows = []
    for g in (.01,.02,.03,.05,.08,.10):
        cells = [next(r for r in nonlinear if r["configuration_key"]==f"manufacturing_giant|benign_globalization|{g}|{m}") for m in ("linear","logistic","piecewise")]
        rows.append([num(g,3)]+[f"{r['n_survived']}/{r['n_runs']}" for r in cells])
    table("stage4_nonlinear_table","Catch-up modes: manufacturing giant, A, benign globalization","tab:nonlinear_catchup","Xrrr",
          ["Assigned frontier growth","Linear","Logistic","Piecewise"],rows,
          "Three seeds per cell; this is 54 executions within the wider 216-run design. Capacity and all other profile inputs remain at their assigned values in this subset.")
    rows = []
    for floor in (.15,.20,.25,.30,.35,.40,.45):
        rs = [r for r in runs if r["batch"]=="threshold" and float(json.loads(r["metadata_json"])["threshold"])==floor]
        rows.append([num(floor,2)]+[ratio([r for r in rs if r["regime"]==g]) for g in "ABC"]+[ratio([r for r in rs if r["regime"]=="C" and r["profile"]=="populous_weak_industry"])])
    table("stage4_floor_table","Absolute welfare-floor scan under benign globalization","tab:b1_threshold","Xrrrr",
          ["Welfare floor","A","B","C","Populous-weak C"],rows,
          "Six profiles, three pure rules and five seeds per cell; 90 runs per floor, 630 overall. The last column is one five-seed setting. Changes in a termination floor are not changes to an empirical poverty line.")
    rows = []
    for tag, label, ps in [("sigma","Coarse substitution",["manufacturing_giant","resource_rich","digital_leapfrogger"]),
                          ("eps_fine","Fine substitution",["manufacturing_giant","digital_leapfrogger"]),
                          ("wemp_eps","Employment interaction",["manufacturing_giant","resource_rich","digital_leapfrogger","post_soviet"])]:
        for p in ps:
            rs = [r for r in runs if r["batch"]==tag and r["profile"]==p and r["regime"]=="B"]
            rows.append([label,PN[p],ratio(rs)])
    table("stage4_sigma_table","B outcomes in three distinct substitution-related scans","tab:b4_interaction","XXr",
          ["Design","Profile","Viable / executions"],rows,
          "Coarse B: 13 levels times five seeds per profile; fine B: 13 levels times five seeds; interaction: five substitution values times five employment exponents times five seeds. The coarse archive also retains A and C (585 runs overall).")
    rows = []
    for inst, skill in ((.65,1.0),(.65,.7),(.4,1.0)):
        rs = [r for r in runs if r["batch"]=="ki" and float(json.loads(r["metadata_json"])["institution_capacity"])==inst and float(json.loads(r["metadata_json"])["H_i"])==skill]
        vals = sorted({json.loads(r["metadata_json"])["K_i"] for r in rs})
        rows.append([num(inst,2),num(skill,2),f"{min(vals)}--{max(vals)}",ratio(rs)])
    table("stage4_ki_table","B industrial-capital grid under three capacity/skill assignments","tab:b5_ki","XXrr",
          ["Capacity","Industrial skill","Capital range","Viable / runs"],rows,
          "15 capital values and five seeds per cell, 75 runs per assignment. Initial industrial labor is independently held fixed. Complete cells and their actual capital values are in configuration\\_endpoints.csv.")
    rows = [r for r in configs if r["batch"]=="exp2"]
    table("stage4_joint_table","Joint B allocation and growth-target adjustment","tab:b_counterfactual","Xrrrr",
          ["Profile / variant","Viable","GDP","Welfare","Mean end"],
          [[PN[r["profile"]]+" / "+("joint change" if r["variant"]=="redistribute" else "baseline"),f"{r['n_survived']}/{r['n_runs']}",
            num(r["final_gdp_mean"]),num(r["final_welfare_mean"],3),num(r["survival_years_mean"],2)] for r in rows],
          "Three seeds per cell. The joint variant changes digital allocation 0.60 to 0.30 and growth target 0.25 to 0.15 together. Means use each run's endpoint; variants are not seed-paired.")
    rows = []
    for target in (.1,.15,.2,.25,.3,.35,.4):
        cells = [r for r in configs if r["batch"]=="extraction" and float(r["configuration_key"].split("|")[-1])==target]
        rows.append([num(target,2)]+[f"{(r:=next(r for r in cells if r['profile']==p))['n_survived']}/{r['n_runs']}" for p in ("manufacturing_giant","resource_rich")])
    table("stage4_target_table","B digital growth-target scan","tab:b_growth_target","Xrr",
          ["Assigned target","Manufacturing giant","Resource-rich"],rows,
          "Five seeds per cell, benign globalization, 70 runs. The target enters a state-dependent intensity formula; it is not an imposed realized extraction rate.")
    rows = [r for r in configs if r["batch"]=="exp5"]
    table("stage4_high_capacity_table","Selected high-capacity tests","tab:high_inst_counterfactual","Xrrrr",
          ["Profile","Rule","Capacity","Viable","Mean end"],
          [[PN[r["profile"]],r["regime"],num(r["configuration_key"].split("|")[-1],2),f"{r['n_survived']}/{r['n_runs']}",num(r["survival_years_mean"],2)] for r in rows],
          "Three seeds per setting; nine runs overall. Other assigned profile and scenario inputs remain in place. These tests do not identify a universal minimum capital requirement.")
    rows = [r for r in configs if r["batch"]=="exp6"]
    table("stage4_weight_table","B welfare-evaluation variants: manufacturing giant, benign globalization","tab:wemp_norm","Xrrr",
          ["Variant","Viable","Mean welfare","Mean end"],
          [[("$\\varepsilon="+r["configuration_key"].split("|")[-1]+"$" if r["configuration_key"].startswith("eps|") else "$w_{emp}="+r["configuration_key"].split("|")[-1]+"$"),
            f"{r['n_survived']}/{r['n_runs']}",num(r["final_welfare_mean"],3),num(r["survival_years_mean"],2)] for r in rows],
          "Three seeds per variant; 30 runs. Employment-exponent changes retain the consumption and skill powers without renormalization. Welfare magnitudes therefore need not have the same cardinal scale.")
    rows = [r for r in configs if r["batch"]=="exp3"]
    table("stage4_selected_mixture_table","Selected mixture and capacity diagnostics","tab:selected_mixture","Xrrr",
          ["Retained variant key","Viable","Mean welfare","Mean end"],
          [[esc(r["configuration_key"]),f"{r['n_survived']}/{r['n_runs']}",num(r["final_welfare_mean"],3),num(r["survival_years_mean"],2)] for r in rows],
          "16 retained variants, three seeds each, 48 runs. Actual metadata and source pointers are in the run inventory. These selected variants do not form a paired pure-rule treatment design.")
    ps = ["manufacturing_giant","resource_rich","digital_leapfrogger","post_soviet"]
    rows = []
    for p in ps:
        for sc, name in (("benign_globalization","Benign"),("tech_cold_war","Cold war")):
            m = [r for r in runs if r["batch"]=="mixture" and r["profile"]==p and json.loads(r["metadata_json"])["scenario"]==sc]
            pure = [r for r in runs if r["batch"]=="census" and r["profile"]==p and json.loads(r["metadata_json"])["scenario"]==sc]
            rows.append([PN[p],name,ratio(m),ratio(pure)])
    table("stage4_mixture_cell_table","Mixtures and pure rules by matched profile--scenario cell","tab:mixed_lhs","XXrr",
          ["Profile","Scenario","Mixture","Pure A/B/C"],rows,
          "Each mixture cell: 40 original inputs times three seeds (120). Each pure cell: three strategies times five seeds (15). Totals 430/960 and 67/120. Original-input groups are not corrected-weight treatment groups.")
    cost = read("B_group_costs.csv")
    groups = ["digital_high_skill","industrial_workers","base_workers","service_workers","unemployed"]
    rows = []
    for p in ps:
        vals = [sum(float(r["modeled_cost"]) for r in cost if r["profile"]==p and r["group"]==g)/30 for g in groups]
        rows.append([PN[p]]+[num(vals[0],3)]+[num(v,1) for v in vals[1:]])
    table("stage4_cost_table","Mean modeled group costs: separate 30-seed B sample","tab:cost_distribution","Xrrrrr",
          ["Profile","Digital","Industry","Primary","Services","Unemployed"],rows,
          "Benign globalization, 30 seeds per profile; cumulative values at each run's own endpoint. Digital costs are exactly zero only in the manufacturing sample; a printed small nonzero value is not treated as zero.")
    print(json.dumps({"generated_tables":len(list(OUT.glob("stage4_*_table.tex")))}))


if __name__=="__main__":
    main()
