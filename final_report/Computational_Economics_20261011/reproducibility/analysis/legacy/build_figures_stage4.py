"""Create traceable data figures and matching editable vector diagrams."""
from pathlib import Path
import csv
import hashlib
import html
import json
import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
STAGE = Path(__file__).resolve().parents[1]
FIRST = STAGE.parent / "阶段01"
os.environ["MPLCONFIGDIR"] = str(STAGE / "reports/matplotlib")
sys.path.insert(0, str(FIRST / "runtime_packages"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

OUT = STAGE / "manuscript"
TABLES = STAGE / "tables"
COLORS = {"A": "#275B8C", "B": "#B45E20", "C": "#287C6F"}
PROFILES = ["manufacturing_giant", "resource_rich", "digital_leapfrogger", "small_open", "post_soviet", "populous_weak_industry"]
NAMES = ["Manufacturing giant", "Resource-rich economy", "Digital leapfrogger", "Small open economy", "Post-Soviet economy", "Populous weakly industrialized"]
PN = dict(zip(PROFILES, NAMES))
GROUPS = ["digital_high_skill", "industrial_workers", "base_workers", "service_workers", "unemployed"]
MANIFEST = {}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
                     "axes.titlesize": 11, "legend.fontsize": 9, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 210, "savefig.bbox": "tight", "savefig.pad_inches": .10})


def read(name):
    with (TABLES / name).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def number(row, k):
    return float(row[k])


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(fig, stem, inputs, source, unit, notes):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / (stem + "." + ext))
    plt.close(fig)
    MANIFEST[stem] = {"kind": "data_plot", "raw_source": source, "statistical_unit": unit,
                      "input_files": {f: digest(TABLES / f) for f in inputs}, "notes": notes,
                      "editable_source": "scripts/build_figures.py", "editable_vector": stem + ".svg"}
    print("generated", stem, flush=True)


def data_figures():
    rows = read("census_profile_strategy.csv")
    mat = np.array([[number(next(r for r in rows if r["profile"] == p and r["regime"] == g), "viability_share") for g in "ABC"] for p in PROFILES])
    fig, ax = plt.subplots(figsize=(6.0, 4.1), layout="constrained")
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(6), [n.replace(" weakly ", "\nweakly ") for n in NAMES], fontsize=10)
    ax.set_xticks(range(3), ["A: gradual", "B: leapfrog", "C: substitution"])
    ax.tick_params(length=0)
    for i in range(6):
        for j in range(3):
            ax.text(j, i, f"{mat[i,j]*100:.1f}%\n{round(mat[i,j]*40)}/40", ha="center", va="center",
                    fontsize=10, color="white" if mat[i,j] < .6 else "#122838")
    fig.colorbar(im, ax=ax, shrink=.84, label="Share viable through nominal T = 50")
    save(fig, "fig_survival_heatmap", ["census_profile_strategy.csv"], "census.json",
         "40 executions per profile-strategy cell: 8 scenarios x 5 seeds", "Assigned scenarios receive equal enumeration weight; not calibrated country failure probabilities.")

    modes = ["industrial_withering", "social_collapse", "tech_lock_in", "jump_induced_deindustrialization", "human_capital_erosion", "premature_deindustrialization"]
    labels = ["Industrial\nwithering", "Welfare\ncollapse", "Technology\nlock-in", "Abrupt\n deindustrialization", "Human-capital\nerosion", "Premature\n deindustrialization"]
    rs = read("failure_prevalence.csv")
    fig, ax = plt.subplots(figsize=(7.4, 3.5), layout="constrained")
    for j, g in enumerate("ABC"):
        vals = [number(next(r for r in rs if r["regime"] == g and r["mode"] == m), "run_prevalence") * 100 for m in modes]
        ax.bar(np.arange(6) + (j-1)*.25, vals, width=.25, color=COLORS[g], label=g)
    ax.set_xticks(range(6), labels, fontsize=9)
    ax.set_ylim(0, 72)
    ax.set_ylabel("Executions with a recorded mode (%)")
    ax.legend(title="Strategy", ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    save(fig, "fig_fracture_fingerprint", ["failure_prevalence.csv"], "census.json",
         "240 runs per strategy; a run may record multiple distinct modes", "Bars use run prevalence. Event composition and all nine monitored modes appear in the supplement.")

    rows = read("t80_failure_windows.csv")
    fig, ax = plt.subplots(figsize=(6.8, 2.8), layout="constrained")
    left = np.zeros(3)
    for k, label, color in [("viable_at_80", "Viable at 80", "#287C6F"), ("fail_before_50", "Failure before 50", "#C6962E"), ("fail_at_50_to_80", "Failure at 50-80", "#6F527D")]:
        vals = np.array([number(r, k) / number(r, "n_runs") * 100 for r in rows])
        ax.barh(range(3), vals, left=left, height=.55, color=color, label=label)
        for i, (row, value) in enumerate(zip(rows, vals)):
            if number(row, k) and value >= 8:
                ax.text(left[i] + value/2, i, row[k], ha="center", va="center", color="white", fontsize=11, weight="bold")
        left += vals
    ax.set_yticks(range(3), [f"{r['regime']} (n = {r['n_runs']})" for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the selected fresh-seed executions (%)")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.03), fontsize=9)
    ax.text(.99, -.40, "Small segments: B, 1 early failure; C, 7 early failures", transform=ax.transAxes, ha="right", fontsize=9)
    save(fig, "fig_t80_windows", ["t80_failure_windows.csv", "t80_run_windows.csv"], "census.json; t80.json",
         "51 configurations selected by >=3/5 census survival, then 5 fresh seeds each", "No paired continuation. Labels are counts; A has 20 late failures, B 18 late/1 early, C 7 early/0 late.")

    rows = read("A_growth_capacity_grid.csv")
    growth = sorted({number(r, "g_world") for r in rows})
    inst = sorted({number(r, "institution_capacity") for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.7), sharey=True, layout="constrained")
    for ax, p in zip(axes, ("manufacturing_giant", "post_soviet")):
        grid = np.array([[number(next(r for r in rows if r["profile"] == p and number(r,"g_world") == g and number(r,"institution_capacity") == v), "viability_share") for v in inst] for g in growth])
        ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
        for i in range(len(growth)):
            for j in range(len(inst)):
                ax.text(j, i, f"{round(grid[i,j]*5)}/5", ha="center", va="center", color="white" if grid[i,j] < .6 else "#122838", fontsize=10)
        ax.set_xticks(range(4), [f"{x:.2f}" for x in inst])
        ax.set_yticks(range(len(growth)), [f"{x*100:g}" for x in growth])
        ax.tick_params(length=0)
        ax.set_xlabel("Assigned institutional capacity")
        ax.set_title(PN[p])
    axes[0].set_ylabel("Assigned frontier growth (%)")
    save(fig, "fig_A_growth_capacity", ["A_growth_capacity_grid.csv"], "gspeed.json",
         "72 settings, 5 seeds each; A, benign globalization; 9 growth values x 4 capacities x 2 profiles", "All retained grid values displayed. Stronger-capacity settings survive to the upper growth limit; no universal critical growth estimated.")

    rows = read("B_diagnostic_timeseries.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
    for rep in range(5):
        rs = [r for r in rows if int(r["rep"]) == rep]
        for ax, k in zip(axes, ("gdp", "welfare_index")):
            ax.plot([number(r,"t") for r in rs], [number(r,k) for r in rs], lw=1.4, label=f"Seed realization {rep+1}")
            ax.plot(number(rs[-1],"t"), number(rs[-1],k), "o", markersize=3)
    axes[0].set_ylabel("Recorded GDP (model units)")
    axes[1].set_ylabel("Aggregate welfare index")
    axes[1].axhline(.30, color="#555555", linestyle="--", linewidth=1, label="Nominal welfare floor")
    for ax in axes:
        ax.set_xlabel("Recorded model time")
        ax.grid(alpha=.18)
    axes[0].set_title("Output")
    axes[1].set_title("Welfare")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].legend(fontsize=8, loc="upper right")
    save(fig, "fig_B_output_welfare", ["B_diagnostic_timeseries.csv", "B_diagnostic_endpoints.csv"], "ts_calibration.json#/results/5..9",
         "5 individual manufacturing-giant B trajectories; benign globalization", "Actual recorded times, no interpolation or pooled tail mean. t=0 is post-update. Each endpoint is its own termination.")

    rows = read("C_selected_cases.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.3), layout="constrained")
    marks = ["o", "s", "^"]
    labels = ["Manufacturing giant / benign", "Resource rich / benign", "Resource rich / containment"]
    colors = ["#287C6F", "#275B8C", "#B45E20"]
    for r, marker, label, color in zip(rows, marks, labels, colors):
        x, y = number(r,"final_gdp_mean"), number(r,"final_welfare_mean")
        ax.errorbar(x, y, xerr=[[x-number(r,"final_gdp_min")],[number(r,"final_gdp_max")-x]],
                    yerr=[[y-number(r,"final_welfare_min")],[number(r,"final_welfare_max")-y]],
                    fmt=marker, color=color, markersize=7, capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Mean endpoint GDP (log scale; model units)")
    ax.set_ylabel("Mean endpoint welfare index")
    ax.set_xlim(25, 6000)
    ax.set_ylim(.25, 2.65)
    ax.grid(alpha=.18)
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "fig_C_endpoint_profiles", ["C_selected_cases.csv"], "census.json, 3 declared C settings",
         "5 seed repetitions per setting; own endpoints, including failed runs", "Markers: means; bars: observed min/max, not confidence intervals. Profiles/scenarios change multiple inputs together.")

    rows = read("B_group_costs.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey=True, layout="constrained")
    for ax, p in zip(axes.flat, ("manufacturing_giant", "post_soviet", "resource_rich", "digital_leapfrogger")):
        data = [[number(r,"modeled_cost") for r in rows if r["profile"] == p and r["group"] == g] for g in GROUPS]
        assert all(len(x) == 30 for x in data)
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=.55)
        for box, color in zip(bp["boxes"], ["#7B8792","#275B8C","#287C6F","#B45E20","#6F527D"]):
            box.set_facecolor(color)
            box.set_alpha(.7)
        for i, vals in enumerate(data):
            jitter = np.linspace(-.12,.12,len(vals))
            ax.scatter(i+1+jitter, vals, s=6, color="#233849", alpha=.45, zorder=3)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(range(1,6), ["Digital","Industry","Primary","Services","Unemployed"], rotation=25, fontsize=8)
        ax.set_title(PN[p] + " (n = 30)", fontsize=10)
        ax.grid(axis="y",alpha=.18)
    axes[0,0].set_ylim(-.05,max(number(r,"modeled_cost") for r in rows)*1.4)
    for ax in axes[:,0]:
        ax.set_ylabel("Cumulative modeled group cost\n(model units; symlog)")
    save(fig, "fig_b_cost_distribution", ["B_group_costs.csv"], "cost_table.json",
         "30 B seeds per profile, benign globalization; all 120 own endpoints", "All points shown; boxes summarize distributions. Digital costs are exactly zero only for manufacturing giant here. No marginal growth-cost ratio.")

    weights = read("mixture_inputs_and_outcomes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.7), layout="constrained")
    for ax, prefix, title in zip(axes, ("drawn_lambda_", "accumulation_lambda_"), ("Original Dirichlet inputs (320)", "Accumulation-object weights (292 unique)")):
        vertices = np.array([[0,0],[1,0],[.5,np.sqrt(3)/2],[0,0]])
        ax.plot(vertices[:,0],vertices[:,1],color="#617485",lw=1)
        for changed, color in (("False","#287C6F"),("True","#B45E20")):
            rs = [r for r in weights if r["changed_by_constraint"] == changed]
            x = [number(r,prefix+"B")+.5*number(r,prefix+"C") for r in rs]
            y = [np.sqrt(3)/2*number(r,prefix+"C") for r in rs]
            ax.scatter(x,y,s=11,alpha=.65,c=color,edgecolor="none",label="Unchanged" if changed=="False" else "Corrected")
        ax.text(-.03,-.04,"A",ha="right",fontsize=11)
        ax.text(1.03,-.04,"B",fontsize=11)
        ax.text(.5,np.sqrt(3)/2+.025,"C",ha="center",fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-.1,1.1)
        ax.set_ylim(-.1,1.01)
        ax.axis("off")
        ax.set_title(title,fontsize=10)
    axes[0].legend(loc="lower center",bbox_to_anchor=(.5,-.05),ncol=2,fontsize=9,frameon=False)
    save(fig, "fig_mixture_weight_mapping", ["mixture_inputs_and_outcomes.csv"], "mixture.json; stage-3 audited weight mapping",
         "320 original inputs; 171 corrected; 292 accumulation vectors at 12 decimal places", "The engine retains original inputs for other rules. Corrected points are not the complete mechanism state. No uniform effective-simplex ranking.")

    rows = read("matched_frame_viability.csv")
    ps = ["manufacturing_giant","resource_rich","digital_leapfrogger","post_soviet"]
    fig, ax = plt.subplots(figsize=(6.8,3.1),layout="constrained")
    for offset, design, color, label in [(-.17,"pure","#275B8C","Pure rules (30 runs/profile)"),(.17,"mixture","#B45E20","Mixtures (240 runs/profile)")]:
        cells = [next(r for r in rows if r["profile"]==p and r["design"]==design) for p in ps]
        values = [number(r,"viability_share")*100 for r in cells]
        ax.barh(np.arange(4)+offset,values,height=.30,color=color,label=label)
        for i,(r,v) in enumerate(zip(cells,values)):
            ax.text(v+1.5,i+offset,f"{r['n_survived']}/{r['n_runs']}",va="center",fontsize=9)
    ax.set_yticks(range(4),[PN[p] for p in ps])
    ax.invert_yaxis()
    ax.set_xlim(0,122)
    ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel("Viability share through nominal T = 50 (%)")
    ax.legend(ncol=1,loc="upper left",bbox_to_anchor=(.02,-.23),fontsize=9)
    ax.grid(axis="x",alpha=.18)
    ax.set_axisbelow(True)
    save(fig, "fig_matched_mixture", ["matched_frame_viability.csv"], "census.json matched subset; mixture.json",
         "same 4 profiles x 2 scenarios; pure 120 runs vs mixture 960, distinct seeds and mechanisms", "Equal profile-scenario enumeration; pure strategy mix of A/B/C. Descriptive frame match, no seed-paired treatment effect.")

    rows = read("selected_representative_paths.csv")
    ps = ["manufacturing_giant","resource_rich","digital_leapfrogger","post_soviet"]
    fig, axes = plt.subplots(2,2,figsize=(7.2,5.1),sharey=True,layout="constrained")
    for ax,p in zip(axes.flat,ps):
        sources = list(dict.fromkeys(r["source"] for r in rows if r["profile"]==p))
        for i,source in enumerate(sources):
            rs = [r for r in rows if r["source"]==source]
            g = rs[0]["regime"]
            style = "--" if rs[0]["scenario"]=="full_containment" else "-"
            label = g + (" / containment" if style=="--" else "")
            ax.plot([number(r,"gdp_per_worker") for r in rs],[number(r,"manufacturing_share_pct") for r in rs],
                    color=COLORS[g],linestyle=style,lw=1.2,label=label)
            ax.plot(number(rs[0],"gdp_per_worker"),number(rs[0],"manufacturing_share_pct"),"o",color=COLORS[g],ms=3)
            ax.plot(number(rs[-1],"gdp_per_worker"),number(rs[-1],"manufacturing_share_pct"),"s",color=COLORS[g],ms=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 3.0), numticks=6))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, position: f"{value:g}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_title(PN[p])
        ax.grid(alpha=.18)
        ax.legend(fontsize=8,loc="upper right")
    fig.supylabel("Manufacturing employment share (%)", fontsize=10)
    fig.supxlabel("GDP per worker (log; model units)", fontsize=10)
    save(fig, "fig_calib_fact1_curves", ["selected_representative_paths.csv"], "ts_viable.json; census seed re-executions",
         "9 selected single-seed paths, 4 profiles", "Eight benign paths and one post-Soviet B path under full containment (dashed). Unsmoothed points; circles first post-update state, squares last. Curve direction is temporal; no population-wide inverted-U or empirical fit.")

    rows = read("directional_configuration_means.csv")
    fig, axes = plt.subplots(1,3,figsize=(7.2,3.2),sharey=True,layout="constrained")
    for ax,g in zip(axes,"ABC"):
        rs = [r for r in rows if r["regime"]==g]
        ax.scatter([number(r,"institution_capacity") for r in rs],[number(r,"final_gdp_mean") for r in rs],color=COLORS[g],s=35)
        ax.set_yscale("log")
        ax.set_title(g + ": six configuration means")
        ax.set_xlabel("Assigned initial capacity",fontsize=9)
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Mean endpoint GDP\n(log; model units)")
    save(fig, "fig_calib_fact2_inst", ["directional_configuration_means.csv"], "calibration.json",
         "18 configuration means, 5 seed repetitions each", "Other profile inputs vary with capacity. Descriptive average-rank Spearman, not a ceteris-paribus institutional effect.")

    rs = [r for r in read("directional_trajectory_metrics.csv") if r["eligible"]=="True"]
    fig,ax = plt.subplots(figsize=(6.3,3.4),layout="constrained")
    for g in "ABC":
        rows = [r for r in rs if r["regime"]==g]
        ax.scatter([number(r,"deindustrialization_extent_pct") for r in rows],
                   [number(r,"full_trajectory_gdp_growth_pct") for r in rows],
                   s=28,alpha=.65,color=COLORS[g],label=g,edgecolor="white",linewidth=.3)
    ax.set_xlabel("Peak-to-last manufacturing share decline (%)")
    ax.set_ylabel("Full-recorded-trajectory GDP growth (%)")
    ax.set_title("60 eligible trajectories: descriptive Pearson r = +0.529")
    ax.legend(title="Strategy",ncol=3,loc="upper left")
    ax.grid(alpha=.18)
    save(fig, "fig_calib_fact3_deind", ["directional_trajectory_metrics.csv"], "ts_calibration.json",
         "60 of 90 trajectories have >=5 positive-GDP points and a positive recorded span", "30 short paths excluded. First-to-last annualized growth, not post-event growth. Expected negative relation not reproduced.")


def diagram(stem, title, boxes, edges, notes, width=700, height=530):
    root = ET.Element("mxfile", host="stage4-native-vector-export")
    page = ET.SubElement(root,"diagram",id=stem,name="Implemented rules")
    graph = ET.SubElement(page,"mxGraphModel",page="1",pageWidth=str(width),pageHeight=str(height))
    cells = ET.SubElement(graph,"root")
    ET.SubElement(cells,"mxCell",id="0")
    ET.SubElement(cells,"mxCell",id="1",parent="0")
    pdf = canvas.Canvas(str(OUT/(stem+".drawio.pdf")),pagesize=(width,height))
    pdf.setTitle(title)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#617485"/></marker></defs>',
           '<rect width="100%" height="100%" fill="white"/>']
    specs = {b["id"]:b for b in boxes}
    for i,(a,b) in enumerate(edges):
        aa,bb=specs[a],specs[b]
        if aa["y"]==bb["y"]:
            coords = [(aa["x"]+aa["w"],aa["y"]+aa["h"]/2),(bb["x"],bb["y"]+bb["h"]/2)]
        else:
            coords = [(aa["x"]+aa["w"]/2,aa["y"]+aa["h"]),(bb["x"]+bb["w"]/2,bb["y"])]
        (x1,y1),(x2,y2)=coords
        pdf.setStrokeColor(HexColor("#617485"));pdf.line(x1,height-y1,x2,height-y2)
        # Arrowhead follows line direction, including diagonal links.
        dx,dy=x2-x1,y2-y1
        length=(dx*dx+dy*dy)**.5
        ux,uy=dx/length,dy/length
        pts=[(x2,y2),(x2-8*ux+3*uy,y2-8*uy-3*ux),(x2-8*ux-3*uy,y2-8*uy+3*ux)]
        path=pdf.beginPath()
        path.moveTo(pts[0][0],height-pts[0][1])
        for x,y in pts[1:]:path.lineTo(x,height-y)
        path.close();pdf.setFillColor(HexColor("#617485"));pdf.drawPath(path,fill=1,stroke=0)
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#617485" marker-end="url(#arrow)"/>')
        edge=ET.SubElement(cells,"mxCell",id=f"edge{i}",edge="1",source=a,target=b,parent="1",style="endArrow=block;strokeColor=#617485;")
        ET.SubElement(edge,"mxGeometry",relative="1",attrib={"as":"geometry"})
    for b in [{"id":"title","x":25,"y":12,"w":width-50,"h":38,"lines":[title],"fill":None}]+boxes:
        x,y,w,h=b["x"],b["y"],b["w"],b["h"]
        value="\n".join(b["lines"])
        style="text;whiteSpace=wrap;html=0;align=left;fontSize=14;" if b["fill"] is None else f'rounded=1;whiteSpace=wrap;html=0;align=left;spacing=12;fontSize=14;fillColor={b["fill"]};strokeColor=#617485;'
        cell=ET.SubElement(cells,"mxCell",id=b["id"],value=value,vertex="1",parent="1",style=style)
        ET.SubElement(cell,"mxGeometry",x=str(x),y=str(y),width=str(w),height=str(h),attrib={"as":"geometry"})
        if b["fill"]:
            pdf.setFillColor(HexColor(b["fill"]));pdf.setStrokeColor(HexColor("#617485"));pdf.roundRect(x,height-y-h,w,h,6,fill=1)
            svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{b["fill"]}" stroke="#617485"/>')
        for j,line in enumerate(b["lines"]):
            size=20 if b["id"]=="title" else (15 if j==0 else 14)
            baseline=y+24+j*19
            if baseline>y+h-8:raise RuntimeError("Diagram text exceeds box: "+b["id"])
            if pdf.stringWidth(line,"Helvetica-Bold" if j==0 else "Helvetica",size)>w-24:
                raise RuntimeError("Diagram line is too wide: "+line)
            pdf.setFont("Helvetica-Bold" if j==0 else "Helvetica",size);pdf.setFillColor(HexColor("#17364D"));pdf.drawString(x+12,height-baseline,line)
            svg.append(f'<text x="{x+12}" y="{baseline}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{"bold" if j==0 else "normal"}" fill="#17364D">{html.escape(line)}</text>')
    notes_lines = [line for note in notes for line in textwrap.wrap(note,width=108)]
    for j,line in enumerate(notes_lines):
        baseline=height-50+j*14
        pdf.setFont("Helvetica",11);pdf.setFillColor(HexColor("#334B5F"));pdf.drawString(25,height-baseline,line)
        svg.append(f'<text x="25" y="{baseline}" font-family="Arial, sans-serif" font-size="11" fill="#334B5F">{html.escape(line)}</text>')
    notes_cell=ET.SubElement(cells,"mxCell",id="notes",value="\n".join(notes_lines),vertex="1",parent="1",style="text;html=0;fontSize=11;align=left;")
    ET.SubElement(notes_cell,"mxGeometry",x="25",y=str(height-65),width=str(width-50),height="60",attrib={"as":"geometry"})
    pdf.showPage();pdf.save()
    svg.append("</svg>")
    (OUT/(stem+".svg")).write_text("\n".join(svg),encoding="utf-8")
    ET.indent(root)
    ET.ElementTree(root).write(OUT/(stem+".drawio"),encoding="utf-8",xml_declaration=True)
    (TABLES/(stem+"_diagram.json")).write_text(json.dumps({"title":title,"width":width,"height":height,"boxes":boxes,"edges":edges,"notes":notes},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    poppler=Path("C:/Users/胡耀宇/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe")
    subprocess.run([str(poppler),"-png","-singlefile","-r","160",str(OUT/(stem+".drawio.pdf")),str(OUT/stem)],check=True,capture_output=True)
    MANIFEST[stem]={"kind":"implementation_diagram","raw_source":"frozen v3.8 implementation; stage-3 implementation audit",
                    "statistical_unit":"specified rules; any endpoint number uses declared diagnostic trajectories",
                    "editable_source":stem+".drawio","editable_vector":stem+".svg",
                    "shared_geometry":stem+"_diagram.json","labels":[b["lines"] for b in boxes],"notes":notes}
    print("generated",stem,flush=True)


def box(id,x,y,w,h,lines,fill="#E9F1F8"):
    return {"id":id,"x":x,"y":y,"w":w,"h":h,"lines":lines,"fill":fill}


def diagrams():
    diagram("fig_model_structure","States and sequential update in the retained model",[
        box("input",25,65,650,84,["Assigned inputs and external state","Six ideal profiles; eight scenarios; pure rules or original mixture inputs","Seeded shocks, foreign equity and external conditions; no fitted countries"]),
        box("prod",25,178,315,120,["Production and labor","Six sector production functions","Wage and skill-filtered labor flows","Shared surplus replaces the fund","Earlier external addition is overwritten"]),
        box("accum",360,178,315,120,["Accumulation and technology","Rule-specific allocation and costs","Absorption, capital and digitalization","Innovation and two diffusion updates","Not a simultaneous equilibrium"]),
        box("hh",25,328,315,120,["Households and evaluation","Five aggregate household groups","Consumption, employment and skills","Atkinson welfare; modeled costs","Enabled political and auxiliary rules"],"#F3EFE7"),
        box("record",360,328,315,120,["Diagnostics and recorded state","SFC proxy residual; no exact closure","Nine detectors; three terminal types","Record after update, before clock step","t = 0 already includes one update"],"#F3EFE7")
    ],[("input","prod"),("prod","accum"),("accum","hh"),("hh","record")],
    ["Source: core/engine.py and the implemented module rules. Diagram labels summarize order and scope.",
     "Welfare, costs, detector events and terminal CDI are distinct measures; red lines do not stop the run."])
    diagram("fig_strategy_framework","Operative A, B and C accumulation rules",[
        box("a",25,65,650,100,["A: capital base plus a digital premium","Capital-proportional weights plus the specified digital premium; then normalized","Industrial-protection and catch-up rules follow investment","Computed tau does not debit a separately accounted investment budget"]),
        box("b",25,185,650,100,["B: digital priority and rate-dependent adjustment","Pre-maturity digital allocation = 0.60; component weights total 1.03","Tau enters additional unemployment and consumption-drag rules","Digital maturity >= 0.50 permits stochastic feedback; it does not ensure success"],"#F3EFE7"),
        box("c",25,305,650,100,["C: scenario-dependent allocation and absorption","Allocation responds to FDI availability, import preference and resource exports","Technology, foreign equity and assigned external shocks affect the state","Component bounds are applied separately; weights are not globally normalized"],"#E5F0EA"),
        box("shared",25,425,650,64,["Shared qualifications: the fund is recomputed from model surplus","SFC is a residual diagnostic. M uses corrected accumulation and original engine weights."],"#F3E8EC")
    ],[],["Source: accumulation.py, mixed_strategy.py, engine.py and sfc/accounting.py."],height=550)
    summary=json.loads((TABLES/"evidence_summary.json").read_text(encoding="utf-8"))["B_diagnostic"]
    means=summary["snapshot_means"]
    diagram("fig_b_welfare_collapse","Strategy B: specified channels and diagnostic observations",[
        box("rule",25,65,315,140,["Specified accumulation package","Digital-priority allocation","Growth target g = 0.25","Tau affects unemployment and costs","Feedback eligibility at DL >= 0.50","Success and execution are separate"]),
        box("welfare",360,65,315,140,["Main welfare calculation","Consumption per person: power 0.5","Employment per person: power 0.3","Skill: power 0.2","Atkinson inequality aversion = 1.5","Sector-quality index is separate"]),
        box("path",25,245,315,140,["Five archived B trajectories",f"Recorded GDP: {means['gdp']['first']:.1f} to {means['gdp']['last']:,.1f}",
              f"Welfare: {means['welfare_index']['first']:.3f} to {means['welfare_index']['last']:.3f}",
              f"Industrial labor: {means['L_i']['first']:.2f} to {means['L_i']['last']:.2f}",
              f"Endpoint years: {summary['survival_years_min']:.1f} to {summary['survival_years_max']:.1f}",
              "0/5 viable; endpoints are not one year"],"#F3EFE7"),
        box("cost",360,245,315,140,["Observed modeled costs","Manufacturing digital cost = 0","Industrial wage-gap cost is positive","Industrial / digital ratio undefined","No marginal cost-per-growth claim","Channels are not isolated effects"],"#F3E8EC")
    ],[("rule","welfare")],
    ["Source of observations: ts_calibration.json, manufacturing giant / B / benign, five seeds (records 5-9).",
     "Top-row link summarizes interacting specified rules. Bottom boxes are observations, not effect estimates."],height=470)


def main():
    data_figures()
    diagrams()
    for stem, entry in MANIFEST.items():
        entry["outputs"]={p.name:{"sha256":digest(p),"bytes":p.stat().st_size} for p in OUT.glob(stem+".*") if p.suffix in {".pdf",".png",".svg",".drawio"}}
    (STAGE/"reports/figure_manifest.json").write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    import reportlab
    (STAGE/"reports/figure_environment.json").write_text(json.dumps(
        {"python":sys.version,"numpy":np.__version__,"matplotlib":matplotlib.__version__,
         "reportlab":reportlab.Version,"dependency_source":"../阶段01/runtime_packages plus bundled runtime",
         "original_experiment_environment_claimed":False},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"figures":len(MANIFEST),"updated_original_figures":9,"new_evidence_figures":6},ensure_ascii=False))


if __name__=="__main__":
    main()
