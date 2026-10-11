"""Prepare 119-mm vector figures with embedded fonts from the frozen figure data."""
from pathlib import Path
import hashlib
import html
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET

STAGE=Path(__file__).resolve().parents[1]
REPRO=STAGE
ART=STAGE/"output/artwork"
REPORTS=STAGE/"output/artwork_reports"
REPORTS.mkdir(parents=True,exist_ok=True)
sys.dont_write_bytecode=True
os.environ["MPLCONFIGDIR"]=str(REPORTS/"matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.markers import MarkerStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from pypdf import PdfReader

WIDTH=119/25.4*72
MANIFEST={}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def save_plot(fig,stem,inputs,source,unit,notes):
    oldw,oldh=fig.get_size_inches()
    fig.set_size_inches(WIDTH/72,max(2.45,oldh/oldw*WIDTH/72+.42))
    axes=fig.axes
    for ax in axes:
        for patch in ax.patches:
            if hasattr(patch,"set_hatch"):
                col=tuple(round(v,2) for v in patch.get_facecolor()[:3])
                patch.set_hatch({(0.15,0.36,0.55):"",(0.71,0.37,0.13):"//",(0.16,0.49,0.44):"..",(0.78,0.59,0.18):"//",(0.44,0.32,0.49):"xx"}.get(col,None))
        ax.xaxis.label.set_text("\n".join(textwrap.wrap(ax.xaxis.label.get_text(),width=40 if len(axes)==1 else 27)))
        if "six configuration means" in ax.title.get_text():
            ax.title.set_text(ax.title.get_text().replace(": six configuration means",": six means"))
        if "Original Dirichlet inputs" in ax.title.get_text():
            ax.title.set_text("(a) Original inputs\n320 Dirichlet draws")
        if "Accumulation-object weights" in ax.title.get_text():
            ax.title.set_text("(b) Accumulation weights\n292 distinct vectors")
        if len(axes)==1 and "Pearson" in ax.title.get_text():
            ax.title.set_text("")
        if stem=="fig_B_output_welfare":
            lines=[line for line in ax.lines if len(line.get_xdata())>5]
            for line,style in zip(lines,("-","--","-.",":",(0,(5,1,1,1,1,1)))):
                line.set_linestyle(style)
            for path,endpoint in zip(ax.lines[:10:2],ax.lines[1:10:2]):
                endpoint.set_color(path.get_color())
            # Update legend handles after assigning line patterns.
            loc="upper left" if ax is axes[0] else "upper right"
            ax.legend(fontsize=8.5,loc=loc)
    for artist in fig.findobj(match=Text):
        old=artist.get_fontsize()
        artist.set_fontsize(max(8.5,min(10,old)))
    # Reflow at the physical publication width; resizing alone overlaps labels.
    rect=(0,0,1,1)
    if stem=="fig_survival_heatmap":
        axes[0].set_xticks(range(3),["A\nGradual","B\nLeapfrog","C\nSubstitution"])
        labels=["Manufacturing\ngiant","Resource-rich\neconomy","Digital\nleapfrogger","Small open\neconomy","Post-Soviet\neconomy","Populous weakly\nindustrialized"]
        axes[0].set_yticks(range(6),labels)
        axes[1].set_ylabel("Viable share (nominal T = 50)")
        fig.set_size_inches(WIDTH/72,3.65)
    elif stem=="fig_fracture_fingerprint":
        ax=axes[0]
        for tick in ax.get_xticklabels():
            tick.set_rotation(55);tick.set_horizontalalignment("right")
        ax.set_ylabel("Executions with\na recorded mode (%)")
        ax.legend(title="Strategy",ncol=3,loc="upper right",fontsize=8.5,title_fontsize=8.5,handlelength=1.4,columnspacing=.8)
        fig.set_size_inches(WIDTH/72,3.25)
    elif stem=="fig_t80_windows":
        ax=axes[0]
        handles,_=ax.get_legend_handles_labels()
        ax.legend(handles,["Viable at 80","Fail < 50","Fail 50–80"],ncol=3,loc="lower center",bbox_to_anchor=(.5,1.03),fontsize=8.5,handlelength=1.3,columnspacing=.8)
        for txt in list(ax.texts):
            if txt.get_text().startswith("Small segments:"):txt.remove()
        ax.set_xlabel("Share of selected fresh-seed executions (%)")
        fig.text(.5,.015,"Small segments: B, 1 early failure;\nC, 7 early failures",ha="center",va="bottom",fontsize=8.5)
        rect=(0,.17,1,.83)
        fig.set_size_inches(WIDTH/72,3.05)
    elif stem=="fig_A_growth_capacity":
        for ax in axes:
            ax.set_title("\n".join(textwrap.wrap(ax.title.get_text(),18)),fontsize=9)
            ax.set_xlabel("Assigned institutional\ncapacity",fontsize=8.5)
        fig.set_size_inches(WIDTH/72,3.9)
    elif stem=="fig_B_output_welfare":
        handles,labels=axes[1].get_legend_handles_labels()
        for ax in axes:
            ax.get_legend().remove()
            ax.set_xlabel("Recorded model time",fontsize=8.5)
        axes[0].set_ylabel("Recorded GDP\n(model units)",fontsize=8.5)
        axes[1].set_ylabel("Aggregate welfare\nindex",fontsize=8.5)
        labels=[label.replace("Seed realization ","Realization ") for label in labels]
        fig.legend(handles,labels,ncol=2,loc="lower center",bbox_to_anchor=(.5,.01),fontsize=8.5,handlelength=2.2,columnspacing=1)
        fig.set_size_inches(WIDTH/72,3.7)
        rect=(0,.24,1,.76)
    elif stem=="fig_b_cost_distribution":
        for ax in axes:
            ax.set_title("\n".join(textwrap.wrap(ax.title.get_text().replace(" (n = 30)",""),18))+"\n(n = 30)",fontsize=8.5)
            for tick in ax.get_xticklabels():
                tick.set_rotation(50);tick.set_horizontalalignment("right")
            if ax.get_ylabel():ax.set_ylabel("Cumulative modeled cost\n(model units; symlog)",fontsize=8.5)
        fig.set_size_inches(WIDTH/72,5.1)
    elif stem=="fig_mixture_weight_mapping":
        for ax in axes:
            ax.set_title(ax.title.get_text(),fontsize=8.5)
            diamond=MarkerStyle("D")
            ax.collections[1].set_paths([diamond.get_path().transformed(diamond.get_transform())])
        handles,labels=axes[0].get_legend_handles_labels()
        axes[0].get_legend().remove()
        fig.legend(handles,labels,ncol=2,loc="lower center",bbox_to_anchor=(.5,.015),fontsize=8.5,frameon=False)
        fig.set_size_inches(WIDTH/72,3.65)
        rect=(0,.15,1,.85)
    elif stem=="fig_matched_mixture":
        ax=axes[0]
        ax.set_yticks(range(4),["Manufacturing\ngiant","Resource-rich\neconomy","Digital\nleapfrogger","Post-Soviet\neconomy"])
        handles,labels=ax.get_legend_handles_labels()
        ax.get_legend().remove()
        fig.legend(handles,labels,ncol=1,loc="lower center",bbox_to_anchor=(.5,.01),fontsize=8.5)
        ax.set_xlabel("Viability through nominal T = 50 (%)")
        fig.set_size_inches(WIDTH/72,3.55)
        rect=(0,.22,1,.78)
    elif stem=="fig_calib_fact1_curves":
        legend_items={}
        for ax in axes:
            ax.set_title("\n".join(textwrap.wrap(ax.title.get_text(),18)),fontsize=8.5)
            for line in ax.lines:
                label=line.get_label()
                if not label.startswith("_"):
                    line.set_linestyle("--" if "containment" in label else {"A":"-","B":"-.","C":":"}[label])
            handles,labels=ax.get_legend_handles_labels()
            legend_items.update(zip(labels,handles))
            ax.get_legend().remove()
        fig.legend([legend_items[label] for label in ("A","B","C","B / containment")],["A","B","C","B / containment"],ncol=2,loc="lower center",bbox_to_anchor=(.5,.008),fontsize=8.5,handlelength=2.2)
        fig.supxlabel("GDP per worker (log; model units)",y=.15,fontsize=8.5)
        fig.set_size_inches(WIDTH/72,5.2)
        rect=(0,.21,1,.79)
    elif stem=="fig_calib_fact2_inst":
        for ax in axes:
            ax.set_title(ax.title.get_text().replace(": six means",": six\nmeans"),fontsize=8.5)
            ax.set_xlabel("Assigned\ninitial capacity",fontsize=8.5)
        axes[0].set_ylabel("Mean endpoint GDP\n(log; model units)",fontsize=8.5)
        fig.set_size_inches(WIDTH/72,3.0)
    elif stem=="fig_calib_fact3_deind":
        for collection,marker in zip(axes[0].collections,("o","s","^")):
            style=MarkerStyle(marker)
            collection.set_paths([style.get_path().transformed(style.get_transform())])
        axes[0].legend(title="Strategy",ncol=3,loc="lower center",bbox_to_anchor=(.5,1.015),fontsize=8.5,title_fontsize=8.5)
        axes[0].set_ylabel("Full-recorded-trajectory\nGDP growth (%)",fontsize=8.5)
        axes[0].set_xlabel("Peak-to-last manufacturing\nshare decline (%)",fontsize=8.5)
        fig.set_size_inches(WIDTH/72,3.0)
    fig.get_layout_engine().set(rect=rect,w_pad=.035,h_pad=.04,wspace=.08,hspace=.1)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[]
    off_scale=set()
    for ax in axes:
        if not ax.axison:
            for axis in (ax.xaxis,ax.yaxis):
                off_scale.update(id(text) for text in axis.findobj(match=Text))
        for axis,limits in ((ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())):
            low,high=sorted(limits)
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not low<=tick.get_loc()<=high:
                    off_scale.update((id(tick.label1),id(tick.label2)))
    for artist in fig.findobj(match=Text):
        if id(artist) in off_scale or not artist.get_visible() or not artist.get_text():continue
        box=artist.get_window_extent(renderer)
        if box.x0 < -2 or box.y0 < -2 or box.x1 > fig.bbox.width+2 or box.y1 > fig.bbox.height+2:
            outside.append(artist.get_text())
    if outside:raise RuntimeError("Text outside figure "+stem+": "+str(outside))
    for ext in ("pdf","svg","eps","png"):
        fig.savefig(ART/(stem+"."+ext),bbox_inches=None,dpi=300)
    plt.close(fig)
    MANIFEST[stem]={"kind":"data_plot","input_sha256":{s:digest(REPRO/"accepted/stage04"/s) for s in inputs},"raw_source":source,"unit":unit,"scope":notes,"width_mm":119,"height_mm":fig.get_size_inches()[1]*25.4,"minimum_label_font_pt":8.5,"text_extents_within_canvas":True,"new_simulations":False}
    print("prepared",stem,flush=True)

def wrap_line(line,font,size,width):
    words=line.split()
    result=[]
    current=""
    for word in words:
        candidate=(current+" "+word).strip()
        if current and pdfmetrics.stringWidth(candidate,font,size)>width:
            result.append(current);current=word
        else:current=candidate
    if current:result.append(current)
    return result

def diagram(stem):
    spec=json.loads((REPRO/"accepted/stage04"/(stem+"_diagram.json")).read_text(encoding="utf-8"))
    rows={}
    for b in spec["boxes"]:rows.setdefault(b["y"],[]).append(b)
    boxes=[];y=4
    for row in sorted(rows):
        original=sorted(rows[row],key=lambda b:b["x"])
        gap=12;w=(WIDTH-8-gap*(len(original)-1))/len(original)
        created=[]
        for i,b in enumerate(original):
            lines=[]
            for j,line in enumerate(b["lines"]):
                font="FigureSansBold" if j==0 else "FigureSans"
                size=9 if j==0 else 8.5
                lines.extend((s,font,size) for s in wrap_line(line,font,size,w-16))
            h=16+11*len(lines)
            created.append(dict(id=b["id"],x=4+i*(w+gap),y=y,w=w,h=h,lines=lines,fill=b["fill"]))
        height=max(b["h"] for b in created)
        for b in created:b["h"]=height
        boxes+=created;y+=height+20
    height=y-16
    if height/72*25.4>195:raise RuntimeError("Diagram exceeds small-journal height")
    pdf=canvas.Canvas(str(ART/(stem+".pdf")),pagesize=(WIDTH,height),pageCompression=1,initialFontName="FigureSans",initialFontSize=8.5)
    pdf.setTitle(spec["title"])
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">','<rect width="100%" height="100%" fill="white"/>']
    xml=ET.Element("mxfile",host="programmatic-vector-export")
    d=ET.SubElement(xml,"diagram",id=stem,name="Implemented rules")
    graph=ET.SubElement(d,"mxGraphModel",page="1",pageWidth=str(WIDTH),pageHeight=str(height))
    cells=ET.SubElement(graph,"root")
    ET.SubElement(cells,"mxCell",id="0");ET.SubElement(cells,"mxCell",id="1",parent="0")
    byid={b["id"]:b for b in boxes}
    for a,b in spec["edges"]:
        aa,bb=byid[a],byid[b]
        if aa["y"]==bb["y"]:
            x1,y1=aa["x"]+aa["w"],aa["y"]+aa["h"]/2
            x2,y2=bb["x"],bb["y"]+bb["h"]/2
        else:
            x1,y1=aa["x"]+aa["w"]/2,aa["y"]+aa["h"]
            x2,y2=bb["x"]+bb["w"]/2,bb["y"]
        pdf.setStrokeColor(HexColor("#617485"));pdf.setLineWidth(.6);pdf.line(x1,height-y1,x2,height-y2)
        angle=math.atan2(y2-y1,x2-x1)
        pts=[(x2,y2),(x2-5*math.cos(angle)+2*math.sin(angle),y2-5*math.sin(angle)-2*math.cos(angle)),(x2-5*math.cos(angle)-2*math.sin(angle),y2-5*math.sin(angle)+2*math.cos(angle))]
        path=pdf.beginPath();path.moveTo(pts[0][0],height-pts[0][1])
        for x,z in pts[1:]:path.lineTo(x,height-z)
        path.close();pdf.setFillColor(HexColor("#617485"));pdf.drawPath(path,fill=1,stroke=0)
        svg.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#617485" fill="none"/><polygon points="'+" ".join(f"{x},{z}" for x,z in pts)+'" fill="#617485"/>')
        edge=ET.SubElement(cells,"mxCell",id="edge_"+a+b,edge="1",source=a,target=b,parent="1",style="endArrow=block;strokeWidth=0.6;")
        ET.SubElement(edge,"mxGeometry",relative="1",attrib={"as":"geometry"})
    for b in boxes:
        x,z,w,h=b["x"],b["y"],b["w"],b["h"]
        pdf.setFillColor(HexColor(b["fill"]));pdf.setStrokeColor(HexColor("#617485"));pdf.setLineWidth(.5);pdf.roundRect(x,height-z-h,w,h,4,fill=1)
        svg.append(f'<rect x="{x}" y="{z}" width="{w}" height="{h}" rx="4" fill="{b["fill"]}" stroke="#617485"/>')
        for j,(line,font,size) in enumerate(b["lines"]):
            baseline=z+13+j*11
            pdf.setFont(font,size);pdf.setFillColor(HexColor("#17364D"));pdf.drawString(x+8,height-baseline,line)
            svg.append(f'<text x="{x+8}" y="{baseline}" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{"bold" if font.endswith("Bold") else "normal"}" fill="#17364D">{html.escape(line)}</text>')
        cell=ET.SubElement(cells,"mxCell",id=b["id"],value="\n".join(s for s,_,_ in b["lines"]),vertex="1",parent="1",style=f'rounded=1;whiteSpace=wrap;html=0;align=left;spacing=8;fontFamily=DejaVu Sans;fontSize=8.5;fillColor={b["fill"]};')
        ET.SubElement(cell,"mxGeometry",x=str(x),y=str(z),width=str(w),height=str(h),attrib={"as":"geometry"})
    pdf.showPage();pdf.save()
    # Matplotlib's PostScript backend embeds the same free TrueType fonts.
    from matplotlib.patches import FancyBboxPatch, Polygon
    fig=plt.figure(figsize=(WIDTH/72,height/72))
    ax=fig.add_axes((0,0,1,1));ax.set_xlim(0,WIDTH);ax.set_ylim(height,0);ax.axis("off")
    for a,b in spec["edges"]:
        aa,bb=byid[a],byid[b]
        if aa["y"]==bb["y"]:
            x1,y1=aa["x"]+aa["w"],aa["y"]+aa["h"]/2;x2,y2=bb["x"],bb["y"]+bb["h"]/2
        else:
            x1,y1=aa["x"]+aa["w"]/2,aa["y"]+aa["h"];x2,y2=bb["x"]+bb["w"]/2,bb["y"]
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",color="#617485",linewidth=.6,shrinkA=0,shrinkB=0))
    for b in boxes:
        ax.add_patch(FancyBboxPatch((b["x"],b["y"]),b["w"],b["h"],boxstyle="round,pad=0,rounding_size=4",facecolor=b["fill"],edgecolor="#617485",linewidth=.5))
        for j,(line,font,size) in enumerate(b["lines"]):
            ax.text(b["x"]+8,b["y"]+13+j*11,line,fontsize=size,fontfamily="DejaVu Sans",weight="bold" if font.endswith("Bold") else "normal",va="baseline",color="#17364D")
    fig.savefig(ART/(stem+".eps"),bbox_inches=None);plt.close(fig)
    svg.append("</svg>");(ART/(stem+".svg")).write_text("\n".join(svg),encoding="utf-8")
    ET.indent(xml);ET.ElementTree(xml).write(ART/(stem+".drawio"),encoding="utf-8",xml_declaration=True)
    MANIFEST[stem]={"kind":"implementation_diagram","frozen_geometry_sha256":digest(REPRO/"accepted/stage04"/(stem+"_diagram.json")),"all_box_labels_retained":True,"width_mm":119,"height_mm":height/72*25.4,"minimum_label_font_pt":8.5,"font":"embedded DejaVu Sans","removed_inside_artwork":["repeated overall title","source notes already in caption and audit"]}
    print("prepared",stem,flush=True)

def main():
    ART.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location("frozen_figures",REPRO/"analysis/legacy/build_figures_stage4.py")
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.TABLES=REPRO/"accepted/stage04";old.save=save_plot
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","savefig.bbox":None})
    old.data_figures()
    fontdir=Path(matplotlib.get_data_path())/"fonts/ttf"
    for name,file in (("FigureSans","DejaVuSans.ttf"),("FigureSansBold","DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
    poppler=Path(os.environ["POPPLER_BIN"]) if os.environ.get("POPPLER_BIN") else None
    for stem in ("fig_model_structure","fig_strategy_framework","fig_b_welfare_collapse"):
        diagram(stem)
        rasterizer=(str(poppler/("pdftoppm.exe" if os.name=="nt" else "pdftoppm")) if poppler else shutil.which("pdftoppm"))
        if rasterizer:
            subprocess.run([rasterizer,"-png","-singlefile","-r","300",str(ART/(stem+".pdf")),str(ART/stem)],check=True,capture_output=True)
    mapping=json.loads((STAGE/"audit/cross_document_references.json").read_text())
    (ART/"submission_files").mkdir(exist_ok=True)
    for key,prefix in (("main_figures","Fig"),("supplement_figures","FigS")):
        for i,source in enumerate(mapping[key]):
            stem=source.removesuffix(".drawio")
            for ext in ("pdf","eps"):
                shutil.copy2(ART/(stem+"."+ext),ART/"submission_files"/(prefix+str(i+1)+"."+ext))
    for stem,entry in MANIFEST.items():
        entry["outputs"]={p.name:digest(p) for p in ART.glob(stem+".*")}
        page=PdfReader(ART/(stem+".pdf")).pages[0]
        assert abs(float(page.mediabox.width)-WIDTH)<.02
    (REPORTS/"artwork_manifest.json").write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"figures":len(MANIFEST),"final_width_mm":119,"all_vector_pdfs_and_eps":True}))

if __name__=="__main__":main()
