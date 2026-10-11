"""Assert seed reproducibility, full trajectories and real evaluation fields.

Five verification runs only; no archived experiment file is overwritten.
Missing YAML, missing CDI/five_dimensions, invalid required summary fields,
same-seed differences, or unchanged different-seed trajectories cause exit 1.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/"runtime_packages"))
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import yaml
from src.core.engine import SimulationEngine
from src.core.config import SimulationConfig
from src.params.schema import get_schema, _find_default_config
from research_scripts.common.runner import build_param_sets


def verify_yaml():
    path=ROOT/"config/default.yaml"
    assert Path(_find_default_config()).resolve()==path.resolve(), "Schema is reading another YAML"
    raw=yaml.safe_load(path.read_text(encoding="utf-8"))
    schema=get_schema()
    found={}
    def flatten(node):
        for k,v in node.items():
            if isinstance(v,dict): flatten(v)
            elif isinstance(v,(int,float)): found[k]=float(v)
    for k in ("structural","strategy","scenario_defaults","meta","simulation","cdi_weights"):
        flatten(raw.get(k,{}))
    checked=0
    ignored=[]
    for pid in schema.all_ids():
        if pid in found:
            lo,hi=schema.get(pid).domain
            if not lo<=found[pid]<=hi:
                ignored.append({"parameter":pid,"yaml_value":found[pid],"registered_domain":[lo,hi],
                                "effective_schema_default":schema.get(pid).default})
                continue
            assert np.isclose(schema.get(pid).default,found[pid],rtol=0,atol=1e-12),pid
            checked+=1
    assert checked>100, "YAML coverage unexpectedly small"
    assert schema.get("A_tau_min").default==0.08, "YAML override sentinel failed"
    return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "default_parameters_checked":checked,"loaded":True,
            "registered_schema_parameter_count":len(schema.all_ids()),
            "out_of_registered_domain_yaml_values_ignored_by_existing_loader":ignored}


def one(seed):
    p=build_param_sets(regime="B",profile="manufacturing_giant",scenario="benign_globalization")
    out=SimulationEngine(params=p,config=SimulationConfig(T_max=50,random_seed=seed,verbose=False),regime="B").run()
    sm=out["summary"]; ev=sm["evaluation"]
    assert ev["enabled"]
    assert set(ev["five_dimensions"])=={"economy","livelihood","resilience","structure","sovereignty"}
    assert np.isfinite([ev["cdi"],*ev["five_dimensions"].values()]).all()
    for k in ("survived","survival_years","final_gdp","final_welfare","composite_dependency","cost_by_group"):
        assert k in sm,k
    assert len(sm["cost_by_group"])==5
    assert np.isfinite([sm["final_gdp"],sm["final_welfare"],*sm["cost_by_group"].values()]).all()
    return out


def compare_values(a,b,rtol,atol,path="summary"):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),path+" keys"
        for k in a: compare_values(a[k],b[k],rtol,atol,path+"."+k)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path+" length"
        for i,(x,y) in enumerate(zip(a,b)): compare_values(x,y,rtol,atol,path+f"[{i}]")
    elif isinstance(a,(int,float,np.number)) and not isinstance(a,bool):
        assert np.isclose(a,b,rtol=rtol,atol=atol,equal_nan=True),path
    else: assert a==b,path


def trajectory_equal(a,b,rtol,atol):
    try:
        pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=rtol,atol=atol,check_dtype=True)
        return True
    except AssertionError: return False


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,default=ROOT.parent/"verification/determinism.json")
    ap.add_argument("--rtol",type=float,default=1e-12)
    ap.add_argument("--atol",type=float,default=1e-10)
    args=ap.parse_args()
    report={"scope":"current-environment seed/trajectory verification; not all-batch replication",
            "is_original_experiment_environment":False,"python":sys.version,"rtol":args.rtol,"atol":args.atol}
    try:
        report["yaml"]=verify_yaml()
        report["versions"]={k:importlib.metadata.version(k) for k in ("numpy","pandas","scipy","PyYAML")}
        outputs=[]
        for seed in (42,42,42,43,44):
            print(f"Running verification seed {seed}",flush=True)
            outputs.append(one(seed))
        for i in (1,2):
            compare_values(outputs[0]["summary"],outputs[i]["summary"],args.rtol,args.atol)
            assert trajectory_equal(outputs[0]["dataframe"],outputs[i]["dataframe"],args.rtol,args.atol), "Same-seed trajectory differs"
        for i in (3,4):
            assert not trajectory_equal(outputs[0]["dataframe"],outputs[i]["dataframe"],args.rtol,args.atol),"Different-seed trajectory unchanged"
            a,b=outputs[0]["dataframe"],outputs[i]["dataframe"]
            assert len(a)!=len(b) or any(not np.allclose(a[k],b[k],rtol=args.rtol,atol=args.atol,equal_nan=True)
                                       for k in ("gdp","welfare_index","K_i")),"No economic trajectory difference"
        report["runs"]=[{"seed":o["summary"]["random_seed"],"trajectory_rows":len(o["dataframe"]),
                         "trajectory_columns":len(o["dataframe"].columns),
                         "survived":bool(o["summary"]["survived"]),"final_gdp":o["summary"]["final_gdp"],
                         "final_welfare":o["summary"]["final_welfare"],"cdi":o["summary"]["evaluation"]["cdi"],
                         "five_dimensions":o["summary"]["evaluation"]["five_dimensions"]} for o in outputs]
        report.update(same_seed_full_summary_and_dataframe_passed=True,different_seeds_change_trajectories=True,passed=True)
    except Exception as exc:
        report.update(passed=False,error=f"{type(exc).__name__}: {exc}")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report["passed"] else 1


if __name__=="__main__": raise SystemExit(main())
