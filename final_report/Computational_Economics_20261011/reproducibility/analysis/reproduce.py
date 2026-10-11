"""Recompute numerical results without simulations or modification of accepted data."""
from pathlib import Path
import csv
import importlib.util
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/"model"))

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"analysis/legacy"/(name+".py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def compare_csv(actual,accepted):
    with actual.open(encoding="utf-8-sig",newline="") as a,accepted.open(encoding="utf-8-sig",newline="") as b:
        aa=list(csv.DictReader(a));bb=list(csv.DictReader(b))
    return aa==bb

def main():
    output=ROOT/"output/recomputed"
    frame=output/"workflow"
    first=frame/"阶段01";third=frame/"阶段03";fourth=frame/"阶段04"
    for path in (first/"results",third/"tables",fourth/"tables"):path.mkdir(parents=True,exist_ok=True)
    module=load("stage1_results")
    module.STAGE=ROOT
    sys.argv=["stage1_results","--runs",str(ROOT/"model/experiments_runs/v38"),"--out",str(first/"results")]
    module.main()
    # This retained aggregate is an explicit input to the later evidence exporter;
    # the raw-record recomputations are independently compared below.
    shutil.copy2(ROOT/"accepted/stage01/tables.json",first/"results/tables.json")
    module=load("export_method_evidence")
    module.STAGE=third;module.FIRST=first;module.MODEL=ROOT/"model"
    module.main()
    module=load("build_evidence")
    module.STAGE=fourth;module.FIRST=first;module.RAW=ROOT/"model/experiments_runs/v38";module.OUT=fourth/"tables"
    module.main()
    checks=[]
    for name,folder in (("stage01",first/"results"),("stage03",third/"tables"),("stage04",fourth/"tables")):
        for p in sorted(folder.glob("*.csv")):
            accepted=ROOT/"accepted"/name/p.name
            if accepted.exists():checks.append({"file":name+"/"+p.name,"passed":compare_csv(p,accepted)})
    exact=json.loads((first/"results/exact_permutation_tests.json").read_text())
    accepted=json.loads((ROOT/"accepted/stage01/exact_permutation_tests.json").read_text())
    checks.append({"file":"stage01/exact_permutation_tests.json","passed":exact==accepted})
    result={"new_simulations":False,"output_directory":str(output),"checks":checks,"passed":all(c["passed"] for c in checks),"check_count":len(checks)}
    (output/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"passed":result["passed"],"check_count":len(checks),"new_simulations":False}))
    if not result["passed"]:raise SystemExit(1)

if __name__=="__main__":main()
