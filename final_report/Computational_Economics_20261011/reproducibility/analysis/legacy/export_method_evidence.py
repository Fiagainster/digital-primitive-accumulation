"""Export effective parameters, initial conditions, seeds and actual mixture weights.

Read-only with respect to the received model and raw experiment archive.
"""
from pathlib import Path
from dataclasses import asdict
import ast
import csv
import hashlib
import json
import sys
import warnings

sys.dont_write_bytecode = True
STAGE = Path(__file__).resolve().parents[1]
FIRST = STAGE.parent / "阶段01"
MODEL = FIRST / "model"
sys.path[:0] = [str(FIRST / "runtime_packages"), str(MODEL)]
import numpy as np
import yaml
from src.params.schema import get_schema
from src.params.country_profiles import COUNTRY_PROFILES
from src.params.presets import SCENARIO_PRESETS
from src.models.mixed_strategy import MixedStrategy
from research_scripts.common.runner import build_param_sets

MASTER_SEED = 20261002

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save_json(name, value):
    (STAGE / "tables" / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def save_csv(name, rows):
    with (STAGE / "tables" / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def seed_for(tag, key, rep):
    return int.from_bytes(hashlib.sha256(f"{MASTER_SEED}|{tag}|{key}|{rep}".encode("utf-8")).digest()[:4], "big")

def main():
    (STAGE / "tables").mkdir(parents=True, exist_ok=True)
    schema = get_schema()
    nominal = {}
    def flatten(obj):
        for k, v in obj.items():
            if isinstance(v, dict):
                flatten(v)
            elif isinstance(v, (int, float)):
                nominal[k] = float(v)
    raw = yaml.safe_load((MODEL / "config" / "default.yaml").read_text(encoding="utf-8"))
    for section in ("structural", "strategy", "scenario_defaults", "meta", "simulation", "cdi_weights"):
        if section in raw:
            flatten(raw[section])
    rows = []
    for pid, spec in schema._params.items():
        declared = nominal.get(pid)
        valid = declared is not None and spec.domain[0] <= declared <= spec.domain[1]
        rows.append({"parameter_id": pid, "symbol": spec.symbol, "description": spec.description,
                     "layer": spec.layer, "lower": spec.domain[0], "upper": spec.domain[1],
                     "effective_default": spec.default, "yaml_nominal": declared,
                     "yaml_status": "accepted" if valid else ("out_of_domain_rejected" if declared is not None else "schema_fallback"),
                     "source": "src/params/schema.py + config/default.yaml"})
    save_csv("effective_parameter_registry.csv", rows)
    save_json("effective_parameter_registry.json", rows)
    profiles = {name: profile.to_params_dict() for name, profile in COUNTRY_PROFILES.items()}
    save_json("profile_initial_conditions.json", profiles)
    ic_rows = []
    for name, values in profiles.items():
        for pid, value in values.items():
            ic_rows.append({"profile": name, "parameter_id": pid, "assigned_value": value, "source": "src/params/country_profiles.py"})
    save_csv("profile_initial_conditions.csv", ic_rows)
    scenarios = {name: fn() for name, fn in SCENARIO_PRESETS.items()}
    save_json("scenario_inputs.json", scenarios)
    save_csv("scenario_inputs.csv", [{"scenario": name, **values} for name, values in scenarios.items()])
    effective = {f"{p}|{s}": build_param_sets("A", s, p) for p in profiles for s in scenarios}
    save_json("effective_profile_scenario_parameters.json", effective)

    batches = []
    seeds = []
    metadata_inventory = {}
    for path in sorted((MODEL / "experiments_runs" / "v38").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "results" not in data:
            continue  # provenance/run-plan metadata are not experiment batches
        records = [r for r in data["results"] if r and r.get("ok")]
        horizons = sorted(set(r["summary"].get("T_max", 80.0 if path.stem in {"t80", "calibration_t80", "exp4"} else 50.0) for r in records))
        batches.append({"batch": path.stem, "records": len(data["results"]), "ok": len(records),
                        "horizon": ";".join(str(x) for x in horizons), "sha256": digest(path)})
        metadata_inventory[path.stem] = {}
        for key in sorted({k for r in records for k in r["meta"]}):
            values = {json.dumps(r["meta"].get(key), sort_keys=True) for r in records}
            if key not in {"key", "lambda_A", "lambda_B", "lambda_C", "idx"}:
                metadata_inventory[path.stem][key] = {"unique_count": len(values), "values": [json.loads(x) for x in sorted(values)] if len(values) <= 32 else "see raw archive"}
        for i, r in enumerate(records):
            m = r["meta"]
            recomputable = "key" in m and "rep" in m and path.stem not in {"ts_viable"}
            derived = seed_for(m.get("batch", path.stem), m.get("key", ""), m.get("rep", 0)) if recomputable else None
            derivation = "master|tag|key|rep" if recomputable else "saved seed; see source selection/driver"
            if path.stem == "ts_calibration":
                parts = f'calib_ts|{m["profile"]}|{m["regime"]}|{m["rep"]}'
                derived = int.from_bytes(hashlib.sha256(parts.encode("utf-8")).digest()[:4], "big")
                recomputable = True
                derivation = "calib_ts|profile|regime|rep (no master prefix)"
            seeds.append({"batch": path.stem, "row": i, "profile": m.get("profile", ""),
                          "scenario": m.get("scenario", ""), "regime": m.get("regime", ""),
                          "key": m.get("key", ""), "rep": m.get("rep", ""), "seed": r["seed"],
                          "derived_seed_candidate": derived,
                          "seed_derivation": derivation,
                          "candidate_matches": derived == r["seed"] if recomputable else "not_applicable"})
    save_csv("batch_inventory.csv", batches)
    save_csv("seed_inventory.csv", seeds)
    save_json("batch_metadata_shapes.json", metadata_inventory)

    mixture = json.loads((MODEL / "experiments_runs" / "v38" / "mixture.json").read_text(encoding="utf-8"))["results"]
    mix_rows = []
    for r in mixture:
        if r["meta"]["rep"] != 0:
            continue
        m = r["meta"]
        drawn = tuple(m["lambda_" + a] for a in "ABC")
        p = build_param_sets("M", m["scenario"], m["profile"], {"mixed_lambda": list(drawn), "mixed_name": "audit"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            actual = MixedStrategy(p, lambda_vec=drawn).lambda_vec
        mix_rows.append({"profile": m["profile"], "scenario": m["scenario"], "draw_index": m["idx"],
                         **{"drawn_lambda_" + a: v for a, v in zip("ABC", drawn)},
                         **{"accumulation_lambda_" + a: v for a, v in zip("ABC", actual)},
                         "changed_by_constraint": not np.allclose(drawn, actual, atol=1e-12, rtol=0),
                         "engine_signature_uses": "drawn_lambda", "accumulation_uses": "corrected_lambda"})
    save_csv("mixture_drawn_and_effective_weights.csv", mix_rows)
    mix_report = {"draws": len(mix_rows), "changed_draws": sum(r["changed_by_constraint"] for r in mix_rows),
                  "unique_accumulation_vectors_rounded_12dp": len({tuple(round(r["accumulation_lambda_"+a], 12) for a in "ABC") for r in mix_rows}),
                  "corrected_weight_limits": {"dominant_min": 0.5, "secondary_max": 0.3, "tertiary_max": 0.2},
                  "engine_signature_uses_original_draw": True,
                  "normalization_cancels_common_allocation_penalty": True,
                  "implication": "Archive samples Dirichlet inputs; it does not implement a uniform unrestricted-simplex accumulation experiment. Counts and outcomes are retained; design interpretation is qualified."}
    save_json("mixture_implementation_audit.json", mix_report)

    source_rows = []
    for path in sorted((MODEL / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                source_rows.append({"file": str(path.relative_to(MODEL)).replace("\\", "/"),
                                    "name": node.name, "kind": type(node).__name__,
                                    "start_line": node.lineno, "end_line": node.end_lineno,
                                    "sha256": digest(path)})
    save_csv("method_source_index.csv", source_rows)
    summary = {"registered_parameters": len(rows), "accepted_yaml": sum(r["yaml_status"] == "accepted" for r in rows),
               "rejected_yaml": [r for r in rows if r["yaml_status"] == "out_of_domain_rejected"],
               "batches": len(batches), "executions": sum(r["records"] for r in batches), "mixture": mix_report}
    save_json("method_evidence_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
