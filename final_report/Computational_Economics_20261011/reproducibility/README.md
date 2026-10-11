# Online Resource 2 — data and code

**Article:** Digital Primitive Accumulation and Development Trade-offs:
A Computational Comparison of Three Strategies
**Target journal:** Computational Economics

This kit preserves the received v3.8 implementation and archive and supplies
the analyses used to audit the manuscript. It is not a new experimental run.
Author identities, sharing authorization and any license must be confirmed by
the authors in the separate centralized form before journal submission.
No public repository DOI or open-source license is asserted for these materials.

## Quick verification

Use Python 3.12.14 in a new environment, then run from this directory:

```text
python -m pip install -r requirements-verification.txt
python analysis/verify_package.py
python analysis/reproduce.py
python artwork/build_artwork.py
```

The package verifier uses only Python's standard library. The analysis command
creates `output/recomputed` and compares 31 checks against accepted CSV results
and exact permutation tests. A failure returns a nonzero exit code.
Figure reproduction writes to `output/artwork` and does not alter accepted data.
The versions above describe the revision verification environment; the original
experiment environment was not recorded and is not reconstructed by these pins.

## Contents and provenance

* `model/src`, `model/config`, `model/research_scripts`, `model/tests`: frozen
  source, configuration and original model programs. Its original
  `requirements.txt` describes broader simulation dependencies. Those extra
  dependencies are unnecessary for the archive-only verification commands.
* `model/experiments_runs/v38`: all 26 received files, including 23 batch JSON
  files containing 5,901 executions, the received `tables.json`, run log and
  analysis report. Keep received summaries distinct from new recomputations.
* `accepted/stage01`: unified aggregate ledger, effective-input inventory,
  seeds, individual-record pointers and statistical checks.
* `accepted/stage03`: implemented-method checks and archive replay material.
* `accepted/stage04`: extended results and frozen figure inputs/diagram geometry.
* `analysis/legacy`: original revision programs, retained without changes.
  `analysis/reproduce.py` redirects their file locations into a new output tree.
  The accepted Stage 1 `tables.json` is an explicit input for the later evidence
  exporter; this copy is not represented as an independently regenerated file.
* `audit`: deterministic checks, archive replay, verified reference scope,
  verification environments, submitted figure provenance and the successful
  31-check recomputation receipt.
* `files.sha256`: file-level SHA-256 list excluding this list itself and generated
  output/cache files. Filenames are UTF-8; do not resave raw JSON files.

## Limits that affect interpretation

The 450 exploratory Latin hypercube sampling records lack realized parameter
vectors. They remain in the preserved archive but are excluded from formal
parameter-robustness claims. Nine known census replays are validation records,
not additional independent observations. Profiles and scenarios are assigned
model cases, not estimated real-country failure probabilities. T80 runs are
selected configurations with fresh seeds, not paired continuations. The model
does not impose a full double-entry stock-flow-consistent accounting identity.
Mixture inputs and implementation-specific transformations remain as documented
in Online Resource 1. The supplied records determine the reported comparisons.

No new simulation is required to reproduce the manuscript's numerical tables.
Run the model's original programs only as a separate study: do not overwrite the
received archive or describe a new run as a reproduction of the unrecorded
original execution environment.
