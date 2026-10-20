# Digital Primitive Accumulation Model (v3.8)

A six-sector political-economy simulation laboratory for the viability, multi-dimensional evaluation, and distribution of adjustment costs under digital primitive accumulation.

This repository contains the model code for the paper *Digital Primitive Accumulation and Development Trade-offs: A Computational Comparison of Three Strategies* (submitted to *Computational Economics*).

## Branches

| Branch | Contents |
|---|---|
| `main` (this branch) | Source code, configuration, tests, and documentation only. **No simulation outputs.** Clone and run to reproduce. |
| `with-results` | The full v3.8 archive: 23 experiment batches, 5,901 executions, the 2026-10 submission package, and verification reports. |

```bash
git clone -b with-results https://github.com/Fiagainster/digital-primitive-accumulation.git
```

## Model overview

- **Six sectors**: primary/resource, industry, digital infrastructure, digital applications, high-skill services, low-skill services.
- **Five aggregate household groups**: digital, industrial, primary, service, and unemployed workers. Groups are aggregate states, not individually simulated agents.
- **Three accumulation rules**: gradual (A), leapfrog (B), substitution (C), plus a restricted mixture rule.
- **Parameters**: 255 registered parameters in `config/default.yaml` with domains and source annotations; 6 country profiles and 8 external scenarios.
- **Evaluation**: five-dimensional evaluation and CDI, a household welfare index, cumulative adjustment costs for five groups, and nine monitored breakdown modes.

Design and equations: `doc/` (Chinese) and `模型公式参考.md`.

## Requirements

Python 3.10+.

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Single-profile demo (three pure strategies + five-dimensional evaluation)
python main.py demo --profile manufacturing_giant

# Multi-profile comparison
python main.py compare --profiles manufacturing_giant,digital_leapfrogger,resource_rich
```

Available profile keys: `manufacturing_giant`, `resource_rich`, `digital_leapfrogger`, `small_open`, `post_soviet`, `populous_weak_industry`.

## Reproducing the paper experiments

```bash
# List the 23 experiment batches
PYTHONIOENCODING=utf-8 python research_scripts/run_paper_experiments.py --list

# Run census and mixture first; t80 selects configurations from the census results
PYTHONIOENCODING=utf-8 python research_scripts/run_paper_experiments.py --only census mixture --workers 8
PYTHONIOENCODING=utf-8 python research_scripts/run_paper_experiments.py --only t80 --workers 8

# Recompute summary tables -> experiments_runs/v38/tables.json
PYTHONIOENCODING=utf-8 python research_scripts/analyze_v38.py

# Rebuild publication figures
PYTHONIOENCODING=utf-8 python research_scripts/make_figures_v38.py --out figures
```

Outputs are written to `experiments_runs/`, which is git-ignored. To verify reported numbers without re-running everything, use the frozen archive on the `with-results` branch.

## Randomness and reproducibility

Each run seeds NumPy from `config.random_seed` at the `SimulationEngine.run()` entry point; batch runners derive per-task sub-seeds with `SeedSequence`. The same configuration and seed reproduce identical values; different seeds produce different trajectories.

## Documentation

| File | Contents |
|---|---|
| `doc/00_设计总纲.md` … `doc/12_Computational_Economics投稿复现_20261011.md` | Design, mathematical framework, parameter system, experiment design, submission reproduction |
| `模型公式参考.md` | Equation-level reference |
| `正式实验方案.md`, `定性校准实验方案.md` | Experiment and calibration protocols |

The documentation is in Chinese. The code, configuration, and data formats are language-neutral.

## Citation

If you use this code, please cite the paper above (under review). Publication details will be added here once available.

## License

Mulan Permissive Software License, Version 2 (Mulan PSL v2). See `LICENSE`.
