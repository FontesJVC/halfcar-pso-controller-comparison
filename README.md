# PSO comparison for a MIMO half-car active suspension

This repository contains the numerical code and run-level data associated with a comparative study of three state-feedback controller parameterizations for a four-degree-of-freedom half-car active suspension:

- **PSO-LQR**: PSO optimizes the diagonal entries of the LQR weighting matrices `Q` and `R` (10 decision variables).
- **PSO-PPC**: PSO optimizes four complex-conjugate closed-loop pole pairs (8 decision variables), and `control.place` maps each requested spectrum to one feedback matrix `K`.
- **PSO-Direct**: PSO optimizes the 16 entries of the `2 x 8` feedback matrix `K` directly.

All three strategies are evaluated using the same half-car model, road disturbance, simulation settings, stability criterion, and outer multiobjective cost.

## Repository contents

```text
halfcar_pso_comparison.py
requirements.txt
README.md
pso_repetitions_results/
    runs.csv
    summary.csv
    experiment_config.json
    best_controllers.json
```

The analysis script also regenerates the manuscript and audit figures (boxplots, run-by-run plots, and best-controller time responses) from the stored numerical results.

### Main outputs

- `runs.csv`: complete run-level results for the 30 independent runs of each method (90 runs total), including seed, total cost, weighted and raw cost components, optimized coordinates, gain matrix `K`, stability result, and runtime.
- `summary.csv`: descriptive statistics used in the manuscript.
- `experiment_config.json`: model parameters, road input, cost weights, PSO settings, bounds, stability tolerance, seeds, and software versions from the experiment environment.
- `best_controllers.json`: human-readable description of the best controller for each method, including:
  - LQR: optimized `Q` and `R` diagonals, `K`, and achieved closed-loop poles;
  - PPC: optimized pole coordinates, requested poles, achieved poles, pole-matching error, and `K`;
  - Direct: optimized `K` and achieved closed-loop poles.

## Installation

The experiment was run with the package versions recorded in `experiment_config.json`. To create a matching Python environment:

```bash
python -m venv .venv
```

Activate the environment and install:

```bash
pip install -r requirements.txt
```

## Reproduce tables and figures without rerunning PSO

The repository includes `runs.csv`, so the numerical summaries, controller exports, and figures can be regenerated without repeating the optimization campaign:

```bash
python halfcar_pso_comparison.py --results-only
```

This mode preserves the recorded `experiment_config.json`.

## Rerun the complete optimization experiment

To execute all 30 independent runs for each of the three methods:

```bash
python halfcar_pso_comparison.py --run-optimization
```

The seeds are 1000--1029. Completed method/seed pairs can be resumed from `runs.csv` when resume mode is enabled.

The nominal search settings are:

| Method | Variables | Particles | Iterations | Nominal evaluations/run |
|---|---:|---:|---:|---:|
| PSO-LQR | 10 | 50 | 100 | 5000 |
| PSO-PPC | 8 | 50 | 100 | 5000 |
| PSO-Direct | 16 | 60 | 110 | 6600 |

These are **method-specific search budgets**, not a strict equal-budget benchmark. The conclusions should be interpreted under the reported optimization settings.

## Objective and numerical conventions

The common outer objective contains four weighted contributions:

1. body vertical and pitch angular accelerations (comfort term);
2. front and rear tire deflections (secondary road-holding penalty);
3. squared actuator forces (control-effort proxy);
4. squared time derivative of actuator force (smoothness penalty).

The controller synthesis criterion of LQR is distinct from this common outer objective. For PPC, the pole-placement routine selects one admissible MIMO gain matrix for each requested pole set according to its internal robust pole-assignment procedure; the resulting `K` is then evaluated using the same outer objective as the other methods.

## Reproducibility

The repository provides the complete optimization/evaluation script, all 90 final optimization outcomes, cost decompositions, optimized controller gains, LQR weights, requested and achieved PPC poles, seeds, numerical configuration, and software versions needed to reproduce the reported numerical summaries and figures.

## Double-blind review note

This is an author-identifying public repository. If the associated manuscript is under double-blind review, do not place this public GitHub URL in the blinded manuscript unless the journal explicitly permits identifying data/code links during review. An anonymized repository or supplementary-file mechanism can be used instead.
