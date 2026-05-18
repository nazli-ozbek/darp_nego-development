# DARP Negotiation

This repository contains a negotiation-based framework for multi-company DARP (Dial-a-Ride Problem), with:

- `heuristic` strategy (classic full-acceptance mechanism)
- `learning` strategy (partial+full acceptance capable mechanism)
- scenario generation (including LLM-based generation)
- negotiation/routing metrics and analysis pipelines

The project has been refactored to a canonical architecture (no extra `python/` layer).

## Repository Layout

Top-level:

- `darp_nego/` core package
- `scenario_generation/` scenario generation scripts and data workflows
- `experiments/` analysis/metrics scripts (`*_impl.py`)
- `metrics/` metrics modules
- `data/` static company/scenario data
- `scripts/` helper scripts
- `stats.py` and `stats/` log-level reporting

Inside `darp_nego/`:

- `core/` domain models and base components
- `protocols/` negotiation mechanism implementations
- `logging/` negotiation/routing logging
- `runners/` runtime entrypoints/config
- `test/` unit tests

See also:

- `ARCHITECTURE.md`
- `RUN.md`

## Quick Start

```bash
cd darp_nego-development_new
python3 -m venv .venv
source .venv/bin/activate
python --version  # use the same Python version on both PCs, preferably Python 3.11
python -m pip install -r requirements.txt
```

## Main Negotiation Run

Canonical entrypoint:

```bash
python -m darp_nego.runners.run_negotiation
```

Strategy selection is configured in:

- `darp_nego/runners/config.py`
  - `RUN_STRATEGY = "heuristic"` or `RUN_STRATEGY = "learning"`
  - `DEBUG_SINGLE_SCENARIO = False/True`

Notes:

- `heuristic` uses classic full-acceptance style flow.
- `learning` supports partial swaps and full agreement.

## Scenario Generation

Standard scenario generation lives under `scenario_generation/`.

LLM-based scenario generation is available via:

- `scenario_generation/run_llm.py`
- `scenario_generation/llm_scenario.py`

Example:

```bash
python scenario_generation/run_llm.py --help
```

## Metrics and Analysis

Canonical analysis scripts are under `experiments/`:

- `compute_metrics_impl.py`
- `compute_joint_metrics_impl.py`
- `solve_and_extract_metrics_impl.py`
- `embedding_cluster_anova_impl.py`
- `embedding_feature_reports_impl.py`
- `visualize_darp_metrics_impl.py`
- `route_visuals_impl.py`

Examples:

```bash
python experiments/solve_and_extract_metrics_impl.py --help
python experiments/compute_metrics_impl.py --help
python experiments/compute_joint_metrics_impl.py --help
```

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s darp_nego/test
```

Current test suite contains legacy assumptions and may fail even on historical variants; use it mainly as a regression signal, not a release gate by itself.

## Packaging

Package config is in:

- `setup.py` (uses `setuptools.find_packages`)
