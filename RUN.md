# Runbook

## 1) Environment

```bash
cd darp_nego-development_new
python3 -m venv .venv
source .venv/bin/activate
python --version  # use the same Python version on both PCs, preferably Python 3.11
python -m pip install -r requirements.txt
```

## 2) Main Negotiation Run

Entry point:

```bash
python -m darp_nego.runners.run_negotiation
```

Strategy selection is configured in:

- `darp_nego/runners/config.py`
  - `RUN_STRATEGY = "heuristic"` or `RUN_STRATEGY = "learning"` or `RUN_STRATEGY = "heuristic_partial"`
  - `DEBUG_SINGLE_SCENARIO = False/True`
  - Fair 10-seed mode:
    - uncomment `FAIR_COMPARISON_MODE = True`
    - keep `FAIR_COMPARISON_SEEDS` identical on both PCs
    - keep `ORTOOLS_NUM_SEARCH_WORKERS = 1`
  - In fair mode, logs include `seedXX` in session names and runner prints scenario SHA256 for cross-PC baseline check.

## 3) Experiment / Analysis Scripts

After `python -m darp_nego.runners.run_negotiation`, logs are written under `logs/<session_id>/`.
Use these scripts for aggregate analysis:

```bash
python stats.py
python analyze.py
```

- `stats.py` reads negotiation logs and reports full/partial acceptance, applied swaps, cost changes, and round-level dynamics.
- `analyze.py` reads routing + negotiation logs and builds cost tables/plots grouped by agent count.

All canonical scripts are under `experiments/` and end with `_impl.py`.

Examples:

```bash
python experiments/solve_and_extract_metrics_impl.py --help
python experiments/compute_metrics_impl.py --help
python experiments/compute_joint_metrics_impl.py --help
python experiments/embedding_cluster_anova_impl.py --help
python experiments/embedding_feature_reports_impl.py --help
python experiments/visualize_darp_metrics_impl.py --help
```

## 4) Tests

```bash
python -m unittest discover -s darp_nego/test
```

## 5) Notes

- Canonical code namespaces:
  - `darp_nego.core.*`
  - `darp_nego.protocols.*`
  - `darp_nego.logging.*`
  - `darp_nego.runners.*`
- `darp_nego/protocol/` remains only for non-code artifacts.
