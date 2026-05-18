# Runbook

## 1) Environment

```bash
cd /Users/nazliozbek/Desktop/darp-nego/darp_nego-development_new
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Main Negotiation Run (Canonical)

Entry point:

```bash
python -m darp_nego.runners.run_negotiation
```

Strategy selection is configured in:

- `darp_nego/runners/config.py`
  - `RUN_STRATEGY = "heuristic"` or `RUN_STRATEGY = "learning"`
  - `DEBUG_SINGLE_SCENARIO = False/True`

## 3) Experiment / Analysis Scripts (Canonical)

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
- Legacy top-level wrapper scripts were removed.
- `darp_nego/protocol/` remains only for non-code artifacts.

