# DARP Negotiation

This repository contains a multi-company DARP (Dial-a-Ride Problem) negotiation framework for comparing heuristic and learning-based negotiation strategies under the same scenario and baseline routing conditions.

The current codebase supports:

- `heuristic`: heuristic proposal generation, swaps only on full acceptance.
- `heuristic_partial`: heuristic proposal generation, swaps on full or partial acceptance.
- `learning`: learning/model-guided proposal generation, swaps on full or partial acceptance.
- deterministic fair-comparison runs with shared seed lists and OR-Tools single-worker solving.
- JSON scenario loading, negotiation/routing logging, per-session metric summaries, and aggregate analysis scripts.

## Repository Layout

- `darp_nego/`: main package.
- `darp_nego/core/`: DARP domain objects, OR-Tools solver integration, negotiators, outcomes, utilities, learning models.
- `darp_nego/protocols/`: negotiation mechanisms for heuristic, heuristic-partial, and learning strategies.
- `darp_nego/logging/`: negotiation and routing log writers.
- `darp_nego/runners/`: run entrypoint and runtime configuration.
- `metrics/`: per-session negotiation/routing metric summaries and optional plots.
- `experiments/`: dataset-level metrics and visualization scripts.
- `scenario_generation/`: scenario generation, visualization, and generated scenario JSONs.
- `data/companies/`: small static/debug data.
- `stats.py`: aggregate negotiation-log analysis.
- `analyze.py`: aggregate routing/cost analysis.
- `RUN.md`: concise command runbook.

## Environment

Use the same Python version on both PCs for fair comparison. Python 3.11 is the intended version.

```bash
cd darp_nego-development_new
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install -r requirements.txt
```

`requirements.txt` pins the main solver and numerical stack, including OR-Tools, NumPy, SciPy, pandas, scikit-learn, NetworkX, and PyTorch.

## Configuration

Runtime configuration is in:

```text
darp_nego/runners/config.py
```

Select one strategy by commenting/uncommenting:

```python
RUN_STRATEGY = "heuristic"
# RUN_STRATEGY = "learning"
# RUN_STRATEGY = "heuristic_partial"
```

Fair-comparison mode:

```python
FAIR_COMPARISON_MODE = True
FAIR_COMPARISON_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
ORTOOLS_NUM_SEARCH_WORKERS = 1
```

For two-PC comparisons, keep the same:

- git commit
- Python version
- `requirements.txt`
- `SCENARIO_JSON_PATH`
- `FAIR_COMPARISON_SEEDS`
- `ORTOOLS_NUM_SEARCH_WORKERS = 1`

Scenario selection:

```python
USE_LATEST_SCENARIO = False
SCENARIO_JSON_PATH = "scenario_generation/data/llm_generated/company_cases_demo_2x3.json"
# SCENARIO_JSON_PATH = "scenario_generation/data/llm_generated/company_cases_merged_4_6_8.json"
```

Relative scenario paths are resolved from the repository root.

## Running Negotiation

Main entrypoint:

```bash
python -m darp_nego.runners.run_negotiation
```

With the current fair-mode config, this runs every selected scenario for each configured seed. Session names include dataset name, case id, strategy, seed, and timestamp.

The runner prints the selected scenario file and SHA256 hash in fair mode. Use that hash to verify that both PCs are using the exact same input file.

## Outputs

Each negotiation session writes logs under:

```text
logs/<session_id>/
```

Important files:

```text
logs/<session_id>/negotiation/negotiation_<session_id>.json
logs/<session_id>/negotiation/negotiation_<session_id>.txt
logs/<session_id>/routing/routing_<session_id>.json
```

Per-session metric summaries are written under:

```text
metrics/<session_id>/negotiation/
metrics/<session_id>/routing/
```

Negotiation metrics include full acceptance, partial acceptance, applied swaps, utility changes, and agreement status. Routing metrics include cost/time/utilization and route-level summaries.

Per-session PNG plots are controlled from `darp_nego/runners/config.py`:

```python
SAVE_PER_SESSION_METRIC_PLOTS = False
SAVE_PER_SESSION_METRIC_TEXT = True
```

For fair multi-seed runs, keeping per-session plots disabled is recommended. The raw JSON/TXT summaries are preserved for every case/seed, and aggregate plots should be generated with `stats.py` and `analyze.py` after the batch finishes.

## Aggregate Analysis

After a run finishes:

```bash
python stats.py
python analyze.py
```

`stats.py` reads negotiation logs and produces aggregate CSVs/plots for:

- full acceptance rate
- partial acceptance rate
- applied swaps
- total proposed/accepted transfers
- cost changes
- mean/std/95% confidence intervals for key aggregate metrics
- round-level negotiation dynamics

`analyze.py` reads routing and negotiation logs and produces:

- average initial/final cost by agent count
- 95% confidence intervals for cost and approval-rate summaries
- cost change tables
- Table 2 style negotiation summary
- cost distribution boxplots
- total-cost-vs-agent-count plots

Both scripts use the dataset selected in `darp_nego/runners/config.py` when `USE_LATEST_SCENARIO = False`.

## Scenario Files

The committed demo scenario is:

```text
scenario_generation/data/llm_generated/company_cases_demo_2x3.json
```

It contains:

- 2 cases with 4 companies
- 2 cases with 6 companies
- 2 cases with 8 companies

Large raw scenario JSONs are intentionally ignored by git because they exceed GitHub's recommended or hard file-size limits. Keep large scenario files locally or share them through external storage, then point `SCENARIO_JSON_PATH` to the local relative path.

Each case stores its own:

- `time_matrix`
- `coordinates`
- `companies`
- vehicles and clients

Do not globalize `time_matrix` or `coordinates` unless every case uses the exact same location universe.

## Strategy Behavior

`heuristic`:

- generates heuristic/farthest-distance proposals.
- applies swaps only when all agents accept.
- partial acceptance is logged as zero.

`heuristic_partial`:

- uses the same heuristic proposal style.
- extracts acceptable sub-swaps.
- applies swaps when the accepted subgraph is valid, even without full acceptance.

`learning`:

- starts from heuristic proposals while cold.
- trains per-agent swap models from observed responses.
- uses model-guided proposals after enough samples.
- supports both full and partial swap application.

## Fair Comparison Protocol

For a fair heuristic-vs-learning comparison:

1. Use the same scenario JSON on both PCs.
2. Confirm the printed scenario SHA256 matches.
3. Use the same git commit.
4. Use the same Python version and pinned `requirements.txt`.
5. Keep `ORTOOLS_NUM_SEARCH_WORKERS = 1`.
6. Use the same `FAIR_COMPARISON_SEEDS`.
7. Run one strategy per PC by changing only `RUN_STRATEGY`.
8. Compare results by matching `case_id + seed`.

The OR-Tools seed and single-worker setting reduce solver nondeterminism. Single-worker solving matters because multi-worker search can produce different paths across machines even with the same seed.

## Scenario Generation

LLM-based scenario generation lives in:

```text
scenario_generation/run_llm.py
scenario_generation/llm_scenario.py
```

Example:

```bash
python scenario_generation/run_llm.py --help
```

Other scenario generation and visualization utilities are under `scenario_generation/`.

## Experiments

Dataset-level metrics and visualizations live under `experiments/`.

Examples:

```bash
python experiments/solve_and_extract_metrics_impl.py --help
python experiments/compute_metrics_impl.py --help
python experiments/compute_joint_metrics_impl.py --help
python experiments/embedding_cluster_anova_impl.py --help
python experiments/embedding_feature_reports_impl.py --help
python experiments/visualize_darp_metrics_impl.py --help
```

## Tests

```bash
python -m unittest discover -s darp_nego/test
```

The tests are useful as regression signals, but some legacy assumptions may not cover the full current fair-comparison workflow.

## Notes

- Use `python -m pip`, not bare `pip`, inside the virtual environment.
- If `.venv` was created before the directory refactor, remove it and recreate it.
- Generated logs, metrics, large local scenario JSONs, and cache files should not be committed.
