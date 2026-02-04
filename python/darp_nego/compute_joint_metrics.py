#!/usr/bin/env python3
"""Compute scenario metrics, DARP solution metrics, and joint embeddings."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from compute_metrics import compute_all_metrics, generate_plots, write_outputs
from metrics.dataset_metrics import build_feature_vectors, pca_embedding, umap_embedding
from metrics.io import load_company_cases
from metrics.plots import save_embedding_scatter, save_embedding_scatter_plain
from solve_and_extract_metrics import solve_and_extract_metrics


def _load_darp_metrics(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    with open(path, "r") as handle:
        data = json.load(handle)
    return pd.DataFrame(data)


def _aggregate_darp_metrics(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=["case_id"])

    df = records.copy()
    df["case_id"] = df["case_name"]

    numeric_cols = [
        "num_vehicles",
        "num_clients",
        "solve_time_sec",
        "solution_cost",
        "total_time",
        "total_waiting_time",
        "vehicles_used",
        "vehicle_utilization",
        "served_clients",
        "total_delays",
        "avg_total_delay",
        "max_total_delay",
        "max_route_time",
        "avg_route_time",
        "capacity_utilization",
        "time_utilization",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    group = df.groupby("case_id")

    status = group["solve_status"].agg(
        solve_ok_rate=lambda s: (s == "ok").mean(),
        solve_infeasible_rate=lambda s: (s == "infeasible").mean(),
    )
    counts = group["company_name"].nunique().rename("company_count")

    available_numeric = [col for col in numeric_cols if col in df.columns]
    numeric_summary = group[available_numeric].agg(["mean", "std"]) if available_numeric else pd.DataFrame()
    if not numeric_summary.empty:
        numeric_summary.columns = [f"{col}_{stat}" for col, stat in numeric_summary.columns]

    combined = pd.concat([counts, status, numeric_summary], axis=1).reset_index()
    return combined


def _build_combined_features(
    case_df: pd.DataFrame,
    company_df: pd.DataFrame,
    darp_case_df: pd.DataFrame,
) -> pd.DataFrame:
    scenario_features = build_feature_vectors(case_df, company_df)
    combined = scenario_features.merge(darp_case_df, on="case_id", how="left")
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    return combined[["case_id"] + numeric_cols]


def _write_combined_outputs(
    out_dir: str,
    darp_case_df: pd.DataFrame,
    combined_features: pd.DataFrame,
    seed: int,
) -> Dict[str, Optional[pd.DataFrame]]:
    combined_dir = os.path.join(out_dir, "combined")
    plots_dir = os.path.join(combined_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not darp_case_df.empty:
        darp_case_df.to_csv(os.path.join(combined_dir, "darp_case_metrics.csv"), index=False)
    if not combined_features.empty:
        combined_features.to_csv(
            os.path.join(combined_dir, "combined_feature_vectors.csv"),
            index=False,
        )

    embeddings: Dict[str, Optional[pd.DataFrame]] = {"pca": None, "umap": None}
    if not combined_features.empty:
        pca_df, _ = pca_embedding(combined_features, seed=seed)
        pca_df.to_csv(os.path.join(combined_dir, "pca_2d.csv"), index=False)
        embeddings["pca"] = pca_df

        umap_df = umap_embedding(combined_features, seed=seed)
        if umap_df is not None:
            umap_df.to_csv(os.path.join(combined_dir, "umap_2d.csv"), index=False)
            embeddings["umap"] = umap_df

        save_embedding_scatter_plain(
            pca_df,
            os.path.join(plots_dir, "pca_scatter.png"),
            "pc1",
            "pc2",
        )
        for color_col in ["gini_requests", "global_density", "solution_cost_mean", "solve_ok_rate"]:
            if color_col in combined_features.columns:
                save_embedding_scatter(
                    pca_df,
                    combined_features,
                    os.path.join(plots_dir, f"pca_scatter_{color_col}.png"),
                    "pc1",
                    "pc2",
                    color_col,
                )
        if umap_df is not None:
            save_embedding_scatter_plain(
                umap_df,
                os.path.join(plots_dir, "umap_scatter.png"),
                "umap1",
                "umap2",
            )
            for color_col in ["gini_requests", "global_density", "solution_cost_mean", "solve_ok_rate"]:
                if color_col in combined_features.columns:
                    save_embedding_scatter(
                        umap_df,
                        combined_features,
                        os.path.join(plots_dir, f"umap_scatter_{color_col}.png"),
                        "umap1",
                        "umap2",
                        color_col,
                    )

    return embeddings


def _natural_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute scenario metrics + DARP solution metrics + joint embeddings."
    )
    parser.add_argument("--input", required=True, help="Path to company_cases.json")
    parser.add_argument("--out", required=True, help="Output directory for all metrics")
    parser.add_argument("--case-id", default=None, help="Optional case id to process")
    parser.add_argument("--case-ids", default=None, help="Comma-separated list of case ids to process")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Process at most this many cases (after filters, deterministic order).",
    )
    parser.add_argument(
        "--darp-metrics",
        default=None,
        help="Optional path to *_darp_metrics_*.json to skip solving",
    )
    parser.add_argument(
        "--solve-repeats",
        type=int,
        default=10,
        help="Number of solver runs per company to average metrics.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    case_ids = [c.strip() for c in args.case_ids.split(",")] if args.case_ids else None
    if case_ids:
        case_ids = sorted(case_ids, key=_natural_key)

    data = load_company_cases(args.input, case_id=args.case_id, case_ids=case_ids)
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise ValueError("--max-cases must be >= 1")
        case_keys = [k for k, v in data.items() if isinstance(v, dict) and "companies" in v]
        case_keys = sorted(case_keys, key=_natural_key)[: args.max_cases]
        data = {k: data[k] for k in case_keys}
    if case_ids is None and args.case_id is None:
        case_ids_for_darp = sorted(
            [k for k, v in data.items() if isinstance(v, dict) and "companies" in v]
            , key=_natural_key
        )
    else:
        case_ids_for_darp = sorted(case_ids, key=_natural_key) if case_ids else case_ids
    results = compute_all_metrics(data, seed=args.seed)

    embeddings = write_outputs(
        args.out,
        results["case"],
        results["company"],
        results["features"],
        seed=args.seed,
    )
    generate_plots(
        data,
        args.out,
        results["case"],
        results["company"],
        results["features"],
        embeddings,
    )

    if args.darp_metrics:
        darp_metrics_path = args.darp_metrics
    else:
        darp_metrics_path = solve_and_extract_metrics(
            args.input,
            args.out,
            case_id=args.case_id,
            case_ids=case_ids_for_darp,
            solve_repeats=args.solve_repeats,
        )

    darp_df = _load_darp_metrics(darp_metrics_path)
    darp_case_df = _aggregate_darp_metrics(darp_df)
    combined_features = _build_combined_features(
        results["case"],
        results["company"],
        darp_case_df,
    )

    _write_combined_outputs(
        args.out,
        darp_case_df,
        combined_features,
        seed=args.seed,
    )

    print(f"Scenario metrics written under: {os.path.join(args.out, 'dataset')}")
    print(f"DARP metrics JSON: {darp_metrics_path}")
    print(f"Combined metrics written under: {os.path.join(args.out, 'combined')}")


if __name__ == "__main__":
    main()
