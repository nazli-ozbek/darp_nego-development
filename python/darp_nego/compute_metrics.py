#!/usr/bin/env python3

"""Compute scenario distribution metrics from company_cases.json."""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd

from metrics.company_metrics import compute_company_metrics
from metrics.case_metrics import compute_case_metrics
from metrics.dataset_metrics import build_feature_vectors, pca_embedding, umap_embedding
from metrics.io import load_company_cases
from metrics.plots import (
    save_case_distributions,
    save_case_metrics_table,
    save_company_distributions,
    save_company_metrics_table,
    save_dbscan_examples,
    save_embedding_scatter,
    save_embedding_scatter_plain,
    save_all_metric_distributions,
    save_feature_correlation,
    save_moran_grids,
)


def _extract_cases(data: Dict) -> Dict:
    return {k: v for k, v in data.items() if isinstance(v, dict) and "companies" in v}


def _natural_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def compute_all_metrics(
    data: Dict,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    company_rows: List[Dict] = []
    case_rows: List[Dict] = []

    cases = _extract_cases(data)
    for case_id in sorted(cases.keys(), key=_natural_key):
        case_data = cases[case_id]
        coordinates = case_data.get("coordinates", {})
        companies = case_data.get("companies", {})
        for company_id, company_data in companies.items():
            company_rows.append(
                compute_company_metrics(case_id, company_id, company_data, coordinates)
            )

        case_rows.append(compute_case_metrics(case_id, case_data, seed=seed))

    company_df = pd.DataFrame(company_rows)
    case_df = pd.DataFrame(case_rows)
    feature_df = build_feature_vectors(case_df, company_df) if not case_df.empty else pd.DataFrame()
    return {
        "company": company_df,
        "case": case_df,
        "features": feature_df,
    }


def write_outputs(
    out_dir: str,
    case_df: pd.DataFrame,
    company_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    def _reorder(df: pd.DataFrame, preferred: List[str]) -> pd.DataFrame:
        if df.empty:
            return df
        ordered = [c for c in preferred if c in df.columns]
        remaining = [c for c in df.columns if c not in ordered]
        return df[ordered + remaining]

    dataset_dir = os.path.join(out_dir, "dataset")
    cases_dir = os.path.join(out_dir, "cases")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(cases_dir, exist_ok=True)
    plots_dir = os.path.join(dataset_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    case_path = os.path.join(dataset_dir, "case_metrics.csv")
    company_path = os.path.join(dataset_dir, "company_metrics.csv")
    case_df = _reorder(
        case_df,
        ["case_id", "global_density", "morans_i", "gini_requests", "avg_overlap"],
    )
    company_df = _reorder(
        company_df,
        [
            "case_id",
            "company_id",
            "request_count",
            "service_point_count",
            "request_density",
            "avg_pairwise_distance",
            "min_pairwise_distance",
            "max_pairwise_distance",
            "dbscan_clusters",
            "avg_cluster_size",
            "noise_ratio",
            "dbscan_eps",
        ],
    )
    case_df.to_csv(case_path, index=False)
    company_df.to_csv(company_path, index=False)
    if not feature_df.empty:
        feature_df = _reorder(
            feature_df,
            [
                "case_id",
                "global_density",
                "avg_company_density",
                "std_company_density",
                "mean_company_avg_distance",
                "std_company_avg_distance",
                "mean_company_min_distance",
                "std_company_min_distance",
                "mean_company_max_distance",
                "std_company_max_distance",
                "morans_i",
                "gini_requests",
                "avg_company_dbscan_clusters",
                "avg_company_noise_ratio",
                "avg_overlap",
            ],
        )
        feature_df.to_csv(os.path.join(dataset_dir, "feature_vectors.csv"), index=False)

    embeddings = {}
    if not feature_df.empty:
        numeric_cols = feature_df.select_dtypes(include="number")
        n_samples = len(feature_df)
        n_features = numeric_cols.shape[1]
        if n_samples < 2 or n_features < 2:
            print(
                f"Skipping PCA/UMAP: need >=2 samples and >=2 numeric features "
                f"(got samples={n_samples}, features={n_features})."
            )
        else:
            pca_df, _ = pca_embedding(feature_df, seed=seed)
            pca_path = os.path.join(dataset_dir, "pca_2d.csv")
            pca_df.to_csv(pca_path, index=False)
            embeddings["pca"] = pca_df

            umap_df = umap_embedding(feature_df, seed=seed)
            if umap_df is not None:
                umap_path = os.path.join(dataset_dir, "umap_2d.csv")
                umap_df.to_csv(umap_path, index=False)
                embeddings["umap"] = umap_df

            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(6, 5))
                plt.scatter(pca_df["pc1"], pca_df["pc2"], s=30, alpha=0.8, color="#2a9d8f")
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.title("Case Metrics PCA")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "pca_scatter.png"), dpi=200)
                plt.close()
            except Exception:
                pass

    if not case_df.empty:
        for case_id in case_df["case_id"].unique():
            case_dir = os.path.join(cases_dir, case_id)
            os.makedirs(case_dir, exist_ok=True)
            case_row = case_df[case_df["case_id"] == case_id]
            company_rows = company_df[company_df["case_id"] == case_id]
            case_row.to_csv(os.path.join(case_dir, "case_metrics.csv"), index=False)
            company_rows.to_csv(os.path.join(case_dir, "company_metrics.csv"), index=False)

    return embeddings


def generate_plots(
    data: Dict,
    out_dir: str,
    case_df: pd.DataFrame,
    company_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    embeddings: Dict[str, pd.DataFrame],
) -> None:
    dataset_dir = os.path.join(out_dir, "dataset")
    save_case_metrics_table(case_df, dataset_dir)
    save_company_metrics_table(company_df, dataset_dir)
    save_case_distributions(case_df, dataset_dir)
    save_company_distributions(company_df, dataset_dir)
    save_feature_correlation(feature_df, dataset_dir)
    save_all_metric_distributions(
        company_df,
        os.path.join(dataset_dir, "plots", "company_all_metrics_distributions.png"),
        exclude_cols=["case_id", "company_id"],
    )
    if not feature_df.empty:
        save_all_metric_distributions(
            feature_df,
            os.path.join(dataset_dir, "plots", "dataset_feature_distributions.png"),
            exclude_cols=["case_id"],
        )

    if "pca" in embeddings:
        save_embedding_scatter(
            embeddings["pca"],
            case_df,
            os.path.join(dataset_dir, "plots", "pca_scatter_gini.png"),
            "pc1",
            "pc2",
            "gini_requests",
        )
        save_embedding_scatter(
            embeddings["pca"],
            case_df,
            os.path.join(dataset_dir, "plots", "pca_scatter_density.png"),
            "pc1",
            "pc2",
            "global_density",
        )

    if "umap" in embeddings:
        save_embedding_scatter_plain(
            embeddings["umap"],
            os.path.join(dataset_dir, "plots", "umap_scatter.png"),
            "umap1",
            "umap2",
        )
        save_embedding_scatter(
            embeddings["umap"],
            case_df,
            os.path.join(dataset_dir, "plots", "umap_scatter_gini.png"),
            "umap1",
            "umap2",
            "gini_requests",
        )
        save_embedding_scatter(
            embeddings["umap"],
            case_df,
            os.path.join(dataset_dir, "plots", "umap_scatter_density.png"),
            "umap1",
            "umap2",
            "global_density",
        )

    save_moran_grids(data, out_dir)
    save_dbscan_examples(data, out_dir)

    if not company_df.empty:
        for case_id in company_df["case_id"].unique():
            case_dir = os.path.join(out_dir, "cases", case_id)
            save_company_metrics_table(
                company_df[company_df["case_id"] == case_id],
                case_dir,
            )


def print_summary(case_df: pd.DataFrame) -> None:
    if case_df.empty:
        print("No cases processed.")
        return

    metrics = ["global_density", "morans_i", "gini_requests", "avg_overlap"]
    summary = case_df[metrics].agg(["mean", "std"]).transpose()

    print("\nProcessed cases:", len(case_df))
    for metric in metrics:
        mean_val = summary.loc[metric, "mean"]
        std_val = summary.loc[metric, "std"]
        print(f"{metric}: mean={mean_val:.4f} std={std_val:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute scenario distribution metrics.")
    parser.add_argument("--input", required=True, help="Path to company_cases.json")
    parser.add_argument("--out", required=True, help="Output directory for metrics CSVs")
    parser.add_argument("--case-id", default=None, help="Optional case id to process")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Process at most this many cases (deterministic natural order).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    data = load_company_cases(args.input, case_id=args.case_id)
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise ValueError("--max-cases must be >= 1")
        case_keys = [k for k, v in data.items() if isinstance(v, dict) and "companies" in v]
        case_keys = sorted(case_keys, key=_natural_key)[: args.max_cases]
        data = {k: data[k] for k in case_keys}
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
    print_summary(results["case"])


if __name__ == "__main__":
    main()
