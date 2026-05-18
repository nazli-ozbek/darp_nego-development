#!/usr/bin/env python3
"""Generate PCA loadings and UMAP correlation reports with plots."""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _ensure_outdir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def _numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    return numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _pca_loadings(feature_df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, PCA]:
    data = _numeric_features(feature_df)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.to_numpy(dtype=float))
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(scaled)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=data.columns,
        columns=["pc1_loading", "pc2_loading"],
    )
    loadings["pc1_abs"] = loadings["pc1_loading"].abs()
    loadings["pc2_abs"] = loadings["pc2_loading"].abs()
    return loadings.sort_values("pc1_abs", ascending=False), pca


def _save_barh(values: pd.Series, title: str, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(values))))
    ax.barh(values.index[::-1], values.values[::-1], color="#2a9d8f", alpha=0.85)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_top_loadings(loadings: pd.DataFrame, out_dir: str, top_n: int) -> None:
    pc1 = loadings["pc1_abs"].head(top_n)
    pc2 = loadings.sort_values("pc2_abs", ascending=False)["pc2_abs"].head(top_n)
    combined = (
        loadings.assign(pca_combined=lambda df: np.sqrt(df["pc1_abs"] ** 2 + df["pc2_abs"] ** 2))
        .sort_values("pca_combined", ascending=False)["pca_combined"]
        .head(top_n)
    )
    _save_barh(pc1, "PCA PC1 Top Loadings (abs)", os.path.join(out_dir, "pca_pc1_top_loadings.png"))
    _save_barh(pc2, "PCA PC2 Top Loadings (abs)", os.path.join(out_dir, "pca_pc2_top_loadings.png"))
    _save_barh(combined, "PCA Combined Influence (PC1+PC2)", os.path.join(out_dir, "pca_combined_influence.png"))


def _umap_correlations(
    features: pd.DataFrame,
    umap_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = features.merge(umap_df, on="case_id", how="inner")
    if merged.empty:
        return pd.DataFrame()
    feature_cols = [c for c in merged.columns if c not in {"case_id", "umap1", "umap2"}]
    corr_rows = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(merged[col]):
            continue
        x = merged[col]
        corr_rows.append(
            {
                "feature": col,
                "spearman_umap1": x.corr(merged["umap1"], method="spearman"),
                "spearman_umap2": x.corr(merged["umap2"], method="spearman"),
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        corr_df["abs_umap1"] = corr_df["spearman_umap1"].abs()
        corr_df["abs_umap2"] = corr_df["spearman_umap2"].abs()
    return corr_df


def _plot_top_umap_corr(corr_df: pd.DataFrame, out_dir: str, top_n: int) -> None:
    if corr_df.empty:
        return
    top1 = corr_df.sort_values("abs_umap1", ascending=False).head(top_n)
    top2 = corr_df.sort_values("abs_umap2", ascending=False).head(top_n)
    combined = (
        corr_df.assign(umap_combined=lambda df: np.sqrt(df["abs_umap1"] ** 2 + df["abs_umap2"] ** 2))
        .sort_values("umap_combined", ascending=False)
        .head(top_n)
    )
    _save_barh(top1.set_index("feature")["abs_umap1"], "UMAP1 Top Spearman |corr|", os.path.join(out_dir, "umap1_top_corr.png"))
    _save_barh(top2.set_index("feature")["abs_umap2"], "UMAP2 Top Spearman |corr|", os.path.join(out_dir, "umap2_top_corr.png"))
    _save_barh(combined.set_index("feature")["umap_combined"], "UMAP Combined Association", os.path.join(out_dir, "umap_combined_association.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA loadings and UMAP correlation reports.")
    parser.add_argument(
        "--features",
        required=True,
        help="Path to combined_feature_vectors.csv",
    )
    parser.add_argument(
        "--out",
        default="metrics_out/combined/embedding_reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--umap",
        default=None,
        help="Optional path to umap_2d.csv (defaults to sibling file)",
    )
    parser.add_argument("--top-n", type=int, default=15, help="Top-N features to plot")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for PCA")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    if "case_id" not in features.columns:
        raise ValueError("features CSV must include case_id")

    out_dir = args.out
    _ensure_outdir(out_dir)

    feature_only = features.drop(columns=["case_id"])
    loadings, pca = _pca_loadings(feature_only, seed=args.seed)
    loadings.to_csv(os.path.join(out_dir, "pca_loadings.csv"))

    _plot_top_loadings(loadings, out_dir, args.top_n)

    umap_path = args.umap
    if umap_path is None:
        candidate = os.path.join(os.path.dirname(args.features), "umap_2d.csv")
        if os.path.exists(candidate):
            umap_path = candidate
    if umap_path and os.path.exists(umap_path):
        umap_df = pd.read_csv(umap_path)
        corr_df = _umap_correlations(features, umap_df)
        if not corr_df.empty:
            corr_df.to_csv(os.path.join(out_dir, "umap_feature_correlations.csv"), index=False)
            _plot_top_umap_corr(corr_df, out_dir, args.top_n)

    print(f"Reports written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
