#!/usr/bin/env python3
"""Cluster PCA/UMAP embeddings and run ANOVA per input feature."""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _ensure_outdir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def _numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    return numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _load_embedding(path: str, x_col: str, y_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"case_id", x_col, y_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Embedding file {path} must include columns {sorted(required)}")
    return df[["case_id", x_col, y_col]].copy()


def _kmeans_labels(embedding: pd.DataFrame, x_col: str, y_col: str, k: int, seed: int) -> pd.Series:
    coords = embedding[[x_col, y_col]].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(coords)
    model = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = model.fit_predict(scaled)
    return pd.Series(labels, index=embedding.index, name="cluster")


def _dbscan_labels(
    scaled: np.ndarray,
    eps: float,
    min_samples: int,
) -> pd.Series:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(scaled)
    return pd.Series(labels, name="cluster")


def _select_dbscan_labels(
    embedding: pd.DataFrame,
    x_col: str,
    y_col: str,
    min_samples: int,
    min_clusters: int,
    min_cluster_size: int,
) -> Tuple[pd.Series, float]:
    coords = embedding[[x_col, y_col]].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(coords)

    eps_base = _median_k_distance(scaled, min_samples)
    eps_min = max(0.05, eps_base * 0.5)
    eps_max = max(eps_min + 0.05, eps_base * 2.0)
    eps_values = np.round(np.arange(eps_min, eps_max + 1e-6, 0.05), 3)

    best_labels: Optional[pd.Series] = None
    best_eps = eps_base
    best_score: Tuple[int, float, int] = (-1, -1.0, -1)

    for eps in eps_values:
        labels = _dbscan_labels(scaled, eps, min_samples)
        counts = labels.value_counts()
        noise = int(counts.get(-1, 0))
        total = len(labels)
        noise_ratio = noise / total if total else 1.0

        cluster_sizes = counts.drop(index=[-1], errors="ignore").to_dict()
        valid_clusters = [size for size in cluster_sizes.values() if size >= min_cluster_size]
        num_clusters = len(valid_clusters)
        min_size = min(valid_clusters) if valid_clusters else 0

        if num_clusters < min_clusters:
            continue

        score = (num_clusters, -noise_ratio, min_size)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_eps = float(eps)

    if best_labels is None:
        fallback = _dbscan_labels(scaled, eps_base, min_samples)
        return fallback, eps_base

    return best_labels, best_eps


def _median_k_distance(scaled: np.ndarray, min_samples: int) -> float:
    k = max(2, int(min_samples))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(scaled)
    distances, _ = nn.kneighbors(scaled)
    k_dist = distances[:, -1]
    median = float(np.median(k_dist))
    return max(median, 0.05)


def _anova_for_features(
    features: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    data = features.copy()
    data["cluster"] = labels.values
    feature_cols = [c for c in data.columns if c not in {"case_id", "cluster"}]

    rows: List[Dict[str, float]] = []
    cluster_ids = sorted(data["cluster"].unique())

    for col in feature_cols:
        values = data[col]
        groups = [values[data["cluster"] == cid].to_numpy(dtype=float) for cid in cluster_ids]
        if len(groups) < 2:
            continue

        try:
            stat, pval = f_oneway(*groups)
        except Exception:
            stat, pval = np.nan, np.nan

        row: Dict[str, float] = {"feature": col, "f_stat": stat, "p_value": pval}
        for cid, grp in zip(cluster_ids, groups):
            row[f"mean_cluster_{cid}"] = float(np.mean(grp)) if len(grp) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _fdr_bh(p_values: Iterable[float]) -> np.ndarray:
    pvals = np.asarray(list(p_values), dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    qvals = np.empty_like(ranked)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = ranked[i] * n / rank
        q = min(q, prev)
        qvals[i] = q
        prev = q
    out = np.empty_like(qvals)
    out[order] = qvals
    return out


def _run_pipeline(
    name: str,
    embedding: pd.DataFrame,
    features: pd.DataFrame,
    x_col: str,
    y_col: str,
    k: Optional[int],
    seed: int,
    clusterer: str,
    dbscan_min_samples: int,
    min_clusters: int,
    min_cluster_size: int,
    out_dir: str,
) -> None:
    if clusterer == "kmeans":
        if k is None or k < 2:
            raise ValueError("k-means requires --k >= 2")
        labels = _kmeans_labels(embedding, x_col, y_col, k, seed)
    elif clusterer == "dbscan":
        labels, selected_eps = _select_dbscan_labels(
            embedding,
            x_col,
            y_col,
            dbscan_min_samples,
            min_clusters,
            min_cluster_size,
        )
        print(f"[{name}] DBSCAN selected eps={selected_eps}")
    else:
        raise ValueError(f"Unknown clusterer: {clusterer}")

    clustered = embedding.copy()
    clustered["cluster"] = labels.values
    clustered.to_csv(os.path.join(out_dir, f"{name}_clusters.csv"), index=False)
    _plot_clusters(clustered, out_dir, name, x_col, y_col)

    # Drop noise (label -1) for DBSCAN-based ANOVA.
    labels = labels.reset_index(drop=True)
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    valid_features = features.loc[valid_mask].reset_index(drop=True)

    cluster_sizes = valid_labels.value_counts().to_dict()
    valid_cluster_ids = [cid for cid, size in cluster_sizes.items() if size >= min_cluster_size]

    if len(valid_cluster_ids) < min_clusters:
        anova_df = pd.DataFrame()
    else:
        filtered_mask = valid_labels.isin(valid_cluster_ids).to_numpy()
        anova_df = _anova_for_features(valid_features.loc[filtered_mask], valid_labels[filtered_mask])

    if not anova_df.empty:
        anova_df["q_value"] = _fdr_bh(anova_df["p_value"].to_numpy())
        anova_df = anova_df.sort_values("q_value", ascending=True)
    anova_df.to_csv(os.path.join(out_dir, f"{name}_anova.csv"), index=False)

    _plot_anova_summary(anova_df, out_dir, name, top_n=15)


def _plot_anova_summary(anova_df: pd.DataFrame, out_dir: str, name: str, top_n: int = 15) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if anova_df.empty:
        return

    df = anova_df.copy()
    df["score"] = -np.log10(df["q_value"].clip(lower=1e-300))
    score_label = "-log10(q)"

    top = df.sort_values("score", ascending=False).head(top_n)
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(top))))
    ax.barh(top["feature"][::-1], top["score"][::-1], color="#1f77b4", alpha=0.85)
    ax.set_xlabel(score_label)
    ax.set_title(f"{name.upper()} ANOVA Top Features")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_anova_top_features.png"), dpi=200)
    plt.close()


def _plot_clusters(embedding: pd.DataFrame, out_dir: str, name: str, x_col: str, y_col: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if embedding.empty or "cluster" not in embedding.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    unique = sorted(embedding["cluster"].unique())
    cmap = plt.get_cmap("tab10")

    for i, cid in enumerate(unique):
        subset = embedding[embedding["cluster"] == cid]
        label = f"cluster {cid}" if cid != -1 else "noise"
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=25,
            alpha=0.8,
            color=cmap(i % 10),
            label=label,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{name.upper()} Clusters")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_clusters_scatter.png"), dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster PCA/UMAP embeddings and run ANOVA for each input feature."
    )
    parser.add_argument("--features", required=True, help="Path to feature_vectors.csv or combined_feature_vectors.csv")
    parser.add_argument("--pca", default=None, help="Optional path to pca_2d.csv (defaults to sibling file)")
    parser.add_argument("--umap", default=None, help="Optional path to umap_2d.csv (defaults to sibling file)")
    parser.add_argument("--out", default="metrics_out/combined/cluster_anova", help="Output directory")
    parser.add_argument("--cluster", choices=["kmeans", "dbscan"], default="kmeans", help="Clustering algorithm")
    parser.add_argument("--k", type=int, default=None, help="Number of clusters for k-means")
    parser.add_argument("--dbscan-min-samples", type=int, default=5, help="DBSCAN min_samples")
    parser.add_argument("--min-clusters", type=int, default=2, help="Minimum clusters required for ANOVA")
    parser.add_argument("--min-cluster-size", type=int, default=2, help="Minimum samples per cluster for ANOVA")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    # Always compute q-values (FDR) for multi-test correction.
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    if "case_id" not in features.columns:
        raise ValueError("features CSV must include case_id")
    features = features[["case_id"] + list(_numeric_features(features).columns)]

    out_dir = args.out
    _ensure_outdir(out_dir)

    base_dir = os.path.dirname(args.features)
    pca_path = args.pca or os.path.join(base_dir, "pca_2d.csv")
    umap_path = args.umap or os.path.join(base_dir, "umap_2d.csv")

    if os.path.exists(pca_path):
        pca_df = _load_embedding(pca_path, "pc1", "pc2")
        merged = features.merge(pca_df, on="case_id", how="inner")
        if not merged.empty:
            _run_pipeline(
                "pca",
                pca_df.loc[pca_df["case_id"].isin(merged["case_id"])].reset_index(drop=True),
                merged.drop(columns=["pc1", "pc2"]),
                "pc1",
                "pc2",
                args.k,
                args.seed,
                args.cluster,
                args.dbscan_min_samples,
                args.min_clusters,
                args.min_cluster_size,
                out_dir,
            )

    if os.path.exists(umap_path):
        umap_df = _load_embedding(umap_path, "umap1", "umap2")
        merged = features.merge(umap_df, on="case_id", how="inner")
        if not merged.empty:
            _run_pipeline(
                "umap",
                umap_df.loc[umap_df["case_id"].isin(merged["case_id"])].reset_index(drop=True),
                merged.drop(columns=["umap1", "umap2"]),
                "umap1",
                "umap2",
                args.k,
                args.seed,
                args.cluster,
                args.dbscan_min_samples,
                args.min_clusters,
                args.min_cluster_size,
                out_dir,
            )

    print(f"Cluster+ANOVA reports written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
