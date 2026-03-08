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


def _kmeans_labels_highdim(features: pd.DataFrame, k: int, seed: int) -> Tuple[pd.Series, float]:
    data = features.drop(columns=["case_id"]).to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(data)
    model = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = model.fit_predict(scaled)
    return pd.Series(labels, index=features.index, name="cluster"), float(model.inertia_)


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
    elbow_min: int,
    elbow_max: int,
) -> None:
    if clusterer == "kmeans":
        n_samples = len(embedding)
        k_min = max(2, int(elbow_min))
        k_max = int(elbow_max)
        k_max = min(k_max, max(2, n_samples - 1))
        if k_max < k_min:
            k_max = k_min

        selected_k = k
        if selected_k is None or selected_k < 2:
            coords = embedding[[x_col, y_col]].to_numpy(dtype=float)
            scaled = StandardScaler().fit_transform(coords)
            k_values = list(range(k_min, k_max + 1))
            inertias: List[float] = []
            for kk in k_values:
                model = KMeans(n_clusters=kk, n_init=20, random_state=seed)
                model.fit(scaled)
                inertias.append(float(model.inertia_))

            if k_values and inertias:
                elbow_df = pd.DataFrame({"k": k_values, "inertia": inertias})
                elbow_df.to_csv(os.path.join(out_dir, f"{name}_elbow.csv"), index=False)
                _plot_elbow(k_values, inertias, out_dir, name)
                selected_k = _select_knee_k(k_values, inertias) or k_min
                print(f"[{name}] Selected k from elbow knee: {selected_k}")
            else:
                selected_k = k_min

        labels = _kmeans_labels(embedding, x_col, y_col, int(selected_k), seed)
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
    _write_global_anova_summary(anova_df, out_dir, name)

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

    # Add compact global summary in the top-right corner.
    if "q_value" in anova_df.columns:
        q = anova_df["q_value"].to_numpy(dtype=float)
        n = len(q)
        if n > 0:
            d1 = float(np.sum(q < 0.05) / n)
            summary = f"D1={d1:.3f}\nN={n}"
            ax.text(
                0.98,
                0.98,
                summary,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.9},
            )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_anova_top_features.png"), dpi=200)
    plt.close()


def _write_global_anova_summary(anova_df: pd.DataFrame, out_dir: str, name: str) -> None:
    if anova_df.empty or "q_value" not in anova_df.columns:
        summary = pd.DataFrame(
            [
                {
                    "n_features": 0,
                    "sig_count_q_lt_0_05": 0,
                    "D1_sig_ratio": np.nan,
                    "mean_q": np.nan,
                    "median_q": np.nan,
                    "min_q": np.nan,
                }
            ]
        )
        summary.to_csv(os.path.join(out_dir, f"{name}_anova_summary.csv"), index=False)
        return

    q = anova_df["q_value"].to_numpy(dtype=float)
    n = int(len(q))
    sig = int(np.sum(q < 0.05))
    d1 = sig / n if n > 0 else np.nan

    summary = pd.DataFrame(
        [
            {
                "n_features": n,
                "sig_count_q_lt_0_05": sig,
                "D1_sig_ratio": d1,
                "mean_q": float(np.mean(q)) if n > 0 else np.nan,
                "median_q": float(np.median(q)) if n > 0 else np.nan,
                "min_q": float(np.min(q)) if n > 0 else np.nan,
            }
        ]
    )
    summary.to_csv(os.path.join(out_dir, f"{name}_anova_summary.csv"), index=False)


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


def _plot_elbow(k_values: List[int], inertias: List[float], out_dir: str, name: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not k_values or not inertias:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, inertias, marker="o", color="#2a5599")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia (SSE)")
    ax.set_title(f"{name.upper()} Elbow")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_elbow.png"), dpi=200)
    plt.close()


def _select_knee_k(k_values: List[int], inertias: List[float]) -> Optional[int]:
    if len(k_values) < 3:
        return None
    x = np.asarray(k_values, dtype=float)
    y = np.asarray(inertias, dtype=float)
    # Normalize to [0, 1] for stable distance computation.
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    # Distance from each point to the line between first and last.
    x1, y1 = x_norm[0], y_norm[0]
    x2, y2 = x_norm[-1], y_norm[-1]
    denom = np.hypot(x2 - x1, y2 - y1) + 1e-12
    distances = np.abs((y2 - y1) * x_norm - (x2 - x1) * y_norm + x2 * y1 - y2 * x1) / denom
    knee_idx = int(np.argmax(distances))
    return int(k_values[knee_idx])


def _run_highdim_pipeline(
    features: pd.DataFrame,
    k: Optional[int],
    seed: int,
    elbow_min: int,
    elbow_max: int,
    out_dir: str,
) -> None:
    n_samples = len(features)
    if n_samples < 3 and (k is None or k < 2):
        raise ValueError("Need at least 3 samples for elbow analysis or specify --k >= 2")

    k_min = max(2, int(elbow_min))
    k_max = int(elbow_max)
    k_max = min(k_max, max(2, n_samples - 1))
    if k_max < k_min:
        k_max = k_min

    k_values = list(range(k_min, k_max + 1))
    inertias: List[float] = []
    for kk in k_values:
        _, inertia = _kmeans_labels_highdim(features, kk, seed)
        inertias.append(inertia)

    if k_values and inertias:
        elbow_df = pd.DataFrame({"k": k_values, "inertia": inertias})
        elbow_df.to_csv(os.path.join(out_dir, "highdim_elbow.csv"), index=False)
        _plot_elbow(k_values, inertias, out_dir, "highdim")

    selected_k = k
    if selected_k is None:
        selected_k = _select_knee_k(k_values, inertias) or k_min
        print(f"[highdim] Selected k from elbow knee: {selected_k}")

    labels, _ = _kmeans_labels_highdim(features, int(selected_k), seed)
    clustered = pd.DataFrame({"case_id": features["case_id"], "cluster": labels.values})
    clustered.to_csv(os.path.join(out_dir, "highdim_clusters.csv"), index=False)

    # 2D UMAP projection for visualization only.
    data = features.drop(columns=["case_id"]).to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(data)
    try:
        import umap
    except Exception:
        print("[highdim] UMAP not available; skipping highdim cluster visualization.")
    else:
        n_samples = scaled.shape[0]
        n_neighbors = max(2, min(15, n_samples - 1))
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors, init="random")
        coords = reducer.fit_transform(scaled)
        vis_df = pd.DataFrame(
            {
                "case_id": features["case_id"],
                "umap1": coords[:, 0],
                "umap2": coords[:, 1],
                "cluster": labels.values,
            }
        )
        _plot_clusters(vis_df, out_dir, "highdim", "umap1", "umap2")

    anova_df = _anova_for_features(features, labels)
    if not anova_df.empty:
        anova_df["q_value"] = _fdr_bh(anova_df["p_value"].to_numpy())
        anova_df = anova_df.sort_values("q_value", ascending=True)
    anova_df.to_csv(os.path.join(out_dir, "highdim_anova.csv"), index=False)
    _write_global_anova_summary(anova_df, out_dir, "highdim")
    _plot_anova_summary(anova_df, out_dir, "highdim", top_n=15)


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
    parser.add_argument("--highdim", action="store_true", help="Run KMeans+ANOVA on full feature vectors")
    parser.add_argument("--elbow-min", type=int, default=2, help="Minimum k for elbow analysis")
    parser.add_argument("--elbow-max", type=int, default=10, help="Maximum k for elbow analysis")
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
                args.elbow_min,
                args.elbow_max,
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
                args.elbow_min,
                args.elbow_max,
            )

    if args.highdim:
        _run_highdim_pipeline(
            features,
            args.k,
            args.seed,
            args.elbow_min,
            args.elbow_max,
            out_dir,
        )

    print(f"Cluster+ANOVA reports written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
