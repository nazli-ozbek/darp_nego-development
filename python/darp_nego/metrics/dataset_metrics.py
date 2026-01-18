"""Dataset-level aggregation and embedding utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def build_feature_vectors(case_metrics: pd.DataFrame, company_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build per-case feature vectors from company and case metrics."""
    company_grouped = company_metrics.groupby("case_id")
    company_stats = company_grouped.agg(
        avg_company_density=("request_density", "mean"),
        std_company_density=("request_density", "std"),
        mean_company_mnnd=("mnnd", "mean"),
        std_company_mnnd=("mnnd", "std"),
        avg_company_dbscan_clusters=("dbscan_clusters", "mean"),
        avg_company_noise_ratio=("noise_ratio", "mean"),
    ).reset_index()

    features = case_metrics.merge(company_stats, on="case_id", how="left")

    return features[
        [
            "case_id",
            "global_density",
            "avg_company_density",
            "std_company_density",
            "mean_company_mnnd",
            "std_company_mnnd",
            "morans_i",
            "gini_requests",
            "avg_company_dbscan_clusters",
            "avg_company_noise_ratio",
            "avg_overlap",
        ]
    ]


def pca_embedding(
    features: pd.DataFrame,
    seed: int = 7,
) -> Tuple[pd.DataFrame, PCA]:
    """Compute a 2D PCA embedding for the case feature vectors."""
    data = features.drop(columns=["case_id"]).fillna(0.0).to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(scaled)
    df = pd.DataFrame({"case_id": features["case_id"], "pc1": coords[:, 0], "pc2": coords[:, 1]})
    return df, pca


def umap_embedding(
    features: pd.DataFrame,
    seed: int = 7,
) -> Optional[pd.DataFrame]:
    """Compute a 2D UMAP embedding if available."""
    try:
        import umap
    except Exception:
        return None

    data = features.drop(columns=["case_id"]).fillna(0.0).to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    reducer = umap.UMAP(n_components=2, random_state=seed)
    coords = reducer.fit_transform(scaled)
    return pd.DataFrame({"case_id": features["case_id"], "umap1": coords[:, 0], "umap2": coords[:, 1]})
