"""Visualization helpers for metrics outputs."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import os

import numpy as np
import pandas as pd

from .company_metrics import _service_points, dbscan_labels


def _safe_save(fig, path: str) -> None:
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=200)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def _color_scale(values: Iterable[float]) -> List[float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return []
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [0.5 for _ in arr]
    vmin = np.min(finite)
    vmax = np.max(finite)
    if np.isclose(vmin, vmax):
        return [0.5 for _ in arr]
    scaled = (arr - vmin) / (vmax - vmin)
    scaled = np.where(np.isfinite(scaled), scaled, 0.5)
    return list(scaled)


def save_case_metrics_table(case_df: pd.DataFrame, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if case_df.empty:
        return

    cols = ["case_id", "global_density", "morans_i", "gini_requests", "avg_overlap"]
    table_df = case_df[cols].copy()
    numeric_cols = cols[1:]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(table_df))))
    ax.axis("off")
    cell_text = table_df.values.tolist()
    table = ax.table(cellText=cell_text, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)

    for j, col in enumerate(numeric_cols, start=1):
        colors = _color_scale(table_df[col].values)
        for i, color in enumerate(colors, start=1):
            table[(i, j)].set_facecolor((1.0 - color, 1.0, 1.0 - color))

    table[(0, 0)].set_facecolor("#2a9d8f")
    for j in range(1, len(cols)):
        table[(0, j)].set_facecolor("#2a9d8f")
    for j in range(len(cols)):
        table[(0, j)].set_text_props(color="white", weight="bold")

    _safe_save(fig, os.path.join(out_dir, "plots", "case_metrics_table.png"))


def save_company_metrics_table(company_df: pd.DataFrame, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if company_df.empty:
        return

    cols = [
        "case_id",
        "company_id",
        "request_density",
        "avg_pairwise_distance",
        "min_pairwise_distance",
        "max_pairwise_distance",
        "dbscan_clusters",
        "noise_ratio",
    ]
    table_df = company_df[cols].copy()
    numeric_cols = cols[2:]

    max_rows = 40
    if len(table_df) > max_rows:
        table_df = table_df.head(max_rows)

    fig, ax = plt.subplots(figsize=(12, max(3, 0.35 * len(table_df))))
    ax.axis("off")
    cell_text = table_df.values.tolist()
    table = ax.table(cellText=cell_text, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.1)

    for j, col in enumerate(numeric_cols, start=2):
        colors = _color_scale(table_df[col].values)
        for i, color in enumerate(colors, start=1):
            table[(i, j)].set_facecolor((1.0 - color, 1.0, 1.0 - color))

    for j in range(len(cols)):
        table[(0, j)].set_facecolor("#2a9d8f")
        table[(0, j)].set_text_props(color="white", weight="bold")

    _safe_save(fig, os.path.join(out_dir, "plots", "company_metrics_table.png"))


def save_case_distributions(case_df: pd.DataFrame, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if case_df.empty:
        return

    metrics = ["global_density", "morans_i", "gini_requests", "avg_overlap"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        values = case_df[metric].dropna()
        ax.hist(values, bins=15, color="#2a9d8f", alpha=0.8)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    _safe_save(fig, os.path.join(out_dir, "plots", "case_metric_distributions.png"))


def save_company_distributions(company_df: pd.DataFrame, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if company_df.empty:
        return

    metrics = [
        "request_density",
        "avg_pairwise_distance",
        "min_pairwise_distance",
        "max_pairwise_distance",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        values = company_df[metric].dropna()
        ax.hist(values, bins=15, color="#264653", alpha=0.8)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    _safe_save(fig, os.path.join(out_dir, "plots", "company_metric_distributions.png"))


def save_all_metric_distributions(df: pd.DataFrame, out_path: str, exclude_cols: List[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if df.empty:
        return

    numeric_cols = [
        col for col in df.columns
        if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)
    ]
    if not numeric_cols:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, numeric_cols):
        values = df[col].dropna()
        ax.hist(values, bins=15, color="#457b9d", alpha=0.85)
        ax.set_title(col, fontsize=9)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(numeric_cols):]:
        ax.axis("off")

    _safe_save(fig, out_path)


def save_feature_correlation(feature_df: pd.DataFrame, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if feature_df.empty:
        return

    data = feature_df.drop(columns=["case_id"]).fillna(0.0)
    corr = data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature Correlation")
    _safe_save(fig, os.path.join(out_dir, "plots", "feature_correlation.png"))


def save_embedding_scatter(
    embedding_df: pd.DataFrame,
    case_df: pd.DataFrame,
    out_path: str,
    x_col: str,
    y_col: str,
    color_col: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    merged = embedding_df.merge(case_df[["case_id", color_col]], on="case_id", how="left")
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        merged[x_col],
        merged[y_col],
        c=merged[color_col],
        cmap="viridis",
        s=40,
        alpha=0.9,
    )
    ax.set_xlabel(x_col.upper())
    ax.set_ylabel(y_col.upper())
    ax.set_title(f"{x_col.upper()} vs {y_col.upper()} colored by {color_col}")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    _safe_save(fig, out_path)


def save_embedding_scatter_plain(
    embedding_df: pd.DataFrame,
    out_path: str,
    x_col: str,
    y_col: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        embedding_df[x_col],
        embedding_df[y_col],
        s=40,
        alpha=0.85,
        color="#2a9d8f",
    )
    ax.set_xlabel(x_col.upper())
    ax.set_ylabel(y_col.upper())
    ax.set_title(f"{x_col.upper()} vs {y_col.upper()}")
    _safe_save(fig, out_path)


def _grid_counts(points: np.ndarray, grid_size: int) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    if np.allclose(mins, maxs):
        return np.zeros((grid_size, grid_size), dtype=float)
    xs = np.linspace(mins[0], maxs[0], grid_size + 1)
    ys = np.linspace(mins[1], maxs[1], grid_size + 1)
    counts, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[xs, ys])
    return counts


def save_moran_grids(data: Dict, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    for case_id, case_data in data.items():
        if not isinstance(case_data, dict) or "companies" not in case_data:
            continue

        coordinates = case_data.get("coordinates", {})
        all_points: List[np.ndarray] = []
        for company_data in case_data.get("companies", {}).values():
            points = _service_points(company_data, coordinates)
            if points.shape[0] > 0:
                all_points.append(points)
        if not all_points:
            continue

        stacked = np.vstack(all_points)
        grid_size = int(round(np.sqrt(stacked.shape[0])))
        grid_size = max(3, min(30, grid_size))
        counts = _grid_counts(stacked, grid_size)

        plots_dir = os.path.join(out_dir, "cases", case_id, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(counts.T, origin="lower", cmap="magma")
        ax.set_title(f"Moran Grid {case_id}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Service points per cell")
        _safe_save(fig, os.path.join(plots_dir, f"moran_grid_{case_id}.png"))


def save_dbscan_examples(data: Dict, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    for case_id, case_data in data.items():
        if not isinstance(case_data, dict) or "companies" not in case_data:
            continue

        coordinates = case_data.get("coordinates", {})
        companies = case_data.get("companies", {})
        if not companies:
            continue

        plots_dir = os.path.join(out_dir, "cases", case_id, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for company_id, company_data in companies.items():
            points = _service_points(company_data, coordinates)
            if points.shape[0] == 0:
                continue

            labels_info = dbscan_labels(points)
            labels = labels_info["labels"]
            eps = labels_info.get("eps")
            min_pts = labels_info.get("min_pts")

            fig, ax = plt.subplots(figsize=(5, 4))
            if labels.size == 0:
                continue
            unique_labels = sorted(set(labels))
            colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_labels), 1)))
            for idx, label in enumerate(unique_labels):
                mask = labels == label
                color = "black" if label == -1 else colors[idx]
                ax.scatter(
                    points[mask, 0],
                    points[mask, 1],
                    s=30,
                    alpha=0.8,
                    color=color,
                    label="noise" if label == -1 else f"cluster {label}",
                )
            ax.set_title(f"DBSCAN {case_id} {company_id}")
            ax.set_xticks([])
            ax.set_yticks([])
            info_lines = [
                "Colored: clusters",
                "Black: noise",
                f"eps={eps:.3f}" if isinstance(eps, float) and np.isfinite(eps) else "eps=nan",
                f"minPts={min_pts}",
                f"n_points={points.shape[0]}",
                f"unique_points={np.unique(points, axis=0).shape[0]}",
            ]
            ax.text(
                0.98,
                0.98,
                "\n".join(info_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
            _safe_save(fig, os.path.join(plots_dir, f"dbscan_{case_id}_{company_id}.png"))
