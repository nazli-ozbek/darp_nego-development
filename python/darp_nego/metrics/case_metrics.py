"""Case-level spatial metrics and aggregations."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

from .company_metrics import _pickup_points
from .geometry import hull_or_bbox_area, polygon_overlap_ratio


def global_density(points: np.ndarray) -> float:
    """Global request density across all companies in a case."""
    area = hull_or_bbox_area(points)
    return float(points.shape[0] / area)


def gini_coefficient(values: List[int]) -> float:
    """Compute Gini coefficient for non-negative values."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))


def _grid_counts(points: np.ndarray, grid_size: int) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    if np.allclose(mins, maxs):
        return np.zeros((grid_size, grid_size), dtype=float)
    xs = np.linspace(mins[0], maxs[0], grid_size + 1)
    ys = np.linspace(mins[1], maxs[1], grid_size + 1)
    counts, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[xs, ys])
    return counts


def morans_i_grid(
    points: np.ndarray,
    grid_size: int | None = None,
    permutations: int = 99,
    seed: int = 7,
) -> Tuple[float, float]:
    """Compute Moran's I using grid cell counts and rook adjacency weights.

    This approach bins pickup points into a grid and computes spatial
    autocorrelation over cell counts (including zeros).
    """
    if points.shape[0] < 2:
        return float("nan"), float("nan")

    if grid_size is None:
        grid_size = int(round(np.sqrt(points.shape[0])))
        grid_size = max(3, min(30, grid_size))

    counts = _grid_counts(points, grid_size)
    values = counts.flatten()
    n = values.size
    mean_val = float(values.mean())
    diffs = values - mean_val
    denom = float(np.sum(diffs ** 2))
    if denom == 0.0:
        return float("nan"), float("nan")

    weights = np.zeros((n, n), dtype=float)
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            neighbors = []
            if i > 0:
                neighbors.append(((i - 1) * grid_size + j))
            if i < grid_size - 1:
                neighbors.append(((i + 1) * grid_size + j))
            if j > 0:
                neighbors.append((i * grid_size + j - 1))
            if j < grid_size - 1:
                neighbors.append((i * grid_size + j + 1))
            for n_idx in neighbors:
                weights[idx, n_idx] = 1.0

    w_sum = float(weights.sum())
    if w_sum == 0.0:
        return float("nan"), float("nan")

    num = float(diffs @ weights @ diffs)
    i_stat = (n / w_sum) * (num / denom)

    if permutations <= 0:
        return i_stat, float("nan")

    rng = np.random.default_rng(seed)
    perm_stats = []
    for _ in range(permutations):
        permuted = rng.permutation(values)
        perm_diffs = permuted - permuted.mean()
        perm_num = float(perm_diffs @ weights @ perm_diffs)
        perm_denom = float(np.sum(perm_diffs ** 2))
        if perm_denom == 0.0:
            perm_stats.append(0.0)
        else:
            perm_stats.append((n / w_sum) * (perm_num / perm_denom))

    perm_stats = np.asarray(perm_stats)
    p_value = float((np.sum(np.abs(perm_stats) >= abs(i_stat)) + 1) / (permutations + 1))
    return i_stat, p_value


def operational_overlap(company_points: Dict[str, np.ndarray]) -> float:
    """Average pairwise overlap of company convex hulls (bbox fallback)."""
    company_ids = list(company_points.keys())
    if len(company_ids) < 2:
        return 0.0
    overlaps = []
    for a_id, b_id in combinations(company_ids, 2):
        overlap = polygon_overlap_ratio(company_points[a_id], company_points[b_id])
        overlaps.append(overlap)
    return float(np.mean(overlaps)) if overlaps else 0.0


def compute_case_metrics(
    case_id: str,
    case_data: Dict,
    permutations: int = 99,
    seed: int = 7,
) -> Dict[str, float]:
    coordinates = case_data.get("coordinates", {})
    company_points: Dict[str, np.ndarray] = {}
    request_counts: List[int] = []
    all_points: List[np.ndarray] = []

    for company_id, company_data in case_data.get("companies", {}).items():
        points = _pickup_points(company_data, coordinates)
        company_points[company_id] = points
        request_counts.append(int(points.shape[0]))
        if points.shape[0] > 0:
            all_points.append(points)

    if all_points:
        stacked = np.vstack(all_points)
    else:
        stacked = np.zeros((0, 2), dtype=float)

    gini = gini_coefficient(request_counts)
    global_d = global_density(stacked) if stacked.shape[0] > 0 else 0.0
    moran_i, moran_p = morans_i_grid(stacked, permutations=permutations, seed=seed)
    overlap = operational_overlap(company_points)

    return {
        "case_id": case_id,
        "global_density": global_d,
        "morans_i": moran_i,
        "morans_p": moran_p,
        "gini_requests": gini,
        "avg_overlap": overlap,
    }
