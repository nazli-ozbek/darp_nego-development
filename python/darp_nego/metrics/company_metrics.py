"""Company-level spatial metrics for DARP scenarios."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .geometry import hull_or_bbox_area


def _service_points(company_data: Dict, coordinates: Dict[str, List[float]]) -> np.ndarray:
    """Return pickup+dropoff points for all clients in a company."""
    points: List[Tuple[float, float]] = []
    for client in company_data.get("clients", []):
        start_id = str(client.get("start_location"))
        end_id = str(client.get("end_location"))
        if start_id in coordinates:
            x, y = coordinates[start_id]
            points.append((x, y))
        if end_id in coordinates:
            x, y = coordinates[end_id]
            points.append((x, y))
    if not points:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def request_density(points: np.ndarray) -> float:
    """Service-point density per unit area using convex hull or bbox fallback."""
    area = hull_or_bbox_area(points)
    return float(points.shape[0] / area)


def mean_nearest_neighbor_distance(points: np.ndarray) -> float:
    """Mean nearest-neighbor Euclidean distance among service points."""
    if points.shape[0] < 2:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    return float(np.mean(distances[:, 1]))


def _estimate_eps(points: np.ndarray, min_pts: int) -> float:
    if points.shape[0] < 2:
        return float("nan")
    k = max(min_pts - 1, 1)
    k = min(k, points.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    kth_distances = distances[:, -1]
    eps = float(np.median(kth_distances))
    return max(eps, 1e-6)


def dbscan_indicators(points: np.ndarray, min_pts: int = 4) -> Dict[str, float]:
    """Compute DBSCAN cluster count, average cluster size, and noise ratio."""
    n_points = points.shape[0]
    if n_points < 2:
        return {
            "dbscan_eps": float("nan"),
            "dbscan_clusters": 0,
            "avg_cluster_size": 0.0,
            "noise_ratio": 1.0,
        }

    if n_points < min_pts:
        min_pts = 3

    eps = _estimate_eps(points, min_pts)
    model = DBSCAN(eps=eps, min_samples=min_pts)
    labels = model.fit_predict(points)
    noise_count = int(np.sum(labels == -1))
    clusters = [label for label in set(labels) if label != -1]
    cluster_sizes = [int(np.sum(labels == label)) for label in clusters]
    avg_cluster_size = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
    return {
        "dbscan_eps": eps,
        "dbscan_clusters": len(clusters),
        "avg_cluster_size": avg_cluster_size,
        "noise_ratio": float(noise_count / n_points),
    }


def dbscan_labels(points: np.ndarray, min_pts: int = 4) -> Dict[str, np.ndarray]:
    """Return DBSCAN labels and epsilon for plotting or diagnostics."""
    n_points = points.shape[0]
    if n_points < 2:
        return {"labels": np.full(n_points, -1), "eps": float("nan"), "min_pts": min_pts}

    if n_points < min_pts:
        min_pts = 3

    eps = _estimate_eps(points, min_pts)
    model = DBSCAN(eps=eps, min_samples=min_pts)
    labels = model.fit_predict(points)
    return {"labels": labels, "eps": eps, "min_pts": min_pts}


def compute_company_metrics(
    case_id: str,
    company_id: str,
    company_data: Dict,
    coordinates: Dict[str, List[float]],
    min_pts: int = 4,
) -> Dict[str, float]:
    points = _service_points(company_data, coordinates)
    request_count = len(company_data.get("clients", []))
    density = request_density(points)
    mnnd = mean_nearest_neighbor_distance(points)
    dbscan_stats = dbscan_indicators(points, min_pts=min_pts)

    return {
        "case_id": case_id,
        "company_id": company_id,
        "request_count": int(request_count),
        "service_point_count": int(points.shape[0]),
        "request_density": density,
        "mnnd": mnnd,
        "dbscan_clusters": dbscan_stats["dbscan_clusters"],
        "avg_cluster_size": dbscan_stats["avg_cluster_size"],
        "noise_ratio": dbscan_stats["noise_ratio"],
        "dbscan_eps": dbscan_stats["dbscan_eps"],
    }
