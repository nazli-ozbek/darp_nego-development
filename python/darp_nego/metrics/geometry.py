"""Geometry helpers for spatial metrics."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

try:
    from shapely.geometry import MultiPoint, Polygon
    _HAS_SHAPELY = True
except Exception:
    MultiPoint = None
    Polygon = None
    _HAS_SHAPELY = False

try:
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except Exception:
    ConvexHull = None
    _HAS_SCIPY = False


def _to_points(points: Iterable[Tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(list(points), dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    return arr.reshape((-1, 2))


def bbox_area(points: Iterable[Tuple[float, float]]) -> float:
    """Return axis-aligned bounding box area for a point set."""
    arr = _to_points(points)
    if arr.shape[0] == 0:
        return 0.0
    min_xy = arr.min(axis=0)
    max_xy = arr.max(axis=0)
    delta = max_xy - min_xy
    return float(max(delta[0], 0.0) * max(delta[1], 0.0))


def hull_area(points: Iterable[Tuple[float, float]]) -> float:
    """Return convex hull area; 0 for degenerate inputs."""
    arr = _to_points(points)
    if arr.shape[0] < 3:
        return 0.0
    if _HAS_SHAPELY:
        hull = MultiPoint(arr).convex_hull
        return float(hull.area)
    if _HAS_SCIPY:
        try:
            hull = ConvexHull(arr)
            return float(hull.volume)
        except Exception:
            return 0.0
    return 0.0


def hull_or_bbox_area(points: Iterable[Tuple[float, float]]) -> float:
    """Return convex hull area with bbox fallback and zero guard."""
    area = hull_area(points)
    if area > 0.0:
        return area
    area = bbox_area(points)
    if area > 0.0:
        return area
    return 1.0


def _bbox_polygon(points: Iterable[Tuple[float, float]]) -> Optional["Polygon"]:
    if not _HAS_SHAPELY:
        return None
    arr = _to_points(points)
    if arr.shape[0] == 0:
        return None
    min_xy = arr.min(axis=0)
    max_xy = arr.max(axis=0)
    return Polygon(
        [
            (min_xy[0], min_xy[1]),
            (max_xy[0], min_xy[1]),
            (max_xy[0], max_xy[1]),
            (min_xy[0], max_xy[1]),
        ]
    )


def hull_polygon(points: Iterable[Tuple[float, float]]) -> Optional["Polygon"]:
    if not _HAS_SHAPELY:
        return None
    arr = _to_points(points)
    if arr.shape[0] == 0:
        return None
    return MultiPoint(arr).convex_hull


def hull_or_bbox_polygon(points: Iterable[Tuple[float, float]]) -> Optional["Polygon"]:
    """Return convex hull polygon or bbox polygon if hull is degenerate."""
    if not _HAS_SHAPELY:
        return None
    hull = hull_polygon(points)
    if hull is not None and hull.area > 0.0:
        return hull
    return _bbox_polygon(points)


def polygon_overlap_ratio(
    points_a: Iterable[Tuple[float, float]],
    points_b: Iterable[Tuple[float, float]],
) -> float:
    """Return intersection-over-union for two point sets' hulls (bbox fallback)."""
    if _HAS_SHAPELY:
        poly_a = hull_or_bbox_polygon(points_a)
        poly_b = hull_or_bbox_polygon(points_b)
        if poly_a is None or poly_b is None:
            return 0.0
        union_area = poly_a.union(poly_b).area
        if union_area == 0.0:
            return 0.0
        return float(poly_a.intersection(poly_b).area / union_area)

    # Bbox-only fallback
    arr_a = _to_points(points_a)
    arr_b = _to_points(points_b)
    if arr_a.shape[0] == 0 or arr_b.shape[0] == 0:
        return 0.0
    min_a = arr_a.min(axis=0)
    max_a = arr_a.max(axis=0)
    min_b = arr_b.min(axis=0)
    max_b = arr_b.max(axis=0)
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_delta = np.maximum(inter_max - inter_min, 0.0)
    inter_area = float(inter_delta[0] * inter_delta[1])
    area_a = float(max(max_a[0] - min_a[0], 0.0) * max(max_a[1] - min_a[1], 0.0))
    area_b = float(max(max_b[0] - min_b[0], 0.0) * max(max_b[1] - min_b[1], 0.0))
    union_area = area_a + area_b - inter_area
    if union_area == 0.0:
        return 0.0
    return inter_area / union_area
