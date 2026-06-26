from __future__ import annotations

import math
from itertools import product
from typing import Iterable

import numpy as np


def simplex_weights(q: int, objectives: int = 2) -> np.ndarray:
    """Generate uniformly spaced simplex weights.

    The paper uses N = C(Q + l - 1, l - 1). For l=2 this returns Q+1
    weights: (0, 1), (1/Q, 1-1/Q), ..., (1, 0).
    """

    if q < 1:
        raise ValueError("q must be >= 1")
    if objectives < 2:
        raise ValueError("objectives must be >= 2")

    if objectives == 2:
        return np.asarray([[i / q, 1.0 - i / q] for i in range(q + 1)], dtype=np.float32)

    values: list[list[float]] = []
    for counts in product(range(q + 1), repeat=objectives):
        if sum(counts) == q:
            values.append([c / q for c in counts])
    return np.asarray(values, dtype=np.float32)


def weighted_sum(point: Iterable[float], weight: Iterable[float]) -> float:
    p = np.asarray(list(point), dtype=np.float32)
    w = np.asarray(list(weight), dtype=np.float32)
    return float(np.dot(p, w))


def hypervolume_2d(points: np.ndarray, ref: tuple[float, float] = (1.0, 1.0)) -> float:
    """Compute dominated hypervolume for normalized minimization points."""

    if len(points) == 0:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[(pts[:, 0] <= ref[0]) & (pts[:, 1] <= ref[1])]
    if len(pts) == 0:
        return 0.0

    order = np.argsort(pts[:, 0])
    pts = pts[order]
    nondom = []
    best_y = math.inf
    for x, y in pts:
        if y < best_y:
            nondom.append((x, y))
            best_y = y

    hv = 0.0
    prev_x = ref[0]
    for x, y in reversed(nondom):
        hv += max(prev_x - x, 0.0) * max(ref[1] - y, 0.0)
        prev_x = x
    return float(hv)

