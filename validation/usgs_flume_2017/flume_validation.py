"""Shared calculations for the USGS 2017 debris-flow flume validation.

The functions in this module intentionally do not depend on Clawpack.  They
turn observations and model samples into the same metrics, so observation
processing can be tested before a D-Claw executable is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SeriesMetrics:
    """Scalar comparison metrics for one modeled time series."""

    n: int
    rmse: float
    mae: float
    bias: float
    correlation: float
    nse: float


def historical_front_envelope(front_m: np.ndarray) -> np.ndarray:
    """Return the furthest position reached up to each sample.

    Missing samples after a detected front retain the historical maximum;
    missing samples before the first detection remain NaN.
    """

    front = np.asarray(front_m, dtype=float)
    if front.ndim != 1:
        raise ValueError("front_m must be a 1-D array")
    result = np.full(front.shape, np.nan)
    maximum = float("nan")
    for index, value in enumerate(front):
        if np.isfinite(value):
            maximum = value if not np.isfinite(maximum) else max(maximum, value)
        if np.isfinite(maximum):
            result[index] = maximum
    return result


def first_persistent_exceedance(
    time_s: np.ndarray,
    values: np.ndarray,
    threshold: float,
    min_duration_s: float,
) -> float:
    """Return the first time a threshold is continuously exceeded.

    NaN values and samples before ``time_s == 0`` cannot trigger arrival.
    ``np.nan`` is returned if no persistent exceedance exists.
    """

    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if time_s.ndim != 1 or values.shape != time_s.shape or time_s.size < 2:
        raise ValueError("time_s and values must be equal-length 1-D arrays")
    dt = float(np.nanmedian(np.diff(time_s)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("time_s must be strictly increasing")
    window = max(1, int(np.ceil(min_duration_s / dt)))
    mask = np.isfinite(values) & (values > threshold) & (time_s >= 0)
    if window == 1:
        indices = np.flatnonzero(mask)
    else:
        held = np.convolve(mask.astype(np.int16), np.ones(window, dtype=np.int16), mode="valid")
        indices = np.flatnonzero(held >= window)
    return float(time_s[indices[0]]) if indices.size else float("nan")


def centerline_axis(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a horizontal centerline and orient it toward increasing elevation."""

    xy = np.column_stack((np.asarray(x, float), np.asarray(y, float)))
    z = np.asarray(z, float)
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(z)
    if valid.sum() < 3:
        raise ValueError("at least three finite lidar points are required")
    xy = xy[valid]
    z = z[valid]
    origin = np.mean(xy, axis=0)
    _, _, vh = np.linalg.svd(xy - origin, full_matrices=False)
    axis = vh[0]
    along = (xy - origin) @ axis
    if np.corrcoef(along, z)[0, 1] < 0:
        axis = -axis
    return origin, axis


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive-exclusive index intervals containing True values."""

    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def initial_pile_and_gate(
    s_m: np.ndarray,
    initial_depth_m: np.ndarray,
    threshold_m: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Identify the principal initial pile and its downslope gate coordinate.

    ``s_m`` increases uphill.  The gate is therefore the minimum ``s`` edge of
    the selected pile component.
    """

    s_m = np.asarray(s_m, float)
    depth = np.asarray(initial_depth_m, float)
    mask = np.isfinite(depth) & (depth > threshold_m)
    runs = contiguous_runs(mask)
    if not runs:
        raise ValueError("no initial pile found above the depth threshold")
    # Select by integrated thickness, which rejects short isolated returns.
    start, stop = max(runs, key=lambda pair: float(np.nansum(depth[pair[0] : pair[1]])))
    pile = np.zeros(mask.shape, dtype=bool)
    pile[start:stop] = True
    return pile, float(s_m[start])


def detect_motion_onset(
    time_s: np.ndarray,
    surface_z_m: np.ndarray,
    initial_surface_z_m: np.ndarray,
    pile_mask: np.ndarray,
    min_change_m: float = 0.02,
    persistence: int = 6,
) -> tuple[float, np.ndarray]:
    """Detect failure onset from persistent changes over the initial pile."""

    change = np.nanmedian(
        np.abs(np.asarray(surface_z_m)[:, pile_mask] - np.asarray(initial_surface_z_m)[pile_mask]),
        axis=1,
    )
    head = change[: max(persistence * 3, min(100, change.size))]
    med = float(np.nanmedian(head))
    mad = float(np.nanmedian(np.abs(head - med)))
    threshold = max(min_change_m, med + 10.0 * 1.4826 * mad)
    active = np.isfinite(change) & (change > threshold)
    held = np.convolve(active.astype(np.int16), np.ones(persistence, dtype=np.int16), mode="valid")
    indices = np.flatnonzero(held >= persistence)
    onset = float(time_s[indices[0]]) if indices.size else float("nan")
    return onset, change


def front_position(
    x_downstream_m: np.ndarray,
    depth_m: np.ndarray,
    threshold_m: float = 0.03,
    support_bins: int = 3,
) -> np.ndarray:
    """Extract a robust downstream front from a depth time-space matrix.

    A candidate must have at least ``support_bins`` wet bins in a centered
    window of width ``2 * support_bins - 1``.  This rejects isolated airborne
    particles and lidar returns ahead of the coherent front.
    """

    x = np.asarray(x_downstream_m, float)
    h = np.asarray(depth_m, float)
    if h.ndim != 2 or h.shape[1] != x.size:
        raise ValueError("depth_m must have shape (time, x)")
    order = np.argsort(x)
    xs = x[order]
    wet = np.isfinite(h[:, order]) & (h[:, order] > threshold_m)
    width = max(1, 2 * support_bins - 1)
    kernel = np.ones(width, dtype=np.int16)
    front = np.full(h.shape[0], np.nan)
    for i, row in enumerate(wet):
        supported = np.convolve(row.astype(np.int16), kernel, mode="same") >= support_bins
        candidates = np.flatnonzero(supported & (xs >= 0))
        if candidates.size:
            front[i] = xs[candidates[-1]]
    return front


def tracked_front_position(
    time_s: np.ndarray,
    x_downstream_m: np.ndarray,
    depth_m: np.ndarray,
    threshold_m: float = 0.03,
    support_bins: int = 3,
    max_speed_m_s: float = 20.0,
    initialization_max_x_m: float = 3.0,
) -> np.ndarray:
    """Track the coherent front while rejecting persistent spatial noise.

    Raw lidar profiles contain returns from saltating grains and small
    bed/profile-registration differences far ahead of the flow.  A front
    candidate must have spatial support and be reachable from the previously
    accepted position at no more than ``max_speed_m_s``. Missing profiles are
    retained as NaN, while the reachable window grows with elapsed time.
    """

    time = np.asarray(time_s, float)
    x = np.asarray(x_downstream_m, float)
    h = np.asarray(depth_m, float)
    if h.shape != (time.size, x.size):
        raise ValueError("depth_m must have shape (time, x)")
    order = np.argsort(x)
    xs = x[order]
    wet = np.isfinite(h[:, order]) & (h[:, order] > threshold_m)
    width = max(1, 2 * support_bins - 1)
    kernel = np.ones(width, dtype=np.int16)
    supported = np.array(
        [np.convolve(row.astype(np.int16), kernel, mode="same") >= support_bins for row in wet]
    )
    front = np.full(time.size, np.nan)
    previous_x = 0.0
    previous_time = 0.0
    initialized = False
    for i, (current_time, row) in enumerate(zip(time, supported)):
        if current_time < 0:
            continue
        candidates = xs[row & (xs >= -0.5)]
        if not initialized:
            near_gate = candidates[candidates <= initialization_max_x_m]
            if near_gate.size:
                front[i] = float(np.max(near_gate))
                previous_x = front[i]
                previous_time = current_time
                initialized = True
            continue
        maximum_reachable = previous_x + max(1.0, max_speed_m_s * (current_time - previous_time))
        reachable = candidates[
            (candidates >= previous_x - 1.0) & (candidates <= maximum_reachable)
        ]
        if reachable.size:
            front[i] = float(np.max(reachable))
            previous_x = front[i]
            previous_time = current_time
    return front


def compare_series(observed: Iterable[float], modeled: Iterable[float]) -> SeriesMetrics:
    """Calculate conventional model-observation goodness-of-fit metrics."""

    obs = np.asarray(list(observed), dtype=float)
    mod = np.asarray(list(modeled), dtype=float)
    if obs.shape != mod.shape:
        raise ValueError("observed and modeled arrays must have identical shape")
    valid = np.isfinite(obs) & np.isfinite(mod)
    obs = obs[valid]
    mod = mod[valid]
    if obs.size == 0:
        return SeriesMetrics(0, *(float("nan"),) * 5)
    residual = mod - obs
    denominator = float(np.sum((obs - np.mean(obs)) ** 2))
    corr = float(np.corrcoef(obs, mod)[0, 1]) if obs.size > 1 else float("nan")
    nse = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else float("nan")
    return SeriesMetrics(
        n=int(obs.size),
        rmse=float(np.sqrt(np.mean(residual**2))),
        mae=float(np.mean(np.abs(residual))),
        bias=float(np.mean(residual)),
        correlation=corr,
        nse=nse,
    )
