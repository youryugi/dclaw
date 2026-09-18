#!/usr/bin/env python3
"""Reduce the raw 60 Hz profile lidar into D-Claw validation arrays.

The published text files contain millions of points.  This script streams
them in chunks, groups every scan line onto a regular along-flume grid, and
then derives the bed, initial pile, motion onset, flow depth, and front path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flume_validation import (
    centerline_axis,
    detect_motion_onset,
    initial_pile_and_gate,
    tracked_front_position,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
USECOLS = ["xM", "yM", "zM", "timestampSEC", "Line"]
DTYPES = {
    "xM": "float64",
    "yM": "float64",
    "zM": "float32",
    "timestampSEC": "float64",
    "Line": "float64",
}


def last_csv_record(path: Path, block_size: int = 8192) -> str:
    """Read the last non-empty CSV record without scanning a multi-GB file."""

    with path.open("rb") as stream:
        stream.seek(0, 2)
        position = stream.tell()
        data = b""
        while position > 0 and data.count(b"\n") < 2:
            take = min(block_size, position)
            position -= take
            stream.seek(position)
            data = stream.read(take) + data
    lines = [line for line in data.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"no data records in {path}")
    return lines[-1].decode("utf-8")


def profile_id_bounds(path: Path) -> tuple[int, int]:
    first = pd.read_csv(path, usecols=["Line"], nrows=1)["Line"].iloc[0]
    last = float(last_csv_record(path).split(",")[-1])
    return int(round(first)), int(round(last))


def fit_geometry(path: Path, sample_rows: int = 500_000) -> tuple[np.ndarray, np.ndarray, float, float]:
    sample = pd.read_csv(path, usecols=USECOLS[:3], nrows=sample_rows, dtype=DTYPES)
    origin, axis = centerline_axis(sample["xM"], sample["yM"], sample["zM"])
    s = (sample[["xM", "yM"]].to_numpy(float) - origin) @ axis
    # Complete profiles occur in the sample; a small margin protects against
    # coordinate jitter and partially longer profiles later in the file.
    return origin, axis, float(np.nanmin(s) - 1.0), float(np.nanmax(s) + 1.0)


def _aggregate_chunk(
    frame: pd.DataFrame,
    surface: np.ndarray,
    time_s: np.ndarray,
    first_line: int,
    origin: np.ndarray,
    axis: np.ndarray,
    s_min: float,
    bin_width_m: float,
) -> None:
    if frame.empty:
        return
    line = np.rint(frame["Line"].to_numpy(float)).astype(np.int64)
    profile = line - first_line
    xy = frame[["xM", "yM"]].to_numpy(float)
    along = (xy - origin) @ axis
    bins = np.floor((along - s_min) / bin_width_m).astype(np.int64)
    valid = (
        (profile >= 0)
        & (profile < surface.shape[0])
        & (bins >= 0)
        & (bins < surface.shape[1])
        & np.isfinite(frame["zM"].to_numpy(float))
    )
    work = pd.DataFrame(
        {
            "profile": profile[valid],
            "bin": bins[valid],
            "z": frame["zM"].to_numpy(float)[valid],
            "time": frame["timestampSEC"].to_numpy(float)[valid],
        }
    )
    if work.empty:
        return
    heights = work.groupby(["profile", "bin"], sort=False, observed=True)["z"].median()
    rows = heights.index.get_level_values(0).to_numpy(int)
    cols = heights.index.get_level_values(1).to_numpy(int)
    surface[rows, cols] = heights.to_numpy(np.float32)
    times = work.groupby("profile", sort=False, observed=True)["time"].median()
    time_s[times.index.to_numpy(int)] = times.to_numpy(float)


def grid_lidar_file(
    path: Path,
    bin_width_m: float,
    chunk_rows: int = 1_000_000,
) -> dict[str, np.ndarray]:
    """Stream a raw point file into a scan-by-distance elevation matrix."""

    first_line, last_line = profile_id_bounds(path)
    origin, axis, s_min, s_max = fit_geometry(path)
    n_profiles = last_line - first_line + 1
    n_bins = int(np.ceil((s_max - s_min) / bin_width_m))
    surface = np.full((n_profiles, n_bins), np.nan, dtype=np.float32)
    time_s = np.full(n_profiles, np.nan, dtype=np.float64)
    carry = pd.DataFrame(columns=USECOLS)

    reader = pd.read_csv(path, usecols=USECOLS, dtype=DTYPES, chunksize=chunk_rows)
    for chunk_number, chunk in enumerate(reader, start=1):
        if not carry.empty:
            chunk = pd.concat((carry, chunk), ignore_index=True)
        line_ids = np.rint(chunk["Line"].to_numpy(float)).astype(np.int64)
        final_line = line_ids[-1]
        complete = line_ids != final_line
        _aggregate_chunk(
            chunk.loc[complete], surface, time_s, first_line, origin, axis, s_min, bin_width_m
        )
        carry = chunk.loc[~complete].copy()
        print(
            f"{path.name}: chunk {chunk_number}, rows {chunk_number * chunk_rows:,}, "
            f"profile {final_line}/{last_line}",
            flush=True,
        )
    _aggregate_chunk(carry, surface, time_s, first_line, origin, axis, s_min, bin_width_m)
    s_m = s_min + (np.arange(n_bins) + 0.5) * bin_width_m
    return {
        "line": np.arange(first_line, last_line + 1, dtype=np.int32),
        "time_scan_s": time_s,
        "s_m": s_m.astype(np.float32),
        "surface_z_m": surface,
        "centerline_origin_xy_m": origin,
        "centerline_uphill_axis": axis,
    }


def fill_and_smooth(profile: np.ndarray, width: int = 5) -> np.ndarray:
    """Linearly fill gaps and apply a short robust running median."""

    values = np.asarray(profile, float).copy()
    index = np.arange(values.size)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return values
    values[~valid] = np.interp(index[~valid], index[valid], values[valid])
    radius = width // 2
    padded = np.pad(values, radius, mode="edge")
    return np.array([np.median(padded[i : i + width]) for i in range(values.size)])


def derive_observations(
    gridded: dict[str, np.ndarray],
    baseline_duration_s: float,
    pile_threshold_m: float,
    depth_threshold_m: float,
    support_bins: int,
    front_max_speed_m_s: float,
) -> tuple[dict[str, np.ndarray], dict]:
    time_scan = gridded["time_scan_s"]
    surface = gridded["surface_z_m"].astype(float)
    valid_profiles = np.isfinite(time_scan)
    if not valid_profiles.all():
        time_scan = time_scan[valid_profiles]
        surface = surface[valid_profiles]
        gridded["line"] = gridded["line"][valid_profiles]
    relative_scan = time_scan - time_scan[0]
    initial_rows = relative_scan <= baseline_duration_s
    with np.errstate(invalid="ignore"):
        initial_surface = np.nanmedian(surface[initial_rows], axis=0)
        preliminary_bed = np.nanpercentile(surface, 2.0, axis=0)
    preliminary_bed = fill_and_smooth(preliminary_bed)
    initial_surface = fill_and_smooth(initial_surface)
    preliminary_depth = np.maximum(initial_surface - preliminary_bed, 0.0)
    pile, gate_s = initial_pile_and_gate(
        gridded["s_m"], preliminary_depth, threshold_m=pile_threshold_m
    )
    # Before release the flume downslope from the gate is bare.  Its initial
    # median is therefore a substantially better bed estimate than a temporal
    # low percentile, which can be biased downward by moving/saltating returns.
    bed = preliminary_bed.copy()
    downslope = gridded["s_m"] < gate_s
    bed[downslope] = initial_surface[downslope]
    bed = fill_and_smooth(bed)
    initial_depth = np.maximum(initial_surface - bed, 0.0)
    onset_scan, motion_score = detect_motion_onset(
        time_scan, surface, initial_surface, pile, min_change_m=0.02, persistence=6
    )
    depth = surface - bed[None, :]
    # Values inside the lidar/bed uncertainty are treated as dry.
    depth[(depth < 0) & (depth > -0.05)] = 0.0
    x_downstream = gate_s - gridded["s_m"].astype(float)
    aligned_time = time_scan - onset_scan
    front = tracked_front_position(
        aligned_time,
        x_downstream,
        depth,
        threshold_m=depth_threshold_m,
        support_bins=support_bins,
        max_speed_m_s=front_max_speed_m_s,
    )
    derived = {
        "line": gridded["line"],
        "time_scan_s": time_scan,
        "time_since_motion_s": aligned_time.astype(np.float32),
        "x_downstream_m": x_downstream.astype(np.float32),
        "surface_z_m": surface.astype(np.float32),
        "bed_z_m": bed.astype(np.float32),
        "initial_surface_z_m": initial_surface.astype(np.float32),
        "initial_depth_m": initial_depth.astype(np.float32),
        "depth_m": depth.astype(np.float32),
        "front_x_m": front.astype(np.float32),
        "motion_score_m": motion_score.astype(np.float32),
        "centerline_origin_xy_m": gridded["centerline_origin_xy_m"],
        "centerline_uphill_axis": gridded["centerline_uphill_axis"],
    }
    finite_front = np.isfinite(front) & (aligned_time >= 0)
    summary = {
        "profile_count": int(surface.shape[0]),
        "distance_bin_count": int(surface.shape[1]),
        "scan_start_s": float(time_scan[0]),
        "scan_end_s": float(time_scan[-1]),
        "median_scan_rate_hz": float(1.0 / np.nanmedian(np.diff(time_scan))),
        "motion_onset_scan_time_s": float(onset_scan),
        "gate_s_m": float(gate_s),
        "initial_pile_length_m": float(np.ptp(gridded["s_m"][pile])),
        "initial_centerline_area_m2": float(
            np.trapezoid(initial_depth[pile], gridded["s_m"][pile])
        ),
        "maximum_coherent_front_x_m": float(np.nanmax(front[finite_front])),
    }
    return derived, summary


def save_diagnostic(path: Path, data: dict[str, np.ndarray], title: str) -> None:
    time = data["time_since_motion_s"]
    x = data["x_downstream_m"]
    order = np.argsort(x)
    active = (time >= -2.0) & (time <= 20.0)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(x[order], data["bed_z_m"][order], label="bed")
    axes[0].plot(x[order], data["initial_surface_z_m"][order], label="initial surface")
    axes[0].set(xlabel="distance downstream from gate (m)", ylabel="elevation (m)")
    axes[0].legend()
    image = axes[1].pcolormesh(
        x[order], time[active], data["depth_m"][active][:, order], shading="auto", vmin=0, vmax=1.5
    )
    axes[1].plot(data["front_x_m"][active], time[active], color="white", linewidth=1.0)
    axes[1].set(
        xlabel="distance downstream from gate (m)",
        ylabel="time since detected motion (s)",
        xlim=(-20, 95),
        title=title,
    )
    fig.colorbar(image, ax=axes[1], label="lidar-derived depth (m)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--date", choices=["2017-05-24", "2017-05-25"], required=True)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated" / "lidar")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run = config["runs"][args.date]
    source = REPO / config["lidar_dataset"] / run["lidar_file"]
    gridded = grid_lidar_file(source, config["lidar_bin_width_m"], args.chunk_rows)
    derived, summary = derive_observations(
        gridded,
        config["lidar_initial_baseline_s"],
        config["lidar_initial_pile_threshold_m"],
        config["lidar_depth_threshold_m"],
        config["lidar_min_front_support_bins"],
        config["lidar_front_max_speed_m_s"],
    )
    summary.update({"date": args.date, "source": str(source.relative_to(REPO))})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / f"{args.date}_lidar_profiles.npz", **derived)
    (args.output_dir / f"{args.date}_lidar_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    save_diagnostic(args.output_dir / f"{args.date}_lidar_diagnostic.png", derived, args.date)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
