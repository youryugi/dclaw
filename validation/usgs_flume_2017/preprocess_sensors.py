#!/usr/bin/env python3
"""Preprocess the 1 kHz USGS flume sensor records into validation products."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from flume_validation import first_persistent_exceedance


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def safe_name(name: str) -> str:
    """Create stable NumPy archive keys from published column labels."""

    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    """Downsample with NaN-aware, non-overlapping block means."""

    n = values.shape[0] // factor
    trimmed = values[: n * factor]
    blocks = trimmed.reshape(n, factor, *values.shape[1:])
    valid = np.isfinite(blocks)
    count = valid.sum(axis=1)
    total = np.nansum(blocks, axis=1)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def process_run(
    csv_path: Path,
    output_path: Path,
    positions_m: list[float],
    threshold_m: float,
    min_duration_s: float,
    output_rate_hz: int,
    pore_fluid_density_kg_m3: float,
) -> dict:
    frame = pd.read_csv(csv_path, encoding="latin1")
    time = frame.iloc[:, 0].to_numpy(float)
    baseline = time < 0
    if baseline.sum() == 0:
        raise ValueError(f"{csv_path} contains no pre-release baseline")

    summary: dict = {
        "source": str(csv_path.relative_to(REPO)),
        "sample_count": int(len(frame)),
        "time_start_s": float(time[0]),
        "time_end_s": float(time[-1]),
        "sample_rate_hz": float(1.0 / np.median(np.diff(time))),
        "all_null_columns": [str(c) for c in frame.columns[frame.isna().all()]],
        "gauges": {},
    }

    initial_window = (time >= -4.0) & (time <= -1.0)
    gate_column = next(column for column in frame if column.startswith("GateAngle"))
    gate_angle = frame[gate_column].to_numpy(float)

    def first_gate_time(angle_deg: float) -> float:
        indices = np.flatnonzero(np.isfinite(gate_angle) & (gate_angle >= angle_deg))
        return float(time[indices[0]]) if indices.size else float("nan")

    upstream_pore_columns = [
        column
        for column in frame
        if column.startswith("PP") and "-2.9m" in column and not frame[column].isna().all()
    ]
    upstream_thickness_column = next(
        column for column in frame if column.startswith("Thickness_-2.3m")
    )
    upstream_normal_column = next(
        column for column in frame if column.startswith("Nstress_-2.9m")
    )
    initial_pore_kpa = float(
        np.nanmean(frame.loc[initial_window, upstream_pore_columns].to_numpy(float))
    )
    initial_depth_m = float(
        np.nanmean(frame.loc[initial_window, upstream_thickness_column].to_numpy(float))
    )
    initial_normal_kpa = float(
        np.nanmean(frame.loc[initial_window, upstream_normal_column].to_numpy(float))
    )
    summary["initial_conditions"] = {
        "averaging_window_s": [-4.0, -1.0],
        "mean_upstream_depth_m": initial_depth_m,
        "mean_upstream_pore_pressure_kpa": initial_pore_kpa,
        "mean_upstream_normal_stress_kpa": initial_normal_kpa,
        "hydrostatic_pore_pressure_ratio": initial_pore_kpa
        * 1000.0
        / (pore_fluid_density_kg_m3 * 9.81 * initial_depth_m),
    }
    summary["gate_opening"] = {
        "first_10_deg_s": first_gate_time(10.0),
        "first_45_deg_s": first_gate_time(45.0),
        "first_89_deg_s": first_gate_time(89.0),
    }
    archive: dict[str, np.ndarray] = {}
    original_rate = summary["sample_rate_hz"]
    factor = max(1, int(round(original_rate / output_rate_hz)))
    archive["time_s"] = block_mean(time[:, None], factor)[:, 0]

    for column in frame.columns[1:]:
        values = frame[column].to_numpy(float)
        if np.isnan(values).all():
            continue
        base = float(np.nanmean(values[baseline]))
        archive[f"raw__{safe_name(column)}"] = block_mean(values[:, None], factor)[:, 0]
        archive[f"delta__{safe_name(column)}"] = block_mean((values - base)[:, None], factor)[:, 0]

    for position in positions_m:
        token = f"{position:g}m"
        candidates = [c for c in frame if c.startswith("Thickness_") and token in c]
        # The published 80 m header omits the decimal point.
        if position == 80.0 and not candidates:
            candidates = [c for c in frame if c.startswith("Thickness_80m")]
        if not candidates:
            continue
        column = candidates[0]
        values = frame[column].to_numpy(float)
        base = float(np.nanmean(values[baseline]))
        depth = values - base
        first_arrival = first_persistent_exceedance(time, depth, threshold_m, 0.0)
        coherent_arrival = first_persistent_exceedance(
            time, depth, threshold_m, min_duration_s
        )
        pore_cols = [c for c in frame if c.startswith("PP") and token in c and not frame[c].isna().all()]
        if position == 80.0 and not pore_cols:
            pore_cols = [c for c in frame if c.startswith("PP") and "80.0m" in c and not frame[c].isna().all()]
        pore_peak = float("nan")
        if pore_cols:
            pore = frame[pore_cols].to_numpy(float)
            pore -= np.nanmean(pore[baseline], axis=0)
            pore_peak = float(np.nanmax(np.nanmean(pore, axis=1)))
        summary["gauges"][f"{position:g}"] = {
            "thickness_column": column,
            "baseline_depth_m": base,
            "first_arrival_time_s": first_arrival,
            "coherent_arrival_time_s": coherent_arrival,
            "peak_depth_change_m": float(np.nanmax(depth)),
            "pore_pressure_channels": len(pore_cols),
            "peak_mean_pore_pressure_change_kpa": pore_peak,
        }

    ordered = sorted(
        (float(x), v["first_arrival_time_s"]) for x, v in summary["gauges"].items()
    )
    segments = []
    for (x0, t0), (x1, t1) in zip(ordered, ordered[1:]):
        if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
            segments.append(
                {
                    "x0_m": x0,
                    "x1_m": x1,
                    "mean_front_velocity_m_s": (x1 - x0) / (t1 - t0),
                }
            )
    summary["front_velocity_segments"] = segments
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **archive)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated" / "sensors")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    dataset = REPO / config["sensor_dataset"]
    all_summaries = {}
    for date, run in config["runs"].items():
        summary = process_run(
            dataset / run["sensor_file"],
            args.output_dir / f"{date}_sensors_100hz.npz",
            config["gauge_positions_m"],
            config["arrival_depth_threshold_m"],
            config["arrival_min_duration_s"],
            config["sensor_output_rate_hz"],
            config["parameter_sets"]["paper2014"]["rho_f_kg_m3"],
        )
        summary["volume_m3"] = run["volume_m3"]
        all_summaries[date] = summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sensor_summary.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(all_summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
