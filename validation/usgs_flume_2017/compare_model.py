#!/usr/bin/env python3
"""Compare one D-Claw flume run with sensors and profile lidar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flume_validation import (
    compare_series,
    first_persistent_exceedance,
    historical_front_envelope,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL_DIR = HERE / "model"
sys.path.insert(0, str(MODEL_DIR))


def read_gauge(path: Path) -> dict[str, np.ndarray]:
    values = np.loadtxt(path, comments="#")
    h = values[:, 2]
    return {
        "time_s": values[:, 1],
        "depth_vertical_m": h,
        "velocity_x_m_s": np.divide(
            values[:, 3], h, out=np.zeros_like(h), where=h > 1.0e-6
        ),
        "solid_fraction": np.divide(
            values[:, 5], h, out=np.zeros_like(h), where=h > 1.0e-6
        ),
        "pore_pressure_kpa": values[:, 6] / 1000.0,
    }


def sensor_key(position: float) -> str:
    return f"{position:g}".replace(".", "_")


def sensor_depth(archive, position: float) -> np.ndarray:
    token = sensor_key(position)
    matches = [
        key
        for key in archive.files
        if key.startswith("delta__Thickness_") and token in key
    ]
    if position == 80.0 and not matches:
        matches = [key for key in archive.files if key.startswith("delta__Thickness_80m")]
    if len(matches) != 1:
        raise ValueError(f"cannot identify thickness field at {position} m: {matches}")
    return archive[matches[0]].astype(float)


def sensor_pore_pressure(archive, position: float) -> np.ndarray | None:
    token = sensor_key(position)
    matches = [
        key
        for key in archive.files
        if key.startswith("delta__PP") and token in key
    ]
    if position == 80.0 and not matches:
        matches = [key for key in archive.files if key.startswith("delta__PP") and "80_0m" in key]
    if not matches:
        return None
    return np.nanmean(np.column_stack([archive[key] for key in matches]), axis=1)


def local_slope_cosine(x: np.ndarray, bed: np.ndarray, position: float) -> float:
    order = np.argsort(x)
    slope = np.gradient(bed[order], x[order])
    local_slope = float(np.interp(position, x[order], slope))
    return float(1.0 / np.sqrt(1.0 + local_slope**2))


def read_model_front(outdir: Path, slope_x: np.ndarray, slope_cos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from clawpack.visclaw.data import ClawPlotData

    plotdata = ClawPlotData()
    plotdata.outdir = str(outdir)
    plotdata.format = "ascii"
    plotdata.verbose = False
    frame_numbers = sorted(int(path.name[-4:]) for path in outdir.glob("fort.t[0-9][0-9][0-9][0-9]"))
    times = []
    fronts = []
    for frame_number in frame_numbers:
        frame = plotdata.getframe(frame_number)
        state = frame.states[0]
        x = state.patch.grid.c_centers[0][:, 0]
        y = state.patch.grid.c_centers[1][0, :]
        j = int(np.argmin(np.abs(y)))
        depth = state.q[0, :, j]
        cosine = np.interp(x, slope_x, slope_cos)
        wet = depth * cosine > 0.03
        times.append(float(frame.t))
        fronts.append(float(np.max(x[wet])) if wet.any() else float("nan"))
    # A front is an arrival envelope: deposition must not make its historical
    # maximum retreat or disappear. This also matches first-passage gauges.
    return np.asarray(times), historical_front_envelope(fronts)


def metric_dict(metrics) -> dict:
    return {
        "n": metrics.n,
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "bias": metrics.bias,
        "correlation": metrics.correlation,
        "nse": metrics.nse,
    }


def json_compatible(value):
    """Replace non-finite floats so output is strict JSON."""

    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_compatible(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def markdown_report(date: str, summary: dict, label: str) -> str:
    rows = []
    for position, item in summary["gauges"].items():
        rows.append(
            f"| {position} | {item['observed_first_arrival_s']:.3f} | "
            f"{item['observed_coherent_arrival_s']:.3f} | "
            f"{item['modeled_arrival_s']:.3f} | {item['arrival_error_s']:.3f} | "
            f"{item['coherent_arrival_error_s']:.3f} | "
            f"{item['observed_peak_depth_m']:.3f} | {item['modeled_peak_depth_m']:.3f} |"
        )
    return "\n".join(
        [
            f"# {date} D-Claw {label} 验证结果",
            "",
            (
                "当前结果使用未校准参数，作用是建立后续参数分析的基线。"
                if label == "baseline"
                else f"当前结果对应参数敏感性方案 `{label}`。"
            ),
            "",
            "| 位置 (m) | 观测首次到达 (s) | 观测持续到达 (s) | 模拟到达 (s) | 对首次误差 (s) | 对持续误差 (s) | 观测峰值流深 (m) | 模拟峰值流深 (m) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "| 综合量 | 数值 |",
            "|---|---:|",
            f"| LiDAR 前缘 RMSE，x≤{summary['front']['comparison_max_m']:g} m (m) | {summary['front']['rmse_m']:.3f} |",
            f"| 观测最大前缘 (m) | {summary['front']['observed_max_m']:.3f} |",
            f"| 模拟 25 s 前缘 (m) | {summary['front']['modeled_final_m']:.3f} |",
            f"| 最大 CFL | {summary['numerics']['maximum_cfl']:.5f} |",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["2017-05-24", "2017-05-25"], required=True)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text())
    outdir = (args.outdir or MODEL_DIR / f"_output_{args.date}").resolve()
    sensor_archive = np.load(HERE / "generated" / "sensors" / f"{args.date}_sensors_100hz.npz")
    sensor_summary = json.loads(
        (HERE / "generated" / "sensors" / "sensor_summary.json").read_text()
    )[args.date]
    lidar = np.load(HERE / "generated" / "lidar" / f"{args.date}_lidar_profiles.npz")
    x_lidar = lidar["x_downstream_m"].astype(float)
    bed = lidar["bed_z_m"].astype(float)
    order = np.argsort(x_lidar)
    slope = np.gradient(bed[order], x_lidar[order])
    slope_cos = 1.0 / np.sqrt(1.0 + slope**2)

    positions = config["gauge_positions_m"]
    gauge_ids = {2.5: 2, 31.7: 3, 65.4: 4, 80.0: 5}
    observation_time = sensor_archive["time_s"].astype(float)
    summaries = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    for ax, position in zip(axes.flat, positions):
        gauge = read_gauge(outdir / f"gauge{gauge_ids[position]:05d}.txt")
        cosine = local_slope_cosine(x_lidar, bed, position)
        model_normal_depth = gauge["depth_vertical_m"] * cosine
        observed_depth = sensor_depth(sensor_archive, position)
        modeled_on_observation = np.interp(
            observation_time, gauge["time_s"], model_normal_depth, left=0.0, right=model_normal_depth[-1]
        )
        window = (observation_time >= 0) & (observation_time <= 20)
        depth_metrics = compare_series(observed_depth[window], modeled_on_observation[window])
        model_arrival = first_persistent_exceedance(
            gauge["time_s"], model_normal_depth, config["arrival_depth_threshold_m"], 0.0
        )
        observed_info = sensor_summary["gauges"][f"{position:g}"]
        first_arrival = observed_info["first_arrival_time_s"]
        coherent_arrival = observed_info["coherent_arrival_time_s"]
        item = {
            "observed_first_arrival_s": first_arrival,
            "observed_coherent_arrival_s": coherent_arrival,
            "modeled_arrival_s": model_arrival,
            "arrival_error_s": model_arrival - first_arrival,
            "coherent_arrival_error_s": model_arrival - coherent_arrival,
            "observed_peak_depth_m": observed_info["peak_depth_change_m"],
            "modeled_peak_depth_m": float(np.nanmax(model_normal_depth)),
            "depth_metrics_0_20s": metric_dict(depth_metrics),
        }
        observed_pore = sensor_pore_pressure(sensor_archive, position)
        if observed_pore is not None:
            modeled_pore = np.interp(
                observation_time,
                gauge["time_s"],
                gauge["pore_pressure_kpa"],
                left=0.0,
                right=gauge["pore_pressure_kpa"][-1],
            )
            item["pore_pressure_metrics_0_20s"] = metric_dict(
                compare_series(observed_pore[window], modeled_pore[window])
            )
            item["modeled_peak_pore_pressure_kpa"] = float(
                np.nanmax(gauge["pore_pressure_kpa"])
            )
        summaries[f"{position:g}"] = item
        ax.plot(observation_time, observed_depth, label="sensor", linewidth=1.0)
        ax.plot(gauge["time_s"], model_normal_depth, label="D-Claw", linewidth=1.0)
        ax.set(title=f"x = {position:g} m", ylabel="bed-normal depth (m)", xlim=(0, 20))
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("time after gate release (s)")
    axes[1, 1].set_xlabel("time after gate release (s)")
    axes[0, 0].legend()
    plot_path = HERE / "generated" / "comparison" / f"{args.date}_{args.label}_gauge_depth.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    model_time, model_front = read_model_front(outdir, x_lidar[order], slope_cos)
    lidar_valid = np.isfinite(lidar["front_x_m"]) & (lidar["time_since_motion_s"] >= 0)
    lidar_time = lidar["time_since_motion_s"][lidar_valid].astype(float)
    lidar_front = lidar["front_x_m"][lidar_valid].astype(float)
    lidar_front = historical_front_envelope(lidar_front)
    # The lidar archive has no common sensor clock and is explicitly expressed
    # relative to detected motion. Align the model front to its first passage
    # through the gate; gauge comparisons above remain on the gate clock.
    onset_candidates = np.flatnonzero(np.isfinite(model_front) & (model_front >= 0.0))
    model_motion_onset = (
        float(model_time[onset_candidates[0]]) if onset_candidates.size else float("nan")
    )
    model_motion_time = model_time - model_motion_onset
    time_window = (
        (model_motion_time >= lidar_time.min())
        & (model_motion_time <= lidar_time.max())
    )
    observed_front_at_model = np.full(model_time.shape, np.nan)
    observed_front_at_model[time_window] = np.interp(
        model_motion_time[time_window], lidar_time, lidar_front
    )
    comparison_max = float(config["lidar_front_comparison_max_m"])
    compare_window = time_window & (observed_front_at_model <= comparison_max)
    front_metrics = compare_series(
        observed_front_at_model[compare_window], model_front[compare_window]
    )
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(lidar_time, lidar_front, ".", markersize=1, alpha=0.5, label="lidar")
    ax.plot(model_motion_time, model_front, "-o", markersize=2, label="D-Claw")
    ax.set(xlabel="time after detected motion (s)", ylabel="coherent front (m)", xlim=(0, 20))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(
        HERE / "generated" / "comparison" / f"{args.date}_{args.label}_front.png",
        dpi=160,
    )
    plt.close(fig)

    maximum_cfl = float("nan")
    fort_amr = outdir / "fort.amr"
    if fort_amr.exists():
        for line in fort_amr.read_text(errors="ignore").splitlines():
            if "maximum Courant number seen" in line:
                maximum_cfl = float(line.split("=")[-1])
    summary = {
        "date": args.date,
        "label": args.label,
        "outdir": str(outdir.relative_to(REPO)),
        "gauges": summaries,
        "front": {
            "rmse_m": front_metrics.rmse,
            "mae_m": front_metrics.mae,
            "bias_m": front_metrics.bias,
            "observed_max_m": float(np.nanmax(lidar_front)),
            "modeled_final_m": float(model_front[-1]),
            "modeled_motion_onset_s_after_gate_clock": model_motion_onset,
            "comparison_max_m": comparison_max,
        },
        "numerics": {"maximum_cfl": maximum_cfl},
    }
    output_json = HERE / "generated" / "comparison" / f"{args.date}_{args.label}_summary.json"
    output_json.write_text(json.dumps(json_compatible(summary), indent=2, allow_nan=False) + "\n")
    report_path = HERE / "results" / f"{args.label}_{args.date}.md"
    report_path.write_text(markdown_report(args.date, summary, args.label))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
