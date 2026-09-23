#!/usr/bin/env python3
"""Compare the D-Claw 2010-SGM reproduction with the paper's aggregate data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL_DIR = HERE / "model"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(REPO / "validation" / "usgs_flume_2017"))

from flume_validation import compare_series  # noqa: E402
from parse_ds03 import load_sections  # noqa: E402
from model_config import PARAMETERS, load_bed  # noqa: E402


GAUGE_IDS = {32.0: 2, 66.0: 3, 90.0: 4}


def read_gauge(path: Path) -> dict[str, np.ndarray]:
    values = np.loadtxt(path, comments="#")
    h = values[:, 2]
    m = np.divide(values[:, 5], h, out=np.zeros_like(h), where=h > 1.0e-6)
    return {
        "time_s": values[:, 1],
        "depth_vertical_m": h,
        "solid_fraction": m,
        "pore_pressure_pa": values[:, 6],
    }


def local_slope_cosine(x: np.ndarray, bed: np.ndarray, position: float) -> float:
    order = np.argsort(x)
    slope = np.gradient(bed[order], x[order])
    local_slope = float(np.interp(position, x[order], slope))
    return float(1.0 / np.sqrt(1.0 + local_slope**2))


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
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_compatible(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    outdir = MODEL_DIR / "_output"
    sections = load_sections()
    x_bed, z_bed = load_bed()

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True)
    summaries = {}
    for row, position in enumerate([32.0, 66.0, 90.0]):
        gauge = read_gauge(outdir / f"gauge{GAUGE_IDS[position]:05d}.txt")
        cosine = local_slope_cosine(x_bed, z_bed, position)
        model_h_normal = gauge["depth_vertical_m"] * cosine
        rho_bar = (
            gauge["solid_fraction"] * PARAMETERS["rho_s_kg_m3"]
            + (1.0 - gauge["solid_fraction"]) * PARAMETERS["rho_f_kg_m3"]
        )
        model_sigma_kpa = rho_bar * 9.81 * model_h_normal * cosine / 1000.0
        model_pbed_kpa = gauge["pore_pressure_pa"] / 1000.0

        table = sections[position]
        obs_t = table["t_s"]
        model_h_on_obs = np.interp(
            obs_t, gauge["time_s"], model_h_normal, left=0.0, right=model_h_normal[-1]
        )
        model_sigma_on_obs = np.interp(
            obs_t, gauge["time_s"], model_sigma_kpa, left=0.0, right=model_sigma_kpa[-1]
        )
        model_pbed_on_obs = np.interp(
            obs_t, gauge["time_s"], model_pbed_kpa, left=0.0, right=model_pbed_kpa[-1]
        )

        h_metrics = compare_series(table["h_m"], model_h_on_obs)
        sigma_metrics = compare_series(table["sigma_kpa"], model_sigma_on_obs)
        pbed_metrics = compare_series(table["pbed_kpa"], model_pbed_on_obs)

        summaries[f"{position:g}"] = {
            "observed_peak_h_m": float(np.nanmax(table["h_m"])),
            "modeled_peak_h_m": float(np.nanmax(model_h_normal)),
            "h_metrics": metric_dict(h_metrics),
            "sigma_metrics_kpa": metric_dict(sigma_metrics),
            "pbed_metrics_kpa": metric_dict(pbed_metrics),
        }

        ax = axes[row, 0]
        ax.plot(obs_t, table["h_m"], label="2010 aggregate (mean of 8)", linewidth=1.0)
        ax.fill_between(
            obs_t, table["h_m"] - table["h_sd_m"], table["h_m"] + table["h_sd_m"], alpha=0.2
        )
        ax.plot(obs_t, model_h_on_obs, label="D-Claw", linewidth=1.0)
        ax.set(title=f"x={position:g} m: flow thickness (m)")
        ax.grid(alpha=0.25)

        ax = axes[row, 1]
        ax.plot(obs_t, table["sigma_kpa"], linewidth=1.0)
        ax.fill_between(
            obs_t,
            table["sigma_kpa"] - table["sigma_sd_kpa"],
            table["sigma_kpa"] + table["sigma_sd_kpa"],
            alpha=0.2,
        )
        ax.plot(obs_t, model_sigma_on_obs, linewidth=1.0)
        ax.set(title=f"x={position:g} m: total normal stress (kPa)")
        ax.grid(alpha=0.25)

        ax = axes[row, 2]
        ax.plot(obs_t, table["pbed_kpa"], linewidth=1.0)
        ax.fill_between(
            obs_t,
            table["pbed_kpa"] - table["pbed_sd_kpa"],
            table["pbed_kpa"] + table["pbed_sd_kpa"],
            alpha=0.2,
        )
        ax.plot(obs_t, model_pbed_on_obs, linewidth=1.0)
        ax.set(title=f"x={position:g} m: basal pore pressure (kPa)")
        ax.grid(alpha=0.25)

    axes[0, 0].legend()
    for ax in axes[-1, :]:
        ax.set_xlabel("time since gate release (s)")
    plot_path = HERE / "generated" / "comparison_2010_sgm.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    maximum_cfl = float("nan")
    fort_amr = outdir / "fort.amr"
    if fort_amr.exists():
        for line in fort_amr.read_text(errors="ignore").splitlines():
            if "maximum Courant number seen" in line:
                maximum_cfl = float(line.split("=")[-1])

    summary = {
        "dataset": "dataset/usgs_flume_2010/jgrf703-sup-0004-ds03.txt (SGM rough bed, N=8 aggregate)",
        "outdir": str(outdir.relative_to(REPO)),
        "gauges": summaries,
        "numerics": {"maximum_cfl": maximum_cfl},
    }
    output_json = HERE / "generated" / "comparison_2010_sgm_summary.json"
    output_json.write_text(json.dumps(json_compatible(summary), indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
