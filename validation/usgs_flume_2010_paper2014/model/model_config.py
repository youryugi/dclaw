"""Configuration for the 2010 SGM rough-bed reproduction case.

Single, fixed case (no per-date branching): the 8-experiment aggregate
published in Iverson et al. (2010) and used by George & Iverson (2014) Part
II as their SGM rough-bed validation target.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = MODEL_DIR.parent
REPO = VALIDATION_DIR.parents[1]
CONFIG = json.loads((VALIDATION_DIR / "config.json").read_text())

PARAMETERS = dict(CONFIG["model"])
RUN = CONFIG["run"]

# Diagnostic-only overrides for cheap near-field sensitivity tests (default
# values reproduce the full validated run). Not part of the paper-sourced
# config; used to isolate candidate causes of the near-gate mismatch found
# at x=2 m (see results/summary.md, round 4).
GATE_HALF_WIDTH_M = float(os.environ.get("DCLAW_GATE_HALF_WIDTH_M", 0.375))
PILE_SHAPE = os.environ.get("DCLAW_PILE_SHAPE", "wedge")
if "DCLAW_ALPHA_C" in os.environ:
    PARAMETERS["compressibility_constant"] = float(os.environ["DCLAW_ALPHA_C"])

LIDAR_PATH = REPO / CONFIG["flume_bed_source"]

X_LOWER = -5.0
# The paper reports the gate-release flow front reaching x=112 m and
# depositing out to roughly x=120 m on the run-out pad beyond the 95 m
# flume proper (figure 15). The reused 2017 LiDAR survey only covers to
# x=92 m; bed_elevation() extrapolates the surveyed near-flat pad slope
# (~-0.04, i.e. ~2.3 deg) beyond that, which is the best approximation
# available without a survey of the actual extended pad.
X_UPPER = float(os.environ.get("DCLAW_DIAG_XUPPER", 130.0))
Y_LOWER = -PARAMETERS["flume_width_m"] / 2.0
Y_UPPER = PARAMETERS["flume_width_m"] / 2.0
DX = 0.25
DY = 0.25
FINAL_TIME = float(os.environ.get("DCLAW_DIAG_TFINAL", 35.0))


def load_bed() -> tuple[np.ndarray, np.ndarray]:
    if not LIDAR_PATH.exists():
        raise FileNotFoundError(
            f"missing {LIDAR_PATH}; run validation/usgs_flume_2017/preprocess_lidar.py"
            " --date 2017-05-25 first (the two setups share the fixed flume geometry)"
        )
    data = np.load(LIDAR_PATH)
    order = np.argsort(data["x_downstream_m"])
    return data["x_downstream_m"][order].astype(float), data["bed_z_m"][order].astype(float)


def bed_elevation(x: np.ndarray) -> np.ndarray:
    """Interpolate the lidar bed and linearly extrapolate short end gaps."""

    source_x, source_z = load_bed()
    values = np.interp(x, source_x, source_z)
    left_slope = np.polyfit(source_x[:20], source_z[:20], 1)[0]
    right_slope = np.polyfit(source_x[-20:], source_z[-20:], 1)[0]
    values = np.where(x < source_x[0], source_z[0] + left_slope * (x - source_x[0]), values)
    values = np.where(x > source_x[-1], source_z[-1] + right_slope * (x - source_x[-1]), values)
    return values


def gate_slope() -> float:
    source_x, source_z = load_bed()
    local = (source_x >= 0) & (source_x <= 10)
    return float(np.polyfit(source_x[local], source_z[local], 1)[0])


def initial_geometry() -> dict[str, float]:
    """Reconstruct the flat-topped wedge reservoir from figure 11 of
    George & Iverson (2014): a vertical rise of ``wedge_gate_height_m`` at
    the gate, a flat top, then a ramp down to the bed, spanning a total
    horizontal extent of ``wedge_horizontal_extent_m``. The flat/ramp split
    is solved so the enclosed volume matches ``volume_m3`` exactly while
    keeping both reported dimensions fixed.
    """

    slope = gate_slope()
    cosine = 1.0 / np.sqrt(1.0 + slope**2)
    gate_height = float(RUN["wedge_gate_height_m"])
    extent = float(RUN["wedge_horizontal_extent_m"])
    width = PARAMETERS["flume_width_m"]
    volume = float(RUN["volume_m3"])

    if PILE_SHAPE == "rect":
        # Diagnostic alternative: a uniform-depth block at the gate height,
        # shortened to match the same volume (no ramp). Isolates whether the
        # wedge's tapered tail (vs a blunt rectangular back) matters for the
        # near-gate release amplitude/duration.
        flat_length = volume / (width * gate_height)
        ramp_length = 0.0
    else:
        # volume = width * gate_height * (flat_length + 0.5 * ramp_length)
        # flat_length + ramp_length = extent
        target = volume / (width * gate_height)
        if not (0.5 * extent <= target <= extent):
            raise ValueError(
                f"volume_m3={volume} is not achievable as a flat-topped wedge of"
                f" height={gate_height} m and extent={extent} m"
            )
        ramp_length = 2.0 * (extent - target)
        flat_length = extent - ramp_length

    return {
        "bed_slope": slope,
        "slope_angle_deg": float(np.degrees(np.arctan(abs(slope)))),
        "slope_cosine": cosine,
        "gate_height_m": gate_height,
        "horizontal_extent_m": extent,
        "flat_length_m": flat_length,
        "ramp_length_m": ramp_length,
    }
