"""Configuration helpers shared by the flume input and runtime setup."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = MODEL_DIR.parent
CONFIG = json.loads((VALIDATION_DIR / "config.json").read_text())
RUN_DATE = os.environ.get("FLUME_RUN_DATE", "2017-05-25")
if RUN_DATE not in ("2017-05-24", "2017-05-25"):
    raise ValueError("FLUME_RUN_DATE must be 2017-05-24 or 2017-05-25")
RUN = CONFIG["runs"][RUN_DATE]
PARAMETERS = dict(CONFIG["model"])
PARAMETER_SET = os.environ.get("DCLAW_PARAMETER_SET", "baseline")
if PARAMETER_SET != "baseline":
    try:
        PARAMETERS.update(CONFIG["parameter_sets"][PARAMETER_SET])
    except KeyError as exc:
        choices = ", ".join(["baseline", *CONFIG.get("parameter_sets", {})])
        raise ValueError(f"unknown DCLAW_PARAMETER_SET={PARAMETER_SET!r}; choose {choices}") from exc
if PARAMETERS.get("use_observed_initial_pore_pressure", False):
    PARAMETERS["initial_pore_pressure_ratio"] = RUN[
        "observed_initial_pore_pressure_ratio"
    ]

# Environment overrides make sensitivity runs reproducible without editing the
# baseline configuration.  Each output directory is paired with the values
# written by setinput.py.
_OVERRIDES = {
    "DCLAW_PINIT_RATIO": "initial_pore_pressure_ratio",
    "DCLAW_PHI_DEG": "basal_friction_angle_deg",
    "DCLAW_KREF_M2": "reference_permeability_m2",
}
for environment_name, parameter_name in _OVERRIDES.items():
    if environment_name in os.environ:
        PARAMETERS[parameter_name] = float(os.environ[environment_name])
if "DCLAW_USE_FINITE_GATE" in os.environ:
    value = os.environ["DCLAW_USE_FINITE_GATE"].strip().lower()
    if value not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError("DCLAW_USE_FINITE_GATE must be true/false or 1/0")
    PARAMETERS["use_finite_gate"] = value in {"1", "true", "yes"}

if not 0.0 <= PARAMETERS["initial_pore_pressure_ratio"] <= 1.0:
    raise ValueError("DCLAW_PINIT_RATIO must be between 0 and 1")
if not 0.0 < PARAMETERS["basal_friction_angle_deg"] < 90.0:
    raise ValueError("DCLAW_PHI_DEG must be between 0 and 90 degrees")
if PARAMETERS["reference_permeability_m2"] <= 0.0:
    raise ValueError("DCLAW_KREF_M2 must be positive")
LIDAR_PATH = VALIDATION_DIR / "generated" / "lidar" / f"{RUN_DATE}_lidar_profiles.npz"

X_LOWER = -5.0
X_UPPER = 95.0
Y_LOWER = -PARAMETERS["flume_width_m"] / 2.0
Y_UPPER = PARAMETERS["flume_width_m"] / 2.0
DX = 0.25
DY = 0.25
FINAL_TIME = 25.0


def load_bed() -> tuple[np.ndarray, np.ndarray]:
    if not LIDAR_PATH.exists():
        raise FileNotFoundError(f"run preprocess_lidar.py first: {LIDAR_PATH}")
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
    """Construct the unseen reservoir prism from volume and normal depth."""

    slope = gate_slope()
    cosine = 1.0 / np.sqrt(1.0 + slope**2)
    normal_depth = float(RUN["initial_normal_depth_m"])
    vertical_depth = normal_depth / cosine
    length = float(RUN["volume_m3"]) / (
        PARAMETERS["flume_width_m"] * vertical_depth
    )
    return {
        "bed_slope": slope,
        "slope_angle_deg": float(np.degrees(np.arctan(abs(slope)))),
        "slope_cosine": cosine,
        "normal_depth_m": normal_depth,
        "vertical_depth_m": vertical_depth,
        "horizontal_length_m": length,
    }
