"""Create topography and initial-state rasters for one flume experiment."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from clawpack.geoclaw import dtopotools, topotools

from model_config import (
    DX,
    DY,
    MODEL_DIR,
    PARAMETER_SET,
    PARAMETERS,
    RUN,
    RUN_DATE,
    X_LOWER,
    X_UPPER,
    Y_LOWER,
    Y_UPPER,
    bed_elevation,
    initial_geometry,
)


GEOMETRY = initial_geometry()
GATE_HALF_WIDTH_M = 0.375


def natural_basal(x, y):
    return bed_elevation(np.asarray(x))


def gate_shape(x):
    """Computational ridge representing the closed, bed-normal headgate."""

    x = np.asarray(x)
    return PARAMETERS["gate_height_m"] * np.clip(
        (GATE_HALF_WIDTH_M - np.abs(x)) / (GATE_HALF_WIDTH_M - 0.125), 0.0, 1.0
    )


def basal(x, y):
    bed = natural_basal(x, y)
    if PARAMETERS["use_finite_gate"]:
        bed = bed + gate_shape(x)
    return bed


def thickness(x, y):
    x = np.asarray(x)
    downslope_edge = -GATE_HALF_WIDTH_M if PARAMETERS["use_finite_gate"] else 0.0
    inside = (
        (x >= downslope_edge - GEOMETRY["horizontal_length_m"])
        & (x <= downslope_edge)
    )
    return np.where(inside, GEOMETRY["vertical_depth_m"], 0.0)


def surface(x, y):
    # The gate is a barrier, not part of the initial free surface.
    return natural_basal(x, y) + thickness(x, y)


def solid_fraction(x, y):
    return np.where(thickness(x, y) > 0, PARAMETERS["solid_volume_fraction"], 0.0)


def write_raster(name, function):
    nx = int(round((X_UPPER - X_LOWER) / (DX / 2))) + 1
    ny = int(round((Y_UPPER - Y_LOWER) / (DY / 2))) + 1
    topotools.topo3writer(
        name, function, X_LOWER, X_UPPER, Y_LOWER, Y_UPPER, nx, ny
    )


def write_gate_dtopo() -> None:
    """Remove the static gate ridge according to the measured gate angle."""

    times = np.arange(0.0, 1.001, 0.01)
    profile = np.asarray(RUN["gate_angle_profile"], dtype=float)
    angle = np.interp(times, profile[:, 0], profile[:, 1], left=0.0, right=90.0)
    removed_fraction = 1.0 - np.cos(np.deg2rad(angle))
    x = np.arange(-0.5, 0.5001, 0.125)
    y = np.arange(Y_LOWER, Y_UPPER + 0.0001, 0.125)
    X, Y = np.meshgrid(x, y)
    shape = gate_shape(X)
    dtopo = dtopotools.DTopography()
    dtopo.x = x
    dtopo.y = y
    dtopo.X = X
    dtopo.Y = Y
    dtopo.times = times
    dtopo.dZ = -removed_fraction[:, None, None] * shape[None, :, :]
    dtopo.write("gate_dtopo.tt3", dtopo_type=3, dZ_format="%.8e")


def main():
    write_raster("basal_topo.tt3", basal)
    write_raster("surface_topo.tt3", surface)
    write_raster("solid_fraction.tt3", solid_fraction)
    if PARAMETERS["use_finite_gate"]:
        write_gate_dtopo()
    volume = (
        GEOMETRY["vertical_depth_m"]
        * GEOMETRY["horizontal_length_m"]
        * PARAMETERS["flume_width_m"]
    )
    summary = {
        "date": RUN_DATE,
        "parameter_set": PARAMETER_SET,
        "target_volume_m3": RUN["volume_m3"],
        "raster_volume_m3": volume,
        "model_parameters": PARAMETERS,
    }
    summary.update(GEOMETRY)
    (MODEL_DIR / "input_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    x = np.linspace(X_LOWER, X_UPPER, 1001)
    y = np.zeros_like(x)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(x, basal(x, y), label="lidar-derived bed")
    ax.plot(x, surface(x, y), label="initial surface")
    ax.set(xlabel="distance downstream from gate (m)", ylabel="elevation (m)", title=RUN_DATE)
    ax.legend()
    fig.savefig("initial_condition.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
