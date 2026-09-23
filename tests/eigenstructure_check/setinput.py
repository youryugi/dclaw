"""Build initial-condition rasters for the two eigenstructure checks.

EIGEN_TEST=contact: a solid-fraction (m) jump in still water on a flat bed.
    With kappa=1 (hardcoded in digclaw_module.f90) and equal h and hydrostatic
    pb on both sides, the net driving force is zero everywhere (eq 2.15/2.16
    of George & Iverson 2014), so the m-interface is a pure contact
    discontinuity (eigenvalue u=0, eigenvector (0,0,0,1,0) in eq 2.26) that
    should stay exactly stationary.

EIGEN_TEST=damfront: a classic wet/dry dam-break on a flat, frictionless bed.
    The nonlinear eigenvalues are lambda = u -/+ sqrt(gz*h) (eq 2.24 with
    kappa=1). For a dry-bed release from rest, the exact frictionless
    shallow-water front speed is 2*sqrt(g*h_L); friction (and any residual
    dilatancy) can only slow the modeled front below this, never exceed it,
    so it is a genuine upper bound on the modeled front position.
"""

from __future__ import annotations

import os

import numpy as np
from clawpack.geoclaw import topotools

TEST = os.environ.get("EIGEN_TEST", "contact")

DX = 0.05
DY = 0.05

if TEST == "contact":
    X_LOWER, X_UPPER = -5.0, 5.0
    Y_LOWER, Y_UPPER = -0.25, 0.25
    H_UNIFORM = 1.0
    M_LEFT, M_RIGHT = 0.55, 0.60
elif TEST == "damfront":
    X_LOWER, X_UPPER = 0.0, 20.0
    Y_LOWER, Y_UPPER = -0.25, 0.25
    H_LEFT, H_RIGHT = 2.0, 0.0
    RESERVOIR_END = 5.0
else:
    raise ValueError(f"unknown EIGEN_TEST={TEST!r}")


def basal(x, y):
    return np.zeros_like(np.asarray(x, dtype=float))


def surface(x, y):
    x = np.asarray(x, dtype=float)
    if TEST == "contact":
        return np.full_like(x, H_UNIFORM)
    return np.where(x < RESERVOIR_END, H_LEFT, H_RIGHT)


def solid_fraction(x, y):
    x = np.asarray(x, dtype=float)
    if TEST == "contact":
        return np.where(x < 0.0, M_LEFT, M_RIGHT)
    return np.full_like(x, 0.55)


def write_raster(name, function):
    nx = int(round((X_UPPER - X_LOWER) / (DX / 2))) + 1
    ny = int(round((Y_UPPER - Y_LOWER) / (DY / 2))) + 1
    topotools.topo3writer(name, function, X_LOWER, X_UPPER, Y_LOWER, Y_UPPER, nx, ny)


def main():
    write_raster("basal_topo.tt3", basal)
    write_raster("surface_topo.tt3", surface)
    write_raster("solid_fraction.tt3", solid_fraction)


if __name__ == "__main__":
    main()
