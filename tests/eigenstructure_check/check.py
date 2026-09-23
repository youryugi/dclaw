#!/usr/bin/env python3
"""Run both eigenstructure sanity checks end-to-end and report pass/fail.

These check the D-Claw solver's characteristic (wave) structure against
closed-form results that are independent of any specific paper's published
run -- they depend only on the model's own governing equations (Iverson &
George 2014, eq 2.17-2.26, with kappa=1 as hardcoded in digclaw_module.f90)
and, for the dam-break case, the classical shallow-water solution. See
setinput.py for the physics of each case.

Usage (from this directory, with CLAW / PYTHONPATH already set up):
    python3 check.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def run(env_extra: dict[str, str], outdir: str) -> None:
    env = dict(os.environ)
    env.update(env_extra)
    for step in (["setinput.py"], ["setrun.py"]):
        subprocess.run(
            [sys.executable, *step], cwd=HERE, env=env, check=True,
            capture_output=True, text=True,
        )
    subprocess.run(
        ["make", "output", f"OUTDIR={outdir}"], cwd=HERE, env=env, check=True,
        capture_output=True, text=True,
    )


def check_contact() -> bool:
    from clawpack.visclaw.data import ClawPlotData

    outdir = "_output_contact"
    run({"EIGEN_TEST": "contact"}, outdir)

    plotdata = ClawPlotData()
    plotdata.outdir = str(HERE / outdir)
    plotdata.format = "ascii"
    plotdata.verbose = False

    frame0 = plotdata.getframe(0)
    n_frames = len(sorted((HERE / outdir).glob("fort.t[0-9][0-9][0-9][0-9]")))
    final = plotdata.getframe(n_frames - 1)

    ok = True
    for label, frame in [("t=0", frame0), ("final", final)]:
        state = frame.states[0]
        h = state.q[0]
        hu = state.q[1]
        u = np.divide(hu, h, out=np.zeros_like(h), where=h > 1e-6)
        max_u = float(np.max(np.abs(u)))
        h_spread = float(h.max() - h.min())
        print(f"  [{label}] t={frame.t:.3f}s max|u|={max_u:.3e} h_spread={h_spread:.3e}")
        if max_u > 1e-9 or h_spread > 1e-9:
            ok = False

    x = frame0.states[0].patch.grid.c_centers[0][:, 0]
    y = frame0.states[0].patch.grid.c_centers[1][0, :]
    j = int(np.argmin(np.abs(y)))

    def m_interface(frame):
        h = frame.states[0].q[0][:, j]
        hm = frame.states[0].q[3][:, j]
        m = np.divide(hm, h, out=np.zeros_like(h), where=h > 1e-6)
        return float(x[np.argmin(np.abs(m - 0.575))])

    x0, xf = m_interface(frame0), m_interface(final)
    print(f"  m-interface position: t=0 -> {x0:+.4f} m, final -> {xf:+.4f} m")
    if abs(xf - x0) > 1e-6:
        ok = False

    return ok


def check_damfront() -> bool:
    from clawpack.visclaw.data import ClawPlotData

    outdir = "_output_damfront"
    run({"EIGEN_TEST": "damfront"}, outdir)

    plotdata = ClawPlotData()
    plotdata.outdir = str(HERE / outdir)
    plotdata.format = "ascii"
    plotdata.verbose = False

    g = 9.81
    h_l = 2.0
    reservoir_end = 5.0
    theory_speed = 2.0 * (g * h_l) ** 0.5
    tolerance_m = 0.1  # grid-scale slack for the front-detection threshold

    n_frames = len(sorted((HERE / outdir).glob("fort.t[0-9][0-9][0-9][0-9]")))
    ok = True
    for fn in range(n_frames):
        frame = plotdata.getframe(fn)
        state = frame.states[0]
        x = state.patch.grid.c_centers[0]
        y = state.patch.grid.c_centers[1]
        j = int(np.argmin(np.abs(y[0, :])))
        h = state.q[0][:, j]
        xr = x[:, j]
        wet = h > 1e-3
        front = float(xr[wet].max()) if wet.any() else reservoir_end
        bound = reservoir_end + theory_speed * frame.t + tolerance_m
        within = front <= bound
        if fn % 10 == 0 or not within:
            print(f"  t={frame.t:.3f}s front={front:.3f}m bound={bound:.3f}m within_bound={within}")
        if not within:
            ok = False
    return ok


def main() -> None:
    print("=== Test A: contact-wave preservation (m-jump in still water) ===")
    ok_a = check_contact()
    print(f"  -> {'PASS' if ok_a else 'FAIL'}\n")

    print("=== Test B: dry dam-break front-speed bound ===")
    ok_b = check_damfront()
    print(f"  -> {'PASS' if ok_b else 'FAIL'}\n")

    if ok_a and ok_b:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
