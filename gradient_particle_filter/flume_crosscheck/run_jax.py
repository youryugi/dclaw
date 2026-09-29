"""dclaw_jax on the 2010 SGM gate-release case, on the same inputs as the
Fortran run in fortran_run/ (fortran_setup.py).

- Grid: uniform 0.0625 m in x, the Fortran run's finest AMR level (0.25 m
  base, 3 levels, ratio 2), which D-Claw's flowgrades put everywhere the
  flow is wet. Everything is uniform across the 2 m flume, so the width
  is two 1 m cells between walls (y fluxes are exactly zero either way).
- Bed, initial surface and solid fraction are read from the Fortran
  run's own .tt3 files. D-Claw sets each cell to the exact cell average
  of the bilinear interpolant; every 0.0625 m cell lies inside one
  0.125 m data cell, where that average equals the bilinear value at the
  cell centre, which is what is used here.
- Gate: solver.gate_bed with the same ridge and angle ramp (0 -> 90 deg
  over 0.85 s). The Fortran dtopo samples 1 - cos(angle) every 0.01 s and
  interpolates linearly in time; the two ridges differ by < 1e-4 m.
- Initial pore pressure: init_ptype=3, init_pratio=1 (hydrostatic,
  rho_f*g*h). Materials: setrun.py.
- Boundaries: walls at both x ends (Fortran: wall upstream, extrapolation
  downstream at x=130 m, which the flow does not reach; checked below).

Writes jax_run.npz: frames every 0.25 s (h, hm, pb along x) and gauge
time series at x = 2, 32, 66, 90 m.
"""

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from clawpack.geoclaw import topotools

import flume_solver as F
import dclaw_jax.model as _model
if "JAX_UTAPER" in os.environ:                  # sensitivity runs only; default keeps the package value
    _model.DILATANCY_U_TAPER = float(os.environ["JAX_UTAPER"])
import dclaw_jax.solver as _solver
if "JAX_HFLOOR" in os.environ:                  # model.DILATANCY_H_FLOOR (m)
    _model.DILATANCY_H_FLOOR = float(os.environ["JAX_HFLOOR"])
if "JAX_RECON" in os.environ:                   # "audusse" (package default) or "chen_noelle"
    _solver.RECONSTRUCTION = os.environ["JAX_RECON"]
from dclaw_jax.model import MaterialParams, qfix
from dclaw_jax.solver import Grid, gate_bed, make_state

RUN = HERE / "fortran_run"
X_LO, X_HI = -5.0, 130.0
DX = float(os.environ.get("JAX_DX", 0.0625))   # grid-refinement runs only; default = Fortran finest level
DRY_TOL = float(os.environ.get("JAX_DRY_TOL", 1.0e-3))   # flume_solver dry-film threshold
NX = int(round((X_HI - X_LO) / DX))
GRID = Grid(nx=NX, ny=2, dx=DX, dy=1.0)
XC = X_LO + (np.arange(NX) + 0.5) * DX
GAUGES_X = np.array([2.0, 32.0, 66.0, 90.0])
T_FINAL, FRAME_DT = 35.0, 0.25

P = MaterialParams(rho_s=2700.0, rho_f=1100.0, m_crit=0.64, mref=0.60, kref=5e-9, phi_deg=40.7, delta=0.001,
                   mu=0.005, alpha_c=0.024, sigma_0=1000.0, c1=1.0, gz=9.81, manning_n=0.0)
GATE_HALF_WIDTH, GATE_H = 0.375, 1.9


def centre_values(name):
    topo = topotools.Topography(str(RUN / name), topo_type=3)
    Z = np.asarray(topo.Z)
    assert np.allclose(Z, Z[:1]), f"{name} is not uniform across the flume"
    return np.interp(XC, np.asarray(topo.x), Z[0])


def inputs():
    basal = centre_values("basal_topo.tt3")                  # includes the closed-gate ridge
    ridge = GATE_H * np.clip((GATE_HALF_WIDTH - np.abs(XC)) / (GATE_HALF_WIDTH - 0.125), 0.0, 1.0)
    surface = centre_values("surface_topo.tt3")
    m = centre_values("solid_fraction.tt3")
    h = np.maximum(surface - basal, 0.0)
    h = np.where(h <= 1.0e-3, 0.0, h)
    col = lambda a: jnp.asarray(np.repeat(a[:, None], GRID.ny, axis=1))
    q0 = make_state(col(h), col(0 * h), col(0 * h), col(h * m), col(P.rho_f * P.gz * h))
    b = gate_bed(col(basal - ridge), col(ridge), [0.0, 0.85], [0.0, 90.0])
    return qfix(q0, P), b, basal


def gauge_cells():
    right = np.searchsorted(XC, GAUGES_X)                   # first centre to the right of the edge
    assert np.allclose(XC[right] - GAUGES_X, DX / 2)
    return jnp.asarray(np.stack([right - 1, right], axis=1))


def main():
    q, b, basal = inputs()
    n_chunk = int(600 * 0.0625 / DX)
    adv = jax.jit(lambda q, t0, t1: F.advance_record(q, t0, t1, b, P, GRID, n_chunk, gauge_cells(), dry_tol=DRY_TOL))
    frames = {"t": [0.0], "h": [np.asarray(q[0, :, 0])], "hm": [np.asarray(q[3, :, 0])], "pb": [np.asarray(q[4, :, 0])]}
    g_t, g_q = [np.zeros(1)], [np.asarray(q[:, gauge_cells()[:, 1], 0])[None]]
    t, n_steps, wall = 0.0, 0, time.time()
    for k in range(1, int(round(T_FINAL / FRAME_DT)) + 1):
        t_end = k * FRAME_DT
        q, t_new, ts, gq = adv(q, t, t_end)
        ts, gq = np.asarray(ts), np.asarray(gq)
        used = np.concatenate([[True], np.diff(ts) > 0]) & (ts > t)
        if not abs(float(t_new) - t_end) < 1e-9:
            raise SystemExit(f"chunk ending {t_end} s only reached {float(t_new)} s: raise n_chunk")
        g_t.append(ts[used]); g_q.append(gq[used]); n_steps += int(used.sum()); t = float(t_new)
        for key, i in (("h", 0), ("hm", 3), ("pb", 4)):
            frames[key].append(np.asarray(q[i, :, 0]))
        frames["t"].append(t)
        if k % 20 == 0:
            front = XC[np.nonzero(frames["h"][-1] > 0.01)[0].max()] if (frames["h"][-1] > 0.01).any() else np.nan
            print(f"t={t:5.2f} s  steps {n_steps}  front {front:6.1f} m  wall {time.time() - wall:5.1f} s", flush=True)
    assert frames["h"][-1][-10:].max() < 1e-6, "flow reached the downstream boundary"
    np.savez(HERE / os.environ.get("JAX_OUT", "jax_run.npz"), x=XC, basal=basal, gauges_x=GAUGES_X, gauge_t=np.concatenate(g_t),
             gauge_q=np.concatenate(g_q), **{f"frame_{k}": np.array(v) for k, v in frames.items()})
    print(f"done: {n_steps} steps, {time.time() - wall:.1f} s wall")


if __name__ == "__main__":
    main()
