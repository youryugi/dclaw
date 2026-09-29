"""Moving headgate (`solver.gate_bed`), the reference's dtopo
representation of the USGS flume gate: a ridge on the bed lowered to
cos(angle) of its height as the gate swings open.

1. Closed, and partly lowered but still above the pond: a still pond
   behind it stays exactly still and nothing leaks past it (a moving bed
   under dry cells must not disturb anything).
2. Opening with the flume's own material (George & Iverson 2014 table 2/3
   parameters, as in validation/usgs_flume_2010_paper2014): material
   gets past the gate, a slower opening lets less of it out early on, and
   mixture mass is conserved up to the dry-film removal. (Neither h nor
   hm is conserved on its own: eq 2.1(a,d) exchange pore fluid through D;
   the mixture mass rho_s*hm + rho_f*(h - hm) is, by those sources.
   Measured per operator over this 4 s run: transport +4e-15, dilatancy
   +7e-6, qfix -1e-8, and -1.4e-2 from solver._clip_dry, which removes
   the thin films this steep flume leaves behind -- a known deviation from
   the reference, which only zeroes h <= 1 mm; see the package README.)
3. Autodiff through the gated release, with respect to the opening
   duration and to phi, matches finite differences (float64). The loss
   is visibly rougher in the opening duration than in phi (residual from a
   quadratic over +-2e-4: 4e-6 vs 2e-8 m3): each step the lowering ridge
   crosses the flow surface at a slightly different point, a kink in the
   hydrostatic reconstruction's max(). The derivative is exact, but only
   a small finite-difference step (1e-6) reproduces it; at 1e-3 the
   finite difference is 1-2% off.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dclaw_jax.model import MaterialParams
from dclaw_jax.solver import Grid, gate_bed, make_state, rollout, rollout_peak

DX = 0.125
X = jnp.arange(-6.0 + 0.5 * DX, 24.0, DX)             # gate at x=0, flow towards +x
GRID = Grid(nx=X.size, ny=2, dx=DX, dy=1.0)
SLOPE = jnp.tan(jnp.deg2rad(31.0))
# 31 deg flume, then a 3 deg run-out pad from x=12 m (heights along x, bed_normal=0)
B_BASE = jnp.where(X < 12.0, -SLOPE * X, -SLOPE * 12.0 - jnp.tan(jnp.deg2rad(3.0)) * (X - 12.0))
B_BASE = jnp.broadcast_to(B_BASE[:, None], (GRID.nx, GRID.ny))
GATE_H = 1.9


def _grid2d(a):
    return jnp.broadcast_to(a[:, None], (GRID.nx, GRID.ny))


def test_closed_gate_holds_still_pond():
    ridge = _grid2d(jnp.where(jnp.abs(X) < 0.125, GATE_H, 0.0))   # a vertical wall, two cells wide
    eta = 1.2                                                        # pond surface, below the wall top
    h = jnp.where(X[:, None] < -0.125, jnp.maximum(eta - B_BASE, 0.0), 0.0)
    p = MaterialParams(phi_deg=0.0, c1=0.0, m_crit=0.64)            # fluid-like: no friction to hide errors
    m = jnp.where(h > 0, 0.5, 0.0)
    q0 = make_state(h, 0 * h, 0 * h, h * m, p.rho_f * p.gz * h)
    # gate turns 0 -> 25 deg over 1 s: the wall drops to cos(25)*1.9 = 1.72 m, still above the pond
    b = gate_bed(B_BASE, ridge, [0.0, 1.0], [0.0, 25.0])
    qf, t, _, _ = rollout(q0, b, p, GRID, t_final=3.0, n_steps=2000)
    wet = qf[0] > 0.01
    max_u = float(jnp.max(jnp.abs(jnp.where(wet, qf[1] / jnp.maximum(qf[0], 1e-9), 0.0))))
    eta_dev = float(jnp.max(jnp.abs(jnp.where(wet, qf[0] + B_BASE - eta, 0.0))))
    leaked = float(jnp.max(qf[0][X > 0.125]))
    print(f"closed gate, t={float(t):.1f}s: max|u| = {max_u:.1e} m/s, max|eta - eta0| = {eta_dev:.1e} m, "
          f"max h past the gate = {leaked:.1e} m")
    return max_u < 1e-8 and eta_dev < 1e-8 and leaked == 0.0


# --- flume release -------------------------------------------------------
# Fortran reference ridge shape (setinput.py gate_shape, half width 0.375 m)
RIDGE = _grid2d(GATE_H * jnp.clip((0.375 - jnp.abs(X)) / (0.375 - 0.125), 0.0, 1.0))
FLUME = MaterialParams(rho_s=2700.0, rho_f=1100.0, m_crit=0.64, mref=0.60, kref=5e-9, phi_deg=40.7,
                       delta=0.001, mu=0.005, alpha_c=0.024, sigma_0=1000.0)


def _wedge_state():
    # 1.9 m thick for 2 m behind the gate, tapering to 0 over the next 1.3 m (the reference's wedge, smaller)
    thick = jnp.where(X <= -0.375, jnp.clip((X + 0.375 + 3.3) / 1.3, 0.0, 1.0) * GATE_H, 0.0)
    h = _grid2d(thick)
    m0 = 0.62
    return make_state(h, 0 * h, 0 * h, m0 * h, FLUME.rho_f * FLUME.gz * h)   # hydrostatic pb


def _release(t_open, t_final, p=FLUME, n_steps=4000):
    b = gate_bed(B_BASE, RIDGE, jnp.stack([jnp.asarray(0.0), t_open]), [0.0, 90.0])
    qf, t, _ = rollout_peak(_wedge_state(), b, p, GRID, t_final=t_final, n_steps=n_steps,
                            bc_x="wall", bc_y="wall", checkpoint_block=50)
    return qf, t


def _mixture_mass(q, p=FLUME):
    return jnp.sum(p.rho_s * q[3] + p.rho_f * (q[0] - q[3])) * GRID.dx * GRID.dy


def _volume_past(q, x0=3.0):
    return jnp.sum(jnp.where(X[:, None] > x0, q[0], 0.0)) * GRID.dx * GRID.dy


def test_release_conserves_and_moves():
    q0 = _wedge_state()
    qf, t = _release(jnp.asarray(0.85), 4.0)
    cell = GRID.dx * GRID.dy
    dM = float(abs(_mixture_mass(qf) - _mixture_mass(q0)) / _mixture_mass(q0))
    dV = float((jnp.sum(qf[0]) - jnp.sum(q0[0])) / jnp.sum(q0[0]))
    front = float(jnp.max(jnp.where(qf[0][:, 0] > 0.01, X, -jnp.inf)))
    past = float(_volume_past(qf))
    slow = float(_volume_past(_release(jnp.asarray(3.0), 4.0)[0]))
    fast = float(_volume_past(_release(jnp.asarray(0.05), 4.0)[0]))
    print(f"release, t={float(t):.1f}s: mixture mass change {dM:.1e} (volume h {dV:+.1e}), "
          f"front at x={front:.1f} m, volume past x=3 m: {past:.3f} m3 "
          f"(opening in 0.05 s: {fast:.3f}, in 3 s: {slow:.3f}; released {float(jnp.sum(q0[0])) * cell:.3f})")
    return dM < 2e-2 and front > 5.0 and past > 0.0 and fast > past > slow


def test_gradient_through_release():
    ok = True
    for name, loss, x0 in [
        ("opening time", lambda T: _volume_past(_release(T, 2.5)[0]), 0.85),
        ("phi_deg", lambda ph: _volume_past(_release(jnp.asarray(0.85), 2.5, p=FLUME._replace(phi_deg=ph))[0]), 40.7),
    ]:
        loss = jax.jit(loss)
        g_rev = float(jax.grad(loss)(x0))
        g_fwd = float(jax.jacfwd(loss)(x0))
        fds = {eps: float((loss(x0 + eps) - loss(x0 - eps)) / (2 * eps)) for eps in (1e-3, 1e-6)}
        print(f"d(volume past x=3 m at t=2.5 s)/d({name}): reverse {g_rev:.6e}, forward {g_fwd:.6e}, "
              + ", ".join(f"FD eps={e:g} {v:.6e}" for e, v in fds.items()))
        ok &= abs(g_rev - g_fwd) <= 1e-8 * abs(g_rev) and abs(g_rev - fds[1e-6]) < 1e-4 * abs(g_rev) and g_rev < 0.0
    return ok


def main():
    results = {f.__name__: f() for f in (test_closed_gate_holds_still_pond, test_release_conserves_and_moves,
                                         test_gradient_through_release)}
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
