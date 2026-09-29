"""Well-balancing: still water over strongly uneven terrain (a bump,
a ridge and a pit, with a dry island) must stay exactly still. A
non-well-balanced scheme (the earlier Rusanov + centred bed-slope source)
fails this and, on real terrain, spreads flows out of their channels.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from dclaw_jax.model import MaterialParams
from dclaw_jax.solver import Grid, make_state, rollout


def main():
    n, dx = 60, 5.0
    grid = Grid(nx=n, ny=n, dx=dx, dy=dx)
    x = (jnp.arange(n) + 0.5) * dx
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    b = (3.0 * jnp.exp(-((X - 100) ** 2 + (Y - 150) ** 2) / 400.0)       # bump
         + 0.02 * X                                                     # tilt
         - 1.5 * jnp.exp(-((X - 200) ** 2 + (Y - 80) ** 2) / 300.0)     # pit
         + 6.0 * jnp.exp(-((X - 220) ** 2 + (Y - 220) ** 2) / 200.0))   # island (pokes above water)
    eta = 5.0
    h = jnp.maximum(eta - b, 0.0)
    p = MaterialParams(phi_deg=0.0, c1=0.0, m_crit=0.64)                # pure fluid-like, no friction to hide errors
    m = jnp.where(h > 0, 0.5, 0.0)
    q0 = make_state(h, 0 * h, 0 * h, h * m, 1000.0 * 9.81 * h)
    qf, t, _, _ = rollout(q0, b, p, grid, t_final=60.0, n_steps=2000, bc_x="wall", bc_y="wall")
    wet = qf[0] > 0.01
    max_u = float(jnp.max(jnp.abs(jnp.where(wet, qf[1] / jnp.maximum(qf[0], 1e-9), 0.0))))
    eta_dev = float(jnp.max(jnp.abs(jnp.where(wet, qf[0] + b - eta, 0.0))))
    print(f"t={float(t):.1f}s  max|u| on wet cells = {max_u:.2e} m/s, max |eta - eta0| = {eta_dev:.2e} m, "
          f"island stays dry: {bool(jnp.all(qf[0][b > eta + 0.01] < 1e-6))}")
    ok = max_u < 1e-8 and eta_dev < 1e-8
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
