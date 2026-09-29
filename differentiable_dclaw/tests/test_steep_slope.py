"""Gravity driving force on a uniform layer at rest on a steep (31 deg)
slope, as a fraction of the exact g*h*tan(theta), for both hydrostatic
reconstructions in solver._hr_fluxes (db = 3.8 cm is the bed step between
two cells, the USGS flume at 6.25 cm cells):

  h >= db:  1 - db/(2 h)   both
  h <  db:  h/(2 db)       Audusse (default) -- 27% for 2 cm
            1 - h/(2 db)   Chen-Noelle       -- 73% for 2 cm

Pins down each scheme's documented behaviour; see _hr_fluxes for why the
default is still Audusse.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from dclaw_jax.model import MaterialParams
import dclaw_jax.solver as solver
from dclaw_jax.solver import Grid, make_state, rhs


def main():
    p = MaterialParams(phi_deg=0.0, c1=0.0)
    n, dx = 64, 0.0625
    grid = Grid(nx=n, ny=2, dx=dx, dy=1.0)
    tan = jnp.tan(jnp.deg2rad(31.0))
    b = jnp.broadcast_to((-tan * jnp.arange(n) * dx)[:, None], (n, 2))
    db = float(tan * dx)
    ok = True
    for scheme in ("audusse", "chen_noelle"):
        solver.RECONSTRUCTION = scheme
        for h0 in (0.002, 0.005, 0.02, 0.035, 0.05, 0.2, 1.0):
            h = jnp.full((n, 2), h0)
            q = make_state(h, 0 * h, 0 * h, 0.6 * h, 1000.0 * 9.81 * h)
            ratio = float(rhs(q, b, p, grid, "open", "wall")[1][n // 2, 0] / (9.81 * h0 * tan))
            if h0 >= db:
                expect = 1.0 - db / (2 * h0)
            else:
                expect = h0 / (2 * db) if scheme == "audusse" else 1.0 - h0 / (2 * db)
            good = abs(ratio - expect) < 1e-10
            ok &= good
            print(f"{scheme:12s} h = {h0:5.3f} m (h/db = {h0 / db:5.2f}): driving force / g h tan(theta) = "
                  f"{ratio:.4f}, expected {expect:.4f}  {'ok' if good else 'MISMATCH'}")
    solver.RECONSTRUCTION = "audusse"
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
