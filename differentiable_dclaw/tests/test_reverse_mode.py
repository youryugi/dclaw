"""Reverse mode with respect to a whole field (the bed elevation of every
cell), through `rollout_peak` with and without gradient checkpointing.

Checks that (1) checkpointing does not change the gradient, and (2) the
gradient's projection on a random smooth direction matches finite
differences -- the standard check when there are too many inputs to
check one by one. Loose material (cv < m_crit) on a slope with a real
wet/dry front, so the liquefaction feedback that once broke this (see
model.pb_relaxation_step) is exercised.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy import ndimage

from dclaw_jax.model import MaterialParams
from dclaw_jax.solver import Grid, make_state, rollout_peak

N, DX = 60, 5.0
GRID = Grid(nx=N, ny=N, dx=DX, dy=DX)
xc = (jnp.arange(N) + 0.5) * DX
X, Y = jnp.meshgrid(xc, xc, indexing="ij")
BED = -0.15 * X + 1.5 * jnp.cos(Y / 30.0)
H0 = jnp.where((X < 60.0) & (jnp.abs(Y - 150.0) < 40.0), 3.0, 0.0)
P = MaterialParams(phi_deg=30.0, kref=3.0e-12, m_crit=0.64, mu=0.005, manning_n=0.06)


def target(bed, block):
    m = jnp.where(H0 > 0, 0.52, 0.0)
    q0 = make_state(H0, 0 * H0, 0 * H0, H0 * m, 1000.0 * 9.81 * H0)
    _, _, hmax = rollout_peak(q0, bed, P, GRID, t_final=40.0, cfl=0.35, bc_x="open", bc_y="open",
                              n_steps=400, checkpoint_block=block)
    return jnp.sum(hmax[40:, :] ** 2)   # depth reaching the lower part of the slope


def main():
    g_plain = jax.jit(jax.grad(target), static_argnums=1)(BED, None)
    g_ckpt = jax.jit(jax.grad(target), static_argnums=1)(BED, 20)
    same = float(jnp.max(jnp.abs(g_plain - g_ckpt)) / jnp.max(jnp.abs(g_plain)))
    print(f"checkpointed vs plain reverse mode, max relative difference: {same:.2e}")

    rng = np.random.default_rng(0)
    v = jnp.asarray(ndimage.gaussian_filter(rng.standard_normal((N, N)), 4.0))
    v = v / jnp.max(jnp.abs(v))
    ad = float(jnp.sum(g_ckpt * v))
    f = jax.jit(target, static_argnums=1)
    fd = (float(f(BED + 1e-4 * v, None)) - float(f(BED - 1e-4 * v, None))) / 2e-4
    rel = abs(ad - fd) / abs(fd)
    print(f"directional derivative along a smooth random bed perturbation: AD {ad:.6e}  FD {fd:.6e}  rel. err {rel:.2e}")

    ok = same < 1e-10 and rel < 1e-3
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
