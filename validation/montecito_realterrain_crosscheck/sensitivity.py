"""Reverse-mode sensitivity of damage-relevant flow depth to the whole terrain.

Target L = sum of simulated peak flow depth at the buildings recorded as
damaged (CalFire damage state != "Unimpacted", Barnhart et al. 2021
buildings.csv) inside this domain, at the calibrated parameters.
jax.grad gives dL/d(bed elevation) for every one of the 118 x 200 cells,
plus dL/dphi and dL/dlog10 kref, in one backward pass.
"""
import sys, time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import ndimage

import run_jax as R
sys.path.insert(0, str(R.HERE.parents[1] / "differentiable_dclaw"))
from dclaw_jax.model import MaterialParams
from dclaw_jax.solver import rollout_peak

inputs = np.load(R.HERE / "model/inputs_20m.npz")
X0, Y0 = float(inputs["x_lower"]), float(inputs["y_lower"])
B = pd.read_csv("/home/yang/.claude/uploads/61b909df-9a48-4264-a419-b4d8b3ab4b3b/88902f97-buildings.csv")
i = ((B.x - X0) // R.DX).astype(int); j = ((B.y - Y0) // R.DX).astype(int)
inside = (i >= 0) & (i < R.NX) & (j >= 0) & (j < R.NY)
damaged = inside & (B.dam_state_CalFire != "Unimpacted")
BI = jnp.asarray(i[damaged].to_numpy()); BJ = jnp.asarray(j[damaged].to_numpy())
print(f"buildings in domain: {int(inside.sum())}, damaged: {int(damaged.sum())}")

THETA = np.load(R.HERE / "calibrated_theta.npy")
N_STEPS = 3000


def target(bed, theta, block):
    p = MaterialParams(phi_deg=theta[0], kref=10.0 ** theta[1], m_crit=0.64, mu=0.005, manning_n=0.06)
    _, _, hmax = rollout_peak(R.q0, bed, p, R.GRID, t_final=1800.0, cfl=0.35, bc_x="open", bc_y="open",
                              n_steps=N_STEPS, checkpoint_block=block)
    return jnp.sum(hmax[BI, BJ])


def peak_mem():
    return jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1e9


if __name__ == "__main__":
    bed, theta = R.bed, jnp.asarray(THETA)
    L = float(jax.jit(target, static_argnums=2)(bed, theta, None))
    print(f"L (sum of peak depth at damaged buildings) = {L:.2f} m")

    block = None if (len(sys.argv) > 1 and sys.argv[1] == 'none') else (int(sys.argv[1]) if len(sys.argv) > 1 else 50)
    g = jax.jit(jax.grad(target, argnums=(0, 1)), static_argnums=2)
    t0 = time.time(); gb, gth = g(bed, theta, block); gb.block_until_ready(); t1 = time.time()
    gb, gth = g(bed, theta, block); gb.block_until_ready(); t2 = time.time()
    print(f"reverse-mode grad, checkpoint_block={block}: first {t1-t0:.1f}s, second {t2-t1:.2f}s, "
          f"peak device memory {peak_mem():.2f} GB")
    if "--timing-only" in sys.argv:
        sys.exit()
    print(f"dL/dphi = {float(gth[0]):.4f}, dL/dlog10 kref = {float(gth[1]):.4f}, "
          f"|dL/dbed| max {float(jnp.abs(gb).max()):.4f} per m")

    # directional-derivative check along a random smooth terrain perturbation (1 m amplitude)
    rng = np.random.default_rng(0)
    v = ndimage.gaussian_filter(rng.standard_normal(bed.shape), 5.0)
    v = jnp.asarray(v / np.abs(v).max())
    f = jax.jit(target, static_argnums=2)
    ad = float(jnp.sum(gb * v))
    fwd = float(jax.jit(lambda b: jax.jvp(lambda bb: target(bb, theta, None), (b,), (v,))[1])(bed))
    for eps in (1e-2, 1e-4, 1e-6):
        fd = (float(f(bed + eps * v, theta, None)) - float(f(bed - eps * v, theta, None))) / (2 * eps)
        print(f"  directional derivative: reverse {ad: .6e}  forward {fwd: .6e}  FD(eps={eps:.0e}) {fd: .6e}")
    np.save(R.HERE / "sens_dL_dbed.npy", np.array(gb))
