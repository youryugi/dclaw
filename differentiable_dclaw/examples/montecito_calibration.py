"""Gradient-based calibration of (phi_deg, kref, cv) against the real 2018
Montecito Creek peak-depth profile (Kean et al. 2019 field survey,
dataset/differentiable/montecito_creek_depth_profile.csv, 20 bins).

Same idealized setup as montecito_real_data.py (231,000 m3 released
instantaneously as a 4 m pile on a uniform 4 deg slope, 203.5 m wide,
Manning n=0.06), bounds from the Barnhart et al. 2021 D-Claw ensemble
ranges for Montecito.

Two stages:
  1. a coarse global grid (100 forward runs, ~2 min on CPU) to pick a
     basin -- the landscape has a regime switch in kref (see README),
     which no local method can see across;
  2. Levenberg-Marquardt / Gauss-Newton from the best grid points, with a
     trust-region step cap and a smoothed autodiff Jacobian (the exact
     forward-mode Jacobian averaged over antithetic pairs of nearby
     parameter points) -- long slowly-creeping deposits leave ~1e-4 m
     structure on a 1e-5 scale in log10 kref that the exact local
     Jacobian would otherwise follow.

Result when last run (~16 min, CPU): best RMS misfit 0.81 m, with kref
driven to its upper bound and phi_deg then irrelevant -- see README
"Real Montecito data". The optimizer is not the limitation there: an
instantaneous pile on a uniform slope can only produce depths that
*decrease* downstream, while the observed ones increase (0.2 m near the
apex to 1.2 m at 3.3 km). Matching that needs real fan topography and an
inflow hydrograph, not different parameters.
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

from dclaw_jax.model import MaterialParams
from dclaw_jax.solver import Grid, make_state, rollout

REPO = Path(__file__).resolve().parents[2]

WIDTH_M = 997_000.0 / 4900.0
PILE_DEPTH_M = 4.0
PILE_LENGTH_M = 231_000.0 / (WIDTH_M * PILE_DEPTH_M)
DX = 25.0
GRID = Grid(nx=216, ny=8, dx=DX, dy=DX)
XC = (np.arange(GRID.nx) + 0.5) * DX - 400.0
X2D = jnp.asarray(XC)[:, None] * jnp.ones((GRID.nx, GRID.ny))
BED = -X2D * jnp.sin(np.radians(4.0))
PILE = jnp.where((X2D >= -PILE_LENGTH_M) & (X2D < 0), PILE_DEPTH_M, 0.0)
T_FINAL, N_STEPS = 1800.0, 2500

profile = pd.read_csv(REPO / "dataset/differentiable/montecito_creek_depth_profile.csv")
OBS_X = profile.downstream_m.to_numpy()
OBS = jnp.asarray(profile.depth_mean.to_numpy())
OBS_IDX = jnp.asarray([int(np.argmin(np.abs(XC - x))) for x in OBS_X])

# theta = (phi_deg, log10 kref, cv); Barnhart et al. 2021 Montecito D-Claw ranges
LO = jnp.array([30.0, -14.0, 0.42])
HI = jnp.array([45.0, -8.0, 0.62])
MAX_STEP = jnp.array([3.0, 0.5, 0.03])
SMOOTH_SIGMA = jnp.array([0.1, 0.02, 0.002])


def peak_profile(theta):
    p = MaterialParams(phi_deg=theta[0], kref=10.0 ** theta[1], manning_n=0.06)
    m0 = jnp.where(PILE > 0, theta[2], 0.0)
    q0 = make_state(PILE, 0 * PILE, 0 * PILE, PILE * m0, jnp.where(PILE > 0, 1000.0 * 9.81 * PILE, 0.0))
    _, _, states, _ = rollout(q0, BED, p, GRID, t_final=T_FINAL, n_steps=N_STEPS, cfl=0.35,
                              bc_x="open", bc_y="wall")
    return jnp.mean(jnp.max(states[:, 0], axis=0), axis=1)[OBS_IDX]


forward = jax.jit(peak_profile)
forward_batch = jax.jit(jax.vmap(peak_profile))
jacobian = jax.jit(jax.jacfwd(peak_profile))


def cost_of(theta):
    r = forward(theta) - OBS
    return r, float(jnp.sum(r**2))


def refine(theta, n_iter=12, lam=1e-2, n_pairs=2, seed=0):
    key = jax.random.PRNGKey(seed)
    r, cost = cost_of(theta)
    for it in range(n_iter):
        key, sub = jax.random.split(key)
        e = jax.random.normal(sub, (n_pairs, 3))
        e = jnp.concatenate([e, -e])
        J = sum(jacobian(jnp.clip(theta + x * SMOOTH_SIGMA, LO, HI)) for x in e) / e.shape[0]
        A, g = J.T @ J, J.T @ r
        while True:
            step = -jnp.linalg.solve(A + lam * jnp.diag(jnp.diag(A) + 1e-12), g)
            step = step / jnp.maximum(1.0, jnp.max(jnp.abs(step) / MAX_STEP))
            trial = jnp.clip(theta + step, LO, HI)
            r_t, c_t = cost_of(trial)
            if np.isfinite(c_t) and c_t < cost:
                theta, r, cost, lam = trial, r_t, c_t, lam * 0.3
                break
            lam *= 10
            if lam > 1e8:
                return theta, cost
        print(f"    it {it + 1:2d}: phi={float(theta[0]):6.2f} log10k={float(theta[1]):7.3f} "
              f"cv={float(theta[2]):.4f} cost={cost:.4f}", flush=True)
    return theta, cost


def main():
    grid_pts = np.array(list(itertools.product(np.linspace(30, 45, 5), np.linspace(-13.5, -8.5, 5),
                                               np.linspace(0.44, 0.60, 4))))
    costs = []
    for i in range(0, len(grid_pts), 5):
        out = np.array(forward_batch(jnp.asarray(grid_pts[i:i + 5])))
        costs += list(((out - np.array(OBS)) ** 2).sum(axis=1))
    costs = np.array(costs)
    order = np.argsort(costs)
    print("best grid points (phi, log10k, cv, cost):")
    for j in order[:5]:
        print(f"  {grid_pts[j]}  {costs[j]:.3f}")

    best = None
    for j in order[:3]:
        print(f"refining from {grid_pts[j]}:")
        theta, cost = refine(jnp.asarray(grid_pts[j]))
        if best is None or cost < best[1]:
            best = (theta, cost)

    theta, cost = best
    print(f"\nbest: phi={float(theta[0]):.2f} deg, kref={10 ** float(theta[1]):.3e} m^2, "
          f"cv={float(theta[2]):.4f}, RMS misfit {np.sqrt(cost / len(OBS_X)):.3f} m")
    model = np.array(forward(theta))
    print(" x (m)  observed  model")
    for x, o, m in zip(OBS_X, np.array(OBS), model):
        print(f" {x:6.0f}  {o:7.3f}  {m:6.3f}")


if __name__ == "__main__":
    main()
