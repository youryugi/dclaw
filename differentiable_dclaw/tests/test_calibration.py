"""End-to-end check of the point of this package: gradient-based
calibration through a full rollout recovers known parameters.

Synthetic twin: a pile of loose material (real Montecito ensemble cv and
kref, see examples/montecito_real_data.py) released on a 10 deg slope with
a moving wet/dry front. The target is the historical peak-depth profile
at 12 points generated with phi_deg=25, kref=3.35e-12; Gauss-Newton with
an exact forward-mode Jacobian starts 10 deg and a factor of 10 away and
must recover both.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dclaw_jax.model import MaterialParams
from dclaw_jax.solver import Grid, make_state, rollout

NX, NY, DX = 80, 4, 2.0
GRID = Grid(nx=NX, ny=NY, dx=DX, dy=DX)
X2D = ((jnp.arange(NX) + 0.5) * DX)[:, None] * jnp.ones((NX, NY))
BED = -X2D * jnp.sin(jnp.radians(10.0))
PROFILE_IDX = jnp.asarray(np.linspace(5, 70, 12).astype(int))
# Trust region: max step per iteration in (phi_deg, log10 kref). Without
# it, an early undamped Gauss-Newton step jumped log10 kref from -11.2 to
# -8.7 -- lower cost, but into the high-permeability basin where the
# mixture dilates to near-pure water and never comes back.
MAX_STEP = jnp.array([3.0, 0.5])


def initial_state():
    h = jnp.where(X2D < 20.0, 3.0, 0.0)
    m = jnp.where(h > 0, 0.512, 0.0)
    return make_state(h, 0 * h, 0 * h, h * m, jnp.where(h > 0, 1000.0 * 9.81 * h, 0.0))


def peak_profile(theta):
    p = MaterialParams(phi_deg=theta[0], kref=10.0 ** theta[1])
    _, _, states, _ = rollout(initial_state(), BED, p, GRID, t_final=10.0, n_steps=320, cfl=0.35,
                              bc_x="open", bc_y="wall")
    return jnp.mean(jnp.max(states[:, 0], axis=0), axis=1)[PROFILE_IDX]


def gauss_newton(theta, target, n_iter=15, lam=1e-3):
    obs = jax.jit(peak_profile)
    jac = jax.jit(jax.jacfwd(peak_profile))
    r = obs(theta) - target
    cost = float(jnp.sum(r**2))
    for it in range(n_iter):
        J = jac(theta)
        A, g = J.T @ J, J.T @ r
        while True:
            step = -jnp.linalg.solve(A + lam * jnp.diag(jnp.diag(A)), g)
            trial = theta + step / jnp.maximum(1.0, jnp.max(jnp.abs(step) / MAX_STEP))
            r_t = obs(trial) - target
            c_t = float(jnp.sum(r_t**2))
            if c_t < cost:
                theta, r, cost, lam = trial, r_t, c_t, lam * 0.3
                break
            lam *= 10
            if lam > 1e12:
                return theta, cost
        print(f"  it {it + 1:2d}: phi={float(theta[0]):8.4f} log10k={float(theta[1]):8.4f} cost={cost:.3e}")
        if cost < 1e-20:
            break
    return theta, cost


def main():
    true = jnp.array([25.0, np.log10(3.35e-12)])
    start = jnp.array([35.0, np.log10(3.35e-12) + 1.0])
    target = jax.jit(peak_profile)(true)
    theta, cost = gauss_newton(start, target)
    err_phi = abs(float(theta[0] - true[0]))
    err_logk = abs(float(theta[1] - true[1]))
    print(f"recovered phi={float(theta[0]):.5f} (true 25), log10k={float(theta[1]):.5f} "
          f"(true {float(true[1]):.5f}); cost {cost:.2e}")
    ok = err_phi < 1e-3 and err_logk < 1e-3
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
