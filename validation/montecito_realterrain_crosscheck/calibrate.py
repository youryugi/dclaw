"""Gradient-based calibration of (phi_deg, log10 kref) against the observed
Montecito Creek inundation (Prescott et al. 2023), on the cross-check setup
(same terrain and initial ponds as the Fortran runs).

Residuals: per-cell soft inundation (sigmoid of peak depth) minus the
observed 0/1 map, plus three summary statistics (downstream and
cross-stream centroid, inundated area) -- cells the simulated flow never
reaches have exactly zero gradient, so without the summary terms nothing
pulls the runout toward places it has not reached yet. Levenberg-Marquardt
with an exact forward-mode Jacobian and a trust-region step cap.
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from compare import MASK, OBS
from run_jax import DX, run

THRESH, WIDTH, W_MOM = 0.1, 0.03, 30.0
IDX = jnp.asarray(np.where(MASK.reshape(-1))[0])
OBSF = jnp.asarray(OBS.reshape(-1).astype(float))[IDX]
nx, ny = OBS.shape
XX = jnp.asarray(np.broadcast_to(((np.arange(nx) + 0.5) * DX)[:, None], (nx, ny)).reshape(-1))[IDX]
YY = jnp.asarray(np.broadcast_to(((np.arange(ny) + 0.5) * DX)[None, :], (nx, ny)).reshape(-1))[IDX]
BASE = float(jax.nn.sigmoid(-THRESH / WIDTH))


def moments(s):
    a = jnp.sum(s)
    return a * DX * DX, jnp.sum(s * XX) / a, jnp.sum(s * YY) / a


A_OBS, X_OBS, Y_OBS = moments(OBSF)


def soft_inundation(theta):
    fr, _, _ = run(theta)
    peak = jnp.max(fr[:, 0], axis=0).reshape(-1)[IDX]
    return jnp.clip((jax.nn.sigmoid((peak - THRESH) / WIDTH) - BASE) / (1.0 - BASE), 0.0, 1.0)


def residual(theta):
    s = soft_inundation(theta)
    a, xm, ym = moments(s)
    mom = W_MOM * jnp.array([(ym - Y_OBS) / 1000.0, (xm - X_OBS) / 1000.0, (a - A_OBS) / A_OBS])
    return jnp.concatenate([s - OBSF, mom])


def csi(s):
    sim = np.array(s) > 0.5
    obs = np.array(OBSF) > 0.5
    tp, fp, fn = (sim & obs).sum(), (sim & ~obs).sum(), (~sim & obs).sum()
    return tp / (tp + fp + fn)


def main(theta0):
    res, jac, soft = jax.jit(residual), jax.jit(jax.jacfwd(residual)), jax.jit(soft_inundation)
    LO, HI, MAX_STEP = jnp.array([15.0, -14.0]), jnp.array([45.0, -8.0]), jnp.array([3.0, 0.5])
    theta = jnp.asarray(theta0)
    r = res(theta); cost = float(jnp.sum(r**2)); lam = 1e-2
    print(f"start phi={float(theta[0]):.2f} log10k={float(theta[1]):.3f} cost={cost:.2f} "
          f"CSI={csi(soft(theta)):.3f}", flush=True)
    for it in range(1, 16):
        t0 = time.time()
        J = jac(theta); A, g = J.T @ J, J.T @ r
        while True:
            s = -jnp.linalg.solve(A + lam * jnp.diag(jnp.diag(A) + 1e-12), g)
            s = s / jnp.maximum(1.0, jnp.max(jnp.abs(s) / MAX_STEP))
            trial = jnp.clip(theta + s, LO, HI)
            r_t = res(trial); c_t = float(jnp.sum(r_t**2))
            if np.isfinite(c_t) and c_t < cost:
                theta, r, cost, lam = trial, r_t, c_t, lam * 0.3
                break
            lam *= 10
            if lam > 1e8:
                print("no improving step; stopping"); return theta
        print(f"  it {it:2d}: phi={float(theta[0]):6.2f} log10k={float(theta[1]):7.3f} cost={cost:.2f} "
              f"CSI={csi(soft(theta)):.3f} moments(y,x,area)={np.round(np.array(r[-3:]), 2)} [{time.time()-t0:.0f}s]",
              flush=True)
    return theta


if __name__ == "__main__":
    start = [float(v) for v in sys.argv[1:3]] if len(sys.argv) > 2 else [37.63, float(np.log10(3.348e-12))]
    theta = main(start)
    np.save("calibrated_theta.npy", np.array(theta))
