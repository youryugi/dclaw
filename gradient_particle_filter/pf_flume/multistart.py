"""Gauss-Newton from many screened, mutually distant starts for one design:
  python multistart.py <obs> [n_screen] [n_starts]
Prints each start's converged mode, -log posterior and Laplace log evidence."""
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import inference as I
from run_twin import problem

obs = sys.argv[1]
n_screen = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
n_starts = int(sys.argv[3]) if len(sys.argv) > 3 else 16
prob, truth = problem(obs, 0)
rng = np.random.default_rng(7)
Z = rng.logistic(size=(n_screen, 3))
ll = prob.loglik(Z)
order = np.argsort(-ll)
span = I.HI - I.LO
starts = []
for i in order:                     # best draws, at least 10% of the prior range apart
    th = np.asarray(I.to_theta(jnp.asarray(Z[i])))
    if all(np.linalg.norm((th - np.asarray(I.to_theta(jnp.asarray(s)))) / span) > 0.1 for s in starts):
        starts.append(Z[i])
    if len(starts) == n_starts:
        break
modes = []
for z0 in starts:
    t0 = time.time()
    z, C, f, it = I.gauss_newton(prob, z0, gn_iters=60)
    th0, th = (np.asarray(I.to_theta(jnp.asarray(v))) for v in (z0, z))
    ev = -f + 0.5 * np.linalg.slogdet(C)[1]
    modes.append((th, f, ev))
    print(f"[{obs}] start {np.round(th0, 3)} -> mode {np.round(th, 3)}  -log post {f:8.3f}  log evidence {ev:8.3f}  "
          f"({it} it, {time.time() - t0:.0f}s)", flush=True)
np.save(f"results/multistart_{obs.replace('@', '_')}.npy", np.array(modes, dtype=object))
