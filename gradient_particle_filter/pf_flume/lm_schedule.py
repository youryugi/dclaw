"""Iterations (sequential batched Jacobian calls) the batched Gauss-Newton needs
under different Levenberg-Marquardt damping schedules, same 16 diverse starts."""
import sys, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import inference as I
from run_twin import problem

obs = sys.argv[1]
prob, _ = problem(obs, 0)
rng = np.random.default_rng(0)
Z = rng.logistic(size=(512, 3)); ll = prob.loglik(Z)
TH = np.asarray(I.to_theta(jnp.asarray(Z))); span = I.HI - I.LO
starts = []
for i in np.argsort(-ll):
    if all(np.linalg.norm((TH[i] - TH[j]) / span) > 0.1 for j in starts):
        starts.append(i)
    if len(starts) == 16:
        break
for lam0, down, up in [(1e-2, 3, 5), (1e-3, 10, 5), (1e-4, 10, 10), (0.0, 10, 10)]:
    t0 = time.time()
    Zm, C, f, conv, it, dec = I.gauss_newton_batch(prob, Z[starts], iters=60, lam0=max(lam0, 1e-12), lam_down=down, lam_up=up)
    modes = sorted({round(float(x), 2) for x in f[conv]})
    print(f"[{obs}] lam0={lam0:g} down={down} up={up}: {it} iterations, {conv.sum()}/16 converged, "
          f"-log post of converged {modes}, {time.time() - t0:.0f}s", flush=True)
