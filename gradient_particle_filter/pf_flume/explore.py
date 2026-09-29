"""Shape of the posterior before sampling: 1-D log-likelihood profiles over the
prior range (others at the truth) and Gauss-Newton from the prior centre and
the 8 corners of the prior box (shrunk 10% inwards).
  python explore.py full|h"""
import itertools
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import flume_model as M
import inference as I
from run_twin import problem

obs = sys.argv[1]
prob, truth = problem(obs, 0)
out = {}
for k, name in enumerate(M.THETA_NAMES):
    grid = np.linspace(I.LO[k], I.HI[k], 97)[1:-1]
    TH = np.repeat(M.THETA_TRUE[None], len(grid), 0); TH[:, k] = grid
    ll = prob.loglik(I.to_z(TH))
    out[name] = (grid, ll)
    i = np.argmax(ll); d2 = np.diff(ll, 2)
    # local maxima of the profile (a crude multimodality check)
    peaks = [j for j in range(1, len(ll) - 1) if ll[j] > ll[j - 1] and ll[j] > ll[j + 1]]
    print(f"[{obs}] profile {name:14s}: max at {grid[i]:.3f} (truth {M.THETA_TRUE[k]:.3f}), ll max {ll[i]:.1f}, "
          f"ll range {ll.min():.0f}..{ll.max():.1f}, local maxima at {np.round(grid[peaks], 3).tolist()}", flush=True)
starts = [0.5 * (I.LO + I.HI)] + [np.array(c) for c in itertools.product(*[(l + 0.1 * (h - l), h - 0.1 * (h - l)) for l, h in zip(I.LO, I.HI)])]
res = []
for s in starts:
    t0 = time.time()
    z, C, f, it = I.gauss_newton(prob, I.to_z(s), gn_iters=40)
    th = np.asarray(I.to_theta(jnp.asarray(z)))
    sd = np.sqrt(np.diag(C)) * (I.HI - I.LO) * 0.25          # rough theta-space sd (sigmoid' <= 1/4)
    res.append((s, th, f))
    print(f"[{obs}] GN from {np.round(s, 2)} -> {np.round(th, 3)}  -log post {f:.2f}  ({it} it, {time.time() - t0:.0f}s)", flush=True)
np.save(f"results/explore_{obs}.npy", np.array([dict(profiles=out, gn=res)], dtype=object))
