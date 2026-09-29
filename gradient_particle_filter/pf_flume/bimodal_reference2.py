"""Stronger reference for the single-gauge (h@1) design: all three local
minima found (A, B and the shallow A' next to A), wider t components
(inflate 2) and 10% prior, 16384 importance samples."""
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import inference as I
from run_twin import problem

prob, _ = problem("h@1", 0)
starts = [np.array([42.099, -8.339, -1.680]), np.array([42.090, -8.034, -2.302]), np.array([42.232, -8.407, -1.706])]
Zm, C, f, conv, it, dec = I.gauss_newton_batch(prob, I.to_z(np.array(starts)), iters=60)
for k in range(3):
    print("mode", np.round(np.asarray(I.to_theta(jnp.asarray(Zm[k]))), 4), "-log post", round(float(f[k]), 3),
          "decrement", f"{dec[k]:.1e}", flush=True)
modes = [(Zm[k], C[k], f[k]) for k in range(3)]
q = I.mixture_proposal(modes, inflate=2.0, prior_weight=0.10)
rng = np.random.default_rng(1)
t0 = time.time()
Z = q.sample(rng, 16384)
logw = prob.loglik(Z) + np.asarray(I.log_prior_z(jnp.asarray(Z))) - q.logpdf(Z)
th = np.asarray(I.to_theta(jnp.asarray(Z)))
np.savez("results/h_1_noise0/reference2_mixture3_N16384.npz", theta=th, z=Z, logw=logw, wall=time.time() - t0,
         ess=I._ess(logw), n_fwd=prob.n_fwd, n_grad=prob.n_grad)
A = np.array([42.099, -8.339, -1.680]); B = np.array([42.090, -8.034, -2.302]); d = B - A
w = np.exp(logw - logw.max()); w /= w.sum()
pB = w[((th - A) @ d / (d @ d)) > 0.6].sum()
# bootstrap standard error of P(B) from the weighted sample
rs = np.random.default_rng(2)
bs = []
for _ in range(200):
    i = rs.integers(0, len(w), len(w)); ww = w[i] / w[i].sum()
    bs.append(ww[((th[i] - A) @ d / (d @ d)) > 0.6].sum())
print(f"reference2: ESS {I._ess(logw):.0f} of 16384, P(B) = {pB:.3f} +- {np.std(bs):.3f} (bootstrap), {time.time() - t0:.0f}s")
