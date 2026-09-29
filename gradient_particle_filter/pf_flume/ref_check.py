"""Decisive reference for the 7d posterior's spread: importance sampling from a
proposal wider than every candidate -- a t (dof 3) centred on the long SMC's
particles with their covariance inflated 1.5x -- plus 5% prior. IS is exact
whatever the proposal; a proposal wider than the posterior can only cost ESS."""
import time
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", str(__import__("pathlib").Path(__file__).resolve().parent / ".jax_cache"))
import jax.numpy as jnp
import numpy as np
import inference as I
from run_twin import problem

prob, _ = problem("full", 0, "7d")
d = np.load("results/full_noise0_p7d/smc_rw_m10_N1024_s5.npz")
Zs = d["z"]
m, C = Zs.mean(0), np.cov(Zs.T)
q = I.MixtureT([m], [1.5 ** 2 * C], [1.0], dof=3.0, prior_weight=0.05)
rng = np.random.default_rng(21)
t0 = time.time()
Z = q.sample(rng, 16384)
logw = prob.loglik(Z) + np.asarray(I.log_prior_z(jnp.asarray(Z))) - q.logpdf(Z)
th = np.asarray(I.to_theta(jnp.asarray(Z)))
w = np.exp(logw - logw.max()); w /= w.sum()
mean = w @ th; sd = np.sqrt(w @ (th - mean) ** 2)
rs = np.random.default_rng(2); bs = []
for _ in range(200):
    i = rs.integers(0, len(w), len(w)); ww = w[i] / w[i].sum(); mm = ww @ th[i]
    bs.append(np.sqrt(ww @ (th[i] - mm) ** 2))
print(f"wide-proposal IS: ESS {1 / (w ** 2).sum():.0f} of 16384, {time.time() - t0:.0f}s")
print("  mean", np.round(mean, 4).tolist())
print("  sd  ", np.round(sd, 4).tolist())
print("  sd bootstrap se", np.round(np.std(bs, 0), 4).tolist())
np.savez("results/full_noise0_p7d/refcheck_wideIS_N16384.npz", theta=th, z=Z, logw=logw, wall=time.time() - t0,
         ess=1 / (w ** 2).sum(), n_fwd=16384, n_grad=0)
