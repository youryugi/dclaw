"""Reference posterior for the single-gauge (h@1) design: importance sampling
from a mixture of the Laplace fits at both modes (+5% prior)."""
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import inference as I
from run_twin import problem

prob, _ = problem("h@1", 0)
modes = []
for th in (np.array([42.099, -8.339, -1.680]), np.array([42.090, -8.034, -2.302])):
    z, C, f, it = I.gauss_newton(prob, I.to_z(th), gn_iters=60)
    modes.append((z, C, f))
    print("mode", np.round(np.asarray(I.to_theta(jnp.asarray(z))), 4), "-log post", round(f, 3), flush=True)
q = I.mixture_proposal(modes)
rng = np.random.default_rng(0)
t0 = time.time()
Z = q.sample(rng, 8192)
logw = prob.loglik(Z) + np.asarray(I.log_prior_z(jnp.asarray(Z))) - q.logpdf(Z)
th = np.asarray(I.to_theta(jnp.asarray(Z)))
np.savez("results/h_1_noise0/reference_mixture_N8192.npz", theta=th, z=Z, logw=logw, wall=time.time() - t0,
         ess=I._ess(logw), n_fwd=prob.n_fwd, n_grad=prob.n_grad)
print(f"reference IS: ESS {I._ess(logw):.0f} of 8192, {time.time() - t0:.0f}s")
