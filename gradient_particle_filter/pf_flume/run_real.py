"""Inference from the real 2010 flume data (realdata.py).
  python run_real.py <robust2|smc_rw> [N] [seed] [--calib 01] [--frac 0.1] [--params 3]
Saves results/real_<calib>_frac<f>_p<params>/<method>_N<N>_s<seed>.npz with the particles and
their simulated series at all three gauges (for posterior predictive checks)."""
import argparse
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", str(Path(__file__).resolve().parent / ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
import jax.numpy as jnp
import numpy as np

import flume_model as M
import inference as I
import realdata as RD

ap = argparse.ArgumentParser()
ap.add_argument("method"); ap.add_argument("N", type=int, nargs="?", default=1024)
ap.add_argument("seed", type=int, nargs="?", default=0)
ap.add_argument("--calib", default="01", help="gauges in the likelihood: digits of 0=32 m, 1=66 m, 2=90 m")
ap.add_argument("--frac", type=float, default=0.1, help="model-error fraction of |obs| in sigma")
ap.add_argument("--params", default="3")
a = ap.parse_args()
prob, R = RD.problem(params=a.params, model_frac=a.frac, calib_gauges=tuple(int(c) for c in a.calib))
out = Path(__file__).resolve().parent / "results" / f"real_{a.calib}_frac{a.frac:g}_p{a.params}"
out.mkdir(parents=True, exist_ok=True)
if a.method == "robust2":
    Z, logw, wall, info = I.robust2_laplace(prob, a.N, seed=a.seed, m_screen=512, k_starts=16, merge_sd=0.1,
                                            wide_n=0, verbose=True)
    extra = dict(is_ess=info["is_ess"], n_modes=len(info["modes"]), modes=np.array([m[0] for m in info["modes"]]))
elif a.method == "robust2_wide":
    Z, logw, wall, info = I.robust2_laplace(prob, 1024, seed=a.seed, m_screen=512, k_starts=16, merge_sd=0.1,
                                            wide_n=a.N, verbose=True)
    extra = dict(is_ess=info["is_ess"], n_modes=len(info["modes"]))
elif a.method == "smc_rw":
    Z, logw, wall, stages = I.tempered_smc(prob, a.N, move="rw", seed=a.seed, verbose=True)
    extra = dict(stages=np.array(stages))
else:
    raise SystemExit(a.method)
theta = np.asarray(I.to_theta(jnp.asarray(Z)))
w = np.exp(logw - logw.max()); w /= w.sum()
print(f"{a.method}: N={len(Z)} ESS={1 / (w ** 2).sum():.1f} wall={wall:.0f}s fwd={prob.n_fwd} grad={prob.n_grad}")
m = w @ theta; sd = np.sqrt(w @ (theta - m) ** 2)
for k, nm in enumerate(M.THETA_NAMES):
    print(f"   {nm:14s} mean {m[k]:8.4f} sd {sd[k]:.4f}   (paper value {M.THETA_TRUE[k]:8.4f})")
# predictive: resample 256 particles by weight and simulate the three gauges
idx = np.random.default_rng(1).choice(len(Z), 256, p=w)
pred = prob.obs(Z[idx])
np.savez(out / f"{a.method}_N{a.N}_s{a.seed}.npz", theta=theta, z=Z, logw=logw, wall=wall, n_fwd=prob.n_fwd,
         n_grad=prob.n_grad, pred=pred, pred_theta=theta[idx], y=prob.y, sigma=prob.sigma,
         mask=np.asarray(prob._mask), obs_times=R.F.obs_times, gauges_x=R.gx, **extra)
