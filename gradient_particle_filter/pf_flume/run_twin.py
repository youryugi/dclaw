"""Identical-twin experiment on the flume.

  python run_twin.py <method> [N] [seed] [--obs full|h] [--noise K]
  method: laplace | bootstrap | smc_rw | smc_mala | laplace_smc | robust_laplace | robust_laplace_is
          | ibis_rw | ibis_laplace
  --obs h: flow depth only (no pore pressure); --obs h@1: depth at gauge 1 (x = 32 m) only;
  --noise K: noise realisation K.
Results go to results/<obs>_noise<K>/<method>_N<N>_s<seed>.npz.
"""
import argparse
import pickle
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
# persistent compilation cache: the model and its derivatives take ~20 s to compile per batch shape
jax.config.update("jax_compilation_cache_dir", str(__import__("pathlib").Path(__file__).resolve().parent / ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
import jax.numpy as jnp
import numpy as np

import flume_model as M
import inference as I

HERE = Path(__file__).resolve().parent
DX, T_OBS, OBS_DT = 0.125, 12.0, 0.5
GAUGES = [0, 1, 2]                      # x = 2, 32, 66 m (90 m stays dry at this grid within 12 s)
SIG_H, SIG_PB = 0.01, 0.1              # noise std: m, kPa


class Sub:
    """Flume restricted to the gauges used."""

    def __init__(self):
        self.F = M.Flume(dx=DX, t_obs=T_OBS, obs_dt=OBS_DT)

    def simulate(self, theta):
        obs, t = self.F.simulate(theta)
        return obs[:, :, GAUGES], t


def problem(obs="full", noise=0, params="3"):
    ps = M.use_param_set(params)
    I.set_prior(ps["lo"], ps["hi"])
    S = Sub()
    truth, _ = jax.jit(S.simulate)(jnp.asarray(M.THETA_TRUE))
    truth = np.asarray(truth)
    sigma = np.broadcast_to(np.array([SIG_H, SIG_PB])[None, :, None], truth.shape).copy()
    y = truth + sigma * np.random.default_rng(noise).normal(size=truth.shape)
    mask = np.ones_like(truth)
    if obs == "h":
        mask[:, 1] = 0.0
    elif "@" in obs:                     # e.g. "h@1": depth at gauge index 1 (x = 32 m) only
        var, g = obs.split("@")
        mask[:] = 0.0
        mask[:, {"h": 0, "pb": 1}[var], int(g)] = 1.0
    return I.Problem(simulate=S.simulate, y=y, sigma=sigma, mask=mask, batch=256), truth


def summary(name, theta, logw, wall, prob):
    w = np.exp(logw - logw.max()); w /= w.sum()
    ess = 1.0 / (w ** 2).sum()
    mean = w @ theta
    sd = np.sqrt(w @ (theta - mean) ** 2)
    print(f"{name}: N={len(theta)} ESS={ess:.1f} wall={wall:.0f}s fwd={prob.n_fwd} grad={prob.n_grad}")
    for k, nm in enumerate(M.THETA_NAMES):
        print(f"   {nm:14s} mean {mean[k]:8.4f} sd {sd[k]:.4f}   truth {M.THETA_TRUE[k]:8.4f}")
    return ess


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("method"); ap.add_argument("N", type=int, nargs="?", default=256)
    ap.add_argument("seed", type=int, nargs="?", default=0)
    ap.add_argument("--obs", default="full"); ap.add_argument("--noise", type=int, default=0)
    ap.add_argument("--moves", type=int, default=5, help="MCMC moves per resampling (IBIS) / per stage (SMC)")
    ap.add_argument("--k", type=int, default=4, help="Gauss-Newton starts (robust_laplace)")
    ap.add_argument("--screen", type=int, default=256, help="prior draws screened (robust_laplace)")
    ap.add_argument("--params", default="3", help="parameter set: 3, 7 or 7d (see flume_model.PARAM_SETS)")
    ap.add_argument("--merge_sd", type=float, default=0.5, help="merge converged GN points closer than this (sd)")
    ap.add_argument("--inflate", type=float, default=1.5, help="Laplace covariance inflation (robust2)")
    ap.add_argument("--dof", type=float, default=5.0, help="t degrees of freedom (robust2)")
    ap.add_argument("--adapt", type=int, default=0, help="adaptive IS rounds (robust2)")
    ap.add_argument("--ridge", type=float, default=None, help="extra inflation along the weak eigendirections (robust2)")
    ap.add_argument("--ridge_k", type=int, default=2, help="number of weak eigendirections to widen")
    ap.add_argument("--wide", type=int, default=0, help="second IS stage: samples from a wide t fitted to the first")
    a = ap.parse_args()
    out = HERE / "results" / (f"{a.obs.replace('@', '_')}_noise{a.noise}" + ("" if a.params == "3" else f"_p{a.params}"))
    out.mkdir(parents=True, exist_ok=True)
    prob, truth = problem(a.obs, a.noise, a.params)
    z_centre = I.to_z(0.5 * (I.LO + I.HI))
    extra = {}
    if a.method.startswith("ibis"):
        hist, snaps = I.ibis(prob, a.N, move=a.method[5:], n_moves=a.moves, seed=a.seed, verbose=True)
        keep = {k: snaps[k] for k in snaps if k in (3, 5, 7, 11, 15, 23)}      # t = 2, 3, 4, 6, 8, 12 s
        with open(out / f"{a.method}_N{a.N}_m{a.moves}_s{a.seed}.pkl", "wb") as f:
            pickle.dump(dict(hist=hist, snaps=keep, y=prob.y, truth=truth, mask=prob._mask), f)
        last = hist[-1]
        th, w, _ = snaps[len(hist) - 1]
        summary(a.method, th, np.log(np.maximum(w, 1e-300)), last["wall"], prob)
        raise SystemExit(0)
    if a.method == "laplace":
        Z, logw, wall, (zmap, C) = I.laplace_is(prob, a.N, z_centre, seed=a.seed, verbose=True)
        extra = dict(zmap=zmap, C=C)
    elif a.method == "bootstrap":
        Z, logw, wall = I.bootstrap_is(prob, a.N, seed=a.seed)
    elif a.method in ("smc_rw", "smc_mala"):
        Z, logw, wall, stages = I.tempered_smc(prob, a.N, move=a.method[4:], n_moves=a.moves, seed=a.seed, verbose=True)
        if a.moves != 5:
            a.method = f"{a.method}_m{a.moves}"
        extra = dict(stages=np.array(stages))
    elif a.method == "laplace_smc":
        Z, logw, wall, stages = I.laplace_smc(prob, a.N, z_centre, seed=a.seed, verbose=True)
        extra = dict(stages=np.array(stages))
    elif a.method in ("robust_laplace", "robust_laplace_is"):
        Z, logw, wall, info = I.robust_laplace(prob, a.N, seed=a.seed, tempering=a.method == "robust_laplace",
                                               m_screen=a.screen, k_starts=a.k, verbose=True)
        a.method = f"{a.method}_k{a.k}_m{a.screen}"
        extra = dict(is_ess=info["is_ess"], t_is=info["t_is"], n_modes=len(info["modes"]),
                     modes=np.array([m[0] for m in info["modes"]]), stages=np.array(info.get("stages", [])))
    elif a.method in ("robust2_laplace_is", "robust2_laplace_smc"):
        Z, logw, wall, info = I.robust2_laplace(prob, a.N, seed=a.seed, m_screen=a.screen, k_starts=a.k,
                                                merge_sd=a.merge_sd, tempering=a.method.endswith("smc"), verbose=True,
                                                inflate=a.inflate, dof=a.dof, adapt_rounds=a.adapt,
                                                ridge_inflate=a.ridge, ridge_k=a.ridge_k, wide_n=a.wide)
        extra = dict(is_ess=info["is_ess"], n_modes=len(info["modes"]), modes=np.array([m[0] for m in info["modes"]]),
                     t_modes=info["t_modes"], n_conv=info["n_conv"])
        a.method = (f"{a.method}_k{a.k}_m{a.screen}" + ("" if a.merge_sd == 0.5 else f"_merge{a.merge_sd:g}")
                    + ("" if (a.inflate, a.dof) == (1.5, 5.0) else f"_infl{a.inflate:g}_dof{a.dof:g}")
                    + ("" if a.adapt == 0 else f"_adapt{a.adapt}")
                    + ("" if a.ridge is None else f"_ridge{a.ridge:g}x{a.ridge_k}")
                    + ("" if a.wide == 0 else f"_wide{a.wide}"))
    else:
        raise SystemExit(f"unknown method {a.method}")
    theta = np.asarray(I.to_theta(jnp.asarray(Z)))
    ess = summary(a.method, theta, logw, wall, prob)
    np.savez(out / f"{a.method}_N{a.N}_s{a.seed}.npz", theta=theta, z=Z, logw=logw, wall=wall, ess=ess,
             n_fwd=prob.n_fwd, n_grad=prob.n_grad, **extra)
