"""MCMC at the posterior (beta = 1) starting from a finished run's particles.
  python mcmc_polish.py <npz> <rw|mala> <n_moves> [n_particles] [--params 7]
Reports per move: acceptance, whitened expected squared jump distance (ESJD,
in units of the particles' own covariance), wall time; saves the final
particles as <npz stem>_polish_<kind><n>.npz. RW: covariance-adaptive random
walk; MALA: preconditioned by the particle covariance, gradient by reverse
mode through the checkpointed scan (flume_model.Flume.simulate(scan_steps=...))."""
import argparse, time
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", str(Path(__file__).resolve().parent / ".jax_cache"))
import jax.numpy as jnp
import numpy as np
import inference as I
from run_twin import problem, GAUGES

ap = argparse.ArgumentParser()
ap.add_argument("npz"); ap.add_argument("kind"); ap.add_argument("n_moves", type=int)
ap.add_argument("n_particles", type=int, nargs="?", default=0); ap.add_argument("--params", default="7")
a = ap.parse_args()
prob, _ = problem("full", 0, a.params)
F = prob.simulate.__self__.F
y, sig = jnp.asarray(prob.y), jnp.asarray(prob.sigma)
d0 = np.load(a.npz)
Z = np.array(d0["z"]); lw = d0["logw"]
rng = np.random.default_rng(11)
if np.ptp(lw) > 0:                                   # weighted sample: resample first
    Z = Z[I._resample(rng, lw)]
if a.n_particles:
    Z = Z[rng.choice(len(Z), a.n_particles, replace=False)]
n, d = Z.shape
C = np.cov(Z.T); Ci = np.linalg.inv(C); L = np.linalg.cholesky(C)
lp = lambda Zc, ll: np.asarray(I.log_prior_z(jnp.asarray(Zc))) + ll

def logpost_scan(z):
    o, _ = F.simulate(I.to_theta(z), scan_steps=5000, block=50)
    r = (o[:, :, GAUGES] - y) / sig
    return -0.5 * jnp.sum(r * r) + I.log_prior_z(z)
vg = jax.jit(jax.vmap(jax.value_and_grad(logpost_scan)))

def val_grad(Zc):
    out = [I._call_padded(vg, Zc[i:i + 128]) for i in range(0, len(Zc), 128)]
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

t0 = time.time()
if a.kind == "mala":
    lpz, g = val_grad(Z)
else:
    lpz = lp(Z, prob.loglik(Z))
scale = 1.0
print(f"start: {n} particles, d={d}, setup {time.time() - t0:.0f}s", flush=True)
t0 = time.time()
tot_esjd = 0.0
for it in range(a.n_moves):
    ts = time.time()
    if a.kind == "rw":
        Zp = Z + scale * 2.38 / np.sqrt(d) * rng.normal(size=Z.shape) @ L.T
        lpp = lp(Zp, prob.loglik(Zp))
        log_a = lpp - lpz
    else:
        h = (scale * 1.65 / d ** (1 / 6)) ** 2
        mean = Z + 0.5 * h * g @ C
        Zp = mean + np.sqrt(h) * rng.normal(size=Z.shape) @ L.T
        lpp, gp = val_grad(Zp)
        meanp = Zp + 0.5 * h * gp @ C
        qf = -0.5 / h * np.einsum("ni,ij,nj->n", Zp - mean, Ci, Zp - mean)
        qb = -0.5 / h * np.einsum("ni,ij,nj->n", Z - meanp, Ci, Z - meanp)
        log_a = lpp - lpz + qb - qf
    acc = np.log(rng.random(n)) < np.nan_to_num(log_a, nan=-np.inf)
    dz = np.where(acc[:, None], Zp - Z, 0.0)
    esjd = float(np.mean(np.einsum("ni,ij,nj->n", dz, Ci, dz)))
    tot_esjd += esjd
    Z = np.where(acc[:, None], Zp, Z); lpz = np.where(acc, lpp, lpz)
    if a.kind == "mala":
        g = np.where(acc[:, None], gp, g)
    scale *= np.exp(acc.mean() - (0.3 if a.kind == "rw" else 0.57))
    print(f"  move {it + 1:3d}: acc {acc.mean():.2f} ESJD {esjd:.3f} scale {scale:.3f} {time.time() - ts:.1f}s", flush=True)
wall = time.time() - t0
th = np.asarray(I.to_theta(jnp.asarray(Z)))
print(f"{a.kind}: {a.n_moves} moves, {wall:.0f}s, mean ESJD/move {tot_esjd / a.n_moves:.3f}, ESJD per second {tot_esjd / wall:.4f} (per particle-second x n: {tot_esjd / wall * n:.2f})")
print("  mean", np.round(th.mean(0), 4).tolist()); print("  sd  ", np.round(th.std(0), 4).tolist())
np.savez(Path(a.npz).with_name(Path(a.npz).stem + f"_polish_{a.kind}{a.n_moves}.npz"), theta=th, z=Z,
         logw=np.zeros(n), wall=wall, ess=n, n_fwd=0, n_grad=0)
