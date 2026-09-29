"""7-parameter set: forward-mode Jacobian of the noise-scaled gauge series vs
central finite differences, at 8 points across the whole prior and 8 near
the truth."""
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import flume_model as M
import inference as I
from run_twin import Sub, SIG_H, SIG_PB

ps = M.use_param_set("7"); I.set_prior(ps["lo"], ps["hi"])
S = Sub()
rng = np.random.default_rng(3)
span = ps["hi"] - ps["lo"]
far = ps["lo"] + span * rng.uniform(0.05, 0.95, (8, 7))
near = ps["truth"] + span * 0.03 * rng.normal(size=(8, 7))
TH = jnp.asarray(np.vstack([far, near]))
f = jax.jit(jax.vmap(lambda th: S.simulate(th)[0]))
obs0 = np.asarray(f(TH))
sig = np.broadcast_to(np.array([SIG_H, SIG_PB])[None, :, None], obs0.shape[1:])
t0 = time.time()
J = np.asarray(jax.jit(jax.vmap(jax.jacfwd(lambda th: S.simulate(th)[0])))(TH)) / sig[None, ..., None]
print(f"jacfwd (7 params) for 16 points: {time.time() - t0:.0f} s incl. compile", flush=True)
for k, name in enumerate(ps["names"]):
    e = 1e-6 * span[k]
    d = jnp.zeros(7).at[k].set(e)
    fd = (np.asarray(f(TH + d)) - np.asarray(f(TH - d))) / (2 * e) / sig[None]
    a, b = J[..., k].reshape(16, -1), fd.reshape(16, -1)
    rel = np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-12)
    print(f"  d/d {name:14s} |J| median {np.median(np.linalg.norm(a, axis=1)):9.1f}   rel.err vs FD (eps=1e-6*range): "
          f"prior-wide points median {np.median(rel[:8]):.1e} max {rel[:8].max():.1e} | near truth median {np.median(rel[8:]):.1e} max {rel[8:].max():.1e}",
          flush=True)
