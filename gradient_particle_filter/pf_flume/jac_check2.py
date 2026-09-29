"""Full observation Jacobian (jacfwd) vs central finite differences at 16
parameter points around the truth; optional DILATANCY_H_FLOOR (argv[1])."""
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import flume_model as M
import dclaw_jax.model as model
if len(sys.argv) > 1:
    model.DILATANCY_H_FLOOR = float(sys.argv[1])
F = M.Flume(dx=float(sys.argv[2]) if len(sys.argv) > 2 else 0.125, t_obs=float(sys.argv[3]) if len(sys.argv) > 3 else 15.0)
rng = np.random.default_rng(1)
th = jnp.asarray((M.THETA_TRUE + rng.normal(0, [1.0, 0.3, 0.2], (64, 3)))[:16])
sig = np.broadcast_to(np.array([0.01, 0.1])[None, :, None], (F.n_obs_t, 2, 4))   # scale: noise std (h m, pb kPa)
f = jax.jit(jax.vmap(lambda x: F.simulate(x)[0]))
J = np.asarray(jax.jit(jax.vmap(jax.jacfwd(lambda x: F.simulate(x)[0])))(th)) / sig[None, ..., None]
print(f"DILATANCY_H_FLOOR = {model.DILATANCY_H_FLOOR}: Jacobian of noise-scaled observations, AD vs FD")
for k, name in enumerate(M.THETA_NAMES):
    row = []
    for e in (1e-3, 1e-4, 1e-6):
        d = jnp.zeros(3).at[k].set(e)
        fd = (np.asarray(f(th + d)) - np.asarray(f(th - d))) / (2 * e) / sig[None]
        a, b = J[..., k].reshape(16, -1), fd.reshape(16, -1)
        rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
        row.append(f"eps={e:g}: rel.err median {np.median(rel):.1e} max {rel.max():.1e}")
    print(f"  d/d {name:14s} |J| median {np.median(np.linalg.norm(J[..., k].reshape(16, -1), axis=1)):8.1f}   " + "   ".join(row), flush=True)
