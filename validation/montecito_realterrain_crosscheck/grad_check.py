"""Autodiff vs finite differences on the real-terrain cross-check setup, for a
smooth inundation-type loss."""
import time
import jax
import jax.numpy as jnp
import numpy as np
from run_jax import run

def loss(theta):
    fr, _, _ = run(theta)
    peak = jnp.max(fr[:, 0], axis=0)
    return jnp.sum(jax.nn.sigmoid((peak - 0.1) / 0.03))

theta0 = jnp.array([37.63, np.log10(3.348e-12)])
f = jax.jit(loss)
t0 = time.time(); g = np.array(jax.jit(jax.jacfwd(loss))(theta0)); print(f"jacfwd {time.time()-t0:.0f}s")
for i, name in enumerate(["phi_deg", "log10 kref"]):
    for e in (1e-3, 1e-5, 1e-7):
        d = jnp.zeros(2).at[i].set(e)
        fd = (float(f(theta0 + d)) - float(f(theta0 - d))) / (2 * e)
        print(f"d loss / d {name}: AD {g[i]: .6e}   FD(eps={e:.0e}) {fd: .6e}", flush=True)
