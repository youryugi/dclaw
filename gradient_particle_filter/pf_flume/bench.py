"""Cost of the flume forward model and its gradients, batched over particles."""
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from flume_model import THETA_TRUE, Flume

dx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.125
F = Flume(dx=dx)
print(f"dx={dx}: {F.grid.nx} cells, n_steps={F.n_steps}, {F.n_obs_t} obs times x 2 vars x 4 gauges")

sim = jax.jit(F.simulate)
obs, t = sim(jnp.asarray(THETA_TRUE))
print(f"truth: t reached {float(t):.3f}; peak h per gauge {np.asarray(obs[:, 0]).max(0).round(3)}; "
      f"peak pb (kPa) {np.asarray(obs[:, 1]).max(0).round(2)}")


def timed(f, *a):
    jax.block_until_ready(f(*a))
    t0 = time.time(); r = jax.block_until_ready(f(*a)); return time.time() - t0, r


rng = np.random.default_rng(0)
for n in (1, 16, 64, 256):
    th = jnp.asarray(THETA_TRUE + rng.normal(0, 0.05, (n, 3)))
    dt_s, (o, tt) = timed(jax.jit(jax.vmap(F.simulate)), th)
    print(f"N={n:5d} forward: {dt_s:6.2f} s ({dt_s / n * 1e3:7.1f} ms/particle), all reached t_obs: {bool((tt > F.t_obs - 1e-9).all())}", flush=True)
for n in (1, 64):
    th = jnp.asarray(THETA_TRUE + rng.normal(0, 0.05, (n, 3)))
    jf = jax.jit(jax.vmap(jax.jacfwd(lambda x: F.simulate(x)[0])))
    dt_s, _ = timed(jf, th)
    print(f"N={n:5d} forward + full Jacobian (jacfwd, 3 params): {dt_s:6.2f} s ({dt_s / n * 1e3:7.1f} ms/particle)", flush=True)
