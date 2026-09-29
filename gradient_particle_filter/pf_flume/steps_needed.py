"""Steps actually used (dt > 0) up to t_obs, at the truth and the corners of the prior box."""
import itertools
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import flume_model as M
from inference import HI, LO
from dclaw_jax.solver import cfl_dt, step

dx, t_obs = float(sys.argv[1]), float(sys.argv[2])
F = M.Flume(dx=dx, t_obs=t_obs)


def count(theta):
    p = M.params(theta)
    q0 = M.qfix(F.q0, p)

    def body(c, _):
        t, q, n = c
        t_next = (jnp.floor(t / F.obs_dt + 1e-9) + 1.0) * F.obs_dt
        dt = jnp.clip(jnp.minimum(jnp.minimum(cfl_dt(q, p, F.grid, 0.4), t_next - t), t_obs - t), 0.0, None)
        q = step(q, F.bed(t), p, F.grid, dt, "wall", "wall")
        return (t + dt, q, n + (dt > 0)), None

    (t, q, n), _ = jax.lax.scan(body, (jnp.asarray(0.0), q0, 0), None, length=F.n_steps)
    return n, t


thetas = [M.THETA_TRUE] + [np.array(c) for c in itertools.product(*zip(LO, HI))]
n, t = jax.jit(jax.vmap(count))(jnp.asarray(np.array(thetas)))
print(f"dx={dx} t_obs={t_obs}: n_steps budget {F.n_steps}; used: truth {int(n[0])}, corners min {int(n[1:].min())} "
      f"max {int(n[1:].max())}; all reached t_obs: {bool((t > t_obs - 1e-9).all())}")
