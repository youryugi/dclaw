"""Step-by-step forward tangent d(state)/d(theta_k) at one parameter point:
where and when does it blow up?"""
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import flume_model as M
from dclaw_jax.solver import cfl_dt, step

pi, k = int(sys.argv[1]), int(sys.argv[2])
F = M.Flume(dx=0.125)
rng = np.random.default_rng(1)
theta = jnp.asarray((M.THETA_TRUE + rng.normal(0, [1.0, 0.3, 0.2], (64, 3)))[pi])
e = jnp.zeros(3).at[k].set(1.0)


def one(q, t, th):
    p = M.params(th)
    t_next = (jnp.floor(t / F.obs_dt + 1e-9) + 1.0) * F.obs_dt
    dt = jnp.clip(jnp.minimum(jnp.minimum(cfl_dt(q, p, F.grid, 0.4), t_next - t), F.t_obs - t), 0.0, None)
    return step(q, F.bed(t), p, F.grid, dt, "wall", "wall"), t + dt


@jax.jit
def jstep(q, dq, t, dt_):
    (qn, tn), (dqn, dtn) = jax.jvp(one, (q, t, theta), (dq, dt_, e))
    return qn, dqn, tn, dtn

q0 = M.qfix(F.q0, M.params(theta))
q, dq = jax.jvp(lambda th: M.qfix(F.q0, M.params(th)), (theta,), (e,))
t, dt_ = jnp.asarray(0.0), jnp.asarray(0.0)
last = 0.0
n = 0
names = ["h", "hu", "hv", "hm", "pb"]
while float(t) < F.t_obs - 1e-9:
    q, dq, t, dt_ = jstep(q, dq, t, dt_)
    n += 1
    # scale-free tangent: relative to typical magnitudes
    mag = np.array([float(jnp.abs(dq[c]).max()) for c in range(5)])
    tot = mag[0] + mag[4] / 1e4
    if tot > 10 * max(last, 1.0) or n % 400 == 0:
        c = int(np.argmax(mag / np.array([1, 1, 1, 1, 1e4])))
        i = int(jnp.argmax(jnp.abs(dq[c]).max(axis=1)))
        h, u = float(q[0, i, 0]), float(q[1, i, 0] / max(float(q[0, i, 0]), 1e-9))
        m = float(q[3, i, 0] / max(h, 1e-9)); lith = (2700 * m + 1100 * (1 - m)) * 9.81 * h
        print(f"step {n:5d} t={float(t):6.3f}  max|dq| h {mag[0]:.2e} hu {mag[1]:.2e} pb {mag[4]:.2e}  worst {names[c]} at "
              f"x={F.x[i]:6.2f} m: h={h * 1e3:7.2f} mm u={u:+.3f} m/s m={m:.3f} pb/lith={float(q[4, i, 0]) / max(lith, 1e-9):.3f}",
              flush=True)
        last = tot
    if tot > 1e12:
        break
