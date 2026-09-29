import jax, jax.numpy as jnp, numpy as np
import run_jax as R
import twin_inversion as T
from dclaw_jax.model import MaterialParams, _material_state
from dclaw_jax.solver import cfl_dt, step
PHI = jnp.asarray(T.PHI_TRUE); KREF = T.KREF
def one(q, t, phi):
    p = MaterialParams(phi_deg=phi, kref=KREF, m_crit=0.64, mu=0.005, manning_n=0.06)
    dt = jnp.clip(jnp.minimum(cfl_dt(q, p, R.GRID, 0.35), 1800.0 - t), 0.0, None)
    return step(q, R.bed, p, R.GRID, dt, "open", "open"), t + dt
def body(c, _):
    q, dq, t = c
    (qn, tn), (dqn, _) = jax.jvp(lambda q_, ph: one(q_, t, ph), (q, PHI), (dq, jnp.ones_like(PHI)))
    a = jnp.abs(dqn[0])                                   # tangent of h only (what the observations see)
    k = jnp.argmax(a.reshape(-1))
    return (qn, dqn, tn), (jnp.max(a), k, tn, qn[0].reshape(-1)[k], qn[1].reshape(-1)[k], qn[2].reshape(-1)[k], qn[3].reshape(-1)[k], qn[4].reshape(-1)[k])
def trace():
    _, out = jax.jit(lambda: jax.lax.scan(body, (R.q0, jnp.zeros_like(R.q0), jnp.asarray(0.0)), None, length=3000))()
    return np.array(out[0])
