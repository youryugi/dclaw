import sys
import jax, jax.numpy as jnp, numpy as np
import run_jax as R
sys.path.insert(0, str(R.HERE.parents[1] / "differentiable_dclaw"))
from dclaw_jax.model import MaterialParams, _material_state
from dclaw_jax.solver import cfl_dt, step
TH = jnp.asarray(np.load(R.HERE / "calibrated_theta.npy"))
def one(q, t, th):
    p = MaterialParams(phi_deg=th[0], kref=10.0 ** th[1], m_crit=0.64, mu=0.005, manning_n=0.06)
    dt = jnp.clip(jnp.minimum(cfl_dt(q, p, R.GRID, 0.35), 1800.0 - t), 0.0, None)
    return step(q, R.bed, p, R.GRID, dt, "open", "open"), t + dt
def body(c, _):
    q, dq, t = c
    (qn, tn), (dqn, _) = jax.jvp(lambda q_, th_: one(q_, t, th_), (q, TH), (dq, jnp.array([1.0, 0.0])))
    a = jnp.abs(dqn) / jnp.array([1.0, 10.0, 10.0, 1.0, 3e4])[:, None, None]
    return (qn, dqn, tn), (jnp.max(a), jnp.argmax(a.reshape(-1)), tn, qn[0].reshape(-1)[jnp.argmax(a.reshape(-1)) % (R.NX * R.NY)])
_, (mx, arg, times, hloc) = jax.jit(lambda: jax.lax.scan(body, (R.q0, jnp.zeros_like(R.q0), jnp.asarray(0.0)), None, length=3000))()
mx, arg, times, hloc = map(np.array, (mx, arg, times, hloc))
prev = 1e-30
for k in range(len(mx)):
    if mx[k] > 10 * prev or k % 300 == 0:
        r, i, j = np.unravel_index(arg[k], (5, R.NX, R.NY))
        print(f"step {k:5d} t={times[k]:7.1f}s max scaled tangent {mx[k]:.2e} row {['h','hu','hv','hm','pb'][r]} cell ({i},{j}) h={hloc[k]:.3e} bed={float(R.bed[i,j]):.1f}")
        prev = max(prev, mx[k])
