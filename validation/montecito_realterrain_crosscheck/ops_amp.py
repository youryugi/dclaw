import sys
import jax, jax.numpy as jnp, numpy as np
import run_jax as R
sys.path.insert(0, str(R.HERE.parents[1] / "differentiable_dclaw"))
import dclaw_jax.solver as S
from dclaw_jax.model import MaterialParams, friction_step, pb_relaxation_step, dilatancy_step, manning_step, qfix, _material_state
TH = np.load(R.HERE / "calibrated_theta.npy")
p = MaterialParams(phi_deg=float(TH[0]), kref=10.0 ** float(TH[1]), m_crit=0.64, mu=0.005, manning_n=0.06)
def body(c, _):
    q, t = c
    dt = jnp.clip(jnp.minimum(S.cfl_dt(q, p, R.GRID, 0.35), 1800.0 - t), 0.0, None)
    return (S.step(q, R.bed, p, R.GRID, dt, "open", "open"), t + dt), dt
SC = jnp.array([1.0, 10.0, 10.0, 1.0, 3e4])[:, None, None]
def transport(q, dt):
    k1 = S.rhs(q, R.bed, p, R.GRID, "open", "open"); q1 = S._clip_dry(q + dt * k1)
    k2 = S.rhs(q1, R.bed, p, R.GRID, "open", "open"); return S._clip_dry(q + 0.5 * dt * (k1 + k2))
ops = {"transport": transport, "qfix": lambda q, dt: qfix(q, p), "friction": lambda q, dt: friction_step(q, p, dt),
       "pb_relax": lambda q, dt: q.at[4].set(pb_relaxation_step(q, p, dt)), "dilatancy": lambda q, dt: dilatancy_step(q, p, dt),
       "full step": lambda q, dt: S.step(q, R.bed, p, R.GRID, dt, "open", "open")}
for n in (1300, 1312, 1470):
    (q, t), dts = jax.jit(lambda: jax.lax.scan(body, (R.q0, jnp.asarray(0.0)), None, length=n))()
    dt = float(dts[-1])
    print(f"\nafter {n} steps, t={float(t):.0f}s dt={dt:.3f}")
    st = _material_state(q, p)
    for name, op in ops.items():
        f = jax.jit(lambda z, op=op: op(z * SC, dt) / SC)
        z0 = q / SC; _, vjp = jax.vjp(f, z0)
        v = jax.random.normal(jax.random.PRNGKey(0), z0.shape)
        for _ in range(30):
            v = v / jnp.linalg.norm(v); v = vjp(jax.jvp(f, (z0,), (v,))[1])[0]
        s = float(jnp.sqrt(jnp.linalg.norm(v)))
        r, i, j = np.unravel_index(int(jnp.argmax(jnp.abs(v))), v.shape)
        h = float(q[0, i, j])
        print(f"  {name:10s} sigma_max={s:9.3e} row {['h','hu','hv','hm','pb'][r]} ({i},{j}) h={h:.3e} "
              f"|u|={float(jnp.hypot(st['u'][i,j], st['v'][i,j])):.2f} m={float(st['m'][i,j]):.3f} "
              f"pb/lith={float(q[4,i,j])/max(float(st['rho'][i,j])*9.81*h,1e-12):.3f} sigma_e={float(st['sigma_e'][i,j]):.1f}")
