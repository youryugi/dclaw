import sys
import jax, jax.numpy as jnp, numpy as np
import run_jax as R
sys.path.insert(0, str(R.HERE.parents[1] / "differentiable_dclaw"))
import dclaw_jax.solver as S
from dclaw_jax.model import MaterialParams, pb_relaxation_step, _material_state
TH = np.load(R.HERE / "calibrated_theta.npy")
p = MaterialParams(phi_deg=float(TH[0]), kref=10.0 ** float(TH[1]), m_crit=0.64, mu=0.005, manning_n=0.06)
def body(c, _):
    q, t = c
    dt = jnp.clip(jnp.minimum(S.cfl_dt(q, p, R.GRID, 0.35), 1800.0 - t), 0.0, None)
    return (S.step(q, R.bed, p, R.GRID, dt, "open", "open"), t + dt), dt
(q, t), dts = jax.jit(lambda: jax.lax.scan(body, (R.q0, jnp.asarray(0.0)), None, length=1300))()
dt = dts[-1]
for (i, j) in [(19, 182), (30, 171)]:
    c = q[:, i, j]
    one = lambda c5: pb_relaxation_step(c5[:, None, None], p, dt)[0, 0]
    J = jax.jacfwd(one)(c)
    s = {k: float(v[0, 0]) for k, v in _material_state(c[:, None, None], p).items()}
    print(f"cell ({i},{j}): h={float(c[0]):.4f} hu={float(c[1]):.3e} hv={float(c[2]):.3e} m={s['m']:.4f} pb/lith={float(c[4])/(s['rho']*9.81*float(c[0])):.6f}")
    print(f"   sigma_e={s['sigma_e']:.3e} vnorm={s['vnorm']:.3e} tanpsi={s['tanpsi']:.4f} alpha={s['alpha']:.3e} zeta={s['zeta']:.3e} k={s['k']:.3e}")
    print(f"   d pb_new / d (h, hu, hv, hm, pb) = {np.array(J)}")
    # which intermediate: derivative of tanpsi and vnorm wrt hu
    tp = lambda c5: _material_state(c5[:, None, None], p)["tanpsi"][0, 0]
    print(f"   d tanpsi / d (h, hu, hv, hm, pb) = {np.array(jax.jacfwd(tp)(c))}")
