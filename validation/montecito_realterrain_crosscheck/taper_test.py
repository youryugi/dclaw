import sys
import dclaw_jax.model as M
M.DILATANCY_U_TAPER = float(sys.argv[1])
import jax, jax.numpy as jnp, numpy as np
import run_jax as R
from compare import csi
import trace_twin_fn as TT
fr, t, q = jax.jit(R.run)(jnp.array([37.63, float(np.log10(3.348e-12))]))
peak = np.array(jnp.max(fr[:, 0], axis=0)); yc = (np.arange(R.NY) + 0.5) * R.DX
fronts = [int(yc[(np.array(fr[k, 0]) > 0.1).any(axis=0)].min()) for k in (15, 30, 60, 90)]
print(f"U_taper={M.DILATANCY_U_TAPER}: cross-check CSI {csi(peak)[0]:.3f}, fronts {fronts} (before: 0.491, [2410,1490,830,450]; Fortran o1 0.519)")
mx = TT.trace()
print(f"   twin config: max |dh/dphi| over run {mx.max():.3e}; at t>1300s {mx[1700:].max():.3e}")
