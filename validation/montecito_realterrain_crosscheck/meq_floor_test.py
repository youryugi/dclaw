import sys
import jax, jax.numpy as jnp, numpy as np
from scipy import ndimage
import run_jax as R
sys.path.insert(0, str(R.HERE.parents[1] / "differentiable_dclaw"))
import dclaw_jax.model as M
from dclaw_jax.model import MaterialParams, EPS_H
from dclaw_jax.solver import rollout_peak
from compare import csi
import sensitivity as SEN

S_DEN = float(sys.argv[1])
def m_eq_floor(sigma_e, shear, p):
    Nden = p.rho_s * (shear * p.delta) ** 2 + sigma_e
    Nnum = p.mu * shear
    ratio = jnp.sqrt(Nden + S_DEN) / (jnp.sqrt(Nden + S_DEN) + jnp.sqrt(jnp.maximum(Nnum, 0.0) + 1e-12) + EPS_H)
    return jnp.where(Nnum > 0.0, p.m_crit * ratio, p.m_crit)
if S_DEN > 0:
    M.m_equilibrium = m_eq_floor

# forward vs Fortran (ensemble median, frame-sampled as in compare.py)
fr, t, q = jax.jit(R.run)(jnp.array([37.63, float(np.log10(3.348e-12))]))
peak = np.array(jnp.max(fr[:, 0], axis=0))
yc = (np.arange(R.NY) + 0.5) * R.DX
fronts = [yc[(np.array(fr[k, 0]) > 0.1).any(axis=0)].min() for k in (15, 30, 60, 90)]
print(f"S_DEN={S_DEN}: CSI {csi(peak)[0]:.3f}, front y at 300/600/1200/1800 s: {[int(f) for f in fronts]} "
      f"(Fortran o1: 2410/1390/410/90)", flush=True)

# directional derivative check on the building target (reverse mode, checkpointed)
bed, th = R.bed, jnp.asarray(SEN.THETA)
g = jax.jit(jax.grad(SEN.target), static_argnums=2)(bed, th, 50)
v = ndimage.gaussian_filter(np.random.default_rng(0).standard_normal(bed.shape), 5.0)
v = jnp.asarray(v / np.abs(v).max())
f = jax.jit(SEN.target, static_argnums=2)
ad = float(jnp.sum(g * v))
fds = [(float(f(bed + e * v, th, None)) - float(f(bed - e * v, th, None))) / (2 * e) for e in (1e-2, 1e-4, 1e-6)]
print(f"   directional: AD {ad: .5e}  FD(1e-2,1e-4,1e-6) " + " ".join(f"{x: .5e}" for x in fds) + f"   max|dL/dbed| {float(jnp.abs(g).max()):.3e}", flush=True)
