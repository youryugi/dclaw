"""How much do the observations (soft inundation + building peak depths)
move per unit perturbation of each candidate field? Forward-mode JVPs at
the twin experiment's truth, for uniform and localized perturbations."""
import jax, jax.numpy as jnp, numpy as np
from scipy import ndimage
import run_jax as R
import twin_inversion as T
from dclaw_jax.model import MaterialParams
from dclaw_jax.solver import rollout_peak

phi0 = jnp.asarray(T.PHI_TRUE); lk0 = jnp.full((T.NX, T.NY), np.log10(T.KREF)); bed0 = R.bed
def obs(phi, logk, bed):
    p = MaterialParams(phi_deg=phi, kref=10.0 ** logk, m_crit=0.64, mu=0.005, manning_n=0.06)
    _, _, h = rollout_peak(R.q0, bed, p, R.GRID, t_final=1800.0, cfl=0.35, bc_x="open", bc_y="open", n_steps=3000)
    s, hb = T.observe(h)
    return jnp.concatenate([s.ravel(), hb])
jvp = jax.jit(lambda dphi, dlk, db: jax.jvp(obs, (phi0, lk0, bed0), (dphi, dlk, db))[1])
Z = jnp.zeros((T.NX, T.NY)); ONE = jnp.ones((T.NX, T.NY))
foot = T.FOOT
blob = np.zeros((T.NX, T.NY)); ii, jj = np.argwhere(foot)[len(np.argwhere(foot)) // 2]
blob[ii, jj] = 1; blob = ndimage.gaussian_filter(blob, 5.0); blob /= blob.max()
blob = jnp.asarray(blob)
print(f"localized perturbations are a gaussian (~100 m) centred on footprint cell ({ii},{jj})")
for name, args in [("phi +1 deg everywhere", (ONE, Z, Z)), ("log10 kref +0.1 everywhere", (Z, 0.1 * ONE, Z)),
                   ("phi +1 deg, local blob", (blob, Z, Z)), ("log10 kref +0.1, local blob", (Z, 0.1 * blob, Z)),
                   ("bed +1 m, local blob", (Z, Z, blob))]:
    d = np.array(jvp(*args))
    ns = int(T.S_OBS.size)
    print(f"{name:30s}: |d obs| = {np.linalg.norm(d):8.3f}   (inundation part {np.linalg.norm(d[:ns]):7.3f}, "
          f"building depths part {np.linalg.norm(d[ns:]):7.3f})", flush=True)
