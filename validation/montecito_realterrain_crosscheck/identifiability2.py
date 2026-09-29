import jax, jax.numpy as jnp, numpy as np
from scipy import ndimage
import identifiability as I
import twin_inversion as T
f = jax.jit(I.obs)
base = np.array(f(I.phi0, I.lk0, I.bed0)); ns = int(T.S_OBS.size)
Z, ONE, blob = I.Z, I.ONE, I.blob
print("actual change of observations for finite perturbations (vs linear prediction amplitude*|JVP|):")
for name, mk, jvp_norm in [("phi everywhere", lambda a: (I.phi0 + a * ONE, I.lk0, I.bed0), 602.4),
                           ("log10 kref everywhere", lambda a: (I.phi0, I.lk0 + a * ONE, I.bed0), 14091.9 / 0.1),
                           ("bed local blob", lambda a: (I.phi0, I.lk0, I.bed0 + a * blob), 806.0)]:
    for a in ([0.001, 0.01, 0.1, 1.0, 3.0] if "phi" in name else [0.001, 0.01, 0.1, 0.3] if "kref" in name else [0.001, 0.01, 0.1, 1.0]):
        d = np.array(f(*mk(a))) - base
        print(f"  {name:22s} step {a:6.3f}: |d obs| = {np.linalg.norm(d):9.3f} (inund {np.linalg.norm(d[:ns]):8.3f}, bldg {np.linalg.norm(d[ns:]):7.3f})"
              f"   linear prediction {a * jvp_norm:10.3f}", flush=True)
