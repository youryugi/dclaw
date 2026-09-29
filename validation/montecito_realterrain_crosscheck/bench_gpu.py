import sys, time
import jax
if "--f32" in sys.argv:
    jax.config.update("jax_enable_x64", False)
if "--cpu" in sys.argv:
    jax.config.update("jax_platforms", "cpu")
import jax.numpy as jnp, numpy as np
import run_jax as R
if "--f32" in sys.argv:
    jax.config.update("jax_enable_x64", False)
f = jax.jit(R.run)
th = jnp.array([37.63, float(np.log10(3.348e-12))])
t0 = time.time(); fr, t, q = f(th); fr.block_until_ready(); t1 = time.time()
fr, t, q = f(th); fr.block_until_ready(); t2 = time.time()
peak = np.array(jnp.max(fr[:, 0], axis=0))
from compare import csi
print(f"{jax.devices()[0].platform} dtype={q.dtype}: first call {t1-t0:.1f}s, second {t2-t1:.2f}s, t={float(t):.0f}s, "
      f"CSI={csi(peak)[0]:.3f}, final volume {float(q[0].sum())*400:.0f} m3")
