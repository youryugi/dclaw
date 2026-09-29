"""float32 vs float64: speed, observation and gradient differences (forward mode)."""
import os, sys, time
import jax
X64 = sys.argv[1] == "64"
jax.config.update("jax_enable_x64", X64)
import jax.numpy as jnp
import numpy as np
import flume_model as M
F = M.Flume(dx=0.125)
rng = np.random.default_rng(1)
th = M.THETA_TRUE + rng.normal(0, [1.0, 0.3, 0.2], (64, 3))
f = jax.jit(jax.vmap(F.simulate)); jf = jax.jit(jax.vmap(jax.jacfwd(lambda x: F.simulate(x)[0])))
T = jnp.asarray(th)
jax.block_until_ready(f(T)); t0 = time.time(); o, t = jax.block_until_ready(f(T)); tf = time.time() - t0
jax.block_until_ready(jf(T[:16])); t0 = time.time(); J = jax.block_until_ready(jf(T[:16])); tj = time.time() - t0
np.savez(f"fp{sys.argv[1]}.npz", obs=np.asarray(o, float), J=np.asarray(J, float))
print(f"float{sys.argv[1]}: forward N=64 {tf:.2f} s ({tf/64*1e3:.1f} ms/particle); jacfwd N=16 {tj:.2f} s ({tj/16*1e3:.1f} ms/particle)")
