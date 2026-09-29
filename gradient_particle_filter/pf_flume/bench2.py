import sys, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import flume_model as M
dx, t_obs = float(sys.argv[1]), float(sys.argv[2])
F = M.Flume(dx=dx, t_obs=t_obs)
o, t = jax.jit(F.simulate)(jnp.asarray(M.THETA_TRUE))
print(f"dx={dx} t_obs={t_obs}: peak h per gauge {np.asarray(o[:, 0]).max(0).round(3)} m, peak pb {np.asarray(o[:, 1]).max(0).round(2)} kPa")
rng = np.random.default_rng(0)
for n, fn, lab in [(256, jax.vmap(F.simulate), "forward"), (64, jax.vmap(jax.jacfwd(lambda x: F.simulate(x)[0])), "jacfwd")]:
    f = jax.jit(fn); th = jnp.asarray(M.THETA_TRUE + rng.normal(0, 0.05, (n, 3)))
    jax.block_until_ready(f(th)); t0 = time.time(); jax.block_until_ready(f(th)); d = time.time() - t0
    print(f"  {lab}: N={n} {d:.2f} s = {d / n * 1e3:.1f} ms/particle", flush=True)
