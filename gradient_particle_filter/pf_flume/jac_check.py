"""Forward-mode Jacobian of the gauge observations vs central finite differences
at the 16 parameter points of fp32_test.py (float64)."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import flume_model as M
F = M.Flume(dx=0.125)
rng = np.random.default_rng(1)
th = (M.THETA_TRUE + rng.normal(0, [1.0, 0.3, 0.2], (64, 3)))[:16]
J = np.load("fp64.npz")["J"]                                  # (16, n_t, 2, 4, 3)
f = jax.jit(lambda x: F.simulate(x)[0])
for i in range(16):
    mx = np.abs(J[i]).reshape(-1, 3).max(0)
    line = f"particle {i:2d} theta {np.round(th[i], 2)} max|dobs/dtheta| {mx}"
    if mx.max() > 1e3:
        k = int(np.argmax(mx))
        for e in (1e-4, 1e-6):
            d = np.zeros(3); d[k] = e
            fd = (np.asarray(f(jnp.asarray(th[i] + d))) - np.asarray(f(jnp.asarray(th[i] - d)))) / (2 * e)
            j = np.unravel_index(np.abs(J[i][..., k]).argmax(), J[i][..., k].shape)
            line += f"\n     param {k} entry {j}: AD {J[i][j + (k,)]:.3e}  FD eps={e:g} {fd[j]:.3e}   |FD| max over obs {np.abs(fd).max():.2e}"
    print(line, flush=True)
