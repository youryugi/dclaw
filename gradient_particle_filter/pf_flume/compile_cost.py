"""Does the batched Jacobian recompile for every new batch size?"""
import time
import jax
jax.config.update("jax_enable_x64", True)
import run_twin  # noqa: F401  (sets the persistent cache)
import jax.numpy as jnp
import numpy as np
import inference as I
from run_twin import problem

prob, _ = problem("h@1", 0)
rng = np.random.default_rng(0)
Z = rng.logistic(size=(16, 3))
for K in (16, 16, 15, 14, 13, 9, 5, 3, 3):
    t0 = time.time(); prob.resid_jac_batch(Z[:K]); dt = time.time() - t0
    print(f"batched Jacobian, K={K:2d}: {dt:6.1f} s", flush=True)
for K in (256, 256, 200):
    Zb = rng.logistic(size=(K, 3))
    t0 = time.time(); prob.loglik(Zb); print(f"batched forward,  K={K:3d}: {time.time() - t0:6.1f} s", flush=True)
