"""A/B: which change tames the exploding tangents at the bad parameter points?"""
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import flume_model as M
import dclaw_jax.model as model

variant = sys.argv[1]
if variant.startswith("floor"):
    orig = model._material_state
    def patched(q, p):
        s = orig(q, p)
        alpha = jnp.maximum(s["alpha"], 1e-5)
        rho, h = s["rho"], s["h"]
        zeta = 3.0 / (2.0 * alpha * jnp.maximum(h, model.EPS_H)) + p.gz * p.rho_f * (rho - p.rho_f) / (4.0 * rho)
        return dict(s, alpha=alpha, zeta=zeta)
    model._material_state = patched
if variant.startswith("newton"):
    model.N_NEWTON_PB = int(variant[6:])
if variant.startswith("hfloor"):
    model.DILATANCY_H_FLOOR = float(variant[6:])
if variant.startswith("taper"):
    model.DILATANCY_U_TAPER = float(variant[5:])
F = M.Flume(dx=0.125)
rng = np.random.default_rng(1)
th = (M.THETA_TRUE + rng.normal(0, [1.0, 0.3, 0.2], (64, 3)))[:16]
bad = [0, 1, 5, 9, 6, 7]
J = jax.jit(jax.vmap(jax.jacfwd(lambda x: F.simulate(x)[0])))(jnp.asarray(th[bad]))
print(variant, " ".join(f"p{i}:{float(jnp.abs(J[k]).max()):.2g}" for k, i in enumerate(bad)), flush=True)
