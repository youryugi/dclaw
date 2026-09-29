"""Cost of the log-likelihood gradient: reverse mode (checkpointed scan) vs
forward mode (jacfwd over the while_loop), 3 and 7 parameters, batched."""
import sys, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import flume_model as M
import inference as I
from run_twin import problem

key = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
prob, _ = problem("full", 0, key)
F = prob.simulate.__self__.F
y, sig = jnp.asarray(prob.y), jnp.asarray(prob.sigma)
G = [0, 1, 2]
def ll_while(th):
    o, _ = F.simulate(th); r = (o[:, :, G] - y) / sig; return -0.5 * jnp.sum(r * r)
def ll_scan(th):
    o, t = F.simulate(th, scan_steps=5000, block=50); r = (o[:, :, G] - y) / sig; return -0.5 * jnp.sum(r * r), t
ps = M.PARAM_SETS[key]
TH = jnp.asarray(ps["truth"] + (ps["hi"] - ps["lo"]) * 0.02 * np.random.default_rng(0).normal(size=(N, len(ps["truth"]))))
def timed(f, x):
    jax.block_until_ready(f(x)); t0 = time.time(); r = jax.block_until_ready(f(x)); return time.time() - t0, r
tf, _ = timed(jax.jit(jax.vmap(ll_while)), TH)
ts, (l2, treach) = timed(jax.jit(jax.vmap(ll_scan)), TH)
tr, g_rev = timed(jax.jit(jax.vmap(jax.grad(lambda th: ll_scan(th)[0]))), TH)
tj, g_fwd = timed(jax.jit(jax.vmap(jax.jacfwd(ll_while))), TH)
print(f"[{key} params, N={N}] per particle: forward (while) {tf / N * 1e3:.1f} ms | forward (scan 5000 steps) {ts / N * 1e3:.1f} ms | "
      f"reverse grad {tr / N * 1e3:.1f} ms ({tr / tf:.1f}x forward) | forward-mode grad {tj / N * 1e3:.1f} ms ({tj / tf:.1f}x forward)")
print(f"   scan reached t_obs for all: {bool((treach > F.t_obs - 1e-9).all())}; max |grad rev - grad fwd| / |grad| = "
      f"{float(jnp.max(jnp.abs(g_rev - g_fwd)) / jnp.max(jnp.abs(g_fwd))):.1e}")
