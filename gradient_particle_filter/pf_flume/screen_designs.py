"""Search for observation designs with a genuinely bimodal posterior.

Simulates N prior draws once (all gauges, h and pb, noiseless) and saves
them; then, for each candidate design (subset of gauges / variables, noise
level), computes the log-likelihood of every draw against the truth's
noiseless observations and asks whether the draws within `delta` of the
best form more than one well-separated cluster in theta.
  python screen_designs.py simulate N     (once; ~2 min per 1000 on the GPU)
  python screen_designs.py analyse
"""
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import flume_model as M
import inference as I
from run_twin import Sub

OUT = Path(__file__).resolve().parent / "results" / "screen_prior.npz"
GAUGE_X = [2, 32, 66]


def simulate(n):
    S = Sub()
    rng = np.random.default_rng(123)
    Z = rng.logistic(size=(n, 3))
    f = jax.jit(jax.vmap(lambda z: S.simulate(I.to_theta(z))[0]))
    O = np.concatenate([np.asarray(f(jnp.asarray(Z[i:i + 256]))) for i in range(0, n, 256)])
    truth = np.asarray(jax.jit(S.simulate)(jnp.asarray(M.THETA_TRUE))[0])
    np.savez(OUT, Z=Z, theta=np.asarray(I.to_theta(jnp.asarray(Z))), O=O, truth=truth)


def clusters(th, span, radius=0.1):
    """Greedy clustering in theta scaled by the prior range: returns centres."""
    X = th / span
    centres = []
    for x in X:
        if all(np.linalg.norm(x - c) > radius for c in centres):
            centres.append(x)
    return np.array(centres) * span


def analyse():
    d = np.load(OUT)
    th, O, truth = d["theta"], d["O"], d["truth"]
    span = I.HI - I.LO
    designs = {}
    for var, vname in ((0, "h"), (1, "pb")):
        for g in range(3):
            designs[f"{vname}@{GAUGE_X[g]}m"] = [(var, g)]
    designs["h@all"] = [(0, g) for g in range(3)]
    designs["h@2+32m"] = [(0, 0), (0, 1)]
    designs["pb@all"] = [(1, g) for g in range(3)]
    designs["all"] = [(v, g) for v in range(2) for g in range(3)]
    sig = {0: 0.01, 1: 0.1}
    print(f"{len(th)} prior draws; for each design: draws within delta of the best log-lik, clustered at 10% of the prior range")
    for name, sel in designs.items():
        for scale in (1, 3):
            ll = np.zeros(len(th))
            for v, g in sel:
                r = (O[:, :, v, g] - truth[None, :, v, g]) / (sig[v] * scale)
                ll += -0.5 * (r * r).sum(1)
            best = ll.max()
            for delta in (5.0, 20.0):
                near = th[ll > best - delta]
                c = clusters(near, span)
                far = [x for x in c if np.linalg.norm((x - M.THETA_TRUE) / span) > 0.2]
                flag = "  <-- far cluster(s)" if far else ""
                print(f"  {name:10s} noise x{scale}  delta {delta:4.0f}: {len(near):4d} draws, {len(c)} cluster(s), "
                      f"far from truth: {[np.round(x, 2).tolist() for x in far][:3]}{flag}")


if __name__ == "__main__":
    if sys.argv[1] == "simulate":
        simulate(int(sys.argv[2]))
    else:
        analyse()
