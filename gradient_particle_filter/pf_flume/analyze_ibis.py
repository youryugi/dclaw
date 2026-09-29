"""IBIS runs: resampling events, diversity, posterior over time, and
predictions of the x = 66 m gauge from early data."""
import pickle
import sys
from pathlib import Path

import numpy as np

import flume_model as M

R = Path(__file__).resolve().parent / "results" / "full_noise0"
OBS_DT = 0.5
for f in sorted(R.glob("ibis_*.pkl")):
    d = pickle.load(open(f, "rb"))
    H = d["hist"]
    ev = [h for h in H if h["moved"]]
    print(f"== {f.stem}: {len(ev)} resample-move events, acceptance {np.round([h['acc'] for h in ev], 2).tolist()}")
    print(f"   final: fwd {H[-1]['n_fwd']} grad {H[-1]['n_grad']} wall {H[-1]['wall']:.0f}s")
    truth = d["truth"]
    for k, (th, w, O) in sorted(d["snaps"].items()):
        uniq = len(np.unique(np.round(th, 10), axis=0))
        m = w @ th; sd = np.sqrt(w @ (th - m) ** 2)
        # predicted x=66 m gauge (index 2): arrival (h > 1 cm) and peak depth, weighted 5/50/95 %
        h66 = O[:, :, 0, 2]
        arr = np.array([OBS_DT * (np.argmax(r > 0.01) + 1) if (r > 0.01).any() else np.nan for r in h66])
        pk = h66.max(1)

        def q(v):
            o = np.argsort(v); c = np.cumsum(w[o])
            return [v[o][np.searchsorted(c, p)] for p in (0.05, 0.5, 0.95)]
        ta = OBS_DT * (np.argmax(truth[:, 0, 2] > 0.01) + 1); tp = truth[:, 0, 2].max()
        print(f"   data to t={OBS_DT * (k + 1):4.1f}s: unique {uniq:4d}/{len(th)}  sd {np.round(sd, 4)}  "
              f"x=66m arrival 5/50/95% {np.round(q(arr), 2)} (truth {ta})  peak h {np.round(q(pk), 3)} (truth {tp:.3f})")
