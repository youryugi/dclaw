"""Mass in mode B (beyond the saddle along A->B) and moments for every run in results/h_1_noise0."""
from pathlib import Path
import numpy as np
A = np.array([42.099, -8.339, -1.680]); B = np.array([42.090, -8.034, -2.302])
d = B - A
R = Path(__file__).resolve().parent / "results" / "h_1_noise0"
print(f"{'run':40s} {'wall':>6s} {'ESS':>7s} {'modes':>5s} {'P(B)':>6s} | mean phi, log kref, log alpha")
for f in sorted(R.glob("*.npz")):
    x = np.load(f)
    lw = x["logw"]; w = np.exp(lw - lw.max()); w /= w.sum()
    s = (x["theta"] - A) @ d / (d @ d)
    pB = w[s > 0.6].sum()
    m = w @ x["theta"]
    nm = int(x["n_modes"]) if "n_modes" in x.files else -1
    print(f"{f.stem:40s} {float(x['wall']):6.0f} {1 / (w ** 2).sum():7.1f} {nm:5d} {pB:6.3f} | {np.round(m, 3)}")

import pickle
for f in sorted(R.glob("ibis_*.pkl")):
    run = pickle.load(open(f, "rb"))
    H = run["hist"]
    print(f"{f.stem:40s} {H[-1]['wall']:6.0f}  events {sum(h['moved'] for h in H)}")
    for k, (th, w, O) in sorted(run["snaps"].items()):
        s = (th - A) @ (B - A) / ((B - A) @ (B - A))
        print(f"     data to t={0.5 * (k + 1):4.1f}s: P(B) {w[s > 0.6].sum():.3f}  mean {np.round(w @ th, 3)}")
