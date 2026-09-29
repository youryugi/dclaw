"""Compare sampler results against the reference posterior (argv[1], default
the Laplace IS run): weighted mean error in reference sd, sd ratio, ESS,
wall time and evaluation counts."""
import sys
from pathlib import Path

import numpy as np

import flume_model as M

R = Path(__file__).resolve().parent / "results" / (sys.argv[2] if len(sys.argv) > 2 else "full_noise0")
ref_file = sys.argv[1] if len(sys.argv) > 1 else "laplace_N1024_s0.npz"


def moments(d):
    lw = d["logw"]; w = np.exp(lw - lw.max()); w /= w.sum()
    m = w @ d["theta"]; s = np.sqrt(w @ (d["theta"] - m) ** 2)
    return m, s, 1.0 / (w ** 2).sum()


ref = np.load(R / ref_file)
mr, sr, er = moments(ref)
print(f"reference: {ref_file} (ESS {er:.0f})")
print("  " + "  ".join(f"{n}: {m:.4f} +- {s:.4f}" for n, m, s in zip(M.THETA_NAMES, mr, sr)))
print(f"{'run':28s} {'wall s':>7s} {'fwd':>6s} {'grad':>6s} {'ESS':>7s} | (mean-ref)/ref_sd per param | sd/ref_sd per param")
for f in sorted(R.glob("*.npz")):
    d = np.load(f)
    m, s, e = moments(d)
    print(f"{f.stem:28s} {float(d['wall']):7.0f} {int(d['n_fwd']):6d} {int(d['n_grad']):6d} {e:7.1f} | "
          + " ".join(f"{v:+6.2f}" for v in (m - mr) / sr) + " | " + " ".join(f"{v:5.2f}" for v in s / sr))
