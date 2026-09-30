"""Posterior predictive check on the real 2010 flume data.
  python plot_real.py <results dir> [method file ...]
2 x 3 panels (h, pbed) x (32, 66, 90 m along the flume); gauges not in the likelihood are marked
'held out'. Prints RMS error of the predictive median and 90% coverage (predictive incl. noise)."""
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import flume_model as M
import realdata as RD
from parse_ds03 import load_sections

INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
SERIES = {"robust2": ("#2a78d6", "gradient: robust Laplace"), "smc_rw": ("#eb6834", "no gradient: tempered SMC-RW"),
          "robust2_wide": ("#2a78d6", "gradient: two-stage Laplace"),
          "esmda_k4": ("#1baf7a", "no gradient: ES-MDA (K=4)"), "esmda_k8": ("#1baf7a", "no gradient: ES-MDA (K=8)")}

d_dir = Path(sys.argv[1])
files = sys.argv[2:] or [f.name for f in sorted(d_dir.glob("*.npz"))]
runs = {f: np.load(d_dir / f) for f in files}
first = next(iter(runs.values()))
t, mask, sigma = first["obs_times"], first["mask"], first["sigma"]
calib = [g for g in range(3) if mask[:, :, g].any()]
# paper-parameter run for reference
_, R = RD.problem(params="3", dx=float(first["dx"]) if "dx" in first.files else 0.125,
                  pb_conv=str(first["pb_conv"]) if "pb_conv" in first.files else "cos2")
paper = np.asarray(jax.jit(R.simulate)(jnp.asarray(M.PARAM_SETS["3"]["truth"]))[0])
S = load_sections()

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), constrained_layout=True, facecolor=SURF)
rng = np.random.default_rng(0)
report = []
for g, xa in enumerate((32, 66, 90)):
    tab = S[float(xa)]
    for v, (key, skey, lab) in enumerate((("h_m", "h_sd_m", "flow thickness h (m)"),
                                          ("pbed_kpa", "pbed_sd_kpa", "basal pore pressure (kPa)"))):
        ax = axes[v, g]; ax.set_facecolor(SURF)
        ax.fill_between(tab["t_s"], tab[key] - tab[skey], tab[key] + tab[skey], color=GRID, lw=0,
                        label="data: mean of 8 releases +-1 SD")
        ax.plot(tab["t_s"], tab[key], color=INK, lw=1.2, label="data mean")
        ax.plot(t, paper[:, v, g], color=INK2, lw=1.2, ls="--", label="model, paper parameters")
        inside = (t >= tab["t_s"].min()) & (t <= tab["t_s"].max())
        yobs = np.interp(t, tab["t_s"], tab[key])
        for f, r in runs.items():
            meth = f.split("_N")[0]
            col, name = SERIES.get(meth, ("#1baf7a", meth))
            P = r["pred"][:, :, v, g]
            med = np.median(P, 0)
            if meth.startswith("robust2"):
                lo, hi = np.percentile(P, [5, 95], 0)
                ax.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(t, med, color=col, lw=2.0, label=f"{name}: posterior median" + (" (band: 90%)" if meth.startswith("robust2") else ""))
            # coverage of the 90% predictive interval for an observation (parameter spread + noise model)
            noisy = P + rng.normal(size=P.shape) * sigma[None, :, v, g]
            plo, phi_ = np.percentile(noisy, [5, 95], 0)
            cov = np.mean((yobs[inside] >= plo[inside]) & (yobs[inside] <= phi_[inside]))
            rms = np.sqrt(np.mean((med[inside] - yobs[inside]) ** 2))
            report.append((xa, lab.split(" (")[0], meth, rms, cov))
        ax.set_xlim(tab["t_s"].min(), tab["t_s"].max())
        ax.grid(color=GRID, lw=0.6); [ax.spines[s].set_visible(False) for s in ("top", "right")]
        held = "" if g in calib else "  -- held out (prediction)"
        ax.set_title(f"{xa} m along the flume{held}", color=INK, fontsize=9.5, loc="left")
        ax.set_ylabel(lab); ax.set_xlabel("time since gate release (s)")
axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.savefig(d_dir / "posterior_predictive.png", dpi=130, facecolor=SURF)
print(f"{'gauge':>5s} {'var':22s} {'method':14s} {'RMS of median':>13s} {'90% coverage':>12s}")
for xa, var, meth, rms, cov in report:
    print(f"{xa:5d} {var:22s} {meth:14s} {rms:13.3f} {cov:12.2f}")
print("wrote", d_dir / "posterior_predictive.png")
