"""Figures for the identical-twin inversion (twin_<tag>.npz)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_jax as R

tag = sys.argv[1] if len(sys.argv) > 1 else "bed_lam0.01_noise0"
d = np.load(R.HERE / f"twin_{tag}.npz")
NX, NY, DX = R.NX, R.NY, R.DX
img = lambda a: np.asarray(a).T[::-1]
ext = [0, NX * DX, 0, NY * DX]
bed = np.array(R.bed)
gy, gx = np.gradient(img(bed), DX)
shade = np.clip(0.5 - 0.5 * (gx - gy) / 0.3, 0, 1)
foot = d["foot"]
field = str(d["field"]); unit = "deg" if field == "phi" else "m"
xt, xr, x0 = d["x_true"], d["x_rec"], float(d["x_start"])
err = xr - xt
name = "friction angle" if field == "phi" else "bed correction"
vmin, vmax = (25, 40) if field == "phi" else (-2, 2)
emax = 5 if field == "phi" else 1.0

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.55])
for k, (arr, title, cmap, lo_, hi_) in enumerate([
        (xt, f"true {name} ({unit})", "viridis" if field == "phi" else "PuOr_r", vmin, vmax),
        (xr, f"recovered ({unit})", "viridis" if field == "phi" else "PuOr_r", vmin, vmax),
        (err, f"recovered - true ({unit})", "RdBu_r", -emax, emax)]):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(shade, cmap="gray", extent=ext)
    im = ax.imshow(img(arr), cmap=cmap, vmin=lo_, vmax=hi_, extent=ext, alpha=0.75)
    ax.contour((np.arange(NX) + 0.5) * DX, (np.arange(NY) + 0.5) * DX, foot.T.astype(float), levels=[0.5],
               colors="k", linewidths=0.8)
    ax.set_title(title); plt.colorbar(im, ax=ax, shrink=0.7)
ax = fig.add_subplot(gs[0, 3])
e_f, e_o = np.abs(err[foot]), np.abs(err[~foot])
ax.hist([e_f, e_o], bins=np.linspace(0, 2 * emax, 33), label=["inside true flow footprint", "outside"], density=True)
ax.set_xlabel(f"|recovered - true| ({unit})"); ax.legend(); ax.set_title(f"where the {name} is recoverable")

h = d["hist"]
ax = fig.add_subplot(gs[1, 0:2])
ax.semilogy(h[:, 0], h[:, 1], label="observation misfit")
ax.set_xlabel("L-BFGS iteration"); ax.set_ylabel("misfit")
ax2 = ax.twinx()
ax2.plot(h[:, 0], h[:, 3], "C1", label="RMSE, flow footprint")
ax2.plot(h[:, 0], h[:, 4], "C2", label="RMSE, everywhere")
ax2.set_ylabel(unit); ax.legend(loc="upper right"); ax2.legend(loc="center right")
ax = fig.add_subplot(gs[1, 2:4])
hb_true = d["h_true"][d["bi"], d["bj"]]
wet = hb_true > 0.01
ax.scatter(hb_true[wet], d["h_start"][d["bi"], d["bj"]][wet], s=8, alpha=0.5, label=f"start ({x0:g} {unit} everywhere)")
ax.scatter(hb_true[wet], d["h_rec"][d["bi"], d["bj"]][wet], s=8, alpha=0.7, label="recovered")
m = max(hb_true.max(), 0.5)
ax.plot([0, m], [0, m], "k--", lw=0.8)
ax.set_xlabel("true peak depth at building (m)"); ax.set_ylabel("simulated (m)"); ax.legend()
ax.set_title("peak depth at buildings reached by the true flow")
fig.suptitle(f"identical-twin inversion of the {name}, 23,600 unknowns ({tag}); black line = true flow footprint")
plt.tight_layout(); plt.savefig(R.HERE / f"results_twin_{tag}.png", dpi=80)
print(f"RMSE: footprint {np.sqrt(np.mean(err[foot]**2)):.3f} {unit} (start {np.sqrt(np.mean((x0-xt[foot])**2)):.3f}), "
      f"outside {np.sqrt(np.mean(err[~foot]**2)):.3f} {unit} (start {np.sqrt(np.mean((x0-xt[~foot])**2)):.3f})")
for lo, hi in [(0, 0.5), (0.5, 2), (2, 99)]:
    sel = foot & (d["h_true"] >= lo) & (d["h_true"] < hi)
    if sel.any():
        print(f"  footprint cells with true peak depth {lo}-{hi} m: n={int(sel.sum())}, RMSE {np.sqrt(np.mean(err[sel]**2)):.3f} {unit}")
