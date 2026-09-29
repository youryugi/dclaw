"""Map of dL/d(bed elevation): L = summed peak depth at damaged buildings."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import run_jax as R
import sensitivity as SEN

g = np.load(R.HERE / "sens_dL_dbed.npy")
bed = np.array(R.bed)
gy, gx = np.gradient(bed.T[::-1], R.DX)
shade = np.clip(0.5 - 0.5 * (gx - gy) / 0.3, 0, 1)
ext = [0, R.NX * R.DX, 0, R.NY * R.DX]
lim = np.percentile(np.abs(g), 99.5)
fig, ax = plt.subplots(figsize=(7, 10))
ax.imshow(shade, cmap="gray", extent=ext, alpha=0.8)
m = np.ma.masked_where(np.abs(g.T[::-1]) < lim * 0.02, g.T[::-1])
im = ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext, alpha=0.9)
ax.plot((np.array(SEN.BI) + 0.5) * R.DX, (np.array(SEN.BJ) + 0.5) * R.DX, "k.", ms=2, label="damaged buildings")
plt.colorbar(im, ax=ax, shrink=0.6, label="dL / d(bed elevation)  [m of summed depth per m]")
ax.set_title("Where raising the ground changes flow depth at damaged buildings\n"
             "blue: raising here lowers it (e.g. a berm); red: raising here raises it")
ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig(R.HERE / "results_sensitivity.png", dpi=90)
i, j = np.unravel_index(np.argsort(g.ravel())[:5], g.shape)
print("five most protective cells to raise (x, y in m from domain corner, dL/db):",
      [(int((a + .5) * R.DX), int((b + .5) * R.DX), round(float(g[a, b]), 3)) for a, b in zip(i, j)])
