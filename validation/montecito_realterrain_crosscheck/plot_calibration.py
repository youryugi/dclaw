"""Observed vs dclaw_jax at the ensemble-median start and at the calibrated parameters."""
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare import MASK, OBS, bed, csi
from run_jax import DX, run

cases = [("start (ensemble median)", [37.63, float(np.log10(3.348e-12))]),
         ("calibrated", list(np.load("calibrated_theta.npy")))]
gy, gx = np.gradient(bed.T[::-1], DX)
shade = np.clip(0.5 - 0.5 * (gx - gy) / 0.3, 0, 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 8))
for ax, (name, th) in zip(axes, cases):
    fr, _, _ = jax.jit(run)(jnp.array(th))
    peak = np.array(jnp.max(fr[:, 0], axis=0))
    c = csi(peak)
    sim = (peak > 0.1).T[::-1]; obs = OBS.T[::-1]
    rgb = np.stack([shade] * 3, -1) * 0.6 + 0.3
    rgb[obs & ~sim] = [0.2, 0.4, 1.0]; rgb[sim & ~obs] = [1.0, 0.3, 0.2]; rgb[sim & obs] = [0.6, 0.2, 0.8]
    ax.imshow(rgb, extent=[0, OBS.shape[0] * DX, 0, OBS.shape[1] * DX])
    ax.set_title(f"{name}\nphi={th[0]:.2f} deg, kref={10**th[1]:.2e} m2\nCSI {c[0]:.3f} (TP {c[1]} FP {c[2]} FN {c[3]})")
    print(f"{name}: phi={th[0]:.2f} log10k={th[1]:.3f} CSI={c[0]:.3f} TP={c[1]} FP={c[2]} FN={c[3]}")
fig.suptitle("dclaw_jax vs observed inundation: blue = observed only, red = simulated only, purple = both")
plt.tight_layout(); plt.savefig("results_calibration.png", dpi=80)
