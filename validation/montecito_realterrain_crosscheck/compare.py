"""Compare Fortran D-Claw (order 1, order 2) and dclaw_jax on identical inputs,
and each against the observed Montecito Creek inundation (Prescott et al. 2023)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fortran_io import frames, read_grid

HERE = Path(__file__).resolve().parent
DX = 20.0
D = np.load(HERE.parents[1] / "dataset/differentiable/montecito_preevent_5m.npz")
inun = D["inun"][0:800, 0:472]
blk = lambda a: a.reshape(200, 4, 118, 4).mean(axis=(1, 3))
OBS = (blk((inun == 1).astype(float)) >= 0.5).T[:, ::-1]
MASK = ~((blk(((inun == 2) | (inun == 3)).astype(float)) >= 0.5).T[:, ::-1])
bed = read_grid(HERE / "model/_output_o2/fort.a0000")[:, :, 0]
yc = (np.arange(OBS.shape[1]) + 0.5) * DX

def csi(peak):
    sim = (peak > 0.1) & MASK; obs = OBS & MASK
    tp, fp, fn = int((sim & obs).sum()), int((sim & ~obs).sum()), int((~sim & obs).sum())
    return tp / (tp + fp + fn), tp, fp, fn


def front(h):
    wet = (h > 0.1).any(axis=0)
    return yc[wet].min() if wet.any() else np.nan




if __name__ == "__main__":
    runs = {}
    for name, d in [("Fortran order 2", "model/_output_o2"), ("Fortran order 1", "model/_output_o1")]:
        fr = frames(HERE / d)
        runs[name] = (np.array([t for t, _ in fr]), np.array([q[:, :, 0] for _, q in fr]))
    jf = np.load(HERE / "jax_frames.npy")
    runs["dclaw_jax"] = (np.arange(jf.shape[0]) * 20.0, jf)


    obs_front = yc[OBS.any(axis=0)].min()
    print(f"observed: {int(OBS.sum())} inundated cells, southernmost at y={obs_front:.0f} m")
    for name, (t, h) in runs.items():
        peak = h.max(axis=0)
        c = csi(peak)
        vols = h.sum(axis=(1, 2)) * DX * DX
        fronts = [front(h[k]) for k in range(len(t))]
        pick = [np.argmin(np.abs(t - T)) for T in (60, 120, 300, 600, 1200, 1800)]
        print(f"\n{name}: CSI={c[0]:.3f} (TP {c[1]} FP {c[2]} FN {c[3]}), inundated cells {int(((peak > 0.1) & MASK).sum())}, "
              f"max peak depth {peak.max():.2f} m")
        print("   t (s):        " + " ".join(f"{t[k]:7.0f}" for k in pick))
        print("   front y (m):  " + " ".join(f"{fronts[k]:7.0f}" for k in pick))
        print("   volume (m3):  " + " ".join(f"{vols[k]:7.0f}" for k in pick))

    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 8))
    gy, gx = np.gradient(bed.T[::-1], DX)
    shade = np.clip(0.5 - 0.5 * (gx - gy) / 0.3, 0, 1)
    for ax, (name, (t, h)) in zip(axes, runs.items()):
        peak = h.max(axis=0)
        sim = (peak > 0.1).T[::-1]; obs = OBS.T[::-1]
        rgb = np.stack([shade] * 3, -1) * 0.6 + 0.3
        rgb[obs & ~sim] = [0.2, 0.4, 1.0]; rgb[sim & ~obs] = [1.0, 0.3, 0.2]; rgb[sim & obs] = [0.6, 0.2, 0.8]
        ax.imshow(rgb, extent=[0, 118 * DX, 0, 200 * DX])
        ax.set_title(f"{name}\nCSI {csi(peak)[0]:.3f}")
    fig.suptitle("peak depth > 0.1 m: blue = observed only, red = simulated only, purple = both")
    plt.tight_layout(); plt.savefig(HERE / "results_crosscheck.png", dpi=80)
