"""Build identical D-Claw (Fortran) inputs for the real-terrain cross-check.

Terrain: Prescott et al. 2023 pre-event 5 m DEM (Zenodo 10.5281/zenodo.7838914),
Montecito Creek sub-domain, nodata filled with the nearest valid elevation.
Initial condition: at each source (Prescott/Kean volumes: Cold Springs
179,000 m3, Hot Springs 52,000 m3) a pond at rest -- every 20 m cell within
150 m of the channel source cell is filled to one common surface level,
solved so the pond volume matches exactly. Solid fraction m0 everywhere.
"""

from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
D = np.load(HERE.parents[1] / "dataset/differentiable/montecito_preevent_5m.npz")
Z5, XLL, YLL = D["z"], float(D["xll"]), float(D["yll"])
NR5 = Z5.shape[0]
ROWS, COLS, F = (0, 800), (0, 472), 4
DX = 5.0 * F
POND_RADIUS_M = 150.0
M0 = 0.512
SOURCES = [("Cold Springs", 3815127.0, 256284.0, 179000.0), ("Hot Springs", 3815100.0, 256878.0, 52000.0)]

valid = np.isfinite(Z5)
Z5F = Z5[tuple(ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True))]
sub5 = Z5F[ROWS[0]:ROWS[1], COLS[0]:COLS[1]]
x_lower = XLL + COLS[0] * 5.0
y_lower = YLL + (NR5 - ROWS[1]) * 5.0
nr20, nc20 = sub5.shape[0] // F, sub5.shape[1] // F
bed20 = sub5.reshape(nr20, F, nc20, F).mean(axis=(1, 3))   # row 0 = north


def write_asc(path, a, dx):
    with open(path, "w") as f:
        f.write(f"ncols {a.shape[1]}\nnrows {a.shape[0]}\nxllcorner {x_lower:.6f}\nyllcorner {y_lower:.6f}\n"
                f"cellsize {dx}\nNODATA_value -9999\n")
        np.savetxt(f, a, fmt="%.4f")


def channel_cell(n, e):
    """Lowest valid 5 m cell within ~60 m below the published source point
    (which sits on the DEM's ragged nodata edge), as a 20 m cell index."""
    c5 = int((e - XLL) / 5.0); r5 = int(NR5 - 1 - (n - YLL) / 5.0)
    r0 = r5 + 12
    win = np.where(valid[r0:r0 + 12, c5 - 12:c5 + 13], Z5[r0:r0 + 12, c5 - 12:c5 + 13], np.inf)
    dr, dc = np.unravel_index(np.argmin(win), win.shape)
    return (r0 + dr - ROWS[0]) // F, (c5 - 12 + dc - COLS[0]) // F


rr, cc = np.meshgrid(np.arange(nr20), np.arange(nc20), indexing="ij")
h0 = np.zeros_like(bed20)
for name, n, e, vol in SOURCES:
    r, c = channel_cell(n, e)
    near = np.hypot(rr - r, cc - c) * DX <= POND_RADIUS_M
    lo, hi = bed20[near].min(), bed20[near].min() + 100.0
    for _ in range(100):
        level = 0.5 * (lo + hi)
        v = np.sum(np.clip(level - bed20[near], 0, None)) * DX * DX
        lo, hi = (level, hi) if v < vol else (lo, level)
    h0[near] = np.maximum(h0[near], np.clip(level - bed20[near], 0, None))
    print(f"{name}: source cell (row {r}, col {c}), bed {bed20[r, c]:.1f} m, pond level {level:.2f} m, "
          f"max depth {np.max(np.clip(level - bed20[near], 0, None)):.2f} m, volume {v:.0f} m3")

write_asc(MODEL / "topo_5m.asc", sub5, 5.0)
write_asc(MODEL / "h0_20m.asc", h0, DX)
write_asc(MODEL / "m0_20m.asc", np.full_like(h0, M0), DX)
np.savez(MODEL / "inputs_20m.npz", bed20=bed20, h0=h0, x_lower=x_lower, y_lower=y_lower, dx=DX)
print(f"domain x [{x_lower:.1f}, {x_lower + nc20 * DX:.1f}], y [{y_lower:.1f}, {y_lower + nr20 * DX:.1f}], "
      f"{nc20} x {nr20} cells at {DX} m; total initial volume {h0.sum() * DX * DX:.0f} m3")
