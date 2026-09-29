"""Fortran D-Claw (fortran_run/) vs dclaw_jax (jax_run.npz) on the 2010
SGM gate-release case. Fortran AMR frames are put on the JAX 0.0625 m grid
finest level first (the flow is uniform across the flume, so the row
nearest y=0 is used). Writes crosscheck.png and prints a summary.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from clawpack.pyclaw import Solution

HERE = Path(__file__).resolve().parent
RUN = HERE / "fortran_run"
J = np.load(HERE / __import__("os").environ.get("JAX_RUN", "jax_run.npz"))
X = J["x"]
DX = X[1] - X[0]
RHO_S, RHO_F, WIDTH = 2700.0, 1100.0, 2.0


def fortran_profile(frame):
    """(t, h, hm, pb) along X, finest level first."""
    sol = Solution(frame, path=str(RUN), file_format="ascii")
    out = np.full((3, X.size), np.nan)
    for st in sorted(sol.states, key=lambda s: -s.patch.level):
        dim_x, dim_y = st.patch.dimensions
        yc = dim_y.centers
        j = int(np.argmin(np.abs(yc)))
        xe = dim_x.edges
        # JAX cells whose centre lies in this patch cell and are still empty
        idx = np.searchsorted(xe, X) - 1
        inside = (idx >= 0) & (idx < dim_x.num_cells) & np.isnan(out[0])
        for k, comp in enumerate((0, 3, 4)):
            out[k, inside] = st.q[comp, idx[inside], j]
    return sol.t, out


def front(h, thresh=0.01):
    wet = np.nonzero(h > thresh)[0]
    return X[wet.max()] if wet.size else np.nan


def mixture_mass(h, hm):
    return np.nansum(RHO_S * hm + RHO_F * (h - hm)) * DX * WIDTH


def fortran_gauge(gid):
    d = np.loadtxt(RUN / f"gauge{gid:05d}.txt", comments="#")
    return d[:, 1], d[:, 2], d[:, 6]      # t, h, pb


def main():
    frames = sorted(int(p.name[6:]) for p in RUN.glob("fort.q*"))
    ft, fh, fhm, fpb = [], [], [], []
    for f in frames:
        t, (h, hm, pb) = fortran_profile(f)
        ft.append(t); fh.append(h); fhm.append(hm); fpb.append(pb)
    ft = np.array(ft)
    jt, jh, jhm = J["frame_t"], J["frame_h"], J["frame_hm"]
    n = min(len(ft), len(jt))

    d0 = np.nanmax(np.abs(fh[0] - jh[0]))
    print(f"initial h, max |Fortran - JAX| = {d0:.2e} m; volume Fortran {np.nansum(fh[0]) * DX * WIDTH:.4f} "
          f"JAX {jh[0].sum() * DX * WIDTH:.4f} m3")
    M0f, M0j = mixture_mass(fh[0], fhm[0]), mixture_mass(jh[0], jhm[0])
    print(" t (s) | front F / JAX (m) | mixture mass change F / JAX")
    rows = []
    for k in range(n):
        if abs(ft[k] - jt[k]) > 1e-6:
            raise SystemExit("frame times differ")
        rows.append((ft[k], front(fh[k]), front(jh[k]), mixture_mass(fh[k], fhm[k]) / M0f - 1,
                     mixture_mass(jh[k], jhm[k]) / M0j - 1))
    rows = np.array(rows)
    for r in rows:
        if np.isclose(r[0] % 2.5, 0) or np.isclose(r[0], 1.0):
            print(f"{r[0]:6.2f} | {r[1]:7.1f} / {r[2]:7.1f} | {r[3]:+.2e} / {r[4]:+.2e}")

    fig, ax = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    ax[0, 0].plot(rows[:, 0], rows[:, 1], label="Fortran D-Claw (AMR, 1st order)")
    ax[0, 0].plot(rows[:, 0], rows[:, 2], "--", label="dclaw_jax")
    ax[0, 0].set(xlabel="t (s)", ylabel="front x (h > 1 cm) (m)"); ax[0, 0].legend()
    ax[0, 1].plot(rows[:, 0], 100 * rows[:, 3], label="Fortran"); ax[0, 1].plot(rows[:, 0], 100 * rows[:, 4], "--", label="JAX")
    ax[0, 1].set(xlabel="t (s)", ylabel="mixture mass change (%)"); ax[0, 1].legend()
    for i, tt in enumerate((2.0, 5.0)):
        k = int(round(tt / 0.25))
        if k < n:
            a = ax[0, 2 + i]
            a.plot(X, fh[k], label="Fortran"); a.plot(X, jh[k], "--", label="JAX")
            a.set(xlabel="x (m)", ylabel="h (m)", title=f"t = {tt:g} s", xlim=(-5, 120)); a.legend()

    gt, gq = J["gauge_t"], J["gauge_q"]
    for g, gx in enumerate(J["gauges_x"]):
        t_f, h_f, pb_f = fortran_gauge(g + 1)
        a, b = ax[1, g], ax[2, g]
        a.plot(t_f, h_f, label="Fortran"); a.plot(gt, gq[:, 0, g], "--", label="JAX")
        a.set(title=f"gauge x = {gx:g} m", xlabel="t (s)", ylabel="h (m)"); a.legend()
        b.plot(t_f, pb_f / 1e3, label="Fortran"); b.plot(gt, gq[:, 4, g] / 1e3, "--", label="JAX")
        b.set(xlabel="t (s)", ylabel="basal pore pressure (kPa)"); b.legend()
        hi = np.interp(t_f, gt, gq[:, 0, g])
        i_f, i_j = np.argmax(h_f), np.argmax(gq[:, 0, g])
        arr_f = t_f[np.argmax(h_f > 0.01)] if (h_f > 0.01).any() else np.nan
        arr_j = gt[np.argmax(gq[:, 0, g] > 0.01)] if (gq[:, 0, g] > 0.01).any() else np.nan
        print(f"gauge x={gx:5.1f}: arrival F {arr_f:5.2f} / JAX {arr_j:5.2f} s; peak h F {h_f[i_f]:.3f} m @ {t_f[i_f]:.2f} s"
              f" / JAX {gq[i_j, 0, g]:.3f} m @ {gt[i_j]:.2f} s; RMS h diff {np.sqrt(np.mean((hi - h_f) ** 2)):.3f} m")
    fig.savefig(HERE / "crosscheck.png", dpi=110)
    print("wrote crosscheck.png")


if __name__ == "__main__":
    main()
