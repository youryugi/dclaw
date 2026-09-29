"""One-line summaries of JAX runs against the Fortran run:
python summarize.py label=file.npz [label=file.npz ...]"""

import sys

import numpy as np

import compare as C

F = {k: C.fortran_profile(k)[1] for k in range(0, 141)}
GF = {g: C.fortran_gauge(g + 1) for g in range(4)}


def row(name, J):
    if J is None:
        x, H, HM = C.X, [F[k][0] for k in range(141)], [F[k][1] for k in range(141)]
        arr = [GF[g][0][np.argmax(GF[g][1] > 0.01)] for g in range(4)]
        rms = ["  -  "] * 4
    else:
        x, H, HM = J["x"], J["frame_h"], J["frame_hm"]
        gt, gq = J["gauge_t"], J["gauge_q"]
        arr = [gt[np.argmax(gq[:, 0, g] > 0.01)] for g in range(4)]
        rms = [f"{np.sqrt(np.mean((np.interp(GF[g][0], gt, gq[:, 0, g]) - GF[g][1]) ** 2)):.3f}" for g in range(4)]
    dx = x[1] - x[0]
    front = lambda h: x[np.nonzero(h > 0.01)[0].max()]
    mass = lambda k: np.nansum(2700 * HM[k] + 1100 * (H[k] - HM[k])) * dx * 2
    fr = [front(H[int(t / 0.25)]) for t in (1, 2.5, 5, 10, 35)]
    res = [H[140][np.argmin(abs(x - gx))] for gx in (32, 66)]
    print(f"{name:28s} front {' '.join(f'{v:6.1f}' for v in fr)} | arrival {' '.join(f'{a:5.2f}' for a in arr)} | "
          f"RMS h {' '.join(rms)} | h@32,66 35s {res[0]:.3f} {res[1]:.3f} | "
          f"mass 15s {100 * (mass(60) / mass(0) - 1):+.1f}% 35s {100 * (mass(140) / mass(0) - 1):+.1f}%")


if __name__ == "__main__":
    print(" " * 29 + "front at t=1,2.5,5,10,35 s        | arrival x=2,32,66,90        | gauge RMS h (m)")
    row("Fortran (0.0625 m AMR)", None)
    for arg in sys.argv[1:]:
        label, path = arg.split("=")
        row(label, np.load(path))
