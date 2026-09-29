"""Side-by-side internal state (Fortran order 1 vs dclaw_jax) at a few times."""
from pathlib import Path
import numpy as np
from fortran_io import frames

HERE = Path(__file__).resolve().parent
RHO_S, RHO_F, G = 2700.0, 1000.0, 9.81
fo = {round(t): np.moveaxis(q[:, :, :5], 2, 0) for t, q in frames(HERE / "model/_output_o1")}
jf = np.load(HERE / "jax_frames_full.npy")
jx = {k * 20: jf[k] for k in range(jf.shape[0])}

def stats(q):
    h = q[0]; wet = h > 0.1
    hs = np.maximum(h, 1e-9)
    u = np.hypot(q[1], q[2]) / hs; m = np.clip(q[3] / hs, 0, 1)
    lith = (RHO_S * m + RHO_F * (1 - m)) * G * hs
    r = q[4] / lith
    mov = wet & (u > 0.1)
    return (f"wet {wet.sum():5d} moving {mov.sum():5d} | |u| p50/p90 {np.median(u[wet]):5.2f}/{np.percentile(u[wet],90):5.2f} "
            f"| m p50 {np.median(m[wet]):.3f} (moving {np.median(m[mov]) if mov.any() else np.nan:.3f}) "
            f"| pb/lith p50 {np.median(r[wet]):.3f} (moving {np.median(r[mov]) if mov.any() else np.nan:.3f}) | max h {h.max():.2f}")

for T in (20, 60, 120, 300, 600):
    print(f"t={T:4d}s  Fortran o1: {stats(fo[T])}")
    print(f"         dclaw_jax : {stats(jx[T])}")
