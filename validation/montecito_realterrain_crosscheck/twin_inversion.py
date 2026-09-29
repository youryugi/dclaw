"""Identical-twin test of high-dimensional inversion: recover a spatially
varying basal friction angle phi(x, y) -- one unknown per cell, 23,600 --
from synthetic observations of the same kinds as the real data.

Truth: a smooth phi field (roughly 26-38 deg) on the real Montecito terrain,
same initial ponds, kref fixed at the calibrated value. Observations from a
forward run with the truth: soft inundation over the scored cells (as
compared with the Prescott et al. map) and peak depth at the 2187
buildings in the domain (as `h_observed_m` in buildings.csv). Inversion
starts from a uniform 35 deg field and minimizes observation misfit +
lam * sum |grad phi|^2 with L-BFGS-B; each gradient is one checkpointed
reverse-mode pass (dclaw_jax.solver.rollout_peak) on the GPU.

A second mode, FIELD=bed, instead recovers a correction to the bed
elevation (true: a +2 m mound in the channel at (1190, 2330) m and a -1.5 m hollow
on the lower fan at (1650, 1110) m,
unknowns 23,600 cells, starting from zero) with phi uniform.

Usage: python twin_inversion.py [lam] [noise_m] [maxiter] [phi|bed]
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import run_jax as R
from compare import MASK
from dclaw_jax.model import MaterialParams
from dclaw_jax.solver import rollout_peak

LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
NOISE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
FIELD = sys.argv[4] if len(sys.argv) > 4 else "phi"
TAG = f"{FIELD}_lam{LAM:g}_noise{NOISE:g}"
NX, NY, DX = R.NX, R.NY, R.DX
KREF = 10.0 ** float(np.load(R.HERE / "calibrated_theta.npy")[1])
THRESH, WIDTH = 0.1, 0.03
PHI_LO, PHI_HI, PHI_START = 20.0, 45.0, 35.0

xc = (np.arange(NX) + 0.5) * DX
yc = (np.arange(NY) + 0.5) * DX
Xg, Yg = np.meshgrid(xc, yc, indexing="ij")
PHI_TRUE = np.clip(32.0 + 4.0 * np.sin(2 * np.pi * Yg / 2400.0) + 2.0 * (Xg - 1200.0) / 1200.0, 25.0, 40.0)
PHI_CAL = float(np.load(R.HERE / "calibrated_theta.npy")[0])
bump = lambda x0, y0, r: np.exp(-((Xg - x0) ** 2 + (Yg - y0) ** 2) / (2 * r ** 2))
DB_TRUE = 2.0 * bump(1190.0, 2330.0, 90.0) - 1.5 * bump(1650.0, 1110.0, 110.0)

inputs = np.load(R.HERE / "model/inputs_20m.npz")
B = pd.read_csv("/home/yang/.claude/uploads/61b909df-9a48-4264-a419-b4d8b3ab4b3b/88902f97-buildings.csv")
bi = ((B.x - float(inputs["x_lower"])) // DX).astype(int)
bj = ((B.y - float(inputs["y_lower"])) // DX).astype(int)
inside = (bi >= 0) & (bi < NX) & (bj >= 0) & (bj < NY)
BI, BJ = jnp.asarray(bi[inside].to_numpy()), jnp.asarray(bj[inside].to_numpy())
SCORED = jnp.asarray(MASK)


def peak(x):
    phi, bed = (x, R.bed) if FIELD == "phi" else (PHI_CAL, R.bed + x)
    p = MaterialParams(phi_deg=phi, kref=KREF, m_crit=0.64, mu=0.005, manning_n=0.06)
    _, _, hmax = rollout_peak(R.q0, bed, p, R.GRID, t_final=1800.0, cfl=0.35, bc_x="open", bc_y="open",
                              n_steps=3000, checkpoint_block=50)
    return hmax


def observe(hmax):
    soft = jax.nn.sigmoid((hmax - THRESH) / WIDTH) * SCORED
    return soft, hmax[BI, BJ]


peak_j = jax.jit(peak)
X_TRUE = PHI_TRUE if FIELD == "phi" else DB_TRUE
X_START, X_LO, X_HI = (PHI_START, PHI_LO, PHI_HI) if FIELD == "phi" else (0.0, -5.0, 5.0)
UNIT = "deg" if FIELD == "phi" else "m"
h_true = peak_j(jnp.asarray(X_TRUE))
S_OBS, HB_OBS = observe(h_true)
if NOISE > 0:
    HB_OBS = jnp.maximum(HB_OBS + NOISE * jax.random.normal(jax.random.PRNGKey(1), HB_OBS.shape) * (HB_OBS > 0), 0.0)
FOOT = np.array(h_true) > 0.1                                   # where the true flow went


def misfit_terms(phi):
    s, hb = observe(peak(phi))
    data = jnp.sum((s - S_OBS) ** 2) + jnp.sum((hb - HB_OBS) ** 2)
    reg = jnp.sum(jnp.diff(phi, axis=0) ** 2) + jnp.sum(jnp.diff(phi, axis=1) ** 2)
    return data + LAM * reg, (data, reg)


vg = jax.jit(jax.value_and_grad(misfit_terms, has_aux=True))
hist = []
t_start = time.time()


def fun(x):
    phi = jnp.asarray(x.reshape(NX, NY))
    (J, (data, reg)), g = vg(phi)
    fun.last = (float(data), float(reg))
    return float(J), np.asarray(g, dtype=np.float64).ravel()


def callback(xk):
    phi = xk.reshape(NX, NY)
    err = phi - X_TRUE
    row = dict(it=len(hist) + 1, data=fun.last[0], reg=fun.last[1],
               rmse_foot=float(np.sqrt(np.mean(err[FOOT] ** 2))), rmse_all=float(np.sqrt(np.mean(err ** 2))),
               t=time.time() - t_start)
    hist.append(row)
    if row["it"] % 5 == 0 or row["it"] == 1:
        print(f"it {row['it']:4d}  data misfit {row['data']:10.4f}  reg {row['reg']:9.2f}  "
              f"RMSE: flow footprint {row['rmse_foot']:.3f} {UNIT}, everywhere {row['rmse_all']:.3f} {UNIT}  "
              f"[{row['t']:.0f}s]", flush=True)


if __name__ == "__main__":
    x0 = np.full(NX * NY, X_START)
    e0 = x0.reshape(NX, NY) - X_TRUE
    print(f"{TAG}: {int(FOOT.sum())} cells in the true flow footprint, {len(BI)} buildings; start RMSE "
          f"footprint {np.sqrt(np.mean(e0[FOOT]**2)):.3f}, everywhere {np.sqrt(np.mean(e0**2)):.3f} {UNIT}", flush=True)
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=[(X_LO, X_HI)] * x0.size,
                   callback=callback, options=dict(maxiter=int(sys.argv[3]) if len(sys.argv) > 3 else 150, maxcor=20))
    phi_rec = res.x.reshape(NX, NY)
    h_rec = np.array(peak_j(jnp.asarray(phi_rec)))
    h_start = np.array(peak_j(jnp.full((NX, NY), X_START)))
    print(f"done: {res.message}; {res.nit} iterations, {res.nfev} gradient evaluations, {time.time()-t_start:.0f}s")
    np.savez(R.HERE / f"twin_{TAG}.npz", field=FIELD, x_true=X_TRUE, x_rec=phi_rec, x_start=X_START, h_true=np.array(h_true), h_rec=h_rec,
             h_start=h_start, foot=FOOT, bi=np.array(BI), bj=np.array(BJ),
             hist=np.array([[r["it"], r["data"], r["reg"], r["rmse_foot"], r["rmse_all"], r["t"]] for r in hist]))
