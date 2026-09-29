"""Forward run against REAL 2018 Montecito
Creek field data (Kean et al. 2019, Geosphere; Table 5 + the 717/633
field-survey points in dataset/differentiable/).

Real numbers used (all cited, none invented):
  - Volume: 231,000 m3 -- Kean et al. 2019 Table 5, OBSERVED (not
    model-predicted) sediment volume for Montecito Creek specifically.
  - Mean flow depth target: 1.7 m -- same table (h-bar).
  - Inundation area: 997,000 m2 -- same table.
  - Profile length: ~4900 m -- Kean et al. 2019 Figure 9's x-axis
    (distance downstream from fan apex).
  - Effective width: area / length = ~200 m (derived from the two facts
    above -- NOT independently measured, see README "what's still an
    assumption" note).
  - Fan channel gradient: 4 degrees average -- cited web-search summary
    of Kean et al. 2019 background section ("channels on the alluvial
    fans have an average gradient of 4 deg"), used as a single
    representative slope for the whole domain since our field data
    covers the fan, not the steep (28 deg) source catchment above it.
  - Material parameters (phi_deg, kref): median of the D-Claw
    "vol_class==3" (Barnhart et al. 2021's own "unbiased volume")
    ensemble members for site=="montecito" in model_parameters.csv.
    This is a real, ensemble-informed starting point, not a guess --
    though note it targets a DIFFERENT volume (that ensemble's own
    log_volume median is ~3.5e5 m3, not the 231,000 m3 used here; see
    the printed comparison at the end of this script).

What's still an assumption (flagged, not hidden):
  - Initial pile SHAPE: a simple rectangular block, not a wedge (no
    physical basis is available for Montecito the way figure 11 gave one
    for the 2014 flume paper) at an assumed 4 m initial depth.
  - Effective width (200 m) is derived from area/length, not an
    independent field measurement -- real channels are far narrower
    than the fan they spread across, so this is a single width standing
    in for a system that actually widens downstream.
  - A single 4 deg slope for the whole 4900 m reach, when the real
    profile (Kean et al. fig 9) visibly steepens toward the apex.
  - Instantaneous release, whereas this event was runoff-driven,
    progressive channel bulking during an intense rain burst -- not a
    single mass failure (see this repo's differentiable_dclaw discussion
    for why an inflow boundary condition would be the more faithful
    alternative; not implemented here).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

from dclaw_jax.model import MaterialParams
from dclaw_jax.solver import Grid, make_state, rollout

REPO = Path(__file__).resolve().parents[3]
UPLOADS = Path("/home/yang/.claude/uploads/61b909df-9a48-4264-a419-b4d8b3ab4b3b")

# --- real numbers -----------------------------------------------------
VOLUME_M3 = 231_000.0
AREA_M2 = 997_000.0
PROFILE_LENGTH_M = 4900.0
WIDTH_M = AREA_M2 / PROFILE_LENGTH_M
SLOPE_DEG = 4.0
TARGET_MEAN_DEPTH_M = 1.7
INITIAL_PILE_DEPTH_M = 4.0
MANNING_N = 0.06  # model_parameters.csv `mannings`: 0.06 for every Montecito D-Claw run
PILE_LENGTH_M = VOLUME_M3 / (WIDTH_M * INITIAL_PILE_DEPTH_M)

print(f"effective width from area/length: {WIDTH_M:.1f} m")
print(f"initial pile: {PILE_LENGTH_M:.1f} m long x {WIDTH_M:.1f} m wide x {INITIAL_PILE_DEPTH_M:.1f} m deep"
      f" = {PILE_LENGTH_M*WIDTH_M*INITIAL_PILE_DEPTH_M:.0f} m3")

# --- material parameters: ensemble-informed starting point -------------
params_csv = pd.read_csv(UPLOADS / "6d0f6d84-model_parameters.csv")
mc_good = params_csv[(params_csv.model == "dclaw") & (params_csv.site == "montecito")
                      & (params_csv.vol_class == 3)]
PHI_DEG_0 = float(mc_good.phi_start.median())
KREF_0 = float(10 ** mc_good.permeability_exp.median())
CV_0 = float(mc_good.cv.median())
print(f"\nensemble-informed starting point (site=montecito, vol_class==3, n={len(mc_good)}):")
print(f"  phi_deg={PHI_DEG_0:.2f}  kref={KREF_0:.3e}  cv={CV_0:.3f}")
print(f"  (that ensemble's own median volume: {10**mc_good.log_volume.median():.0f} m3,"
      f" vs the {VOLUME_M3:.0f} m3 used here from Kean et al.'s direct field measurement)")

# --- grid / geometry ----------------------------------------------------
SLOPE_RAD = np.radians(SLOPE_DEG)
X_LOWER, X_UPPER = -400.0, 5000.0
DX = 25.0
nx = int(round((X_UPPER - X_LOWER) / DX))
Y_HALF = WIDTH_M / 2.0
DY = 25.0
ny = max(4, int(round(2 * Y_HALF / DY)))
grid = Grid(nx=nx, ny=ny, dx=DX, dy=DY)
print(f"\ngrid: {nx} x {ny} cells, dx=dy={DX} m, domain x=[{X_LOWER},{X_UPPER}] y=[{-Y_HALF},{Y_HALF}]")

x_centers = (np.arange(nx) + 0.5) * DX + X_LOWER
x2d = jnp.asarray(x_centers)[:, None] * jnp.ones((nx, ny))


def build_initial_state(p: MaterialParams):
    bed = -x2d * jnp.sin(SLOPE_RAD)
    thickness = jnp.where((x2d >= -PILE_LENGTH_M) & (x2d < 0.0), INITIAL_PILE_DEPTH_M, 0.0)
    m = jnp.where(thickness > 0, CV_0, 0.0)
    pb = jnp.where(thickness > 0, p.rho_f * p.gz * thickness, 0.0)
    q0 = make_state(thickness, jnp.zeros_like(thickness), jnp.zeros_like(thickness),
                     thickness * m, pb)
    return q0, bed


def simulate(phi_deg, kref, t_final, n_steps):
    p = MaterialParams(phi_deg=phi_deg, kref=kref, rho_f=1000.0, rho_s=2700.0,
                        m_crit=0.64, mu=0.005, alpha_c=0.01, sigma_0=1000.0,
                        delta=0.01, manning_n=MANNING_N)
    q0, bed = build_initial_state(p)
    qf, t, states, times = rollout(q0, bed, p, grid, t_final=t_final, n_steps=n_steps,
                                    cfl=0.35, bc_x="open", bc_y="wall")
    return qf, t, states, times


def mean_wet_depth(qf):
    h = qf[0]
    wet = h > 0.01
    return jnp.where(jnp.any(wet), jnp.sum(h * wet) / jnp.maximum(jnp.sum(wet), 1), 0.0)


T_FINAL = 1800.0
N_STEPS = 2500  # ~800-1200 needed across the ensemble parameter range with Manning on

print("\n=== forward run at the ensemble-informed starting point (open outflow BC) ===")
qf0, t0, states0, times0 = simulate(PHI_DEG_0, KREF_0, T_FINAL, N_STEPS)
print(f"t reached: {t0:.1f} s")
h0 = qf0[0][:, 0]
wet0 = h0 > 0.01
print(f"wet extent: [{float(x_centers[np.array(wet0)].min()) if np.any(wet0) else float('nan'):.0f},"
      f" {float(x_centers[np.array(wet0)].max()) if np.any(wet0) else float('nan'):.0f}] m")
print(f"mean depth over wet cells: {float(mean_wet_depth(qf0)):.3f} m  (real target: {TARGET_MEAN_DEPTH_M} m)")
print(f"max depth: {float(qf0[0].max()):.3f} m")

# Track front position through real time (real event: 2-4.5 m/s from
# superelevation, Kean et al. 2019 Fig. 13) to get a clean front-speed
# comparison, using the actually-recorded per-step times.
states0_np = np.array(states0[:, 0, :, 0])  # (n_steps, nx) at y=0
times0_np = np.array(times0)
print("\nfront position vs simulation time:")
prev_front, prev_t = None, None
for i in range(0, states0_np.shape[0], max(1, states0_np.shape[0] // 25)):
    h_i = states0_np[i]
    wet_i = h_i > 0.01
    front = float(x_centers[wet_i].max()) if np.any(wet_i) else float("nan")
    ti = float(times0_np[i])
    speed_str = ""
    if prev_front is not None and ti > prev_t:
        speed_str = f"  (avg speed since last row: {(front-prev_front)/(ti-prev_t):.2f} m/s)"
    print(f"  t={ti:7.1f} s: front_x={front:8.1f} m{speed_str}")
    prev_front, prev_t = front, ti
