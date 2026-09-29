"""Real data: the 2010 USGS SGM rough-bed aggregate (mean and SD of 8 gate
releases, Iverson et al. 2010 JGR supplement ds03) at x = 32, 66, 90 m.

Unit conventions (the model uses bed_normal = 0, vertical depths):
  - measured h is the flow thickness normal to the bed:  h_normal = h_vertical * cos(theta)
  - measured pbed is the basal pore pressure normal to the bed; for a
    slope-parallel hydrostatic column it is rho_f g cos(theta) h_normal
    = rho_f g h_vertical cos(theta)^2, while the model's hydrostatic pb is
    rho_f g h_vertical:  pbed = pb_model * cos(theta)^2  (an assumption --
    the earlier Fortran comparison compared pb unconverted)
  theta is the local bed slope at each gauge (from the model bed).
Observation times: every obs_dt within each gauge's recorded window
(32 m: 0-20 s, 66 m: 5-25 s, 90 m: 10-30 s); entries outside are masked.
Noise model: sigma^2 = SD_between_experiments^2 + (model-error floor + fraction * |obs|)^2.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import flume_model as M
import inference as I

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "validation" / "usgs_flume_2010_paper2014"))
from parse_ds03 import load_sections  # noqa: E402

ALONG_FLUME = np.array([32.0, 66.0, 90.0])     # gauge positions as published: distance along the flume


def horizontal_positions(along, smooth_m=2.5):
    """The model's x is HORIZONTAL distance from the gate (bed drops 19.1 m over x = 0-32 m: 31 deg
    horizontal, not along-slope); the published gauge positions are along the flume. Map them via
    the arc length of the (lightly smoothed) lidar bed profile: 32 -> 27.50, 66 -> 56.65,
    90 -> 79.19 m. The earlier Fortran validation and flume_crosscheck put the gauges at x = 32,
    66, 90 m -- 4.5, 9.4 and 10.8 m too far downstream. Returns (x, local slope angle in deg)."""
    d = np.load(REPO / "validation/usgs_flume_2017/generated/lidar/2017-05-25_lidar_profiles.npz")
    x, z = d["x_downstream_m"], d["bed_z_m"]; o = np.argsort(x); x, z = x[o], z[o]
    xs = np.linspace(0, 125, 25001); zs = np.interp(xs, x, z)
    n = int(round(smooth_m / (xs[1] - xs[0])))
    k = np.ones(2 * n + 1) / (2 * n + 1)
    zsm = np.convolve(np.pad(zs, n, mode="edge"), k, mode="valid")
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(zsm)))])
    xh = np.interp(along, s, xs)
    slope = np.degrees(np.arctan(-np.interp(xh, xs, np.gradient(zsm, xs))))
    return xh, slope


class RealFlume:
    def __init__(self, dx=0.125, t_obs=30.0, obs_dt=0.5):
        self.gx, self.slope_deg = horizontal_positions(ALONG_FLUME)
        self.F = M.Flume(dx=dx, t_obs=t_obs, obs_dt=obs_dt, gauges_x=self.gx)
        self.cos = np.cos(np.radians(self.slope_deg))
        self.fac = jnp.asarray(np.stack([self.cos, self.cos ** 2]))[None]   # (1, 2 vars, 3 gauges)

    def simulate(self, theta):
        obs, t = self.F.simulate(theta)
        return obs * self.fac, t


def observations(obs_times, model_floor=(0.01, 0.1), model_frac=0.1):
    """y, sigma, mask arrays of shape (n_t, 2, 3): h [m], pbed [kPa]."""
    S = load_sections()
    y = np.zeros((len(obs_times), 2, 3)); sd = np.ones_like(y); mask = np.zeros_like(y)
    for g, x in enumerate((32.0, 66.0, 90.0)):
        tab = S[x]
        inside = (obs_times >= tab["t_s"].min()) & (obs_times <= tab["t_s"].max())
        for v, (key, skey) in enumerate((("h_m", "h_sd_m"), ("pbed_kpa", "pbed_sd_kpa"))):
            y[:, v, g] = np.interp(obs_times, tab["t_s"], tab[key])
            sd[:, v, g] = np.interp(obs_times, tab["t_s"], tab[skey])
            mask[inside, v, g] = 1.0
    floor = np.array(model_floor)[None, :, None]
    sigma = np.sqrt(sd ** 2 + (floor + model_frac * np.abs(y)) ** 2)
    return y, sigma, mask


def problem(params="3", model_floor=(0.01, 0.1), model_frac=0.1, dx=0.125, calib_gauges=(0, 1, 2)):
    """calib_gauges: which gauges enter the likelihood (0, 1, 2 = along-flume 32, 66, 90 m); the
    others are simulated but masked -- held out for prediction."""
    ps = M.use_param_set(params)
    I.set_prior(ps["lo"], ps["hi"])
    R = RealFlume(dx=dx)
    y, sigma, mask = observations(R.F.obs_times, model_floor, model_frac)
    for g in range(3):
        if g not in calib_gauges:
            mask[:, :, g] = 0.0
    return I.Problem(simulate=R.simulate, y=y, sigma=sigma, mask=mask, batch=256), R


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    prob, R = problem()
    print("gauges (along-flume 32/66/90 m) at horizontal x =", np.round(R.gx, 2), "slope (deg)", np.round(R.slope_deg, 1))
    o, t = jax.jit(R.simulate)(jnp.asarray(M.THETA_TRUE))
    o = np.asarray(o); y, m = prob.y, np.asarray(prob._mask)
    for g, x in enumerate((32, 66, 90)):
        for v, name in enumerate(("h (m)", "pbed (kPa)")):
            sel = m[:, v, g] > 0
            yo, mo = y[sel, v, g], o[sel, v, g]
            arr = lambda s: R.F.obs_times[sel][np.argmax(s > (0.01 if v == 0 else 0.1))] if (s > (0.01 if v == 0 else 0.1)).any() else np.nan
            print(f"x={x:2d} m {name:10s}: peak data {yo.max():.3f} model {mo.max():.3f} | arrival data {arr(yo):5.1f} s model {arr(mo):5.1f} s | "
                  f"RMS diff {np.sqrt(np.mean((yo - mo) ** 2)):.3f}")
    print(f"log-lik at the paper's parameters: {float(prob.loglik(I.to_z(M.THETA_TRUE[None]))[0]):.1f} "
          f"({int(m.sum())} observations)")
