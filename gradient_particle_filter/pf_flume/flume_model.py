"""Forward model for the flume twin experiments: theta -> gauge time series.

The 2010 SGM gate release of ../flume_crosscheck (same terrain, pile and
gate, read from its Fortran inputs; run ../flume_crosscheck/fortran_setup.py
once to create them), with the package defaults (Audusse reconstruction,
no alpha floor). theta = (phi_deg, log10 kref, log10 alpha_c); all other
material parameters are the paper's.

Observations: flow depth h and basal pore pressure pb at the four gauges
(x = 2, 32, 66, 90 m) every obs_dt up to t_obs. Time steps are clamped to
land exactly on each observation time, so the observations are smooth
functions of theta (no interpolation between adaptive steps).

`Flume.simulate(theta)` is written for one theta; use jax.vmap for particles.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "differentiable_dclaw"))

import jax
import jax.numpy as jnp
import numpy as np

from dclaw_jax.model import MaterialParams, qfix
from dclaw_jax.solver import Grid, cfl_dt, gate_bed, make_state, step

FORTRAN_INPUTS = HERE.parent / "flume_crosscheck" / "fortran_run"
X_LO, X_HI = -5.0, 130.0
GAUGES_X = np.array([2.0, 32.0, 66.0, 90.0])
GATE_HALF_WIDTH, GATE_H = 0.375, 1.9
# Parameter sets: theta, its truth (the paper's values) and the uniform prior box.
#   "3": phi_deg, log10 kref, log10 alpha_c (everything else at the paper's values)
#   "7": + m0 (initial solid volume fraction), m_crit, p_ratio (initial pore
#        pressure / hydrostatic) and vol (source volume factor: the initial
#        pile's thickness is scaled by it)
PARAM_SETS = {
    "3": dict(names=("phi_deg", "log10 kref", "log10 alpha_c"),
              truth=np.array([40.7, np.log10(5e-9), np.log10(0.024)]),
              lo=np.array([34.0, -10.0, -2.5]), hi=np.array([46.0, -7.0, -1.0])),
    "7": dict(names=("phi_deg", "log10 kref", "log10 alpha_c", "m0", "m_crit", "p_ratio", "vol"),
              truth=np.array([40.7, np.log10(5e-9), np.log10(0.024), 0.62, 0.64, 1.0, 1.0]),
              lo=np.array([34.0, -10.0, -2.5, 0.55, 0.60, 0.6, 0.8]),
              hi=np.array([46.0, -7.0, -1.0, 0.66, 0.68, 1.2, 1.2])),
    # as "7" but with dm = m_crit - m0 in place of m_crit: the 7-parameter posterior is a
    # ridge along which m0 and m_crit move together (only their difference is pinned)
    "7d": dict(names=("phi_deg", "log10 kref", "log10 alpha_c", "m0", "dm", "p_ratio", "vol"),
               truth=np.array([40.7, np.log10(5e-9), np.log10(0.024), 0.62, 0.02, 1.0, 1.0]),
               lo=np.array([34.0, -10.0, -2.5, 0.55, -0.03, 0.6, 0.8]),
               hi=np.array([46.0, -7.0, -1.0, 0.66, 0.08, 1.2, 1.2])),
}
REPARAM_DM = False          # set by use_param_set: theta[4] is dm = m_crit - m0
THETA_TRUE = PARAM_SETS["3"]["truth"]
THETA_NAMES = PARAM_SETS["3"]["names"]


def use_param_set(key):
    """Switch THETA_TRUE / THETA_NAMES (the prior box is set in inference.set_prior)."""
    global THETA_TRUE, THETA_NAMES, REPARAM_DM
    THETA_TRUE, THETA_NAMES = PARAM_SETS[key]["truth"], PARAM_SETS[key]["names"]
    REPARAM_DM = key == "7d"
    return PARAM_SETS[key]


def params(theta):
    if theta.shape[0] >= 7:
        m_crit = theta[3] + theta[4] if REPARAM_DM else theta[4]
    else:
        m_crit = 0.64
    return MaterialParams(rho_s=2700.0, rho_f=1100.0, m_crit=m_crit, mref=0.60, kref=10.0 ** theta[1],
                          phi_deg=theta[0], delta=0.001, mu=0.005, alpha_c=10.0 ** theta[2], sigma_0=1000.0,
                          c1=1.0, gz=9.81, manning_n=0.0)


def _centre_values(name, xc):
    from clawpack.geoclaw import topotools

    topo = topotools.Topography(str(FORTRAN_INPUTS / name), topo_type=3)
    Z = np.asarray(topo.Z)
    assert np.allclose(Z, Z[:1]), f"{name} is not uniform across the flume"
    return np.interp(xc, np.asarray(topo.x), Z[0])


class Flume:
    """Grid, initial state, moving bed and gauge cells at resolution dx."""

    def __init__(self, dx=0.125, t_obs=15.0, obs_dt=0.5, n_steps=None, gauges_x=None):
        self.dx = dx
        nx = int(round((X_HI - X_LO) / dx))
        self.grid = Grid(nx=nx, ny=2, dx=dx, dy=1.0)
        self.x = X_LO + (np.arange(nx) + 0.5) * dx
        basal = _centre_values("basal_topo.tt3", self.x)
        ridge = GATE_H * np.clip((GATE_HALF_WIDTH - np.abs(self.x)) / (GATE_HALF_WIDTH - 0.125), 0.0, 1.0)
        surface = _centre_values("surface_topo.tt3", self.x)
        m = _centre_values("solid_fraction.tt3", self.x)
        h = np.maximum(surface - basal, 0.0)
        h = np.where(h <= 1.0e-3, 0.0, h)
        col = lambda a: jnp.asarray(np.repeat(a[:, None], 2, axis=1))
        rho_f, gz = 1100.0, 9.81
        self.q0 = make_state(col(h), col(0 * h), col(0 * h), col(h * m), col(rho_f * gz * h))
        # the dry cut is applied to the base pile, before any volume scaling, so the
        # initial state is smooth in vol (no cell switching on/off as vol changes)
        self.h_base, self.m_base, self.rho_f, self.gz = col(h), col(m), rho_f, gz
        self.bed = gate_bed(col(basal - ridge), col(ridge), [0.0, 0.85], [0.0, 90.0])
        # gauges: linear interpolation between the two cell centres around each gauge when both are
        # wet (the reference's gauges_module), else the nearer cell. Default: GAUGES_X, which sit on
        # cell edges (weight 1/2 -- identical to the earlier plain average).
        gx = np.asarray(GAUGES_X if gauges_x is None else gauges_x, float)
        right = np.searchsorted(self.x, gx)
        self.gl, self.gr = jnp.asarray(right - 1), jnp.asarray(right)
        self.gw = jnp.asarray((gx - self.x[right - 1]) / dx)          # weight of the right cell
        self.gnear = jnp.asarray(np.where(self.gw >= 0.5, right, right - 1))
        self.gauges_x = gx
        self.t_obs, self.obs_dt = t_obs, obs_dt
        self.n_obs_t = int(round(t_obs / obs_dt))
        self.obs_times = obs_dt * np.arange(1, self.n_obs_t + 1)
        # dt >= ~0.35*dx/(speed + sqrt(g h)) ~ dx/40 for this flow; 2x margin
        self.n_steps = n_steps or int(2 * 40 * t_obs / dx)

    def simulate(self, theta, scan_steps=None, block=50):
        """-> (obs, t_reached): obs has shape (n_obs_t, 2, 4) = (time, [h in m, pb in kPa], gauge).
        scan_steps=None: while_loop to t_obs (forward-mode differentiable only);
        scan_steps=n: n fixed steps with checkpointing (reverse-mode differentiable;
        steps past t_obs are no-ops, so n only has to be large enough -- check t_reached)."""
        p = params(theta)
        if theta.shape[0] >= 7:               # m0, p_ratio, vol are initial-condition parameters
            h = self.h_base * theta[6]
            m = jnp.where(self.h_base > 0, theta[3], 0.0)
            q0 = make_state(h, 0 * h, 0 * h, h * m, theta[5] * self.rho_f * self.gz * h)
        else:
            q0 = self.q0
        q0 = qfix(q0, p)
        grid, dt_obs, t_end = self.grid, self.obs_dt, self.t_obs

        def gauge(q):
            qa, qb = q[:, self.gl, 0], q[:, self.gr, 0]
            wet = (qa[0] > 1e-3) & (qb[0] > 1e-3)
            g = jnp.where(wet[None], (1.0 - self.gw) * qa + self.gw * qb, q[:, self.gnear, 0])
            return jnp.stack([g[0], g[4] / 1e3])

        def body(carry, _):
            t, q, obs = carry
            t_next = (jnp.floor(t / dt_obs + 1e-9) + 1.0) * dt_obs
            dt = jnp.clip(jnp.minimum(jnp.minimum(cfl_dt(q, p, grid, 0.4), t_next - t), t_end - t), 0.0, None)
            q = step(q, self.bed(t), p, grid, dt, "wall", "wall")
            t = t + dt
            k = jnp.round(t / dt_obs).astype(int) - 1
            hit = (jnp.abs(t - (k + 1) * dt_obs) < 1e-9) & (dt > 0)
            obs = obs.at[k].set(jnp.where(hit, gauge(q), obs[k]))
            return (t, q, obs), None

        if scan_steps is not None:
            # fixed-length scan with blocked checkpointing: reverse-mode differentiable
            # (the while_loop below is not), memory ~ (n/block + block) states
            obs0 = jnp.zeros((self.n_obs_t, 2, len(self.gauges_x)), dtype=q0.dtype)
            carry = (jnp.asarray(0.0, q0.dtype), q0, obs0)
            if scan_steps % block:
                raise ValueError("scan_steps must be a multiple of block")
            inner = jax.checkpoint(lambda c: jax.lax.scan(jax.checkpoint(body), c, None, length=block)[0])
            carry, _ = jax.lax.scan(lambda c, _: (inner(c), None), carry, None, length=scan_steps // block)
            t, q, obs = carry
            return obs, t
        obs0 = jnp.zeros((self.n_obs_t, 2, len(self.gauges_x)), dtype=q0.dtype)
        # while_loop, not scan: stops at t_obs (a fixed-length scan wastes the
        # steps left over after the fastest-stepping particle), and forward-mode
        # derivatives (jvp / jacfwd), which is what the samplers use, go through
        # it. Reverse mode does not; use simulate_scan for that.
        t, q, obs = jax.lax.while_loop(lambda c: c[0] < t_end - 1e-9, lambda c: body(c, None)[0],
                                       (jnp.asarray(0.0, q0.dtype), q0, obs0))
        return obs, t
