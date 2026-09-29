"""A minimal, fully-differentiable finite-volume solver for the digclaw
five-equation system on a single uniform rectangular grid.

Numerics (deliberately simple, see this package's README for the exact
justification of each simplification relative to the Fortran reference):
  - h, hu, hv: conservative update with an HLL flux on hydrostatically reconstructed interface states (Audusse et
    al. 2004, SIAM J. Sci. Comput. 25(6); Chen & Noelle 2017 optional, see _hr_fluxes), which carries the bed slope
    (well-balanced: still water over uneven terrain stays still). With
    kappa=1 the momentum flux has no direct pb-gradient term (eq B1 of
    Iverson & George 2014), so this is a standard shallow-water flux.
  - hm: conservative, carried with the h flux at the upwind cell's m.
  - pb: advected in primitive (non-conservative) form using upwind
    differencing, from eq 2.1(a,e); the paper describes pb as carried
    across the linearly degenerate (contact) characteristic field (eq
    2.26).
  - Explicit SSP-RK2 time stepping. Rollout uses a fixed number of
    lax.scan steps with a CFL-limited, differentiable adaptive step size
    (NOT lax.while_loop, whose trip count cannot depend on differentiated
    inputs under reverse-mode autodiff).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .model import (MaterialParams, bulk_density, chi, dilatancy_step, friction_step, pb_relaxation_step,
                    manning_step, primitive, qfix)


class Grid(NamedTuple):
    nx: int
    ny: int
    dx: float
    dy: float


def make_state(h, hu, hv, hm, pb) -> jnp.ndarray:
    return jnp.stack([h, hu, hv, hm, pb])


def _pad_axis(q: jnp.ndarray, axis: int, bc: str) -> jnp.ndarray:
    """Add one ghost cell on each side of spatial `axis` (0=x, 1=y). `q`
    has shape (5, nx, ny); ghost cells are inserted along axis+1.
    """

    a = axis + 1
    if bc == "periodic":
        left = jnp.take(q, jnp.array([-1]), axis=a)
        right = jnp.take(q, jnp.array([0]), axis=a)
    elif bc in ("open", "wall"):
        left = jnp.take(q, jnp.array([0]), axis=a)
        right = jnp.take(q, jnp.array([-1]), axis=a)
        if bc == "wall":
            comp = 1 if axis == 0 else 2  # hu flips at x-walls, hv at y-walls
            left = left.at[comp].multiply(-1.0)
            right = right.at[comp].multiply(-1.0)
    else:
        raise ValueError(f"unknown bc {bc!r}")
    return jnp.concatenate([left, q, right], axis=a)


def flux_x(q: jnp.ndarray, p: MaterialParams) -> jnp.ndarray:
    h, u, v, m, pb, rho = primitive(q, p)
    hu = q[1]
    z = jnp.zeros_like(h)
    return jnp.stack([hu, hu * u + 0.5 * p.gz * h**2, hu * v, z, z])


def flux_y(q: jnp.ndarray, p: MaterialParams) -> jnp.ndarray:
    h, u, v, m, pb, rho = primitive(q, p)
    hv = q[2]
    z = jnp.zeros_like(h)
    return jnp.stack([hv, hv * u, hv * v + 0.5 * p.gz * h**2, z, z])


def wave_speed(q: jnp.ndarray, p: MaterialParams) -> jnp.ndarray:
    """|u| + sqrt(gz*h). Dry cells (h=0, most of any real domain) make
    the sqrt argument exactly 0; the forward value is fine there, but
    sqrt's derivative 1/(2*sqrt(x)) is infinite at x=0, and this feeds
    the CFL dt (and the same form is used in _hll), so an unguarded sqrt would
    NaN essentially every gradient through the solver. Same reasoning
    as the SQRT_EPS floor in model.m_equilibrium.
    """
    h, u, v, m, pb, rho = primitive(q, p)
    SQRT_EPS = 1.0e-12
    c = jnp.sqrt(p.gz * jnp.maximum(h, 0.0) + SQRT_EPS)
    return jnp.sqrt(u**2 + v**2 + SQRT_EPS) + c


def _hll(qL: jnp.ndarray, qR: jnp.ndarray, p: MaterialParams, flux_fn, axis: int) -> jnp.ndarray:
    """HLL flux with Einfeldt-type wave-speed bounds -- less diffusive
    than Rusanov, which applies the largest wave speed to every
    component. (On the real Montecito DEM this alone changed little; the
    big differences from the Fortran reference were well-balancing and
    the pb compression term, see _hr_fluxes and rhs.)"""
    hL, uL, vL = primitive(qL, p)[:3]
    hR, uR, vR = primitive(qR, p)[:3]
    unL, unR = (uL, uR) if axis == 0 else (vL, vR)
    SQRT_EPS = 1.0e-12
    cL = jnp.sqrt(p.gz * jnp.maximum(hL, 0.0) + SQRT_EPS)
    cR = jnp.sqrt(p.gz * jnp.maximum(hR, 0.0) + SQRT_EPS)
    sL = jnp.minimum(jnp.minimum(unL - cL, unR - cR), 0.0)
    sR = jnp.maximum(jnp.maximum(unL + cL, unR + cR), 0.0)
    fL, fR = flux_fn(qL, p), flux_fn(qR, p)
    return (sR * fL - sL * fR + sL * sR * (qR - qL)) / jnp.maximum(sR - sL, SQRT_EPS)


def _tracer_flux(f_h: jnp.ndarray, qL: jnp.ndarray, qR: jnp.ndarray, p: MaterialParams) -> jnp.ndarray:
    """Solid-volume flux carried with the h mass flux, at the upwind
    cell's m. Unlike a Rusanov flux of hm itself, this adds no extra
    diffusion of m across a stationary contact (h flux is 0 there)."""
    mL = primitive(qL, p)[3]
    mR = primitive(qR, p)[3]
    return f_h * jnp.where(f_h > 0.0, mL, mR)


# Hydrostatic reconstruction used by _hr_fluxes: "audusse" (default) or
# "chen_noelle". See _hr_fluxes for what each costs.
RECONSTRUCTION = "audusse"


def _hr_fluxes(q: jnp.ndarray, b: jnp.ndarray, axis: int, bc: str, p: MaterialParams, flux_fn):
    """Hydrostatic reconstruction. Returns the interface fluxes as seen
    from the left cell and from the right cell (they differ only in the
    normal-momentum row), shape (5, n+1, ...) along `axis`.

    RECONSTRUCTION = "audusse" (Audusse et al. 2004, SIAM J. Sci. Comput.
    25(6)): interface bed max(bL, bR), depths max(h + b - max(bL, bR), 0).
    RECONSTRUCTION = "chen_noelle" (Chen & Noelle 2017, SIAM J. Numer.
    Anal. 55(2); IGPM report 440, eqs 2.15-2.18): interface bed
    z = min(max(bL, bR), min(wL, wR)) with w = h + b, depths
    h* = min(w - z, h), bed-slope source g*(h + h*)/2*(z - b_cell) per
    side. The two are identical wherever the interface is fully wet
    (min(wL, wR) > max(bL, bR)), so both keep lake at rest, wet/dry
    shorelines and positivity (tests/test_lake_at_rest.py passes with
    either).

    They differ for a layer thinner than the bed step between two cells
    running downhill. Audusse's source there is g*h^2/2 per cell instead
    of g*h*db -- a fraction h/(2 db) of the gravity driving force, 27% for
    a 2 cm layer on the 31 deg USGS flume at 6.25 cm cells (db = 3.8 cm)
    -- which pins thin tails to steep slopes and slows thin fronts; the
    error is first order in dx. Chen-Noelle gives 1 - h/(2 db)
    (tests/test_steep_slope.py). Against the Fortran reference on that
    flume (gradient_particle_filter/flume_crosscheck) it removes the
    pinned 2 cm tails and cuts the depth error at 2 m below the gate from
    0.044 to 0.009 m RMS, but it is not the default: the thin films it
    sets moving keep crossing _clip_dry's 1-2 mm ramp, which makes losses
    that integrate over the flow or depend on a thin front rough at small
    scales (autodiff vs finite differences off by 2-10x down to eps=1e-7,
    where Audusse matches to 6 digits), and _clip_dry then removes 13-18%
    of the mixture mass in 35 s (Fortran: 7%, Audusse: 8%). Raising the
    dry threshold to 2 mm restores exact gradients but removes 30%. A
    mass-conserving, differentiable dry-cell treatment would be needed
    to have both.

    Without any well-balanced reconstruction (an earlier version: Rusanov
    on raw h plus a centred -g*h*db/dx source) diffusing h across a
    sloping bed pushed material out of channels onto their banks. On the
    real Montecito DEM at 20 m that spread the flow across the upper fan
    and stopped it ~2.6 km short, where the Fortran reference on
    identical inputs followed the channel to the coast.
    """
    a = axis + 1
    qp = _pad_axis(q, axis, bc)
    bp = _pad_axis(jnp.broadcast_to(b, q.shape), axis, bc)[0]
    sl = lambda arr, lo, hi, ax: jax.lax.slice_in_dim(arr, lo, arr.shape[ax] + hi if hi <= 0 else hi, axis=ax)
    qL, qR = sl(qp, 0, -1, a), sl(qp, 1, 0, a)
    bL, bR = sl(bp, 0, -1, axis), sl(bp, 1, 0, axis)
    hL, hR = qL[0], qR[0]
    if RECONSTRUCTION == "chen_noelle":
        wL, wR = hL + bL, hR + bR
        zf = jnp.minimum(jnp.maximum(bL, bR), jnp.minimum(wL, wR))
        hLs = jnp.minimum(wL - zf, hL)            # >= 0 since zf <= min(wL, wR)
        hRs = jnp.minimum(wR - zf, hR)
    elif RECONSTRUCTION == "audusse":
        zf = jnp.maximum(bL, bR)
        hLs = jnp.maximum(hL + bL - zf, 0.0)
        hRs = jnp.maximum(hR + bR - zf, 0.0)
    else:
        raise ValueError(f"unknown RECONSTRUCTION {RECONSTRUCTION!r}")
    qLs = qL * (hLs / jnp.maximum(hL, 1.0e-12))[None]
    qRs = qR * (hRs / jnp.maximum(hR, 1.0e-12))[None]
    f = _hll(qLs, qRs, p, flux_fn, axis)
    f = f.at[3].set(_tracer_flux(f[0], qL, qR, p))
    mom = 1 + axis
    if RECONSTRUCTION == "chen_noelle":
        fL = f.at[mom].add(0.5 * p.gz * (hL + hLs) * (zf - bL))
        fR = f.at[mom].add(0.5 * p.gz * (hR + hRs) * (zf - bR))
    else:
        fL = f.at[mom].add(0.5 * p.gz * (hL**2 - hLs**2))
        fR = f.at[mom].add(0.5 * p.gz * (hR**2 - hRs**2))
    return fL, fR


def _upwind_grad(field: jnp.ndarray, vel: jnp.ndarray, axis: int, dxi: float, bc: str) -> jnp.ndarray:
    padded = _pad_axis(field[None], axis, bc)[0]
    fwd = (jnp.roll(padded, -1, axis=axis) - padded) / dxi
    bwd = (padded - jnp.roll(padded, 1, axis=axis)) / dxi
    interior = tuple(slice(1, -1) if i == axis else slice(None) for i in range(2))
    return jnp.where(vel >= 0.0, bwd[interior], fwd[interior])


def rhs(q: jnp.ndarray, b: jnp.ndarray, p: MaterialParams, grid: Grid, bc_x: str, bc_y: str) -> jnp.ndarray:
    h, u, v, m, pb, rho = primitive(q, p)

    # --- conservative rows: h, hu, hv, hm via well-balanced fluxes ---
    fxL, fxR = _hr_fluxes(q, b, 0, bc_x, p, flux_x)
    dfx = (fxL[:, 1:, :] - fxR[:, :-1, :]) / grid.dx
    fyL, fyR = _hr_fluxes(q, b, 1, bc_y, p, flux_y)
    dfy = (fyL[:, :, 1:] - fyR[:, :, :-1]) / grid.dy

    dh = -dfx[0] - dfy[0]
    dhu = -dfx[1] - dfy[1]
    dhv = -dfx[2] - dfy[2]

    # hm (solid volume) is conservative -- see _tracer_flux. Advecting m
    # in primitive form instead (an earlier version)
    # gave newly wetted cells at a wet/dry edge far too little solid
    # (m=0.08 next to an m=0.512 pile on the very first step), and those
    # fake-dilute cells sit right on the reference's steep m~0.1 friction
    # taper, where they stick-slipped and wrecked every gradient.
    dhm = -dfx[3] - dfy[3]

    # --- pb: upwind advection in primitive form plus the compression term
    # gamma*rho*gz*h*div(u) (gamma = chi with kappa=1), as in the
    # reference's Riemann solver (riemannsolvers_dclaw.f, del(4)); its
    # sources are split steps. h*div(u) is taken from the actual numerical
    # mass change, h*div(u) = div(hu) - u.grad(h) = -dh/dt - u.grad(h),
    # which is how the reference ties the pb wave to the h wave: a cell
    # that receives dh gains gamma*rho*gz*dh of pore pressure. (An earlier
    # version used a centred div(u) and was missing rho -- ~1900x too
    # small -- so piled-up material lost its liquefaction and stopped:
    # on the real Montecito DEM the Fortran run stayed at pb/lith ~1 and
    # reached the coast while this one fell to ~0.2 and stalled.)
    dpb_dx = _upwind_grad(pb, u, 0, grid.dx, bc_x)
    dpb_dy = _upwind_grad(pb, v, 1, grid.dy, bc_y)
    h_adv = u * _upwind_grad(h, u, 0, grid.dx, bc_x) + v * _upwind_grad(h, v, 1, grid.dy, bc_y)
    dpb = chi(rho, p) * rho * p.gz * (dh + h_adv) - (u * dpb_dx + v * dpb_dy)

    return jnp.stack([dh, dhu, dhv, dhm, dpb])


def cfl_dt(q: jnp.ndarray, p: MaterialParams, grid: Grid, cfl: float) -> jnp.ndarray:
    smax = jnp.max(wave_speed(q, p))
    return cfl * jnp.minimum(grid.dx, grid.dy) / jnp.maximum(smax, 1.0e-6)


DRY_TOLERANCE = 1.0e-3  # matches geo_data.dry_tolerance used throughout this repo


def _clip_dry(q: jnp.ndarray) -> jnp.ndarray:
    """Dry-cell handling, made continuous. The reference hard-zeroes the
    whole state wherever h <= dry_tolerance (src2.f90/qfix). Here the
    whole state is instead scaled by a smoothstep weight going from 0 at
    1x to 1 at 2x the dry tolerance: same end result for thin films
    (they are removed), but continuous. A hard reset was the one
    discontinuous operator in the scheme -- every time some cell crossed
    the threshold a step earlier or later the whole solution jumped,
    which stalled Gauss-Newton on the full-scale Montecito problem even
    with exact gradients. Keeping h and only ramping momentum/pressure
    (an intermediate version) was worse: residual films of h~1e-7 m (down
    to 1e-303) persisted wherever flow had passed, and the 1/h, 1/h^2
    source terms there amplified sensitivities ~4x per step on real
    terrain.
    """

    h = jnp.maximum(q[0], 0.0)
    s = jnp.clip((h - DRY_TOLERANCE) / DRY_TOLERANCE, 0.0, 1.0)
    w = s * s * (3.0 - 2.0 * s)
    return q.at[0].set(h) * w[None]


def step(q: jnp.ndarray, b: jnp.ndarray, p: MaterialParams, grid: Grid, dt: jnp.ndarray, bc_x: str, bc_y: str) -> jnp.ndarray:
    """One step: explicit SSP-RK2 (Heun) for the transport part (flux
    divergence, slope, m/pb advection), then three Godunov-split
    reaction stages in the reference's src2.f90 order, all stiff:
    basal friction (model.friction_step -- must be able to stop a cell
    without reversing it), pb relaxation (model.pb_relaxation_step --
    blows up within a handful of explicit steps for realistic
    permeability/compressibility), and the D-driven volume change of
    h/hu/hv/hm (model.dilatancy_step, using the relaxed pb), then Manning
    friction (model.manning_step, a no-op unless manning_n > 0).
    """

    k1 = rhs(q, b, p, grid, bc_x, bc_y)
    q1 = _clip_dry(q + dt * k1)
    k2 = rhs(q1, b, p, grid, bc_x, bc_y)
    q_transport = _clip_dry(q + 0.5 * dt * (k1 + k2))

    # qfix at the same three points the reference's src2/mp_update call it.
    q_fric = friction_step(qfix(q_transport, p), p, dt)
    q_relaxed = qfix(q_fric.at[4].set(pb_relaxation_step(q_fric, p, dt)), p)
    return _clip_dry(qfix(manning_step(dilatancy_step(q_relaxed, p, dt), p, dt), p))


def _bed_at(b, t):
    return b(t) if callable(b) else b


def gate_bed(b_base: jnp.ndarray, gate_shape: jnp.ndarray, gate_t, gate_deg):
    """Moving bed for a headgate release, in the same representation the
    reference's flume setups use (validation/usgs_flume_2010_paper2014/
    model/setinput.py, write_gate_dtopo): the closed gate is a ridge
    `gate_shape` on top of `b_base`, and as the gate swings open to
    angle(t) -- piecewise linear through (gate_t, gate_deg), held at the
    end values outside them -- the ridge is lowered to cos(angle) of its
    height. Returns b(t) for `rollout` / `rollout_peak`; differentiable
    with respect to all four inputs.
    """

    gate_t = jnp.asarray(gate_t)
    gate_deg = jnp.asarray(gate_deg)

    def b(t):
        angle = jnp.interp(t, gate_t, gate_deg, left=gate_deg[0], right=gate_deg[-1])
        return b_base + jnp.cos(jnp.deg2rad(angle)) * gate_shape

    return b


def rollout(q0: jnp.ndarray, b, p: MaterialParams, grid: Grid,
            t_final: float, cfl: float = 0.4, bc_x: str = "wall", bc_y: str = "wall",
            n_steps: int = 2000):
    """Advance from q0 towards t_final over a *fixed* number of adaptive-dt
    steps (lax.scan, so this is reverse-mode differentiable end to end).
    Once t reaches t_final, dt is clamped to 0 and the state stops
    changing, so n_steps only needs to be large enough, not exact.
    Returns (q_final, t_reached, states, times) where `states`/`times`
    have one entry per scan step for inspection/plotting (times are the
    simulation time *after* that step, so they pair up directly with
    states[i]).

    `b` is either a fixed bed array or a function t -> bed array (a
    moving bed, the reference's dtopo; see `gate_bed`). A moving bed is
    evaluated once per step, at the start of the step, and h is left
    unchanged when it moves, as GeoClaw does.
    """

    def body(carry, _):
        t, q = carry
        dt_cfl = cfl_dt(q, p, grid, cfl)
        dt = jnp.clip(jnp.minimum(dt_cfl, t_final - t), 0.0, None)
        q_new = step(q, _bed_at(b, t), p, grid, dt, bc_x, bc_y)
        t_new = t + dt
        return (t_new, q_new), (q_new, t_new)

    (t_final_reached, q_final), (states, times) = jax.lax.scan(
        body, (jnp.asarray(0.0), q0), xs=None, length=n_steps
    )
    return q_final, t_final_reached, states, times


def rollout_peak(q0: jnp.ndarray, b, p: MaterialParams, grid: Grid,
                 t_final: float, cfl: float = 0.4, bc_x: str = "wall", bc_y: str = "wall",
                 n_steps: int = 2000, checkpoint_block: int | None = None):
    """Like `rollout`, but keeps only the running per-cell peak depth
    instead of every state -- the form reverse-mode differentiation with
    respect to many inputs (e.g. the whole bed b) needs. Returns
    (q_final, t_reached, h_peak). `b` may be a function of t, as in
    `rollout`.

    With `checkpoint_block=k` (n_steps must be a multiple of k) time is
    split into blocks of k steps: the backward pass stores only the carry
    at block boundaries and, inside each block, only the carry at each
    step (step internals are recomputed), so memory is ~(n_steps/k + k)
    states rather than every intermediate of every step.
    """

    def body(carry, _):
        t, q, hmax = carry
        dt = jnp.clip(jnp.minimum(cfl_dt(q, p, grid, cfl), t_final - t), 0.0, None)
        q = step(q, _bed_at(b, t), p, grid, dt, bc_x, bc_y)
        return (t + dt, q, jnp.maximum(hmax, q[0])), None

    carry = (jnp.asarray(0.0, dtype=q0.dtype), q0, q0[0])
    if checkpoint_block is None:
        carry, _ = jax.lax.scan(body, carry, None, length=n_steps)
    else:
        if n_steps % checkpoint_block:
            raise ValueError("n_steps must be a multiple of checkpoint_block")
        inner = jax.checkpoint(lambda c: jax.lax.scan(jax.checkpoint(body), c, None, length=checkpoint_block)[0])
        carry, _ = jax.lax.scan(lambda c, _: (inner(c), None), carry, None, length=n_steps // checkpoint_block)
    t, q, hmax = carry
    return q, t, hmax
