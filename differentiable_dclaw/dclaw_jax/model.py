"""Core digclaw physics as pure, differentiable JAX functions.

Implements the five-equation hyperbolic system of Iverson & George (2014),
"A depth-averaged debris-flow model that includes the effects of evolving
dilatancy. I. Physical basis", eq 2.1-2.14, with kappa=1 (as hardcoded in
the Fortran reference, digclaw_module.f90, and as used for all flume
comparisons in the companion paper). State is q = (h, hu, hv, hm, pb).

This is a deliberately reduced re-implementation for gradient-based
calibration research, not a drop-in replacement for D-Claw:
- No AMR: a single uniform rectangular grid.
- No topography-file reading: the bed b(x, y) is any array you construct
  or load yourself (including a real DEM) and pass in directly.
- No moving-gate mechanism.
- Coulomb + viscous basal friction are applied as a separate split step
  (`friction_step`), exactly as the Fortran reference does in src2.f90.
  The friction-bounds-by-driving-force static check in the reference's
  Riemann solver (eq 2.15-2.16, calc_taudir) is not ported.

Every closure here was checked line-by-line against
clawpack-runtime/dclaw/src/2d/dig/digclaw_module.f90 (kperm, alphainv,
m_eq/tanpsi) during the same investigation that produced
tests/eigenstructure_check at the repo root -- see this package's README
for the correspondence table.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class MaterialParams(NamedTuple):
    """Physical parameters, table 2/eq 2.2-2.14 of Iverson & George (2014)."""

    rho_s: float = 2700.0
    rho_f: float = 1000.0
    m_crit: float = 0.64
    mref: float = 0.60
    kref: float = 1.0e-9        # k0 in eq 2.7 (m^2)
    phi_deg: float = 40.0       # basal friction angle (degrees)
    delta: float = 0.01         # grain-collision length scale (m)
    mu: float = 0.005           # pore-fluid viscosity (Pa s)
    alpha_c: float = 0.01       # 'a' in eq 2.8
    sigma_0: float = 1000.0     # Pa, eq 2.8 (alphamethod=0 / "case 0")
    c1: float = 1.0             # dilatancy-angle scale, eq 2.12
    gz: float = 9.81            # bed-normal gravity (bed_normal=0: = g)
    manning_n: float = 0.0      # geo_data.manning_coefficient; 0 = off (see manning_step)


EPS_H = 1.0e-6        # dry tolerance for divisions by h


def bulk_density(m: jnp.ndarray, p: MaterialParams) -> jnp.ndarray:
    """eq 2.2: rho = rho_s * m + rho_f * (1 - m)."""
    return p.rho_s * m + p.rho_f * (1.0 - m)


def chi(rho: jnp.ndarray, p: MaterialParams) -> jnp.ndarray:
    """eq 2.3, with kappa=1 this is the only kappa-dependent quantity used."""
    return (p.rho_f + 3.0 * rho) / (4.0 * rho)


def permeability(m: jnp.ndarray, p: MaterialParams) -> jnp.ndarray:
    """eq 2.7: k = k0 * exp((mref - m) / 0.04)."""
    return p.kref * jnp.exp(-(m - p.mref) / 0.04)


def effective_stress(rho: jnp.ndarray, h: jnp.ndarray, pb: jnp.ndarray, p: MaterialParams) -> jnp.ndarray:
    """sigma_e = rho * gz * h - pb, clipped to >= 0 (eq 2.11)."""
    return jnp.clip(rho * p.gz * h - pb, min=0.0)


def dilation_rate(k: jnp.ndarray, h: jnp.ndarray, mu_eff: jnp.ndarray, pb: jnp.ndarray, p: MaterialParams) -> jnp.ndarray:
    """eq 2.6: D = -(2k / (h*mu)) * (pb - rho_f*gz*h).

    The paper notes pb relaxes toward hydrostatic "at the same rate as
    D -> 0" -- i.e. this term is a relaxation with rate 2k/(h*mu), which
    is why the reference scheme solves it as an *exact* exponential
    relaxation rather than with explicit time-stepping (Iverson & George
    2014 eq 3.9 discussion). Our explicit RK2 has no such special
    treatment, so h is floored at the grain diameter `delta` here (not
    the generic dry tolerance): a film thinner than one grain has no
    physical meaning anyway, and this keeps the relaxation rate finite
    enough for explicit stepping to remain stable.
    """
    h_floor = jnp.maximum(h, p.delta)
    return -(2.0 * k / (h_floor * mu_eff)) * (pb - p.rho_f * p.gz * h)


def m_equilibrium(sigma_e: jnp.ndarray, shear: jnp.ndarray, p: MaterialParams) -> jnp.ndarray:
    """eq 2.13-2.14: quasi-static equilibrium solid fraction.

    sqrt's derivative, 1/(2*sqrt(x)), is infinite at x=0 -- forward
    values are fine there (jnp.where correctly selects m_crit when
    Nnum<=0), but reverse-mode autodiff evaluates *both* branches, and
    0 * inf = nan leaks through the select regardless of which branch is
    "chosen". Both Nden and Nnum can be exactly 0 at rest (shear=0), so
    both square roots need an epsilon floor purely to keep the gradient
    finite; SQRT_EPS is far below any physically meaningful value of
    either quantity.
    """
    SQRT_EPS = 1.0e-12
    Nden = p.rho_s * (shear * p.delta) ** 2 + sigma_e
    Nnum = p.mu * shear
    ratio = jnp.sqrt(Nden + SQRT_EPS) / (
        jnp.sqrt(Nden + SQRT_EPS) + jnp.sqrt(jnp.maximum(Nnum, 0.0) + SQRT_EPS) + EPS_H
    )
    return jnp.where(Nnum > 0.0, p.m_crit * ratio, p.m_crit)


def primitive(q: jnp.ndarray, p: MaterialParams):
    """q = (h, hu, hv, hm, pb) -> (h, u, v, m, pb, rho).

    m is clipped to the physical range [0, 1]: the explicit RK2 predictor
    stage can transiently overshoot a sharp m gradient (e.g. at a wet/dry
    front) before the corrector stage pulls it back, and an out-of-range
    m fed into the permeability's exp() (eq 2.7) would otherwise blow up
    and, through phi1, contaminate h itself.
    """

    h, hu, hv, hm, pb = q
    h_safe = jnp.maximum(h, EPS_H)
    u = hu / h_safe
    v = hv / h_safe
    m = jnp.clip(hm / h_safe, min=0.0, max=1.0)
    rho = bulk_density(m, p)
    return h, u, v, m, pb, rho


def _material_state(q: jnp.ndarray, p: MaterialParams) -> dict:
    """Shared intermediate quantities for `friction_step`, `dilatancy_step` (h, hu, hv, hm)
    and `pb_relaxation_step` (pb). Kept in one place so the two never drift
    out of sync on the definitions of D, alpha, zeta, tanpsi, etc.
    """

    h, u, v, m, pb, rho = primitive(q, p)
    vnorm = jnp.sqrt(u**2 + v**2 + 1.0e-12)  # see wave_speed: sqrt'(0) is inf
    shear = 2.0 * vnorm / jnp.maximum(h, EPS_H)

    k = permeability(m, p)
    sigma_e = effective_stress(rho, h, pb, p)
    D = dilation_rate(k, h, p.mu, pb, p)

    m_eq = m_equilibrium(sigma_e, shear, p)
    # Clipped to +-1 (psi = +-45 deg): far outside any physically meaningful
    # dilatancy angle for a real granular material, but for very loose
    # material (m well below m_crit -- confirmed directly with real
    # Montecito Creek parameters, cv=0.512 vs m_crit=0.64, a bigger gap
    # than any lab-flume case in this repo) tanpsi can get large enough
    # that tan(phi_deg + psi) goes negative, i.e. "friction" that
    # accelerates rather than resists motion. That is a real mechanism in
    # the full model (matches the physical picture of contraction-driven
    # mobility), but explicit time-stepping runs away under it: confirmed
    # directly (unclipped, this exact scenario reached t=16s before the
    # front had already crossed the whole 5.4 km domain with h up to
    # 171 m, vs. a physically sane run with this clip in place).
    tanpsi = p.c1 * (m - m_eq)
    psi = jnp.arctan(tanpsi)
    tan_phi_psi = jnp.tan(jnp.deg2rad(p.phi_deg) + psi)

    m_safe = jnp.maximum(m, EPS_H)
    # eq 2.8, no floor -- as the reference (digclaw_module.f90 setvars,
    # alphainv = m*(sig_eff + sig_0)/alpha_c). An earlier floor at 1e-5
    # 1/Pa, left over from explicit pb stepping, silently overrode realistic
    # values (2e-6 for the USGS flume, similar at Montecito): dilatancy
    # forcing and pb relaxation, both ~1/alpha, came out ~5x too weak, pore
    # pressure fell where the reference's rose, and alpha_c barely mattered.
    alpha = p.alpha_c / (m_safe * (sigma_e + p.sigma_0))
    zeta = 3.0 / (2.0 * alpha * jnp.maximum(h, EPS_H)) + p.gz * p.rho_f * (rho - p.rho_f) / (4.0 * rho)

    return dict(h=h, u=u, v=v, m=m, pb=pb, rho=rho, vnorm=vnorm, k=k,
                sigma_e=sigma_e, D=D, tanpsi=tanpsi, tan_phi_psi=tan_phi_psi,
                alpha=alpha, zeta=zeta)


def dilatancy_step(q: jnp.ndarray, p: MaterialParams, dt: jnp.ndarray) -> jnp.ndarray:
    """eq 2.4 (a)-(d): the D-driven volume change of h, hu, hv, hm, as a
    split step run right after `pb_relaxation_step`, ported from the
    reference's default source method (mp_update.f90,
    mp_update_relax_Dclaw4, src2method=0):

        hm <- hm * exp(-dt*D*rho_f/(rho*h))
        h, hu, hv <- (h, hu, hv) * exp(dt*D*(rho-rho_f)/(rho*h))

    with D (eq 2.6) evaluated from the *already relaxed* pb, so it stays
    bounded. Stepping these inside the explicit RK2 instead (an earlier
    version of this solver) used D from un-relaxed pb, which scales like
    1/h^2 in thin frontal cells; that stiffness amplified sensitivities
    of m=hm/h by up to ~1e6 per step there. The reference advances h
    explicitly (h += h*krate*dt) and hu, hv exponentially; all three
    use the same exponential here so u = hu/h is exactly unchanged,
    matching the reference's own note that velocity is constant through
    this stage.
    """

    s = _material_state(q, p)
    h, rho, D = s["h"], s["rho"], s["D"]
    rhoh = rho * jnp.maximum(h, EPS_H)
    hm_factor = jnp.exp(-dt * D * p.rho_f / rhoh)
    h_factor = jnp.exp(dt * D * (rho - p.rho_f) / rhoh)
    return (q.at[0].multiply(h_factor).at[1].multiply(h_factor).at[2].multiply(h_factor)
            .at[3].multiply(hm_factor))


def _phi1_stable(x: jnp.ndarray) -> jnp.ndarray:
    """(1 - exp(-x)) / x, stable and equal to 1 at x=0 (a truncated
    Taylor series is used there instead of dividing by ~0)."""

    small = jnp.abs(x) < 1.0e-6
    safe_x = jnp.where(small, 1.0, x)
    return jnp.where(small, 1.0 - x / 2.0 + x**2 / 6.0, (1.0 - jnp.exp(-x)) / safe_x)


N_NEWTON_PB = 6
# Dilatancy forcing is tapered smoothly to zero below this speed (m/s).
# The reference has none at exactly u=0; with a small floor on |u| (needed
# for finite derivatives) thin, nearly static, liquefied deposits kept a
# forcing with coefficient 3/(alpha*h) ~ 3.5e6 Pa, closing a speed ->
# pore-pressure -> friction -> speed loop with enormous gain: on the real
# Montecito terrain dh/dphi grew to ~3e10 in 5-10 cm deposits creeping at
# <2 cm/s, while forward results were unaffected. At 0.05 m/s the tangent
# stays below ~1 and the real-terrain results are unchanged (CSI 0.490 vs
# 0.491, identical front positions); 0.01 m/s still let it grow to ~9.
DILATANCY_U_TAPER = 0.05
# Floor on h (m) in the dilatancy forcing's 3/(alpha*h). In a thin (~1 cm)
# fast front that is dilating (tanpsi > 0), the forcing drives pb down,
# which raises sigma_e and lowers alpha, which strengthens the forcing: a
# positive feedback with dS/dpb ~ 3|u|tanpsi*m/(alpha_c*h) ~ 1e3 1/s at
# h = 1 cm, u = 8 m/s -- more than 1/dt. The forward solution stays bounded
# by the clip to [0, lithostatic], but the tangent grows by that factor
# every step: on the USGS flume, d(gauge series)/d(theta) reached 1e4-1e19
# at half of 16 parameter points near the paper's values (finite
# differences: O(1-10)); pb_relaxation_step alone contributed +15 decades
# over 0.7 s. At 2 cm the gain stays below 1: the full observation
# Jacobian matches finite differences (eps=1e-6) to <1e-3 at all 16 points,
# and the flume gauge series change by <= 1.1 mm and 18 Pa. More Newton
# iterations, a larger DILATANCY_U_TAPER, or the old alpha floor did not
# help. See gradient_particle_filter/pf_flume.
DILATANCY_H_FLOOR = 0.02


def pb_relaxation_step(q: jnp.ndarray, p: MaterialParams, dt: jnp.ndarray) -> jnp.ndarray:
    """eq 2.4e's pb source over one step, dpb/dt = -rate*(pb - p_hydro) + S(pb)
    with S = -(3/(alpha*h))*|u|*tanpsi the dilatancy forcing. This is the
    Godunov-split "reaction" stage, called after the transport stage.

    The linear relaxation (rate, p_hydro frozen at the input state) is
    integrated exactly, as the reference's exponential relaxation does.
    The dilatancy forcing S is treated *implicitly*: S depends on pb
    itself (through sigma_e -> alpha and m_eq -> tanpsi), and in
    liquefied flow that dependence is extremely stiff -- 3/(alpha*h) is
    ~2e5 Pa and m_eq ~ sqrt(sigma_e) near sigma_e=0. Freezing S at the
    input state (explicit, as the reference does before clipping pb to
    [0, lithostatic] in qfix) left the forward result bounded only
    because of that clip, while the per-step derivative dpb_new/dpb was
    ~ -2e6: on the real Montecito DEM reverse- and forward-mode gradients
    agreed with each other but not with finite differences (4.8e3 vs ~-2)
    until this was made implicit. Implicitly,
    dpb_new/dpb = 1/(1 - dt*phi1*dS/dpb) is bounded.

    Below DILATANCY_U_TAPER the forcing is scaled smoothly to zero (see
    that constant).

    Solved per cell by Newton on
        r(pb) = pb - [pb0*e^-x + p_hydro*(1 - e^-x) + dt*phi1(x)*S(pb)],
    starting from the explicit value and kept inside [0, lithostatic]; the
    derivative is floored at 1, which is exact whenever dS/dpb <= 0.
    """

    s = _material_state(q, p)
    h, pb, zeta, k, rho = s["h"], s["pb"], s["zeta"], s["k"], s["rho"]
    rate = zeta * (2.0 * k) / (jnp.maximum(h, p.delta) * p.mu)  # >= 0 always
    p_hydro = p.rho_f * p.gz * h
    lith = rho * p.gz * jnp.maximum(h, 0.0)
    x = rate * dt
    ex = jnp.exp(-x)
    base = pb * ex + p_hydro * (1.0 - ex)
    w = dt * _phi1_stable(x)

    def forcing(pb_):
        st = _material_state(q.at[4].set(pb_), p)
        vn = st["vnorm"]
        if DILATANCY_U_TAPER > 0:
            r = jnp.clip(vn / DILATANCY_U_TAPER, 0.0, 1.0)
            vn = vn * r * r * (3.0 - 2.0 * r)
        return -(3.0 / (st["alpha"] * jnp.maximum(h, DILATANCY_H_FLOOR))) * vn * st["tanpsi"]

    def resid(pb_):
        return pb_ - base - w * forcing(pb_)

    pb_new = jnp.clip(base + w * forcing(pb), 0.0, lith)
    for _ in range(N_NEWTON_PB):
        r, dr = jax.jvp(resid, (pb_new,), (jnp.ones_like(pb_new),))  # cells are independent
        pb_new = jnp.clip(pb_new - r / jnp.maximum(dr, 1.0), 0.0, lith)
    return pb_new


def friction_step(q: jnp.ndarray, p: MaterialParams, dt: jnp.ndarray) -> jnp.ndarray:
    """Coulomb + viscous basal friction as a separate split step, ported
    from the Fortran reference (src2.f90, the `hvnorm0>0` block):

        vnorm <- max(0, vnorm - dt*tau/(rho*h))            exact Coulomb
        vnorm <- vnorm * exp(-(1-m)*2*mu*dt/(rho*h^2))     exact viscous

    applied to the speed with the flow direction kept, so friction can
    bring a cell to rest but never reverse it. This must not be folded
    into the explicit RK2 right-hand side: tau/(rho*h) is ~1-3 m/s^2 for
    real debris, so one explicit step changes u by far more than any
    smoothing width around u=0, and a quasi-static cell then chatters
    through zero every step. That chattering was the root cause of both
    the rough (piecewise-jumping) loss landscape and the reverse-mode
    gradient blow-up seen with an earlier regularized-closure version --
    confirmed directly: two runs 1e-4 deg apart in phi_deg fell one
    chatter period out of phase at the pile's static tail and diverged
    from there.

    tau includes the reference's dilute-mixture taper
    0.5*(1+tanh(100*(m-0.1))) (digclaw_module.f90 setvars).
    """

    s = _material_state(q, p)
    h, m, rho = s["h"], s["m"], s["rho"]
    hu, hv = q[1], q[2]
    h_safe = jnp.maximum(h, EPS_H)

    tau = s["sigma_e"] * s["tan_phi_psi"] * 0.5 * (1.0 + jnp.tanh(100.0 * (m - 0.1)))
    # sqrt'(0) is infinite; the floor keeps d/d(hu) finite for cells at rest.
    hvnorm0 = jnp.sqrt(hu**2 + hv**2 + 1.0e-24)
    vnorm = hvnorm0 / h_safe
    vnorm = jnp.maximum(0.0, vnorm - dt * tau / (rho * h_safe))
    vnorm = vnorm * jnp.exp(-(1.0 - m) * 2.0 * p.mu * dt / (rho * h_safe**2))
    ratio = h_safe * vnorm / hvnorm0
    return q.at[1].set(ratio * hu).at[2].set(ratio * hv)


def qfix(q: jnp.ndarray, p: MaterialParams) -> jnp.ndarray:
    """Project onto admissible states, ported from the reference
    (digclaw_module.f90 qfix): m in [0, 1] with the conserved hm
    rewritten as h*m, and 0 <= pb <= rho*gz*h (pore pressure can reach
    but not exceed lithostatic). Without it, the exponential hm update in
    `dilatancy_step` let hm grow past h in thin cells, and pb could
    exceed lithostatic -- both seen directly at full Montecito scale as
    15 m peak-depth spikes and blown-up runs across the parameter space.
    Dry cells are left to solver._clip_dry.
    """

    h = q[0]
    m = jnp.clip(q[3] / jnp.maximum(h, EPS_H), 0.0, 1.0)
    rho = bulk_density(m, p)
    pb = jnp.clip(q[4], 0.0, rho * p.gz * jnp.maximum(h, 0.0))
    return q.at[3].set(h * m).at[4].set(pb)


def manning_step(q: jnp.ndarray, p: MaterialParams, dt: jnp.ndarray) -> jnp.ndarray:
    """Implicit Manning friction, ported from the reference (src2.f90,
    friction_forcing block, beta=1, no depth-dependent coefficient):
    hu, hv <- (hu, hv) / (1 + dt*|hu|*gz*n^2/h^(7/3)). This is the only
    resistance left once tau is tapered off for dilute cells (m < ~0.1,
    `friction_step`); without it, a thin water film on a 4 deg slope has
    no meaningful terminal velocity, which collapsed the CFL step to
    ~0.07 s in full-scale Montecito runs. Barnhart et al. 2021 used
    n = 0.06 for every Montecito D-Claw run.
    """

    h_safe = jnp.maximum(q[0], EPS_H)
    hvnorm = jnp.sqrt(q[1] ** 2 + q[2] ** 2 + 1.0e-24)
    denom = 1.0 + dt * hvnorm * p.gz * p.manning_n ** 2 / h_safe ** (7.0 / 3.0)
    return q.at[1].divide(denom).at[2].divide(denom)
