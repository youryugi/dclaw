"""dclaw_jax's step / rollout with the dry-film threshold as a parameter.

Everything physical (fluxes, friction, pb relaxation, dilatancy, Manning,
qfix) is imported from `differentiable_dclaw/dclaw_jax` unchanged; only
the time-step wrapper is repeated here, so the package -- and the
Montecito results built on it -- are untouched.

`solver._clip_dry` scales the state by a smoothstep from 0 at h = tol to
1 at h = 2*tol, with tol = 1 mm fixed. On the steep flume that removed
1.4% of the mixture mass in 4 s (differentiable_dclaw/README.md, "Moving
headgate"). The Fortran reference keeps everything above 1 mm and zeroes
the rest; `dry_tol=5e-4` (ramp 0.5-1 mm) is the closest continuous match.
With `dry_tol=1e-3` this is bit-identical to `solver.rollout_peak`.

Measured by check_dry_tol.py (4 s flume release), before the package's
alpha-floor fix (same conclusion expected): lowering the threshold
cuts the mass loss (1.4% at 1e-3, 0.68% at 5e-4, 0.12% at 1e-4) but
breaks the gradient -- at 5e-4, d/dphi comes out 3e12 against a finite
difference of -0.13, the thin-film 1/h, 1/h^2 amplification described in
the package README (item 7). Keep 1e-3 for anything differentiated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "differentiable_dclaw"))

import jax
import jax.numpy as jnp

from dclaw_jax.model import MaterialParams, dilatancy_step, friction_step, manning_step, pb_relaxation_step, qfix
from dclaw_jax.solver import Grid, _bed_at, cfl_dt, rhs


def clip_dry(q: jnp.ndarray, tol: float) -> jnp.ndarray:
    h = jnp.maximum(q[0], 0.0)
    s = jnp.clip((h - tol) / tol, 0.0, 1.0)
    w = s * s * (3.0 - 2.0 * s)
    return q.at[0].set(h) * w[None]


def step(q, b, p: MaterialParams, grid: Grid, dt, bc_x: str, bc_y: str, dry_tol: float):
    """solver.step with clip_dry(., dry_tol) in place of solver._clip_dry."""
    k1 = rhs(q, b, p, grid, bc_x, bc_y)
    q1 = clip_dry(q + dt * k1, dry_tol)
    k2 = rhs(q1, b, p, grid, bc_x, bc_y)
    q_transport = clip_dry(q + 0.5 * dt * (k1 + k2), dry_tol)
    q_fric = friction_step(qfix(q_transport, p), p, dt)
    q_relaxed = qfix(q_fric.at[4].set(pb_relaxation_step(q_fric, p, dt)), p)
    return clip_dry(qfix(manning_step(dilatancy_step(q_relaxed, p, dt), p, dt), p), dry_tol)


def rollout_peak(q0, b, p: MaterialParams, grid: Grid, t_final: float, cfl: float = 0.4,
                 bc_x: str = "wall", bc_y: str = "wall", n_steps: int = 2000,
                 checkpoint_block: int | None = None, dry_tol: float = 1.0e-3):
    """solver.rollout_peak with a `dry_tol` argument. Returns (q_final, t_reached, h_peak)."""

    def body(carry, _):
        t, q, hmax = carry
        dt = jnp.clip(jnp.minimum(cfl_dt(q, p, grid, cfl), t_final - t), 0.0, None)
        q = step(q, _bed_at(b, t), p, grid, dt, bc_x, bc_y, dry_tol)
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


def advance_record(q, t0, t_end, b, p: MaterialParams, grid: Grid, n_steps: int, gauge_cells,
                   cfl: float = 0.4, bc_x: str = "wall", bc_y: str = "wall", dry_tol: float = 1.0e-3):
    """Advance q from t0 to t_end in at most n_steps CFL steps (dt is clamped
    to 0 past t_end), recording the state at gauges after every step.

    gauge_cells: int array (n_gauges, 2) of the two x-cells either side of
    each gauge (the gauges sit on cell edges), read in y-row 0. As the
    reference's gauges_module does, the two cells are averaged when both
    are wet and the cell containing the gauge (the right one) is used
    otherwise. Returns (q, t, times, gauge_q) with gauge_q of shape
    (n_steps, 5, n_gauges); steps past t_end repeat the last time.
    """

    gl, gr = gauge_cells[:, 0], gauge_cells[:, 1]

    def body(carry, _):
        t, q = carry
        dt = jnp.clip(jnp.minimum(cfl_dt(q, p, grid, cfl), t_end - t), 0.0, None)
        q = step(q, _bed_at(b, t), p, grid, dt, bc_x, bc_y, dry_tol)
        qa, qb = q[:, gl, 0], q[:, gr, 0]
        both_wet = (qa[0] > dry_tol) & (qb[0] > dry_tol)
        return (t + dt, q), (t + dt, jnp.where(both_wet[None], 0.5 * (qa + qb), qb))

    (t, q), (times, gq) = jax.lax.scan(body, (jnp.asarray(t0, dtype=q.dtype), q), None, length=n_steps)
    return q, t, times, gq
