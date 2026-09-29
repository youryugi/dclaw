"""Port of tests/eigenstructure_check's dry dam-break front-speed bound
(repo root) to this JAX solver: a reservoir released from rest onto a
dry, frictionless bed cannot advance faster than the classical
frictionless shallow-water front speed 2*sqrt(g*h_L). See that test's
README for the closed-form derivation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from dclaw_jax.model import MaterialParams
from dclaw_jax.solver import Grid, make_state, rollout


def main():
    # phi_deg=0 alone is not "frictionless": the dilatancy angle psi (from
    # loose, m<m_crit material contracting under shear) adds to the
    # effective friction angle tan(phi_deg + psi), and psi is *negative*
    # for contracting material -- i.e. it can make the effective friction
    # coefficient negative, accelerating the flow rather than resisting
    # it. That is a real mechanism in the full model (not a bug), but it
    # breaks the assumption behind the classical bound, which is a pure
    # hyperbolic system with no accelerating source. c1=0 disables the
    # dilatancy-angle contribution entirely (tanpsi = c1*(m-m_eq) = 0
    # regardless of m), giving a fair like-for-like comparison.
    p = MaterialParams(phi_deg=0.0, c1=0.0)
    nx = 400
    grid = Grid(nx=nx, ny=4, dx=0.1, dy=0.1)
    x = (jnp.arange(nx)[:, None] + 0.5) * grid.dx  # cell centers, x in [0, 40]
    x = x * jnp.ones((nx, 4))

    reservoir_end = 3.0
    h_l = 1.0
    h = jnp.where(x < reservoir_end, h_l, 0.0)
    m = jnp.where(h > 0, 0.55, 0.0)
    pb = jnp.where(h > 0, p.rho_f * p.gz * h, 0.0)
    q0 = make_state(h, jnp.zeros_like(h), jnp.zeros_like(h), h * m, pb)
    b = jnp.zeros_like(h)

    t_final = 1.0  # front stays well clear of the domain wall at 40 m
    theory_speed = 2.0 * jnp.sqrt(p.gz * h_l)
    print(f"theoretical frictionless front speed 2*sqrt(g*h_L) = {theory_speed:.4f} m/s")

    qf, t, _, _ = rollout(q0, b, p, grid, t_final=t_final, n_steps=1500,
                        bc_x="wall", bc_y="wall")

    xr = x[:, 0]
    h_final = qf[0][:, 0]
    wet = h_final > 1e-3
    final_front = float(jnp.max(xr[wet])) if bool(jnp.any(wet)) else reservoir_end
    bound_final = reservoir_end + theory_speed * float(t) + 0.1
    print(f"t reached: {t}")
    print(f"front at t_final: {final_front:.3f} m, bound: {bound_final:.3f} m")

    ok = final_front <= bound_final
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
