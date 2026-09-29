"""Verify autodiff gradients through the full rollout match finite
differences. This is the actual point of a *differentiable* D-Claw --
if this doesn't hold, nothing downstream (gradient-based calibration)
can be trusted, regardless of how physically reasonable the forward
simulation looks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

# JAX defaults to float32. Finite-difference verification of a parameter
# like kref (~1e-9) needs perturbations many orders of magnitude smaller
# still; float32 rounds those away entirely (confirmed directly: FD
# collapsed to exactly 0 below eps~1e-11 in float32, while float64 matches
# autodiff to 6 significant figures across eps=1e-10..1e-14). This must be
# set before any other JAX code runs.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from dclaw_jax.model import MaterialParams
from dclaw_jax.solver import Grid, make_state, rollout


def make_problem(phi_deg: float, kref: float):
    """A fully-wet, periodic domain: h > 0 everywhere, so there is no
    wet/dry free boundary anywhere in this problem. Free boundaries are
    inherently non-smooth (a cell's wet/dry state is a discontinuous
    function of the state), which makes both autodiff *and* finite
    differences unreliable near them -- not something specific to this
    solver. This scenario isolates whether the reacting physics itself
    (friction, dilatancy, permeability) is cleanly differentiable,
    deferring the free-boundary question.
    """
    p = MaterialParams(phi_deg=phi_deg, kref=kref)
    grid = Grid(nx=30, ny=4, dx=0.1, dy=0.1)
    x = jnp.arange(30)[:, None] * jnp.ones((30, 4))
    h = 1.0 + 0.3 * jnp.sin(2 * jnp.pi * x / 30.0)
    m = 0.6 + 0.02 * jnp.sin(2 * jnp.pi * x / 30.0 + 1.0)
    u0 = 0.2 * jnp.ones_like(h)
    pb = 0.8 * p.rho_f * p.gz * h  # sub-hydrostatic: drives real dynamics
    q0 = make_state(h, h * u0, jnp.zeros_like(h), h * m, pb)
    b = jnp.zeros((30, 4))
    return q0, b, grid, p


def loss_fn(phi_deg, kref, t_final):
    q0, b, grid, p = make_problem(phi_deg, kref)
    qf, t, _, _ = rollout(q0, b, p, grid, t_final=t_final, n_steps=400,
                        bc_x="periodic", bc_y="periodic")
    # a scalar summary a calibration loss would plausibly use: mean
    # squared flow depth (a stand-in for "match an observed depth field").
    return jnp.mean(qf[0] ** 2)


def finite_diff_grad(f, x0, eps):
    return (f(x0 + eps) - f(x0 - eps)) / (2 * eps)


def check_one(name, phi_deg, kref, t_final, eps_phi, eps_kref, rtol):
    print(f"=== {name}: phi_deg={phi_deg}, kref={kref:.2e}, t_final={t_final} ===")

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    g_phi, g_kref = grad_fn(phi_deg, kref, t_final)

    fd_phi = finite_diff_grad(lambda x: loss_fn(x, kref, t_final), phi_deg, eps_phi)
    fd_kref = finite_diff_grad(lambda x: loss_fn(phi_deg, x, t_final), kref, eps_kref)

    print(f"  d(loss)/d(phi_deg): autodiff={g_phi: .6e}  finite-diff={fd_phi: .6e}")
    print(f"  d(loss)/d(kref):    autodiff={g_kref: .6e}  finite-diff={fd_kref: .6e}")

    ok_phi = jnp.isclose(g_phi, fd_phi, rtol=rtol, atol=1e-6)
    ok_kref = jnp.isclose(g_kref, fd_kref, rtol=rtol, atol=1e-6)
    print(f"  PASS phi_deg: {bool(ok_phi)}   PASS kref: {bool(ok_kref)}")
    return bool(ok_phi) and bool(ok_kref)


def main():
    results = []
    results.append(check_one("short rollout", 35.0, 1.0e-9, 0.1, 1e-4, 1e-12, 1e-3))
    results.append(check_one("longer rollout", 30.0, 5.0e-9, 0.3, 1e-6, 1e-12, 1e-3))

    if all(results):
        print("\nALL GRADIENT CHECKS PASSED")
    else:
        print("\nSOME GRADIENT CHECKS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
