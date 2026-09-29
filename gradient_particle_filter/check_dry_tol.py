"""Dry-film threshold on the flume release of differentiable_dclaw/tests/test_gate.py:
1. dry_tol=1e-3 reproduces solver.rollout_peak bit for bit;
2. mixture-mass loss, runout and volume past x=3 m for smaller thresholds;
3. d(volume past x=3 m)/dphi still matches finite differences.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "differentiable_dclaw" / "tests"))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import flume_solver as F
import test_gate as T
from dclaw_jax.solver import gate_bed, rollout_peak

B = gate_bed(T.B_BASE, T.RIDGE, [0.0, 0.85], [0.0, 90.0])
Q0 = T._wedge_state()


def run(dry_tol, p=T.FLUME, t_final=4.0):
    return F.rollout_peak(Q0, B, p, T.GRID, t_final=t_final, n_steps=4000, checkpoint_block=50, dry_tol=dry_tol)[0]


def main():
    ref = rollout_peak(Q0, B, T.FLUME, T.GRID, t_final=4.0, n_steps=4000, checkpoint_block=50)[0]
    same = bool(jnp.array_equal(ref, run(1e-3)))
    print(f"dry_tol=1e-3 bit-identical to solver.rollout_peak: {same}")

    M0 = T._mixture_mass(Q0)
    for tol in (1e-3, 5e-4, 2e-4, 1e-4):
        q = run(tol)
        front = float(jnp.max(jnp.where(q[0][:, 0] > 0.01, T.X, -jnp.inf)))
        print(f"dry_tol={tol:.0e}: mixture mass change {float((T._mixture_mass(q) - M0) / M0):+.2e}, "
              f"front {front:.1f} m, volume past x=3 m {float(T._volume_past(q)):.3f} m3")

    for tol in (1e-3, 5e-4):
        loss = jax.jit(lambda ph: T._volume_past(run(tol, T.FLUME._replace(phi_deg=ph), 2.5)))
        g = float(jax.grad(loss)(40.7))
        fd = {e: float((loss(40.7 + e) - loss(40.7 - e)) / (2 * e)) for e in (1e-3, 1e-6)}
        print(f"dry_tol={tol:.0e}: d(volume past 3 m, t=2.5 s)/dphi reverse {g:.6e}, "
              + ", ".join(f"FD eps={e:g} {v:.6e}" for e, v in fd.items()))
    if not same:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
