"""dclaw_jax on exactly the Fortran run's terrain and initial state (read
back from its own frame 0 and aux output), same parameters, a frame
every 20 s (time steps clamped to land on each output time)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "differentiable_dclaw"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from dclaw_jax.model import MaterialParams
from dclaw_jax.solver import Grid, cfl_dt, step
from fortran_io import read_grid

REF = HERE / "model" / "_output_o2"
q0f = read_grid(REF / "fort.q0000")
bed = jnp.asarray(read_grid(REF / "fort.a0000")[:, :, 0])
q0 = jnp.asarray(np.moveaxis(q0f[:, :, :5], 2, 0))
NX, NY = bed.shape
DX = 20.0
GRID = Grid(NX, NY, DX, DX)
T_FINAL, FRAME_DT, N_STEPS = 1800.0, 20.0, 3000  # <=3000 reach 1800 s across phi 20-44, kref 1e-13.5..1e-10
N_FRAMES = int(T_FINAL / FRAME_DT) + 1


def run(theta):
    p = MaterialParams(phi_deg=theta[0], kref=10.0 ** theta[1], m_crit=0.64, mu=0.005, manning_n=0.06)

    def body(carry, _):
        t, q, fr = carry
        t_next = (jnp.floor(t / FRAME_DT + 1e-9) + 1.0) * FRAME_DT
        dt = jnp.clip(jnp.minimum(jnp.minimum(cfl_dt(q, p, GRID, 0.35), t_next - t), T_FINAL - t), 0.0, None)
        q = step(q, bed, p, GRID, dt, "open", "open")
        t = t + dt
        k = jnp.round(t / FRAME_DT).astype(int)
        hit = (jnp.abs(t - k * FRAME_DT) < 1e-6) & (dt > 0)
        fr = fr.at[k].set(jnp.where(hit, q, fr[k]))
        return (t, q, fr), None

    fr0 = jnp.zeros((N_FRAMES, 5, NX, NY)).at[0].set(q0)
    (t, q, fr), _ = jax.lax.scan(body, (jnp.asarray(0.0), q0, fr0), None, length=N_STEPS)
    return fr, t, q


if __name__ == "__main__":
    fr, t, q = jax.jit(run)(jnp.array([37.63, np.log10(3.348e-12)]))
    print(f"JAX reached t={float(t):.1f} s; initial volume {float(q0[0].sum()) * DX * DX:.0f} m3, "
          f"final {float(q[0].sum()) * DX * DX:.0f} m3")
    np.save(HERE / "jax_frames.npy", np.array(fr[:, 0]))
    np.save(HERE / "jax_frames_full.npy", np.array(fr))
