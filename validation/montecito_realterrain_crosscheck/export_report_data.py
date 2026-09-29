"""Collect every number and field shown in the validation report, from the
current code (GPU), into report_data.json."""
import json, re, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
import run_jax as R
import sensitivity as SEN
from compare import OBS, MASK, csi
from fortran_io import frames

HERE = R.HERE
NX, NY, DX = R.NX, R.NY, R.DX
to_img = lambda a: np.asarray(a).T[::-1]          # (x, y south-up) -> rows north->south
bed = np.array(R.bed)
gy, gx = np.gradient(to_img(bed), DX)
hill = np.clip(0.55 - 0.45 * (gx - gy) / 0.25, 0, 1)

th_med = jnp.array([37.63, float(np.log10(3.348e-12))])
th_cal = jnp.asarray(np.load(HERE / "calibrated_theta.npy"))
run = jax.jit(R.run)
fr_med, _, _ = run(th_med); fr_cal, _, _ = run(th_cal)
fr_med, fr_cal = np.array(fr_med), np.array(fr_cal)
peak_cal = fr_cal[:, 0].max(axis=0); peak_med = fr_med[:, 0].max(axis=0)
fo1 = frames(HERE / "model/_output_o1"); fo2 = frames(HERE / "model/_output_o2")
peak_f1 = np.max([q[:, :, 0] for _, q in fo1], axis=0)
peak_f2 = np.max([q[:, :, 0] for _, q in fo2], axis=0)
yc = (np.arange(NY) + 0.5) * DX
def front(h):
    w = (h > 0.1).any(axis=0); return float(yc[w].min()) if w.any() else None
fronts = {
    "fortran_o1": [front(q[:, :, 0]) for _, q in fo1],
    "fortran_o2": [front(q[:, :, 0]) for _, q in fo2],
    "jax_median": [front(fr_med[k, 0]) for k in range(fr_med.shape[0])],
    "jax_calibrated": [front(fr_cal[k, 0]) for k in range(fr_cal.shape[0])],
}
times = [t for t, _ in fo1]
obs_front = float(yc[OBS.any(axis=0)].min())

def state_stats(q):
    h = q[0]; wet = h > 0.1; hs = np.maximum(h, 1e-9)
    u = np.hypot(q[1], q[2]) / hs; m = np.clip(q[3] / hs, 0, 1)
    r = q[4] / ((2700 * m + 1000 * (1 - m)) * 9.81 * hs)
    return dict(u=float(np.median(u[wet])), m=float(np.median(m[wet])), r=float(np.median(r[wet])), wet=int(wet.sum()))
state = {"fortran_o1": [], "jax_median": []}
for k in range(0, 91, 3):
    state["fortran_o1"].append(state_stats(np.moveaxis(fo1[k][1][:, :, :5], 2, 0)))
    state["jax_median"].append(state_stats(fr_med[k]))
state_times = [times[k] for k in range(0, 91, 3)]

# gradient checks on the real terrain, final solver
bedj = R.bed
g_bed, g_th = jax.jit(jax.grad(SEN.target, argnums=(0, 1)), static_argnums=2)(bedj, th_cal, 50)
from scipy import ndimage
v = ndimage.gaussian_filter(np.random.default_rng(0).standard_normal(bed.shape), 5.0)
v = jnp.asarray(v / np.abs(v).max())
f = jax.jit(SEN.target, static_argnums=2)
rev = float(jnp.sum(g_bed * v))
fwd = float(jax.jit(lambda b: jax.jvp(lambda bb: SEN.target(bb, th_cal, None), (b,), (v,))[1])(bedj))
eps_list = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
fd_dir = [(float(f(bedj + e * v, th_cal, None)) - float(f(bedj - e * v, th_cal, None))) / (2 * e) for e in eps_list]
param_checks = []
for idx, name in [(0, "phi"), (1, "log10 kref")]:
    d = jnp.zeros(2).at[idx].set(1e-5)
    fd = (float(f(bedj, th_cal + d, None)) - float(f(bedj, th_cal - d, None))) / 2e-5
    param_checks.append(dict(name=name, ad=float(g_th[idx]), fd=fd))

# timing: forward CPU (recorded) vs GPU, reverse with/without checkpointing (recorded)
t0 = time.time(); _ = run(th_cal)[0].block_until_ready(); fwd_gpu = time.time() - t0
cal_log = (HERE / "calibrate_gpu.log").read_text()
start = re.search(r"start phi=([\d.]+) log10k=([-\d.]+) cost=([\d.]+) CSI=([\d.]+)", cal_log)
iters = [dict(it=0, phi=float(start[1]), logk=float(start[2]), cost=float(start[3]), csi=float(start[4]))]
for m_ in re.finditer(r"it\s+(\d+): phi=\s*([\d.]+) log10k=\s*([-\d.]+) cost=([\d.]+) CSI=([\d.]+)", cal_log):
    iters.append(dict(it=int(m_[1]), phi=float(m_[2]), logk=float(m_[3]), cost=float(m_[4]), csi=float(m_[5])))

c = lambda p: csi(p)
out = dict(
    nx=NX, ny=NY, dx=DX,
    hill=(to_img(hill) * 255).astype(np.uint8).ravel().tolist() if False else np.round(hill * 255).astype(int).ravel().tolist(),
    bed=np.round(to_img(bed), 1).ravel().tolist(),
    obs=to_img(OBS).astype(int).ravel().tolist(),
    scored=to_img(MASK).astype(int).ravel().tolist(),
    peak_cal=np.round(to_img(peak_cal), 2).ravel().tolist(),
    peak_f1=np.round(to_img(peak_f1), 2).ravel().tolist(),
    peak_f2=np.round(to_img(peak_f2), 2).ravel().tolist(),
    peak_med=np.round(to_img(peak_med), 2).ravel().tolist(),
    sens=[float(f"{x:.4g}") for x in to_img(np.array(g_bed)).ravel()],
    buildings=[[int(i), int(NY - 1 - j)] for i, j in zip(np.array(SEN.BI), np.array(SEN.BJ))],
    csi={"fortran_o1": c(peak_f1), "fortran_o2": c(peak_f2), "jax_median": c(peak_med), "jax_calibrated": c(peak_cal)},
    times=times, fronts=fronts, obs_front=obs_front,
    state_times=state_times, state=state,
    grad=dict(reverse=rev, forward=fwd, eps=eps_list, fd=fd_dir, params=param_checks,
              L=float(f(bedj, th_cal, None)), n_buildings=len(SEN.BI), sens_max=float(jnp.abs(g_bed).max())),
    calib=dict(iters=iters, theta=[float(th_cal[0]), float(th_cal[1])]),
    perf=dict(forward_cpu_s=19.0, forward_gpu_s=round(fwd_gpu, 2), grad_time_s=7.05, grad_mem_gb=0.21,
              grad_mem_nockpt_gb=617.6, calib_cpu_min=40, calib_gpu_s=140),
)
(HERE / "report_data.json").write_text(json.dumps(out, separators=(",", ":")))
print("CSI", {k: round(v[0], 3) for k, v in out["csi"].items()})
print("grad", rev, fwd, [round(x, 5) for x in fd_dir], param_checks)
print("fwd gpu", fwd_gpu, "size MB", (HERE / "report_data.json").stat().st_size / 1e6)
