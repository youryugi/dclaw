"""Per-operator amplification of the forward tangent at one parameter point,
accumulated (log10 of the norm ratio) over steps t0..t1."""
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import flume_model as M
import dclaw_jax.model as Mo
import dclaw_jax.solver as S

pi, k, t0, t1 = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
F = M.Flume(dx=0.125)
rng = np.random.default_rng(1)
theta = jnp.asarray((M.THETA_TRUE + rng.normal(0, [1.0, 0.3, 0.2], (64, 3)))[pi])
e = jnp.zeros(3).at[k].set(1.0)
W = jnp.array([1.0, 1.0, 1.0, 1.0, 1e-4])[:, None, None]      # pb in units of ~10 kPa
nrm = lambda d: jnp.sqrt(jnp.sum((W * d) ** 2))


def ops(q, t, th):
    p = M.params(th)
    g = F.grid
    t_next = (jnp.floor(t / F.obs_dt + 1e-9) + 1.0) * F.obs_dt
    dt = jnp.clip(jnp.minimum(jnp.minimum(S.cfl_dt(q, p, g, 0.4), t_next - t), F.t_obs - t), 0.0, None)
    b = F.bed(t)
    out = [q]
    k1 = S.rhs(q, b, p, g, "wall", "wall"); q1 = S._clip_dry(q + dt * k1)
    k2 = S.rhs(q1, b, p, g, "wall", "wall"); qt = q + 0.5 * dt * (k1 + k2); out.append(qt)
    qt = S._clip_dry(qt); out.append(qt)
    qa = Mo.qfix(qt, p); out.append(qa)
    qf = Mo.friction_step(qa, p, dt); out.append(qf)
    qr = qf.at[4].set(Mo.pb_relaxation_step(qf, p, dt)); out.append(qr)
    qr = Mo.qfix(qr, p); out.append(qr)
    qd = Mo.dilatancy_step(qr, p, dt); out.append(qd)
    qm = Mo.manning_step(qd, p, dt); out.append(qm)
    qq = Mo.qfix(qm, p); out.append(qq)
    qo = S._clip_dry(qq); out.append(qo)
    return tuple(out), t + dt

names = ["transport", "clip_dry", "qfix1", "friction", "pb_relax", "qfix2", "dilatancy", "manning", "qfix3", "clip_dry3"]


@jax.jit
def jstep(q, dq, t, dt_):
    (outs, tn), (douts, dtn) = jax.jvp(ops, (q, t, theta), (dq, dt_, e))
    ratios = jnp.array([nrm(douts[i + 1]) / jnp.maximum(nrm(douts[i]), 1e-300) for i in range(len(names))])
    return outs[-1], douts[-1], tn, dtn, ratios

q, dq = jax.jvp(lambda th: Mo.qfix(F.q0, M.params(th)), (theta,), (e,))
t, dt_ = jnp.asarray(0.0), jnp.asarray(0.0)
acc = np.zeros(len(names)); n = 0
while float(t) < t1:
    q, dq, t, dt_, r = jstep(q, dq, t, dt_)
    if float(t) >= t0:
        acc += np.log10(np.asarray(r)); n += 1
print(f"particle {pi}, d/dtheta_{k}, {n} steps in t = {t0}..{t1} s: log10 tangent growth by operator")
for nm, a in zip(names, acc):
    print(f"  {nm:10s} {a:+8.2f}")
print(f"  total      {acc.sum():+8.2f}   (norm at end {float(nrm(dq)):.2e})")
