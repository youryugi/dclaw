import jax, jax.numpy as jnp, numpy as np
import identifiability as I
import twin_inversion as T
f = jax.jit(I.obs)
ns = int(T.S_OBS.size)
jv = np.array(I.jvp(I.ONE, I.Z, I.Z))
print(f"JVP (phi +1 everywhere): |total| {np.linalg.norm(jv):.3f}, |inund| {np.linalg.norm(jv[:ns]):.3f}, |bldg| {np.linalg.norm(jv[ns:]):.3f}")
big = np.argsort(-np.abs(jv))[:5]
print("largest JVP components (index, value):", [(int(i), round(float(jv[i]), 2)) for i in big])
for e in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    fd = (np.array(f(I.phi0 + e * I.ONE, I.lk0, I.bed0)) - np.array(f(I.phi0 - e * I.ONE, I.lk0, I.bed0))) / (2 * e)
    cos = fd @ jv / (np.linalg.norm(fd) * np.linalg.norm(jv) + 1e-300)
    print(f"eps {e:.0e}: |FD| {np.linalg.norm(fd):9.3f} (inund {np.linalg.norm(fd[:ns]):8.3f}, bldg {np.linalg.norm(fd[ns:]):7.3f}), "
          f"cos(FD, JVP) {cos:.3f}, FD at largest-JVP components {[round(float(fd[i]), 2) for i in big]}", flush=True)
