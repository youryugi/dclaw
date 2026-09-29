"""-log posterior along the straight line between the two modes of the
single-gauge (h@1) design, plus a 2-D (log10 kref, log10 alpha_c) slice at phi = 42.09."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import inference as I
from run_twin import problem

prob, _ = problem("h@1", 0)
A = np.array([42.099, -8.339, -1.680]); B = np.array([42.090, -8.034, -2.302])
s = np.linspace(-0.3, 1.3, 33)
TH = A[None] + s[:, None] * (B - A)[None]
Z = I.to_z(TH)
nlp = -(prob.loglik(Z) + np.asarray(I.log_prior_z(jnp.asarray(Z))))
for si, v in zip(s, nlp):
    print(f"  s={si:+.2f} theta {np.round(A + si * (B - A), 3)}  -log post {v:8.3f}")
kk = np.linspace(-8.7, -7.7, 41); aa = np.linspace(-2.6, -1.3, 41)
K, Aa = np.meshgrid(kk, aa, indexing="ij")
TH2 = np.stack([np.full(K.size, 42.09), K.ravel(), Aa.ravel()], 1)
Z2 = I.to_z(TH2)
nlp2 = -(prob.loglik(Z2) + np.asarray(I.log_prior_z(jnp.asarray(Z2)))).reshape(K.shape)
np.savez("results/ridge_scan_h_1.npz", s=s, nlp=nlp, kk=kk, aa=aa, nlp2=nlp2, A=A, B=B)
print("2-D slice min", nlp2.min().round(3), "at", K.ravel()[nlp2.argmin()].round(3), Aa.ravel()[nlp2.argmin()].round(3))
