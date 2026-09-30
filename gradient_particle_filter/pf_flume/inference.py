"""Posterior for theta = (phi_deg, log10 kref, log10 alpha_c) from flume gauge
time series, and the samplers compared on it.

Parameterisation: a uniform box prior on theta, sampled in unconstrained
z = logit((theta - lo) / (hi - lo)), where the prior density is
sum(log s + log(1 - s)), s = sigmoid(z). Gradient-based moves never leave
the box.

Every sampler takes the same `Problem` and a PRNG key and returns
(particles z, log weights, number of forward-model evaluations, number of
forward+gradient evaluations, wall time). Cost is compared in wall time on
one GPU, where batching over particles is what makes this affordable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

LO = np.array([34.0, -10.0, -2.5])
HI = np.array([46.0, -7.0, -1.0])


def set_prior(lo, hi):
    """Switch the uniform prior box (and with it the parameter dimension).
    Call before building a Problem: to_theta is traced into its jitted functions."""
    global LO, HI
    LO, HI = np.asarray(lo, float), np.asarray(hi, float)


def dim():
    return len(LO)


def to_theta(z):
    return LO + (HI - LO) * jax.nn.sigmoid(z)


def to_z(theta):
    s = (np.asarray(theta) - LO) / (HI - LO)
    return np.log(s / (1 - s))


def log_prior_z(z):
    return jnp.sum(jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z), axis=-1)


@dataclass
class Problem:
    simulate: callable          # theta -> (obs, t)
    y: np.ndarray               # observed (n_t, ...)
    sigma: np.ndarray           # noise std, broadcastable to y
    batch: int = 256            # particles per device call
    mask: np.ndarray = None     # 1 = observed, 0 = not used (default: all)
    n_t_used: int = None        # use only the first n_t_used observation times (default: all)
    ar1: float = 0.0            # AR(1) correlation of consecutive residuals in each series (0 = independent)

    def __post_init__(self):
        y, sig = jnp.asarray(self.y), jnp.asarray(self.sigma)
        mask = jnp.ones_like(y) if self.mask is None else jnp.asarray(self.mask, dtype=y.dtype)
        self._mask = mask

        rho = float(self.ar1)

        def whiten(r):
            """AR(1) residuals along time within each observed series: the first observed entry
            stays, later ones become (r_t - rho r_{t-1}) / sqrt(1 - rho^2) (exact for a stationary
            AR(1) with unit marginal variance; the log-determinant is a constant)."""
            if rho == 0.0:
                return r
            prev_r = jnp.concatenate([jnp.zeros_like(r[:1]), r[:-1]])
            prev_m = jnp.concatenate([jnp.zeros_like(mask[:1]), mask[:-1]])
            w = jnp.where(prev_m > 0, (r - rho * prev_r) / jnp.sqrt(1.0 - rho ** 2), r)
            return w * mask

        def terms(z):
            """log-likelihood contribution of each observation time, shape (n_t,)."""
            obs, _ = self.simulate(to_theta(z))
            r = whiten((obs - y) / sig * mask)
            return -0.5 * jnp.sum((r * r).reshape(r.shape[0], -1), axis=1)

        def loglik(z):
            tr = terms(z)
            n = tr.shape[0] if self.n_t_used is None else self.n_t_used
            return jnp.sum(tr[:n])

        def resid(z):
            """all observation times, flattened time-major (callers slice a prefix)."""
            obs, _ = self.simulate(to_theta(z))
            return whiten((obs - y) / sig * mask).ravel()

        self._terms = jax.jit(jax.vmap(terms))
        self._obs = jax.jit(jax.vmap(lambda z: self.simulate(to_theta(z))[0]))

        self._ll = jax.jit(jax.vmap(loglik))
        # forward mode: the model's while_loop is not reverse-differentiable,
        # and with 3 parameters forward mode is the cheaper one anyway
        def ll_and_grad(z):
            g, ll = jax.jacfwd(lambda z: (loglik(z), loglik(z)), has_aux=True)(z)
            return ll, g

        self._ll_grad = jax.jit(jax.vmap(ll_and_grad))
        self._res_jac = jax.jit(lambda z: (resid(z), jax.jacfwd(resid)(z)))
        self._res_jac_b = jax.jit(jax.vmap(lambda z: (resid(z), jax.jacfwd(resid)(z))))
        self.n_fwd = 0
        self.n_grad = 0

    def _batched(self, f, Z):
        out = [_call_padded(f, Z[i:i + self.batch]) for i in range(0, len(Z), self.batch)]
        return jax.tree_util.tree_map(lambda *a: np.concatenate([np.asarray(x) for x in a]), *out)

    def loglik(self, Z):
        self.n_fwd += len(Z)
        return self._batched(self._ll, Z)

    def obs(self, Z):
        """(N, n_t, ...) simulated observations."""
        self.n_fwd += len(Z)
        return self._batched(self._obs, Z)

    def terms_from_obs(self, O):
        m = np.asarray(self._mask)[None]
        r = (O - self.y[None]) / self.sigma[None] * m
        if self.ar1:
            pr = np.concatenate([np.zeros_like(r[:, :1]), r[:, :-1]], 1)
            pm = np.concatenate([np.zeros_like(m[:, :1]), m[:, :-1]], 1)
            r = np.where(pm > 0, (r - self.ar1 * pr) / np.sqrt(1 - self.ar1 ** 2), r) * m
        return -0.5 * (r * r).reshape(r.shape[0], r.shape[1], -1).sum(-1)

    def loglik_terms(self, Z):
        """(N, n_t) per-time log-likelihood contributions (ignores n_t_used)."""
        self.n_fwd += len(Z)
        return self._batched(self._terms, Z)

    def loglik_grad(self, Z):
        self.n_grad += len(Z)
        return self._batched(self._ll_grad, Z)

    def resid_jac_batch(self, Z, n_t=None):
        """Residuals and Jacobians for several z at once, first n_t
        observation times (default: n_t_used, else all)."""
        self.n_grad += len(Z)
        r, J = _call_padded(self._res_jac_b, np.asarray(Z))
        n_t = n_t or self.n_t_used or self.y.shape[0]
        rows = n_t * (r.shape[1] // self.y.shape[0])
        return np.array(r[:, :rows]), np.array(J[:, :rows])          # writable copies

    def resid_jac(self, z, n_t=None):
        """Residuals and Jacobian for the first n_t observation times
        (default: n_t_used, else all)."""
        self.n_grad += 1
        r, J = self._res_jac(jnp.asarray(z))
        n_t = n_t or self.n_t_used or self.y.shape[0]
        rows = n_t * (r.shape[0] // self.y.shape[0])
        return np.asarray(r)[:rows], np.asarray(J)[:rows]


def _call_padded(f, Z):
    """Call a jitted, vmapped f on Z padded (with copies of its first row) to
    the next power of two, and drop the padding. Every new batch shape
    recompiles the whole model (and its forward-mode derivative) -- ~20 s
    each on the flume -- and the batched Gauss-Newton shrinks its batch by one
    whenever a start converges (16, 15, ..., 1): up to ~300 s of compilation
    per call site that would otherwise dominate the gradient-based methods."""
    Z = np.asarray(Z)
    n = len(Z)
    m = 1 << max(0, (n - 1).bit_length())
    if m > n:
        Z = np.concatenate([Z, np.repeat(Z[:1], m - n, axis=0)])
    out = f(jnp.asarray(Z))
    return jax.tree_util.tree_map(lambda x: np.asarray(x)[:n], out)


def _ess(logw):
    w = np.exp(logw - logw.max())
    return w.sum() ** 2 / (w ** 2).sum()


def _resample(rng, logw):
    """Systematic resampling -> indices."""
    n = len(logw)
    w = np.exp(logw - logw.max()); w /= w.sum()
    u = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), u), n - 1)


def _next_temperature(ll, beta, target):
    """Largest beta' in (beta, 1] with ESS of weights (beta'-beta)*ll >= target."""
    if _ess((1.0 - beta) * ll) >= target:
        return 1.0
    lo, hi = beta, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _ess((mid - beta) * ll) >= target:
            lo = mid
        else:
            hi = mid
    return lo


# ----------------------------------------------------------------- samplers
def bootstrap_is(prob: Problem, n: int, seed: int = 0):
    """Importance sampling from the prior."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    Z = rng.logistic(size=(n, dim()))                 # uniform prior in theta <=> logistic in z
    logw = prob.loglik(Z)
    return Z, logw, time.time() - t0


def tempered_smc(prob: Problem, n: int, move: str = "rw", n_moves: int = 5, seed: int = 0, max_stages: int = 200,
                 verbose: bool = False):
    """Adaptive likelihood tempering (ESS = n/2 per stage), systematic
    resampling, then n_moves MCMC moves per stage targeting
    prior * lik^beta: move="rw" random-walk Metropolis with covariance
    2.38^2/d * particle covariance, move="mala" preconditioned MALA with the
    same covariance as preconditioner. Step scales adapt to acceptance."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    Z = rng.logistic(size=(n, dim()))
    if move == "mala":
        ll, g = prob.loglik_grad(Z)
    else:
        ll, g = prob.loglik(Z), None
    beta, scale, stages = 0.0, 1.0, []
    while beta < 1.0 and len(stages) < max_stages:
        new_beta = _next_temperature(ll, beta, n / 2)
        logw = (new_beta - beta) * ll
        idx = _resample(rng, logw)
        Z, ll = Z[idx], ll[idx]
        g = g[idx] if g is not None else None
        beta = new_beta
        C = np.cov(Z.T) + 1e-9 * np.eye(dim())
        L = np.linalg.cholesky(C)
        acc_tot = 0.0
        for _ in range(n_moves):
            lp = log_prior_z(Z) + beta * ll
            if move == "rw":
                eps = rng.normal(size=Z.shape) @ L.T
                Zp = Z + scale * 2.38 / np.sqrt(dim()) * eps
                llp = prob.loglik(Zp)
                lpp = log_prior_z(Zp) + beta * llp
                log_a = lpp - lp
            else:
                # preconditioned MALA: z' = z + h/2 C grad + sqrt(h) L xi
                h = (scale * 1.65 / dim() ** (1 / 6)) ** 2
                grad = np.asarray(jax.grad(lambda z: jnp.sum(log_prior_z(z)))(jnp.asarray(Z))) + beta * g
                mean = Z + 0.5 * h * grad @ C
                Zp = mean + np.sqrt(h) * rng.normal(size=Z.shape) @ L.T
                llp, gp = prob.loglik_grad(Zp)
                lpp = log_prior_z(Zp) + beta * llp
                gradp = np.asarray(jax.grad(lambda z: jnp.sum(log_prior_z(z)))(jnp.asarray(Zp))) + beta * gp
                meanp = Zp + 0.5 * h * gradp @ C
                Ci = np.linalg.inv(C)
                q_fwd = -0.5 / h * np.einsum("ni,ij,nj->n", Zp - mean, Ci, Zp - mean)
                q_bwd = -0.5 / h * np.einsum("ni,ij,nj->n", Z - meanp, Ci, Z - meanp)
                log_a = lpp - lp + q_bwd - q_fwd
            accept = np.log(rng.random(n)) < np.nan_to_num(log_a, nan=-np.inf)
            Z = np.where(accept[:, None], Zp, Z)
            ll = np.where(accept, llp, ll)
            if g is not None:
                g = np.where(accept[:, None], gp, g)
            a = accept.mean(); acc_tot += a
            target = 0.3 if move == "rw" else 0.57
            scale *= np.exp(a - target)
        stages.append((beta, acc_tot / n_moves, scale))
        if verbose:
            print(f"  stage {len(stages):3d}: beta={beta:.3e} acc={acc_tot / n_moves:.2f} scale={scale:.3f} "
                  f"t={time.time() - t0:.0f}s", flush=True)
    return Z, np.zeros(n), time.time() - t0, stages


def gauss_newton(prob: Problem, z_start, n_t=None, gn_iters: int = 30, tol: float = 1e-6, verbose: bool = False):
    """Levenberg-Marquardt to the posterior mode using the first n_t
    observation times. Returns (z_mode, GN covariance (J^T J + prior
    Hessian approx)^-1, -log posterior, iterations used)."""
    z = np.asarray(z_start, float)
    lam = 1e-2

    def neg_logpost(z, r):
        return 0.5 * r @ r - float(log_prior_z(jnp.asarray(z)))

    r, J = prob.resid_jac(z, n_t)
    f = neg_logpost(z, r)
    it = 0
    for it in range(gn_iters):
        gp = np.asarray(jax.grad(lambda z: log_prior_z(z))(jnp.asarray(z)))
        H = J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-9)
        dz = np.linalg.solve(H, -(J.T @ r) + gp)
        zn = z + dz
        rn, Jn = prob.resid_jac(zn, n_t)
        fn = neg_logpost(zn, rn)
        if fn < f:
            z, r, J, f, lam = zn, rn, Jn, fn, lam / 3
            if np.abs(dz).max() < tol:
                break
        else:
            lam *= 5
            if lam > 1e8:
                break
        if verbose:
            print(f"  GN {it:2d}: -log post {f:.3f} lam {lam:.1e} theta {np.asarray(to_theta(jnp.asarray(z)))}", flush=True)
    # prior curvature (logistic density in z) keeps C finite along unidentified directions
    hp = -np.asarray(jax.hessian(lambda z: log_prior_z(z))(jnp.asarray(z)))
    C = np.linalg.inv(J.T @ J + hp + 1e-9 * np.eye(len(z)))
    return z, C, f, it + 1


def mvt_sample(rng, mean, C, n, dof):
    L = np.linalg.cholesky(C)
    u = rng.chisquare(dof, n) / dof
    X = rng.normal(size=(n, len(mean))) @ L.T / np.sqrt(u)[:, None]
    return mean + X


def mvt_logpdf(Z, mean, C, dof):
    """up to a constant (the same for every call with the same C)."""
    X = Z - mean
    q = np.einsum("ni,ij,nj->n", X, np.linalg.inv(C), X)
    return -0.5 * (dof + len(mean)) * np.log1p(q / dof)


def laplace_is(prob: Problem, n: int, z_start, gn_iters: int = 30, inflate: float = 1.5, dof: float = 5.0,
               seed: int = 0, verbose: bool = False):
    """Gauss-Newton to the posterior mode from z_start, then importance
    sampling from a multivariate t centred there with the Gauss-Newton
    covariance (inflated). Weights make it exact."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    z, C0, _, _ = gauss_newton(prob, z_start, gn_iters=gn_iters, verbose=verbose)
    C = inflate ** 2 * C0
    Z = mvt_sample(rng, z, C, n, dof)
    logw = prob.loglik(Z) + np.asarray(log_prior_z(jnp.asarray(Z))) - mvt_logpdf(Z, z, C, dof)
    return Z, logw, time.time() - t0, (z, C)


def laplace_smc(prob: Problem, n: int, z_start, n_moves: int = 5, inflate: float = 1.5, dof: float = 5.0,
                seed: int = 0, verbose: bool = False, max_stages: int = 100):
    """Laplace proposal q (Gauss-Newton mode, t-distribution), then adaptive
    tempering from q to the posterior pi along q^(1-beta) pi^beta with
    random-walk moves -- exact like tempered SMC, but starting where the
    posterior is instead of at the prior. If the Laplace fit is good the
    first stage already reaches beta = 1 and this is Laplace IS plus moves."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    zm, C0, _, _ = gauss_newton(prob, z_start, verbose=verbose)
    C = inflate ** 2 * C0
    Z = mvt_sample(rng, zm, C, n, dof)
    lq = mvt_logpdf(Z, zm, C, dof)
    lpi = prob.loglik(Z) + np.asarray(log_prior_z(jnp.asarray(Z)))
    beta, scale, stages = 0.0, 1.0, []
    while beta < 1.0 and len(stages) < max_stages:
        d = lpi - lq
        d = np.where(np.isfinite(d), d, -np.inf)
        new_beta = _next_temperature(d, beta, n / 2)
        idx = _resample(rng, (new_beta - beta) * d)
        Z, lq, lpi, beta = Z[idx], lq[idx], lpi[idx], new_beta
        Ck = np.cov(Z.T) + 1e-9 * np.eye(dim())
        L = np.linalg.cholesky(Ck)
        acc = 0.0
        for _ in range(n_moves):
            Zp = Z + scale * 2.38 / np.sqrt(dim()) * rng.normal(size=Z.shape) @ L.T
            lqp = mvt_logpdf(Zp, zm, C, dof)
            lpip = prob.loglik(Zp) + np.asarray(log_prior_z(jnp.asarray(Zp)))
            log_a = ((1 - beta) * lqp + beta * lpip) - ((1 - beta) * lq + beta * lpi)
            a = np.log(rng.random(n)) < np.nan_to_num(log_a, nan=-np.inf)
            Z, lq, lpi = np.where(a[:, None], Zp, Z), np.where(a, lqp, lq), np.where(a, lpip, lpi)
            acc += a.mean(); scale *= np.exp(a.mean() - 0.3)
        stages.append((beta, acc / n_moves, scale))
        if verbose:
            print(f"  stage {len(stages):3d}: beta={beta:.3e} acc={acc / n_moves:.2f} t={time.time() - t0:.0f}s", flush=True)
    return Z, np.zeros(n), time.time() - t0, stages


def ibis(prob: Problem, n: int, move: str = "rw", n_moves: int = 5, seed: int = 0, inflate: float = 1.5,
         dof: float = 5.0, verbose: bool = False):
    """Iterated batch importance sampling (Chopin 2002) over observation
    times: particles in theta (from the prior) are reweighted by each new
    time's likelihood; when ESS < n/2 they are resampled and moved with
    MCMC targeting p(theta | y_1..y_k):
      move="rw":      n_moves random-walk Metropolis moves (particle covariance)
      move="laplace": Gauss-Newton on y_1..y_k from the particle mean, then
                      n_moves independent Metropolis-Hastings moves with the
                      t(mode, inflate^2 C) proposal.
    The model is deterministic, so every particle's whole observation series
    is simulated once and later times cost nothing until a move.
    Returns history per time (mean, sd, ESS, evaluations, wall) and the
    particles' simulated series at every time (for predictions)."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    n_t = prob.y.shape[0]
    Z = rng.logistic(size=(n, dim()))
    O = prob.obs(Z)
    T = prob.terms_from_obs(O)
    logw = np.zeros(n)
    hist = []
    snaps = {}
    for k in range(n_t):
        logw = logw + T[:, k]
        ess = _ess(logw)
        moved, acc = False, np.nan
        if ess < n / 2:
            idx = _resample(rng, logw)
            Z, O, T, logw = Z[idx], O[idx], T[idx], np.zeros(n)
            lp = lambda Zc, Tc: np.asarray(log_prior_z(jnp.asarray(Zc))) + Tc[:, :k + 1].sum(1)
            accs = []
            if move == "rw":
                L = np.linalg.cholesky(np.cov(Z.T) + 1e-9 * np.eye(dim()))
                for _ in range(n_moves):
                    Zp = Z + 2.38 / np.sqrt(dim()) * rng.normal(size=Z.shape) @ L.T
                    Op = prob.obs(Zp); Tp = prob.terms_from_obs(Op)
                    a = np.log(rng.random(n)) < np.nan_to_num(lp(Zp, Tp) - lp(Z, T), nan=-np.inf)
                    Z, O, T = np.where(a[:, None], Zp, Z), np.where(a[:, None, None, None], Op, O), np.where(a[:, None], Tp, T)
                    accs.append(a.mean())
            elif move == "laplace_multi":
                # several diverse high-posterior particles as starts, batched Gauss-Newton on y_1..y_k,
                # mixture of all converged modes (+5% prior) as the independent proposal: a single-mode
                # proposal would move particles out of every other mode (they propose into the one mode
                # and are accepted because q is ~0 where they are)
                lpc = lp(Z, T)
                TH = np.asarray(to_theta(jnp.asarray(Z)))
                span = HI - LO
                starts = []
                for i in np.argsort(-lpc):     # 16 starts >= 10% of the prior range apart (8 at 5% found only
                    if all(np.linalg.norm((TH[i] - TH[j]) / span) > 0.1 for j in starts):   # mode A at one step)
                        starts.append(i)
                    if len(starts) == 16:
                        break
                Zm, Cm, fm, conv, _, dec = gauss_newton_batch(prob, Z[starts], iters=40, n_t=k + 1)
                if not conv.any():
                    conv = fm == fm.min()
                modes = []
                for j in np.argsort(fm):
                    if conv[j] and not any(np.max(np.abs(Zm[j] - m[0]) / np.sqrt(np.diag(m[1]))) < 0.5 for m in modes):
                        modes.append((Zm[j], Cm[j], fm[j]))
                modes = [m for m in modes if m[2] - modes[0][2] < 20.0]
                qm = mixture_proposal(modes, inflate=inflate, dof=dof, prior_weight=0.05)
                lq = qm.logpdf(Z)
                for _ in range(n_moves):
                    Zp = qm.sample(rng, n)
                    Op = prob.obs(Zp); Tp = prob.terms_from_obs(Op)
                    lqp = qm.logpdf(Zp)
                    log_a = lp(Zp, Tp) - lp(Z, T) + lq - lqp
                    a = np.log(rng.random(n)) < np.nan_to_num(log_a, nan=-np.inf)
                    Z, O, T, lq = (np.where(a[:, None], Zp, Z), np.where(a[:, None, None, None], Op, O),
                                   np.where(a[:, None], Tp, T), np.where(a, lqp, lq))
                    accs.append(a.mean())
                if verbose:
                    print(f"    {len(modes)} mode(s): " + ", ".join(f"{np.round(np.asarray(to_theta(jnp.asarray(m[0]))), 3)} ({m[2]:.2f})" for m in modes), flush=True)
            else:
                # from the most probable particle: after resampling the particles are in the right basin,
                # while their mean or a fixed start can sit in a local minimum (see explore.py)
                zm, C0, _, _ = gauss_newton(prob, Z[np.argmax(lp(Z, T))], n_t=k + 1)
                C = inflate ** 2 * C0
                lq = mvt_logpdf(Z, zm, C, dof)
                for _ in range(n_moves):
                    Zp = mvt_sample(rng, zm, C, n, dof)
                    Op = prob.obs(Zp); Tp = prob.terms_from_obs(Op)
                    lqp = mvt_logpdf(Zp, zm, C, dof)
                    log_a = lp(Zp, Tp) - lp(Z, T) + lq - lqp
                    a = np.log(rng.random(n)) < np.nan_to_num(log_a, nan=-np.inf)
                    Z, O, T, lq = (np.where(a[:, None], Zp, Z), np.where(a[:, None, None, None], Op, O),
                                   np.where(a[:, None], Tp, T), np.where(a, lqp, lq))
                    accs.append(a.mean())
            moved, acc = True, float(np.mean(accs))
        w = np.exp(logw - logw.max()); w /= w.sum()
        th = np.asarray(to_theta(jnp.asarray(Z)))
        m = w @ th
        hist.append(dict(k=k, ess=ess, moved=moved, acc=acc, mean=m, sd=np.sqrt(w @ (th - m) ** 2),
                         n_fwd=prob.n_fwd, n_grad=prob.n_grad, wall=time.time() - t0))
        snaps[k] = (th, w, O)
        if verbose:
            print(f"  t_k {k:2d}: ESS before {ess:7.1f} {'moved acc %.2f' % acc if moved else '':16s} mean {np.round(m, 3)} "
                  f"sd {np.round(hist[-1]['sd'], 4)} fwd {prob.n_fwd} grad {prob.n_grad} t={time.time() - t0:.0f}s", flush=True)
    return hist, snaps


class MixtureT:
    """Mixture of multivariate t's (Laplace fits at several modes) plus an
    optional defensive prior component, with normalised log density."""

    def __init__(self, means, covs, weights, dof=5.0, prior_weight=0.0):
        self.means, self.covs, self.dof = [np.asarray(m) for m in means], [np.asarray(c) for c in covs], dof
        w = np.asarray(weights, float); w = w / w.sum() * (1.0 - prior_weight)
        self.w = np.append(w, prior_weight)

    def _t_logpdf(self, Z, m, C):
        from scipy.special import gammaln
        d, nu = len(m), self.dof
        X = Z - m
        q = np.einsum("ni,ij,nj->n", X, np.linalg.inv(C), X)
        return (gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * np.log(nu * np.pi)
                - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * (nu + d) * np.log1p(q / nu))

    def logpdf(self, Z):
        comps = [np.log(w) + self._t_logpdf(Z, m, C) for w, m, C in zip(self.w[:-1], self.means, self.covs) if w > 0]
        if self.w[-1] > 0:
            comps.append(np.log(self.w[-1]) + np.asarray(log_prior_z(jnp.asarray(Z))))
        return np.logaddexp.reduce(np.stack(comps), axis=0)

    def sample(self, rng, n):
        c = rng.choice(len(self.w), size=n, p=self.w)
        Z = np.empty((n, len(self.means[0])))
        for j in range(len(self.w)):
            idx = np.nonzero(c == j)[0]
            if len(idx) == 0:
                continue
            Z[idx] = rng.logistic(size=(len(idx), Z.shape[1])) if j == len(self.w) - 1 else \
                mvt_sample(rng, self.means[j], self.covs[j], len(idx), self.dof)
        return Z


def find_modes(prob: Problem, rng, m_screen=256, k_starts=4, keep_within=20.0, n_t=None, verbose=False):
    """Screen m_screen prior draws by likelihood, run Gauss-Newton from the k
    best, merge duplicates, keep modes whose -log posterior is within
    keep_within of the best. Returns [(z_mode, C, -log post)] best first."""
    Z = rng.logistic(size=(m_screen, dim()))
    if n_t is None:
        ll = prob.loglik(Z)
    else:
        ll = prob.loglik_terms(Z)[:, :n_t].sum(1)
    order = np.argsort(-ll)
    modes = []
    for i in order[:k_starts]:
        z, C, f, it = gauss_newton(prob, Z[i], n_t=n_t, gn_iters=40)
        if verbose:
            print(f"  GN from screened start {np.round(np.asarray(to_theta(jnp.asarray(Z[i]))), 3)} -> "
                  f"{np.round(np.asarray(to_theta(jnp.asarray(z))), 3)} -log post {f:.2f} ({it} it)", flush=True)
        # duplicate only if the two Gauss-Newton fixed points (nearly) coincide: along a flat ridge
        # two genuinely distinct modes can be < 3 posterior sd apart (the single-gauge flume design:
        # modes 2-3 sd apart were merged, and k = 4 starts behaved exactly like k = 1)
        if any(np.max(np.abs(z - m[0]) / np.sqrt(np.diag(m[1]))) < 0.1 for m in modes):
            continue
        modes.append((z, C, f))
    modes.sort(key=lambda m: m[2])
    return [m for m in modes if m[2] - modes[0][2] < keep_within]


def ridge_inflated(C, inflate=1.5, ridge_inflate=None, ridge_k=2):
    """inflate^2 * C, and along the ridge_k largest-variance eigendirections of C
    (the weakly identified ones -- the Gauss-Newton Hessian's smallest
    eigenvalues) ridge_inflate^2 * C instead. Uniform widening wastes samples in
    the well-identified directions (7d flume: 2.5x everywhere dropped the IS
    ESS to ~20); widening only the ridge targets where the Laplace fit is short."""
    if ridge_inflate is None:
        return inflate ** 2 * C
    lam, V = np.linalg.eigh(C)
    f2 = np.full(len(lam), inflate ** 2)
    f2[np.argsort(lam)[-ridge_k:]] = ridge_inflate ** 2
    return (V * (f2 * lam)) @ V.T


def mixture_proposal(modes, inflate=1.5, dof=5.0, prior_weight=0.05, ridge_inflate=None, ridge_k=2):
    # component weights ~ Laplace evidence exp(-f) sqrt(det C)
    logw = np.array([-f + 0.5 * np.linalg.slogdet(C)[1] for _, C, f in modes])
    return MixtureT([m[0] for m in modes], [ridge_inflated(m[1], inflate, ridge_inflate, ridge_k) for m in modes],
                    np.exp(logw - logw.max()), dof=dof, prior_weight=prior_weight)


def robust_laplace(prob: Problem, n: int, seed: int = 0, n_moves: int = 5, m_screen=256, k_starts=4,
                   tempering=True, verbose: bool = False, max_stages: int = 100):
    """Screened multi-start Gauss-Newton -> mixture-of-t (+5% prior)
    proposal -> importance weights; with tempering=True, then adaptive
    tempering from the proposal q to the posterior along q^(1-b) pi^b with
    random-walk moves (stops at once when the weights are already good)."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    modes = find_modes(prob, rng, m_screen, k_starts, verbose=verbose)
    q = mixture_proposal(modes)
    Z = q.sample(rng, n)
    lq = q.logpdf(Z)
    lpi = prob.loglik(Z) + np.asarray(log_prior_z(jnp.asarray(Z)))
    info = dict(modes=[(np.asarray(to_theta(jnp.asarray(m[0]))), m[2]) for m in modes],
                is_ess=_ess(lpi - lq), t_is=time.time() - t0)
    if verbose:
        print(f"  {len(modes)} mode(s); IS ESS {info['is_ess']:.1f} of {n} after {info['t_is']:.0f}s", flush=True)
    if not tempering:
        return Z, lpi - lq, time.time() - t0, info
    beta, scale, stages = 0.0, 1.0, []
    while beta < 1.0 and len(stages) < max_stages:
        d = lpi - lq
        new_beta = _next_temperature(d, beta, n / 2)
        idx = _resample(rng, (new_beta - beta) * d)
        Z, lq, lpi, beta = Z[idx], lq[idx], lpi[idx], new_beta
        L = np.linalg.cholesky(np.cov(Z.T) + 1e-9 * np.eye(dim()))
        acc = 0.0
        for _ in range(n_moves):
            Zp = Z + scale * 2.38 / np.sqrt(dim()) * rng.normal(size=Z.shape) @ L.T
            lqp = q.logpdf(Zp)
            lpip = prob.loglik(Zp) + np.asarray(log_prior_z(jnp.asarray(Zp)))
            log_a = ((1 - beta) * lqp + beta * lpip) - ((1 - beta) * lq + beta * lpi)
            a = np.log(rng.random(n)) < np.nan_to_num(log_a, nan=-np.inf)
            Z, lq, lpi = np.where(a[:, None], Zp, Z), np.where(a, lqp, lq), np.where(a, lpip, lpi)
            acc += a.mean(); scale *= np.exp(a.mean() - 0.3)
        stages.append((beta, acc / n_moves, scale))
        if verbose:
            print(f"  stage {len(stages):3d}: beta={beta:.3e} acc={acc / n_moves:.2f} t={time.time() - t0:.0f}s", flush=True)
    info["stages"] = stages
    return Z, np.zeros(n), time.time() - t0, info


def gauss_newton_batch(prob: Problem, Z0, iters: int = 100, tol: float = 0.05, n_t=None, verbose: bool = False,
                       lam0: float = 1e-4, lam_down: float = 10.0, lam_up: float = 10.0):
    """Levenberg-Marquardt from several starts at once (one batched Jacobian
    call per iteration for the starts still running). A start counts as
    converged when the Newton decrement g^T H^-1 g of -log posterior is
    below tol. The decrement is ~2x the remaining gap to the minimum in
    -log posterior; tol = 0.05 means within ~0.025 of it, far inside the
    posterior's own scale (1 sd ~ 0.5). Absolute tolerances much below that
    fail on the objective's numerical noise floor, which grows with the
    number of observations: 1e-8 dropped the single-gauge flume's second
    mode (5 of 5 starts stalled just above it), 1e-3 rejected all 16 starts
    that had reached the full-data mode (decrements stalled at 6e-3-1e-2).
    The rest are returned with converged=False.
    Damping: lam0 = 1e-4, /10 on success, x10 on failure (lm_schedule.py:
    16 and 12 iterations on the single-gauge and full-data flume, against 60+
    and 10 for 1e-2, /3, x5 -- the number of sequential batched Jacobian
    calls, not their size, is what costs time on the GPU).
    Returns z (K,d), C (K,d,d), f (K,), converged (K,), iterations used,
    final Newton decrements (K,)."""
    Z = np.array(Z0, float)
    K = len(Z)
    lam = np.full(K, lam0)
    lp_grad = jax.jit(jax.vmap(jax.grad(lambda z: log_prior_z(z))))
    lp_hess = jax.jit(jax.vmap(jax.hessian(lambda z: log_prior_z(z))))

    def nlp(r, Zc):
        return 0.5 * (r * r).sum(1) - np.asarray(log_prior_z(jnp.asarray(Zc)))

    r, J = prob.resid_jac_batch(Z, n_t)
    f = nlp(r, Z)
    conv = np.zeros(K, bool)
    active = np.ones(K, bool)
    it = 0
    for it in range(iters):
        idx = np.nonzero(active)[0]
        if len(idx) == 0:
            break
        gp = np.asarray(lp_grad(jnp.asarray(Z[idx])))
        JtJ = np.einsum("kmi,kmj->kij", J[idx], J[idx])
        g = -np.einsum("kmi,km->ki", J[idx], r[idx]) + gp                       # = -grad(-log post)
        H = JtJ - np.asarray(lp_hess(jnp.asarray(Z[idx])))
        dec = np.einsum("ki,ki->k", g, np.linalg.solve(H, g[..., None])[..., 0])  # Newton decrement
        done = dec < tol
        conv[idx[done]] = True
        active[idx[done]] = False
        idx, g, JtJ = idx[~done], g[~done], JtJ[~done]
        if len(idx) == 0:
            break
        Hd = JtJ + lam[idx, None, None] * np.stack([np.diag(np.diag(A) + 1e-9) for A in JtJ])
        Zn = Z[idx] + np.linalg.solve(Hd, g[..., None])[..., 0]
        rn, Jn = prob.resid_jac_batch(Zn, n_t)
        fn = nlp(rn, Zn)
        better = fn < f[idx]
        ib, iw = idx[better], idx[~better]
        Z[ib], r[ib], J[ib], f[ib] = Zn[better], rn[better], Jn[better], fn[better]
        lam[ib] /= lam_down
        lam[iw] *= lam_up
        active[iw[lam[iw] > 1e10]] = False
        if verbose:
            print(f"  batched GN it {it:3d}: running {active.sum():2d}, converged {conv.sum():2d}, best -log post {f.min():.3f}",
                  flush=True)
    H = np.einsum("kmi,kmj->kij", J, J) - np.asarray(lp_hess(jnp.asarray(Z)))
    C = np.linalg.inv(H + 1e-9 * np.eye(Z.shape[1]))
    g = -np.einsum("kmi,km->ki", J, r) + np.asarray(lp_grad(jnp.asarray(Z)))
    dec = np.einsum("ki,ki->k", g, np.linalg.solve(H, g[..., None])[..., 0])
    return Z, C, f, conv, it + 1, dec


def find_modes_diverse(prob: Problem, rng, m_screen=512, k_starts=16, min_sep=0.1, keep_within=20.0,
                       iters=40, merge_sd=0.5, verbose=False):
    """Screen m_screen prior draws; take the best-likelihood draws that are at
    least min_sep (fraction of the prior range) apart as k_starts starts;
    batched Gauss-Newton; keep converged modes only, merge coinciding ones
    (< 0.1 sd), drop modes more than keep_within above the best.
    iters=40: on the single-gauge flume every start that converged at all did
    so within 29 iterations (the rest never did in 100)."""
    Z = rng.logistic(size=(m_screen, dim()))
    ll = prob.loglik(Z)
    span = HI - LO
    TH = np.asarray(to_theta(jnp.asarray(Z)))
    starts = []
    for i in np.argsort(-ll):
        if all(np.linalg.norm((TH[i] - TH[j]) / span) > min_sep for j in starts):
            starts.append(i)
        if len(starts) == k_starts:
            break
    Zm, C, f, conv, it, dec = gauss_newton_batch(prob, Z[starts], iters=iters, verbose=verbose)
    modes = []
    if not conv.any():
        print(f"  WARNING: no Gauss-Newton start converged (smallest Newton decrement {dec.min():.1e}); "
              f"using the lowest -log posterior point", flush=True)
        conv = f == f.min()
    for k in np.argsort(f):
        if not conv[k]:
            continue
        if any(np.max(np.abs(Zm[k] - m[0]) / np.sqrt(np.diag(m[1]))) < merge_sd for m in modes):
            continue                  # < merge_sd: same mode (A and B of the single-gauge flume are 2-3 sd apart);
                                      # a small merge_sd keeps several points along a flat ridge as separate components
        modes.append((Zm[k], C[k], f[k]))
    info = dict(n_conv=int(conv.sum()), n_starts=len(starts), gn_iters=it)
    if verbose:
        for k in range(len(starts)):
            print(f"  start {np.round(TH[starts[k]], 3)} -> {np.round(np.asarray(to_theta(jnp.asarray(Zm[k]))), 3)} "
                  f"-log post {f[k]:.3f} Newton decrement {dec[k]:.1e} {'converged' if conv[k] else 'NOT converged'}", flush=True)
    return [m for m in modes if m[2] - modes[0][2] < keep_within], info


def robust2_laplace(prob: Problem, n: int, seed: int = 0, m_screen=512, k_starts=16, merge_sd=0.5,
                    tempering=False, n_moves=5, verbose=False, max_stages=100, inflate=1.5, dof=5.0,
                    adapt_rounds=0, adapt_n=512, ridge_inflate=None, ridge_k=2, wide_n=0, wide_inflate=2.0,
                    wide_dof=3.0):
    """find_modes_diverse -> mixture-of-t (+5% prior) importance sampling;
    with tempering=True, then adaptive tempering from that proposal q to the
    posterior along q^(1-b) pi^b with random-walk moves (for posteriors a
    mixture of Gaussians fits badly, e.g. the 7-parameter flume's flat ridge)."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    modes, info = find_modes_diverse(prob, rng, m_screen, k_starts, merge_sd=merge_sd, verbose=verbose)
    info["t_modes"] = time.time() - t0
    q = mixture_proposal(modes, inflate=inflate, dof=dof, ridge_inflate=ridge_inflate, ridge_k=ridge_k)
    # adaptive IS (population Monte Carlo): a mixture of Gaussians fitted at the modes covers a
    # long, curved ridge badly (samples bunch in the middle; the 7-parameter flume's sd came out
    # up to 35% too small). Each round draws adapt_n samples, and the next proposal adds a t
    # component with the importance-weighted mean and covariance of that round -- the weighted
    # sample shows how far the ridge actually extends.
    for r in range(adapt_rounds):
        Za = q.sample(rng, adapt_n)
        lwa = prob.loglik(Za) + np.asarray(log_prior_z(jnp.asarray(Za))) - q.logpdf(Za)
        wa = np.exp(lwa - lwa.max()); wa /= wa.sum()
        mw = wa @ Za
        Cw = (wa[:, None] * (Za - mw)).T @ (Za - mw) + 1e-9 * np.eye(dim())
        q = MixtureT(q.means + [mw], q.covs + [inflate ** 2 * Cw],
                     np.append(0.5 * q.w[:-1] / q.w[:-1].sum(), 0.5), dof=dof, prior_weight=q.w[-1])
        if verbose:
            print(f"  adapt round {r + 1}: ESS {1 / (wa ** 2).sum():.1f} of {adapt_n}; weighted sd "
                  f"{np.round(np.sqrt(np.diag(Cw)), 3).tolist()} (z)", flush=True)
    Z = q.sample(rng, n)
    logw = prob.loglik(Z) + np.asarray(log_prior_z(jnp.asarray(Z))) - q.logpdf(Z)
    info.update(modes=[(np.asarray(to_theta(jnp.asarray(m[0]))), m[2]) for m in modes], is_ess=_ess(logw))
    if verbose:
        print(f"  {len(modes)} converged mode(s) from {info['n_conv']}/{info['n_starts']} converged starts; "
              f"IS ESS {info['is_ess']:.1f} of {n}; {time.time() - t0:.0f}s", flush=True)
    if wide_n:
        # second stage: one wide, heavy-tailed t fitted to this weighted sample (a global picture
        # of the ridge, if too narrow) -- local Gauss-Newton curvature cannot see how a curved
        # ridge bends away from its tangents, the sample's global covariance can
        w = np.exp(logw - logw.max()); w /= w.sum()
        mw = w @ Z
        Cw = (w[:, None] * (Z - mw)).T @ (Z - mw) + 1e-9 * np.eye(dim())
        q = MixtureT([mw], [wide_inflate ** 2 * Cw], [1.0], dof=wide_dof, prior_weight=0.05)
        info["is_ess_stage1"] = info["is_ess"]
        Z = q.sample(rng, wide_n)
        logw = prob.loglik(Z) + np.asarray(log_prior_z(jnp.asarray(Z))) - q.logpdf(Z)
        info["is_ess"] = _ess(logw)
        if verbose:
            print(f"  wide second stage: IS ESS {info['is_ess']:.1f} of {wide_n}; {time.time() - t0:.0f}s", flush=True)
    if not tempering:
        return Z, logw, time.time() - t0, info
    lq = q.logpdf(Z)
    lpi = logw + lq
    beta, scale, stages = 0.0, 1.0, []
    while beta < 1.0 and len(stages) < max_stages:
        d = lpi - lq
        new_beta = _next_temperature(d, beta, n / 2)
        idx = _resample(rng, (new_beta - beta) * d)
        Z, lq, lpi, beta = Z[idx], lq[idx], lpi[idx], new_beta
        L = np.linalg.cholesky(np.cov(Z.T) + 1e-9 * np.eye(dim()))
        acc = 0.0
        for _ in range(n_moves):
            Zp = Z + scale * 2.38 / np.sqrt(dim()) * rng.normal(size=Z.shape) @ L.T
            lqp = q.logpdf(Zp)
            lpip = prob.loglik(Zp) + np.asarray(log_prior_z(jnp.asarray(Zp)))
            log_a = ((1 - beta) * lqp + beta * lpip) - ((1 - beta) * lq + beta * lpi)
            a = np.log(rng.random(n)) < np.nan_to_num(log_a, nan=-np.inf)
            Z, lq, lpi = np.where(a[:, None], Zp, Z), np.where(a, lqp, lq), np.where(a, lpip, lpi)
            acc += a.mean(); scale *= np.exp(a.mean() - 0.3)
        stages.append((beta, acc / n_moves, scale))
        if verbose:
            print(f"  stage {len(stages):3d}: beta={beta:.3e} acc={acc / n_moves:.2f} t={time.time() - t0:.0f}s", flush=True)
    info["stages"] = stages
    return Z, np.zeros(n), time.time() - t0, info


def esmda(prob: Problem, n: int, n_assim: int = 4, seed: int = 0, verbose: bool = False):
    """Ensemble smoother with multiple data assimilation (Emerick & Reynolds 2013) -- the
    gradient-free workhorse of geoscience inversion. An ensemble of n parameter vectors (in z,
    drawn from the prior) is updated n_assim times with the data, each time with the
    observation-error covariance inflated by alpha = n_assim (sum 1/alpha = 1):
        z_j <- z_j + C_zd (C_dd + alpha R)^-1 (y + sqrt(alpha) R^1/2 eps_j - g(z_j)),
    with C_zd, C_dd the ensemble (cross-)covariances of parameters and simulated data. Uses only
    forward runs (n * n_assim). Exact for a linear-Gaussian problem; for a nonlinear, curved or
    multimodal posterior an approximation. Returns (Z, equal log weights, wall time)."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    m = np.asarray(prob._mask).astype(bool)
    y = prob.y[m]
    sig = prob.sigma[m]
    Z = rng.logistic(size=(n, dim()))
    alpha = float(n_assim)
    for k in range(n_assim):
        D = prob.obs(Z)[:, m]                                       # (n, n_obs)
        dz = Z - Z.mean(0)
        dd = D - D.mean(0)
        Czd = dz.T @ dd / (n - 1)
        Cdd = dd.T @ dd / (n - 1)
        pert = y[None] + np.sqrt(alpha) * sig[None] * rng.normal(size=D.shape)
        K = np.linalg.solve(Cdd + alpha * np.diag(sig ** 2), Czd.T).T   # (d, n_obs)
        Z = Z + (pert - D) @ K.T
        if verbose:
            misfit = np.mean(np.sum(((D - y[None]) / sig[None]) ** 2, 1))
            print(f"  ES-MDA step {k + 1}/{n_assim}: mean chi^2 before update {misfit:.1f} ({len(y)} obs), "
                  f"t={time.time() - t0:.0f}s", flush=True)
    return Z, np.zeros(n), time.time() - t0
