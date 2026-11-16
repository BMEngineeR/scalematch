"""Eigendecomposition-free analytic spectral null for the v3 swap / p_in tests (Chebyshev / KPM). Standalone; no pipeline file is changed.

Setting (as analytic_null/code/specnull.py): L = L_sym of the section graph, spectrum in [0, 2]; Chebyshev variable x = lambda - 1, so
T_j(L - I) is the j-th Chebyshev polynomial of the shifted Laplacian Lt = L - I (spectrum in [-1, 1]).
Bands: LOW = exp(-t_L L), MID = exp(-t_S L) - exp(-t_L L), HIGH = I - exp(-t_S L), t = 2 (sigma / pitch)^2.
  exp(-t L) = sum_j c_j(t) T_j(Lt),  c_j(t) = (2 - [j=0]) (-1)^j e^{-t} I_j(t)   (generating function of the modified Bessel functions).
Context C' (genes kept fixed), band residual r_b of the pipeline's perband full-data fit (standardised columns, ridge 1e-8 N),
gene g (raw log-CP10K field), statistic T_b = <h_b(L) g, r_b> = x_b' r_b (x_b = band b of g; exact, from the pipeline's features).
Random-sign null (specnull.py, version (a)): Sigma_bb' = sum_{k>=1} g_hat_k^2 h_b h_b'(lam_k) r_hat_bk r_hat_b'k, mean mu_b = g_hat_0 h_b(0) r_hat_b0.

Variant 1 (Chebyshev only, smoothed-spectrum Gaussian null): g_hat_k^2 -> p_g(lam_k), the per-mode power of g smoothed in lambda
  (KPM: density of the spectral measure of g divided by the eigenvalue density rho, both Jackson-damped at order K, regularised
  where rho is small by shrinkage towards the mean power, refitted as a degree-K Chebyshev polynomial). Then
  Sigma_bb' = q_b' p_g(L) q_b'  with q_b = h_b(L) r_b,   = sum_j c_j(p_g) m_j(b, b'),   m_j(b, b') = q_b' T_j(Lt) q_b'.
  The DC mode (u_0 ~ sqrt(degree)) is handled exactly (deflated from g, q and the rho probes; mu_b exact).
  m_j for every context by one Chebyshev recurrence on the 3 q vectors (doubling: K/2 sparse products), p_g coefficients per gene once:
  per test O(K) scalar work. q_b is formed without any sparse product from precomputed h_b(L)^2 [X, y] and h_b(L) 1.
Variant 2 (hybrid): the lowest K_low eigenpairs (eigsh) are treated exactly (random-sign variance over k = 1..K_low-1, DC fixed); the
  remainder is Variant 1 on g, q and the rho probes deflated by those eigenvectors.
Exact (validation only): K_low = N (dense eigenvectors supplied), no remainder; identical to specnull version (a).
p values: per band 2 Phi(-|z_b|), joint Q = d' Sigma^+ d vs chi2(rank) (the version (a) rule; two-sided in T_b)."""
import os, time, numpy as np, scipy.sparse as sp, numba as nb
from scipy import stats
from scipy.special import ive
from scipy.sparse.linalg import eigsh

BANDS = ("LOW", "MID", "HIGH")
RIDGE = 1e-8
PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]          # band pairs of the 3x3 covariance (upper triangle)


# ------------------------------------------------------------------ numba kernels ------------------------------------------------------------------
@nb.njit(parallel=True, cache=True)
def _step(indptr, indices, data, Xc, Xp, out, mode):
    """mode 0: out = Lt Xc;  mode 1: out = 2 Lt Xc - Xp."""
    n, m = Xc.shape
    for i in nb.prange(n):
        for c in range(m): out[i, c] = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]; v = data[k]
            for c in range(m): out[i, c] += v * Xc[j, c]
        if mode == 1:
            for c in range(m): out[i, c] = 2.0 * out[i, c] - Xp[i, c]


@nb.njit(parallel=True, cache=True)
def _step_acc(indptr, indices, data, Xc, Xp, out, mode, acc, coef):
    """_step, then acc[o] += coef[o] * out for every output o."""
    n, m = Xc.shape; no = coef.shape[0]
    for i in nb.prange(n):
        for c in range(m): out[i, c] = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]; v = data[k]
            for c in range(m): out[i, c] += v * Xc[j, c]
        if mode == 1:
            for c in range(m): out[i, c] = 2.0 * out[i, c] - Xp[i, c]
        for o in range(no):
            co = coef[o]
            if co != 0.0:
                for c in range(m): acc[o, i, c] += co * out[i, c]


@nb.njit(parallel=True, cache=True)
def _pairdots(A, B, ia, ib, nchunk):
    """sum over rows of A[:, ia[p]] * B[:, ib[p]] for every pair p (row-chunk parallel partial sums)."""
    n = A.shape[0]; P = ia.shape[0]; part = np.zeros((nchunk, P)); step = (n + nchunk - 1) // nchunk
    for ch in nb.prange(nchunk):
        r0 = ch * step; r1 = min(n, r0 + step)
        for i in range(r0, r1):
            for p in range(P): part[ch, p] += A[i, ia[p]] * B[i, ib[p]]
    return part.sum(0)


def nthreads():
    return nb.get_num_threads()


def pairdots(A, B, ia, ib):
    return _pairdots(A, B, np.asarray(ia, np.int64), np.asarray(ib, np.int64), max(4 * nthreads(), 1))


class Op:
    """Shifted Laplacian Lt = L - I (csr) with the numba products."""

    def __init__(self, Lsym):
        Lt = (sp.csr_matrix(Lsym) - sp.eye(Lsym.shape[0], format="csr")).tocsr(); Lt.eliminate_zeros(); Lt.sort_indices()
        self.indptr, self.indices, self.data = Lt.indptr.astype(np.int64), Lt.indices.astype(np.int64), Lt.data.astype(np.float64)
        self.n = Lt.shape[0]; self.nnz = Lt.nnz; self.n_matvec_cols = 0

    def step(self, Xc, Xp, out, first):
        _step(self.indptr, self.indices, self.data, Xc, Xp if Xp is not None else Xc, out, 0 if first else 1); self.n_matvec_cols += Xc.shape[1]

    def step_acc(self, Xc, Xp, out, first, acc, coef):
        _step_acc(self.indptr, self.indices, self.data, Xc, Xp if Xp is not None else Xc, out, 0 if first else 1, acc, coef); self.n_matvec_cols += Xc.shape[1]


# ------------------------------------------------------------------ heat filters ------------------------------------------------------------------
def heat_coefs(t, tol=1e-13, kmax=2000):
    """Chebyshev coefficients of exp(-t lambda) in x = lambda - 1, truncated where the remaining |c_j| sum < tol."""
    j = np.arange(kmax + 1); c = (2.0 - (j == 0)) * (-1.0) ** j * ive(j, t)
    tail = np.cumsum(np.abs(c)[::-1])[::-1]; K = int(np.argmax(tail < tol)) if (tail < tol).any() else kmax
    return c[:max(K, 1)]


def heat_multi(op, X, times, tol=1e-13):
    """{t: exp(-t L) X} for several t from one Chebyshev recurrence. Returns dict and the degree used."""
    X = np.ascontiguousarray(X, np.float64); cs = [heat_coefs(t, tol) for t in times]; K = max(len(c) for c in cs) - 1
    C = np.zeros((len(times), K + 1))
    for o, c in enumerate(cs): C[o, :len(c)] = c
    acc = np.empty((len(times), *X.shape)); acc[:] = C[:, 0][:, None, None] * X[None]
    if K >= 1:
        prev, cur, nxt = X, np.empty_like(X), np.empty_like(X)
        op.step_acc(X, None, cur, True, acc, np.ascontiguousarray(C[:, 1]))
        for j in range(2, K + 1):
            op.step_acc(cur, prev, nxt, False, acc, np.ascontiguousarray(C[:, j]))
            prev, cur, nxt = cur, nxt, (prev if prev is not X else np.empty_like(X))
    return {t: acc[o] for o, t in enumerate(times)}, K


def tvals(sig, pitch_um):
    return 2.0 * (sig[0] / pitch_um) ** 2, 2.0 * (sig[1] / pitch_um) ** 2


def band_features(op, X, sig, pitch_um, squared=False, tol=1e-13):
    """Bands of the columns of X (dict band -> (N, P)); with squared=True also h_b(L)^2 X (dict) for the q vectors."""
    tS, tL = tvals(sig, pitch_um); times = [tS, tL] + ([2 * tS, 2 * tL, tS + tL] if squared else [])
    H, K = heat_multi(op, X, times, tol)
    Xb = {"LOW": H[tL], "MID": H[tS] - H[tL], "HIGH": X - H[tS]}
    if not squared: return Xb, K
    Fq = {"LOW": H[2 * tL], "MID": H[2 * tS] - 2 * H[tS + tL] + H[2 * tL], "HIGH": X - 2 * H[tS] + H[2 * tS]}
    return Xb, Fq, K


def filters(lam, sig, pitch_um):
    tS, tL = tvals(sig, pitch_um); lam = np.asarray(lam, float)
    return np.stack([np.exp(-tL * lam), np.exp(-tS * lam) - np.exp(-tL * lam), 1.0 - np.exp(-tS * lam)])


# ------------------------------------------------------------------ KPM ------------------------------------------------------------------
def jackson(K):
    N = K + 1; n = np.arange(N); a = np.pi / (N + 1)
    return ((N - n + 1) * np.cos(a * n) + np.sin(a * n) / np.tan(a)) / (N + 1)


def moments_pairs(op, V0, K, ia, ib):
    """Chebyshev moments m_j = V0[:, ia]' T_j(Lt) V0[:, ib], j = 0..K, for column pairs, by the doubling identities
    T_2i = 2 T_i^2 - T_0, T_2i+1 = 2 T_i+1 T_i - T_1 (ceil(K/2) sparse products of the block). Returns (P, K+1)."""
    V0 = np.ascontiguousarray(V0, np.float64); ia = np.asarray(ia); ib = np.asarray(ib); P = len(ia)
    m = np.zeros((P, K + 1)); n = int(np.ceil(K / 2))
    m[:, 0] = pairdots(V0, V0, ia, ib)
    if K == 0: return m
    prev, cur, nxt = V0, np.empty_like(V0), np.empty_like(V0); op.step(V0, None, cur, True)
    m[:, 1] = pairdots(cur, V0, ia, ib)
    for i in range(1, n + 1):                                     # cur = V_i, prev = V_{i-1}
        if 2 * i <= K: m[:, 2 * i] = 2 * pairdots(cur, cur, ia, ib) - m[:, 0]
        if 2 * i + 1 <= K:
            op.step(cur, prev, nxt, False)                        # nxt = V_{i+1}
            m[:, 2 * i + 1] = 2 * pairdots(nxt, cur, ia, ib) - m[:, 1]
            prev, cur, nxt = cur, nxt, (prev if prev is not V0 else np.empty_like(V0))
    return m


def moments_self(op, V0, K):
    idx = np.arange(V0.shape[1]); return moments_pairs(op, V0, K, idx, idx)


def deflate(Q, V):
    """V - Q (Q' V) for orthonormal columns Q."""
    return V - Q @ (Q.T @ V) if Q is not None and Q.shape[1] else V


def rho_moments(op, K, n_probe, seed, Q=None):
    """Hutchinson estimate of tr_R T_j(Lt) (trace over the complement of span Q), Rademacher probes, fixed seed. Returns (K+1,), per-probe."""
    Z = np.random.default_rng(seed).choice([-1.0, 1.0], size=(op.n, n_probe)); Z = np.ascontiguousarray(deflate(Q, Z))
    mp = moments_self(op, Z, K); return mp.mean(0), mp


def cheb_nodes(M):
    th = np.pi * (np.arange(M) + 0.5) / M; return np.cos(th), th


def power_poly(mu_g, tau, K, eps):
    """Per-mode power p_g(x) = nu_g(x) / rho(x) (Jackson-damped KPM densities; the 1/(pi sqrt(1-x^2)) weight cancels), regularised by
    shrinkage towards the mean power: p = (nu + kappa Pbar) / (rho + kappa), kappa = eps * mean(rho) on the grid; refitted as a degree-K
    Chebyshev series (Gauss-Chebyshev projection on 4(K+1) nodes). mu_g (G, >=K+1), tau (>=K+1). Returns coefficients (G, K+1)."""
    gj = jackson(K); wj = np.where(np.arange(K + 1) == 0, 1.0, 2.0); M = 4 * (K + 1); x, th = cheb_nodes(M)
    Tm = np.cos(np.outer(th, np.arange(K + 1)))                                    # (M, K+1) T_j(x_i)
    num = np.maximum((mu_g[:, :K + 1] * gj * wj) @ Tm.T, 0.0); den = np.maximum((tau[:K + 1] * gj * wj) @ Tm.T, 0.0)
    kap = eps * den.mean(); Pbar = mu_g[:, 0] / tau[0]
    p = (num + kap * Pbar[:, None]) / (den[None] + kap)
    return (p @ Tm) * (wj / M)                                                       # (G, K+1)


# ------------------------------------------------------------------ low eigenpairs ------------------------------------------------------------------
def dc_vector(deg):
    d = np.sqrt(np.asarray(deg, float).ravel()); return d / np.linalg.norm(d)


def low_eigs(Lsym, k, u0, sigma=-1e-3, tol=1e-10, seed=0):
    """Lowest k eigenpairs by shift-invert Lanczos (fixed start vector); the null vector replaced by the exact u0."""
    v0 = np.random.default_rng(seed).standard_normal(Lsym.shape[0])
    w, V = eigsh(sp.csc_matrix(Lsym), k=k, sigma=sigma, which="LM", tol=tol, v0=v0); o = np.argsort(w); w, V = w[o], V[:, o]
    V[:, 0] = u0; w[0] = 0.0
    V[:, 1:] -= np.outer(u0, u0 @ V[:, 1:])                                       # tiny re-orthogonalisation against the exact u0
    return w, V


# ------------------------------------------------------------------ context fit ------------------------------------------------------------------
def colstats(Xb):
    """Full-data mean / sd of every band column (engine zstats with all units in training)."""
    out = {}
    for b in BANDS:
        mu = Xb[b].mean(0); sd = Xb[b].std(0); out[b] = (mu, np.where(sd < 1e-10, 1.0, sd))
    return out


def context_fit(Xb, yb, cols, cs):
    """Per-band full-data perband fit (standardised columns, ridge 1e-8 N; = specnull.context_residual / slotlib.fullfit).
    Returns residual r (3, N), coefficients on the raw band columns beta/sd (3, |C|) and intercepts c0 (3,) so that
    r_b = y_b - X_b[:, C] (beta/sd) - c0_b."""
    N = len(yb["LOW"]); r = np.empty((3, N)); Bs = np.zeros((3, len(cols))); c0 = np.empty(3)
    for i, b in enumerate(BANDS):
        ybar = yb[b].mean(); t = yb[b] - ybar
        if len(cols) == 0: r[i] = t; c0[i] = ybar; continue
        mu, sd = cs[b][0][cols], cs[b][1][cols]; Z = (Xb[b][:, cols] - mu) / sd
        beta = np.linalg.solve(Z.T @ Z + RIDGE * N * np.eye(len(cols)), Z.T @ t); r[i] = t - Z @ beta
        Bs[i] = beta / sd; c0[i] = ybar - mu @ Bs[i]
    return r, Bs, c0


# ------------------------------------------------------------------ the null ------------------------------------------------------------------
class SpecNull:
    """method: 'v1' (Chebyshev only), 'v2' (hybrid, K_low eigenpairs), 'exact' (dense eigenvectors U, lam).
    Inputs: Lsym (csr), deg (node degrees of the adjacency, for u0 ~ sqrt(deg)), Xraw (N, P) raw fields of every gene that can be tested or enter a context,
    Xb / Fq (dict band -> (N, P)) bands and h_b^2 X of those columns, yb / Fy (dict band -> (N,)) target bands and h_b^2 y, H1 (dict band ->
    (N,)) h_b(L) 1, sig, pitch."""

    def __init__(self, method, Lsym, deg, Xraw, Xb, Fq, yb, Fy, H1, sig, pitch_um, K=100, n_probe=30, eps=1e-2, K_low=100, seed=20261120,
                 U=None, lam=None, op=None, log=None, rho_tau=None):
        self.method, self.K, self.eps, self.sig, self.pitch = method, K, eps, sig, pitch_um
        self.Xraw, self.Xb, self.Fq, self.yb, self.Fy, self.H1 = Xraw, Xb, Fq, yb, Fy, H1
        self.op = op or Op(Lsym); self.cs = colstats(Xb); self.tim = {}; log = log or (lambda *a: None); self.N = Lsym.shape[0]
        t = time.time(); u0 = dc_vector(deg)
        if method == "exact":
            self.lam, self.Q = np.asarray(lam, float), U
            assert abs(abs(self.Q[:, 0] @ u0) - 1) < 1e-6, "U[:, 0] is not the null vector"
        elif method == "v2":
            self.lam, self.Q = low_eigs(Lsym, K_low, u0, seed=seed)
        else:
            self.lam, self.Q = np.zeros(1), u0[:, None]
        self.tim["low_eigs_s"] = time.time() - t; self.Hlow = filters(self.lam, sig, pitch_um)       # (3, K_low)
        t = time.time(); self.ghat_low = self.Q.T @ Xraw; self.tim["gene_low_projection_s"] = time.time() - t
        self.rem = method != "exact"
        if self.rem:
            t = time.time()
            if rho_tau is None: self.tau, self.tau_probes = rho_moments(self.op, K, n_probe, seed, self.Q)
            else: self.tau, self.tau_probes = rho_tau
            self.tim["rho_s"] = time.time() - t; t = time.time()
            GR = np.ascontiguousarray(Xraw - self.Q @ self.ghat_low); self.mu_g = moments_self(self.op, GR, K); del GR
            self.tim["gene_moments_s"] = time.time() - t; t = time.time()
            self.cpoly = power_poly(self.mu_g, self.tau, K, eps); self.tim["power_poly_s"] = time.time() - t
        log("SpecNull", method, {k: round(v, 2) for k, v in self.tim.items()})
        self.ctx = {}

    # ---- contexts ----
    def add_contexts(self, ctx_list, mask=None):
        """Fit every context (list of column-index lists), form q_b = h_b(L) r_b, low projections and remainder moments (one batched
        Chebyshev recurrence). mask: optional 0/1 vector; then r_b -> mask * r_b and q_b by a direct filter pass."""
        new = [tuple(c) for c in dict.fromkeys(tuple(c) for c in ctx_list) if tuple(c) not in self.ctx]
        if not new: return
        t = time.time(); R, Qv = [], []
        for c in new:
            r, Bs, c0 = context_fit(self.Xb, self.yb, list(c), self.cs)
            if mask is None:
                q = np.stack([self.Fy[b] - self.Fq[b][:, list(c)] @ Bs[i] - c0[i] * self.H1[b] for i, b in enumerate(BANDS)])
            else:
                r = r * mask[None]; q = None
            R.append(r); Qv.append(q)
        if mask is not None:
            Rm = np.ascontiguousarray(np.concatenate([r.T for r in R], 1)); Bm, _ = band_features(self.op, Rm, self.sig, self.pitch)
            for j in range(len(new)): Qv[j] = np.stack([Bm[b][:, 3 * j + i] for i, b in enumerate(BANDS)])
        self.tim["context_fit_s"] = self.tim.get("context_fit_s", 0) + time.time() - t; t = time.time()
        RH = self.Q.T @ np.concatenate([r.T for r in R], 1); Rhat = [RH[:, 3 * j:3 * j + 3] for j in range(len(new))]   # (K_low, 3) each
        self.tim["context_low_projection_s"] = self.tim.get("context_low_projection_s", 0) + time.time() - t
        if self.rem:
            t = time.time()
            V = np.empty((self.N, 3 * len(new)))
            for j, (q, rh) in enumerate(zip(Qv, Rhat)):
                V[:, 3 * j:3 * j + 3] = q.T - self.Q @ (self.Hlow.T * rh)          # deflated q (h_b commutes with the projector)
            ia = np.concatenate([[3 * j + a for a, b in PAIRS] for j in range(len(new))]); ib = np.concatenate([[3 * j + b for a, b in PAIRS] for j in range(len(new))])
            Mm = moments_pairs(self.op, V, self.K, ia, ib).reshape(len(new), len(PAIRS), self.K + 1); del V
            self.tim["context_moments_s"] = self.tim.get("context_moments_s", 0) + time.time() - t
        for j, c in enumerate(new):
            self.ctx[c] = dict(r=R[j], rhat=Rhat[j], m=Mm[j] if self.rem else None)

    # ---- tests ----
    def test(self, c, genes, want_D=False):
        """Tests of genes (column indices) against context c. Returns dict of arrays (len(genes))."""
        c = tuple(c); C = self.ctx[c]; genes = np.asarray(genes, int)
        T = np.stack([self.Xb[b][:, genes].T @ C["r"][i] for i, b in enumerate(BANDS)], 1)                 # (G, 3)
        hr = self.Hlow * C["rhat"].T                                                                          # (3, K_low)
        gl = self.ghat_low[:, genes]                                                                          # (K_low, G)
        mu = gl[0][:, None] * hr[:, 0][None]                                                                 # (G, 3)
        W = np.stack([hr[a, 1:] * hr[b, 1:] for a, b in PAIRS], 1)                                           # (K_low-1, 6)
        S6 = (gl[1:] ** 2).T @ W                                                                              # (G, 6)
        if self.rem: S6 = S6 + self.cpoly[genes] @ C["m"].T
        V = np.zeros((len(genes), 3, 3))
        for k, (a, b) in enumerate(PAIRS): V[:, a, b] = S6[:, k]; V[:, b, a] = S6[:, k]
        d = T - mu; var = np.einsum("gii->gi", V).copy(); z = d / np.sqrt(np.maximum(var, 1e-300))
        w, P = np.linalg.eigh(V); keep = w > 1e-10 * np.maximum(w.max(1, keepdims=True), 1e-300)
        winv = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0); y = np.einsum("gij,gi->gj", P, d)
        Q = (winv * y ** 2).sum(1); rk = keep.sum(1)
        out = dict(T=T, mu=mu, var=var, z=z, p_band=2 * stats.norm.sf(np.abs(z)), Q=Q, rank=rk, p_joint=stats.chi2.sf(Q, rk), V=V)
        if want_D:
            a2 = (gl[1:] ** 2)[:, :, None] * (hr[:, 1:].T ** 2)[:, None, :]                                 # (K_low-1, G, 3)
            Db = a2.max(0) / np.maximum(var, 1e-300); out["D_band"] = Db; out["D_joint"] = Db[np.arange(len(genes)), np.argmax(np.abs(z), 1)]
        return out
