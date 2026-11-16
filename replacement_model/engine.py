"""Forward selection and the held-out score of the scalematch perband design, from per-fold sufficient statistics.

  G[f, b]  = Z_b,tr' Z_b,tr        (train Gram, genes standardised with the fold's training mean/sd, as scalematch fit._zstats)
  w[f, b]  = Z_b,tr' t_b,tr        (t_b = band b of the target, centred on the training mean)
  H[f]     = [Z_L Z_M Z_H]_te' [Z_L Z_M Z_H]_te   (test cross-Gram over the three bands stacked)
  v[f]     = [Z_L Z_M Z_H]_te' (y_te - mean_tr y)
  c0[f]    = sum (y_te - mean_tr y)^2 = held-out SST of fold f
Fold index F (= n_folds) holds the full-data statistics. The held-out SSE on y of the perband fit of a set S in fold f is
  c0 - 2 beta'v[S3] + beta' H[S3,S3] beta,   beta_b = (G_b[S,S] + ridge*ntr*I)^-1 w_b[S],
which equals scalematch fit.evaluate_fixed; forward selection equals scalematch fit.forward_select (same score, same OLS refit, same ridge).
score(S) = 1 - pooled held-out NMSE."""
import numpy as np

BANDS = ("LOW", "MID", "HIGH")
RIDGE = 1e-8


def zstats(B, tr):
    mu = B[tr].mean(0); sd = B[tr].std(0); sd = np.where(sd < 1e-10, 1.0, sd)
    return mu, sd


def build_cache(Xb, block, Y, Yb):
    """Xb: band -> (N, P); block: strip label per unit; Y: (N, M) targets; Yb: band -> (N, M) target bands."""
    folds = np.unique(block); F = len(folds); P = Xb[BANDS[0]].shape[1]; M = Y.shape[1]
    G = np.zeros((F + 1, 3, P, P)); w = np.zeros((F + 1, 3, P, M)); ntr = np.zeros(F + 1)
    H = np.zeros((F, 3 * P, 3 * P)); v = np.zeros((F, 3 * P, M)); c0 = np.zeros((F, M))
    for fi in range(F + 1):
        tr = (block != folds[fi]) if fi < F else np.ones(len(block), bool); te = ~tr
        ntr[fi] = tr.sum(); Zte = []
        for bi, b in enumerate(BANDS):
            mu, sd = zstats(Xb[b], tr); Z = (Xb[b] - mu) / sd
            G[fi, bi] = Z[tr].T @ Z[tr]
            T = Yb[b] - Yb[b][tr].mean(0)
            w[fi, bi] = Z[tr].T @ T[tr]
            if fi < F: Zte.append(Z[te])
        if fi < F:
            Zte = np.hstack(Zte); H[fi] = Zte.T @ Zte
            r0 = Y[te] - Y[tr].mean(0); v[fi] = Zte.T @ r0; c0[fi] = (r0 ** 2).sum(0)
    return dict(G=G, w=w, ntr=ntr, H=H, v=v, c0=c0)


class Target:
    """One target column m of a cache."""

    def __init__(self, C, m=0):
        self.G, self.H, self.ntr = C["G"], C["H"], C["ntr"]
        self.w = C["w"][..., m]; self.v = C["v"][..., m]; self.c0 = C["c0"][:, m]
        self.F = self.H.shape[0]; self.P = self.G.shape[-1]

    def fs(self, fi, kmax):
        """Forward selection in fold fi (fi = F: all units). Returns the ordered set and the per-step coefficients."""
        G, w, ntr = self.G[fi], self.w[fi], self.ntr[fi]
        avail = np.ones(self.P, bool); kmax = int(min(kmax, self.P)); S, betas = [], []; corr = w.copy()
        for k in range(1, kmax + 1):
            score = (corr ** 2).sum(0) / ntr; score[~avail] = -np.inf; j = int(np.argmax(score))
            S.append(j); avail[j] = False; B = np.empty((3, k))
            for b in range(3):
                beta = np.linalg.solve(G[b][np.ix_(S, S)] + RIDGE * ntr * np.eye(k), w[b][S])
                corr[b] = w[b] - G[b][:, S] @ beta; B[b] = beta
            betas.append(B)
        return S, betas

    def _sse(self, fi, S, B):
        idx3 = np.concatenate([np.asarray(S) + b * self.P for b in range(3)]); be = B.ravel()
        return self.c0[fi] - 2 * be @ self.v[fi][idx3] + be @ self.H[fi][np.ix_(idx3, idx3)] @ be

    def cv_curve(self, kmax=30):
        """Held-out curve of forward selection, selection redone in each training fold. Returns pooled NMSE(k) and per-strip e(k)."""
        rows = []
        for fi in range(self.F):
            S, betas = self.fs(fi, kmax)
            rows.append([self._sse(fi, S[:k + 1], betas[k]) for k in range(len(S))])
        K = min(len(r) for r in rows); sse = np.array([r[:K] for r in rows])
        return sse.sum(0) / self.c0.sum(), sse / self.c0[:, None]

    def score_sets(self, idx, chunk=None):
        """idx: (n, k) gene indices. Returns held-out sse (n, F)."""
        idx = np.atleast_2d(np.asarray(idx, int)); n, k = idx.shape; P = self.P
        chunk = chunk or max(1, int(4e7 // (9 * k * k * 8)))
        out = np.zeros((n, self.F))
        for s0 in range(0, n, chunk):
            I = idx[s0:s0 + chunk]; m = len(I)
            I3 = np.concatenate([I + b * P for b in range(3)], axis=1)
            for fi in range(self.F):
                beta = np.empty((m, 3, k))
                for b in range(3):
                    Gs = self.G[fi, b][I[:, :, None], I[:, None, :]] + RIDGE * self.ntr[fi] * np.eye(k)
                    beta[:, b] = np.linalg.solve(Gs, self.w[fi, b][I][..., None])[..., 0]
                be = beta.reshape(m, 3 * k)
                Hs = self.H[fi][I3[:, :, None], I3[:, None, :]]
                out[s0:s0 + m, fi] = self.c0[fi] - 2 * np.einsum("ni,ni->n", be, self.v[fi][I3]) + np.einsum("ni,nij,nj->n", be, Hs, be)
        return out


def score(T, sets):
    """1 - pooled held-out NMSE for a list of equal-size sets; the empty set scores 0."""
    sets = [sorted(s) for s in sets]
    if len(sets[0]) == 0: return np.zeros(len(sets))
    return 1 - T.score_sets(np.array(sets)).sum(1) / T.c0.sum()


def score_any(T, sets):
    """Scores for sets of mixed size."""
    out = np.empty(len(sets)); by = {}
    for i, s in enumerate(sets): by.setdefault(len(s), []).append(i)
    for k, ii in by.items(): out[ii] = score(T, [sets[i] for i in ii])
    return out
