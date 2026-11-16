"""Exact random-sign surrogate genes for the calibration gate (V3 spec section 6).

x~ = U diag(eps) U^T x, eps_k Rademacher for every mode except the null mode(s) (lambda = 0, one per connected component), so the surrogate keeps
the gene's spectrum exactly and loses its alignment with the target. U = eigenvectors of L_sym.
  * Bipartite graphs (the 4-neighbour bin lattice): with M = I - L = [[0, B], [B^T, 0]] and B = W S Z^T (full SVD), the eigenvectors are
    [W_k; +-Z_k]/sqrt2 (lambda = 1 -+ s_k) plus zero-padded extra columns of W or Z (lambda = 1). U is never formed; cost = one SVD of an
    (n/2 x n/2) dense block (about 45 min for 53k bins on 48 cores, about 6 min for 24k).
  * Other graphs: dense eigh of L (only sensible for a few thousand units)."""
import numpy as np, scipy.sparse as sp
from scipy.sparse import csgraph


def _two_colouring(M):
    n = M.shape[0]; col = -np.ones(n, int); ncomp, lab = csgraph.connected_components(M, directed=False)
    for c in range(ncomp):
        r0 = int(np.flatnonzero(lab == c)[0]); order, pred = csgraph.breadth_first_order(M, r0, directed=False, return_predecessors=True); col[r0] = 0
        for v in order[1:]: col[v] = 1 - col[pred[v]]
    Mc = M.tocoo(); bad = int((col[Mc.row] == col[Mc.col]).sum())
    return col, ncomp, bad


def random_sign_surrogates(L, X, seed=20261117, dense_limit=8000):
    """L: L_sym (sparse, N x N). X: (N, m) gene fields. Returns (X~ (N, m), info)."""
    L = sp.csr_matrix(L); n = L.shape[0]; X = np.asarray(X, float); rng = np.random.default_rng(seed)
    M = (sp.identity(n, format="csr") - L).tocsr(); M.eliminate_zeros()
    col, ncomp, bad = _two_colouring(M)
    if bad == 0 and abs(M.diagonal()).max() < 1e-12:
        P, Q = np.flatnonzero(col == 0), np.flatnonzero(col == 1); n1, n2 = len(P), len(Q); r = min(n1, n2)
        W, S, Zt = np.linalg.svd(M[P][:, Q].toarray(), full_matrices=True); Z = Zt.T; del Zt
        h = 1 / np.sqrt(2); a, b = W[:, :r].T @ X[P], Z[:, :r].T @ X[Q]; gp, gm = (a + b) * h, (a - b) * h
        ge = W[:, r:].T @ X[P] if n1 > n2 else (Z[:, r:].T @ X[Q] if n2 > n1 else np.zeros((0, X.shape[1])))
        ep = rng.choice([-1.0, 1.0], gp.shape); em = rng.choice([-1.0, 1.0], gm.shape); ee = rng.choice([-1.0, 1.0], ge.shape)
        null = np.abs(1 - S) < 1e-9; assert null.sum() == ncomp, (null.sum(), ncomp); ep[null] = 1.0
        cp, cm, ce = ep * gp, em * gm, ee * ge; Xs = np.empty_like(X)
        Xs[P] = W[:, :r] @ ((cp + cm) * h) + (W[:, r:] @ ce if n1 > n2 else 0)
        Xs[Q] = Z[:, :r] @ ((cp - cm) * h) + (Z[:, r:] @ ce if n2 > n1 else 0)
        info = dict(method="bipartite SVD", components=ncomp)
    else:
        if n > dense_limit: raise ValueError(f"graph is not bipartite and has {n} units (> dense_limit {dense_limit}); dense eigh would be too slow")
        lam, U = np.linalg.eigh(L.toarray()); gh = U.T @ X; e = rng.choice([-1.0, 1.0], gh.shape); e[lam < 1e-9] = 1.0
        Xs = U @ (e * gh); info = dict(method="dense eigh", components=int((lam < 1e-9).sum()))
    info["max_rel_norm_err"] = float(np.max(np.abs(np.linalg.norm(Xs, axis=0) / np.maximum(np.linalg.norm(X, axis=0), 1e-300) - 1)))
    return Xs, info
