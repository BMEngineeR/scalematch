"""Band-matched sparse regression with held-out strips.

The target ``y`` and every gene are split into the same bands. In the ``perband`` design each
band of ``y`` is fitted only by the same band of the genes; one gene is one group across the
bands, so the three fits share a gene set. Genes enter one at a time (group forward selection,
most useful first) with an OLS refit after each step. Everything is scored on spatial strips
that were held out of the fit.
"""
import numpy as np
from .bands import BANDS, split

DESIGNS = ("perband", "raw", "concat")


def make_strips(coords, n=5, along=None):
    """Fold labels: ``n`` strips of equal count, perpendicular to an axis.

    ``along`` is a 2-vector giving the axis; by default the long axis of the section. For a
    region target pass the principal axis of the region's own coordinates (``region_axis``), so
    that every held-out strip contains part of the region and the fit is tested by
    interpolation rather than by extrapolation to a part of the tissue it never saw.
    """
    coords = np.asarray(coords, float)
    if along is None:
        ax = int(np.argmax(np.ptp(coords, axis=0)))
        proj = coords[:, ax]
    else:
        along = np.asarray(along, float) / np.linalg.norm(along)
        proj = (coords - coords.mean(0)) @ along
    qs = np.quantile(proj, np.arange(1, n) / n)
    return np.digitize(proj, qs)


def region_axis(coords, mask):
    """Principal axis of the coordinates inside a 0/1 region."""
    c = np.asarray(coords, float)[np.asarray(mask, bool)]
    c = c - c.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[0]


def _zstats(B, tr):
    mu = B[tr].mean(0)
    sd = B[tr].std(0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    return mu, sd


def _blocks(design, X, Xbands):
    if design == "perband":
        return [Xbands[b] for b in BANDS]
    if design == "raw":
        return [X]
    if design == "concat":
        return [np.hstack([Xbands[b] for b in BANDS])]
    raise ValueError(f"unknown design {design!r}")


def forward_select(Z_tr, y_tr, kmax, ridge=1e-8):
    """Group forward selection with OLS refit.

    ``Z_tr`` is a list of standardised training blocks (one per band, or one block), ``y_tr``
    the list of matching centred training targets. Returns the selected columns (shared across
    blocks) and, for each block, the coefficient vector at every step.
    """
    ntr = Z_tr[0].shape[0]
    P = Z_tr[0].shape[1]
    G = [z.T @ z for z in Z_tr]
    Zy = [z.T @ t for z, t in zip(Z_tr, y_tr)]
    corr = [zy.copy() for zy in Zy]
    S, avail, betas = [], np.ones(P, bool), []
    for k in range(1, kmax + 1):
        score = sum(c ** 2 for c in corr) / ntr
        score[~avail] = -np.inf
        j = int(np.argmax(score))
        S.append(j)
        avail[j] = False
        step = []
        for b in range(len(Z_tr)):
            beta = np.linalg.solve(G[b][np.ix_(S, S)] + ridge * ntr * np.eye(k), Zy[b][S])
            corr[b] = Zy[b] - G[b][:, S] @ beta
            step.append(beta)
        betas.append(step)
    return S, betas


def evaluate(design, X, Xbands, y, ybands, block, kmax=30):
    """Held-out curve for one design.

    Returns a dict with ``nmse`` (length ``kmax + 1``, index 0 is the mean predictor), ``A``
    (mean over k of max(0, 1 - NMSE)), ``best``, ``at_kmax`` and the out-of-fold prediction
    ``oof`` at ``kmax``.
    """
    y = np.asarray(y, float)
    N = len(y)
    blocks = _blocks(design, X, Xbands)
    sse = np.zeros(kmax + 1)
    sst = 0.0
    oof = np.zeros(N)
    for f in np.unique(block):
        te = block == f
        tr = ~te
        ym = y[tr].mean()
        sst += ((y[te] - ym) ** 2).sum()
        sse[0] += ((y[te] - ym) ** 2).sum()
        if design == "perband":
            targets = [ybands[b] - ybands[b][tr].mean() for b in BANDS]
        else:
            targets = [y - ym]
        Z = []
        for B in blocks:
            mu, sd = _zstats(B, tr)
            Z.append((B - mu) / sd)
        S, betas = forward_select([z[tr] for z in Z], [t[tr] for t in targets], kmax)
        for k in range(1, kmax + 1):
            pred = np.full(int(te.sum()), ym)
            for b in range(len(blocks)):
                pred += Z[b][te][:, S[:k]] @ betas[k - 1][b]
            sse[k] += ((y[te] - pred) ** 2).sum()
        oof[te] = pred
    nmse = sse / sst
    curve = np.maximum(0, 1 - nmse[1:])
    return dict(nmse=nmse, A=float(curve.mean()), best=float(1 - nmse[1:].min()),
                at_kmax=float(1 - nmse[kmax]), k_best=int(np.argmin(nmse[1:]) + 1), oof=oof)


def band_explained(L, y, oof, ybands, pitch_um, sigmas):
    """Held-out explained variance per band: the residual is split with the same bands."""
    r = np.asarray(y, float) - oof
    rb = split(L, r, pitch_um, sigmas)
    out = {}
    for b in BANDS:
        num = ((rb[b] - rb[b].mean()) ** 2).sum()
        den = ((ybands[b] - ybands[b].mean()) ** 2).sum()
        out[b] = float(1 - num / den)
    return out
