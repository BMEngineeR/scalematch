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


def evaluate(design, X, Xbands, y, ybands, block, kmax=30, per_gene=False, groups=None, ridge=1e-8):
    """Held-out curve for one design.

    Returns a dict with ``nmse`` (length ``kmax + 1``, index 0 is the mean predictor), ``A``
    (mean over k of max(0, 1 - NMSE)), ``best``, ``at_kmax`` and the out-of-fold prediction
    ``oof`` at ``kmax``.

    With ``per_gene`` (perband design) it also returns, for every dictionary gene, ``heldout``:
    the rise in held-out error when the gene is dropped from the fold's selected set and the
    remaining genes are refitted, as a fraction of the target's held-out variance; and
    ``heldout_band``, the same within each band (the band refitted without the gene, against
    the band's own variance). ``groups`` (gene index -> indices of the genes it stands for)
    adds ``heldout_module``: the whole group dropped from the selected set at once. A gene
    not selected in a fold contributes nothing there.
    """
    y = np.asarray(y, float)
    N = len(y)
    blocks = _blocks(design, X, Xbands)
    P = blocks[0].shape[1]
    sse = np.zeros(kmax + 1)
    sst = 0.0
    oof = np.zeros(N)
    hg, hgm = np.zeros(P), np.zeros(P)
    hgb = {b: np.zeros(P) for b in BANDS}
    hgmb = {b: np.zeros(P) for b in BANDS}
    sstb = {b: 0.0 for b in BANDS}
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
        if per_gene and design == "perband":
            nb = len(blocks)
            ntr = int(tr.sum())
            Ztr = [Z[b][tr][:, S] for b in range(nb)]
            Zte = [Z[b][te][:, S] for b in range(nb)]
            G = [z.T @ z for z in Ztr]
            Zy = [z.T @ targets[b][tr] for b, z in enumerate(Ztr)]
            base = [Zte[b] @ betas[kmax - 1][b] for b in range(nb)]
            sse_full = ((y[te] - pred) ** 2).sum()
            sse_band = [((targets[b][te] - base[b]) ** 2).sum() for b in range(nb)]
            for b in range(nb):
                sstb[BANDS[b]] += (targets[b][te] ** 2).sum()

            def dropped(mask):
                preds = []
                for b in range(nb):
                    beta = np.linalg.solve(G[b][np.ix_(mask, mask)] + ridge * ntr * np.eye(int(mask.sum())), Zy[b][mask])
                    preds.append(Zte[b][:, mask] @ beta)
                return preds

            pos = {j: i for i, j in enumerate(S)}
            for i, j in enumerate(S):
                m = np.ones(kmax, bool)
                m[i] = False
                preds = dropped(m)
                hg[j] += ((y[te] - ym - sum(preds)) ** 2).sum() - sse_full
                for b in range(nb):
                    hgb[BANDS[b]][j] += ((targets[b][te] - preds[b]) ** 2).sum() - sse_band[b]
                if groups is not None and j in groups:
                    idx = [pos[g] for g in groups[j] if g in pos]
                    if len(idx) > 1:
                        m = np.ones(kmax, bool)
                        m[idx] = False
                        preds = dropped(m)
                    hgm[j] += ((y[te] - ym - sum(preds)) ** 2).sum() - sse_full
                    for b in range(nb):
                        hgmb[BANDS[b]][j] += ((targets[b][te] - preds[b]) ** 2).sum() - sse_band[b]
    nmse = sse / sst
    curve = np.maximum(0, 1 - nmse[1:])
    out = dict(nmse=nmse, A=float(curve.mean()), best=float(1 - nmse[1:].min()),
               at_kmax=float(1 - nmse[kmax]), k_best=int(np.argmin(nmse[1:]) + 1), oof=oof)
    if per_gene and design == "perband":
        out["heldout"] = hg / sst
        out["heldout_band"] = {b: hgb[b] / max(sstb[b], 1e-12) for b in BANDS}
        out["heldout_module"] = hgm / sst
        out["heldout_module_band"] = {b: hgmb[b] / max(sstb[b], 1e-12) for b in BANDS}
    return out


def evaluate_fixed(X, Xbands, y, ybands, block, S, ridge=1e-8):
    """Held-out score of a fixed gene set ``S`` (perband design, OLS refit per fold, no selection)."""
    y = np.asarray(y, float)
    N = len(y)
    S = list(S)
    sse, sst = 0.0, 0.0
    oof = np.zeros(N)
    for f in np.unique(block):
        te = block == f
        tr = ~te
        ym = y[tr].mean()
        sst += ((y[te] - ym) ** 2).sum()
        pred = np.full(int(te.sum()), ym)
        for b in BANDS:
            mu, sd = _zstats(Xbands[b], tr)
            Z = (Xbands[b][:, S] - mu[S]) / sd[S]
            t = ybands[b] - ybands[b][tr].mean()
            G = Z[tr].T @ Z[tr] + ridge * tr.sum() * np.eye(len(S))
            beta = np.linalg.solve(G, Z[tr].T @ t[tr])
            pred += Z[te] @ beta
        sse += ((y[te] - pred) ** 2).sum()
        oof[te] = pred
    nmse = sse / sst
    return dict(nmse=nmse, best=float(1 - nmse), oof=oof)


def alone_scores(Xbands, y, ybands, block):
    """Held-out 1 - NMSE of every dictionary gene on its own (perband design, one slope per band per fold).

    Returns ``alone`` (P,) against the target's held-out variance and ``alone_band`` (band -> (P,))
    against each band's own variance.
    """
    y = np.asarray(y, float)
    P = Xbands[BANDS[0]].shape[1]
    sse = np.zeros(P)
    sseb = {b: np.zeros(P) for b in BANDS}
    slopes = {b: np.zeros(P) for b in BANDS}
    sst = 0.0
    sstb = {b: 0.0 for b in BANDS}
    for f in np.unique(block):
        te = block == f
        tr = ~te
        ym = y[tr].mean()
        sst += ((y[te] - ym) ** 2).sum()
        pred = np.zeros((int(te.sum()), P))
        for b in BANDS:
            mu, sd = _zstats(Xbands[b], tr)
            Ztr = (Xbands[b][tr] - mu) / sd
            Zte = (Xbands[b][te] - mu) / sd
            t = ybands[b] - ybands[b][tr].mean()
            slope = (Ztr * t[tr][:, None]).sum(0) / np.maximum((Ztr ** 2).sum(0), 1e-12)
            slopes[b] += slope
            pb = Zte * slope[None, :]
            pred += pb
            sseb[b] += ((t[te][:, None] - pb) ** 2).sum(0)
            sstb[b] += (t[te] ** 2).sum()
        sse += ((y[te][:, None] - ym - pred) ** 2).sum(0)
    nf = len(np.unique(block))
    return dict(alone=1 - sse / sst, alone_band={b: 1 - sseb[b] / max(sstb[b], 1e-12) for b in BANDS},
                slope_band={b: slopes[b] / nf for b in BANDS})


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
