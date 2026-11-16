"""New V3 pieces only (spec sections 3e-3g, 4, 5, 6). Everything else is imported from the existing code:
bands (chebnull.band_features), forward selection and score (engine.Target, slotlib.score_any), the random-sign null with V1 / V2 variance
(chebnull.SpecNull), the all-bins band fit (chebnull.context_fit), contributions / shares (scalematch attribution.contributions)."""
import numpy as np, pandas as pd
from scipy import stats
BANDS = ("LOW", "MID", "HIGH")


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out


def clopper_pearson(k, n, a=0.05):
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0; hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def describe(Xb, yb, cols, genes, CN, attribution):
    """3f: least-squares fit of S_orig on all bins, band by band, standardised band signals (chebnull.context_fit = the all-bins perband fit),
    then scalematch attribution.contributions for coef / rms / share / dominant."""
    cs = CN.colstats(Xb); _, Bs, _ = CN.context_fit(Xb, yb, list(cols), cs)
    Z = {b: (Xb[b] - cs[b][0]) / cs[b][1] for b in BANDS}
    beta = {b: Bs[i] * cs[b][1][list(cols)] for i, b in enumerate(BANDS)}                     # coefficients on standardised columns
    _, tab = attribution.contributions(list(cols), Z, beta, genes)
    return tab


def substitution_table(Xb, yb, S, cand, genes, rel_thr=0.15, cover_thr=0.5):
    """Step 4: rel_b = corr(h_b, y_b); cover_b = R^2 of h_b on {g_b, g in S} (OLS with intercept); b+ = argmax |rel_b|;
    rep = argmax_g |corr(h_b+, g_b+)|."""
    rel, cov, cor = {}, {}, {}
    for b in BANDS:
        Z = Xb[b] - Xb[b].mean(0); sd = np.sqrt((Z ** 2).sum(0)); Z = Z / np.where(sd < 1e-12, 1.0, sd)
        y = yb[b] - yb[b].mean(); y = y / np.linalg.norm(y)
        rel[b] = Z[:, cand].T @ y
        Zs = Z[:, S]; coef, *_ = np.linalg.lstsq(Zs, Z[:, cand], rcond=None); res = Z[:, cand] - Zs @ coef
        cov[b] = 1 - (res ** 2).sum(0) / np.maximum((Z[:, cand] ** 2).sum(0), 1e-300)
        cor[b] = Z[:, cand].T @ Zs                                                                     # (|cand|, |S|)
    R = np.stack([rel[b] for b in BANDS], 1); Cv = np.stack([cov[b] for b in BANDS], 1)
    bd = np.argmax(np.abs(R), 1); ix = np.arange(len(cand))
    rmax = np.abs(R[ix, bd]); cmax = Cv[ix, bd]
    rep = [genes[S[int(np.argmax(np.abs(cor[BANDS[bd[i]]][i])))]] for i in ix]
    lab = np.where((rmax >= rel_thr) & (cmax >= cover_thr), np.array([f"covered by {r}" for r in rep], dtype=object),
                   np.where(rmax >= rel_thr, "relevant, not covered", "unrelated"))
    return pd.DataFrame(dict(gene=[genes[j] for j in cand], **{f"rel_{b}": R[:, i] for i, b in enumerate(BANDS)},
                             **{f"cover_{b}": Cv[:, i] for i, b in enumerate(BANDS)}, rep=rep, label=lab,
                             band_dagger=[BANDS[i] for i in bd], max_abs_rel=rmax, cover_dagger=cmax))


def strip_layouts(coords, y, fit):
    """The 16 strip layouts of the stored noise floor (v3e_noise.py): n strips 4/5/6/8 x offset 0, 1/4, 1/2, 3/4 strip width along the
    target-region axis; (5, 0) is fit.make_strips."""
    from scipy.stats import rankdata
    axis = fit.region_axis(coords, y > 0.5); axis = axis / np.linalg.norm(axis); proj = (coords - coords.mean(0)) @ axis
    u = (rankdata(proj, method="ordinal") - 0.5) / len(proj); out = []
    for n in (4, 5, 6, 8):
        for d in (0.0, 0.25, 0.5, 0.75):
            out.append(((n, d), fit.make_strips(coords, n, along=axis) if d == 0 else np.minimum(np.floor(((u + d / n) % 1.0) * n), n - 1).astype(int)))
    return out


def masking_partners(Xb, S, genes, dominant, not_ssc, thr=0.8):
    """Section 5 (description only): for each S_orig gene that is not an SSC, partners in S_orig with |corr| >= thr in its dominant band."""
    rows = []
    for g in not_ssc:
        b = dominant[g]; i = S[[genes[j] for j in S].index(g)]; x = Xb[b][:, i]
        for j in S:
            if j == i: continue
            r = float(np.corrcoef(x, Xb[b][:, j])[0, 1])
            if abs(r) >= thr: rows.append(dict(gene=g, dominant=b, partner=genes[j], corr=r))
    return pd.DataFrame(rows, columns=["gene", "dominant", "partner", "corr"])


def minimal_panel(score_fn, S_orig, score_ref, bar=-np.inf, eta=0.1):
    """Old backward reduction (v3d search_min.Search.reduce, as in fastv3d.py). Stand-alone; NOT called by run_v3.py.
    score_fn(list of index lists) -> scores. Returns the reduced index list (unsorted)."""
    thr = max((1 - eta) * score_ref, bar); S = list(S_orig)
    while len(S) > 1:
        cand = [[x for x in S if x != g] for g in S]; sc = np.asarray(score_fn([sorted(c) for c in cand])); ok = sc >= thr
        if not ok.any(): break
        S = cand[int(np.flatnonzero(ok)[np.argmax(sc[ok])])]
    return S
