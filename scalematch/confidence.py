"""Gene-level confidence for a Result.

The question a gene list raises is which of its genes can be trusted. Three readings are
computed from the band-split dictionary kept in the Result, and the first one carries the p.

* ``alone``: the gene on its own, fitted band by band on the training strips and scored on the
  held-out strips (1 - NMSE). The same is computed for random rigid placements of the target,
  and ``p`` is the share of placements in which some dictionary gene did as well on its own.
  A gene is *supported* when ``p`` is at or below ``alpha``: it reproduces more of the target
  on unseen tissue than the best single gene of a randomly placed target. ``alone_<band>``
  is the gene alone within one band.
* ``drop``: the rise in held-out error when the gene is dropped from the fold's selected set
  and the rest is refitted (necessity). A gene with a small ``drop`` and a large ``alone`` is
  one of several that carry the same signal; the module columns say which.
* ``freq``: the gene's selection frequency among the first ``k_sel`` genes when the section is
  half-sampled by tiles (stability selection on complementary halves); ``sign_consistency``
  how often its coefficient keeps its sign. Random placements produce stable genes too, so
  the frequencies are reported next to the highest frequency a random placement reached.

Modules are groups of selected genes that stand in for one another (each gene regressed on
the rest of the selected set, bands stacked; R² at or above ``cover`` links it to its
representative; connected components). A module has its own ``alone`` (the members fitted
together), ``p`` (against the best module of each random placement) and ``drop``.

The supported genes, fitted together on the held-out strips, give the score of the set.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from . import attribution, bands, fit, nulls
from .bands import BANDS


@dataclass
class Confidence:
    table: pd.DataFrame          # one row per gene of interest
    modules: pd.DataFrame        # one row per module of the selected set
    supported: list              # genes with p <= alpha
    stable: list                 # genes with freq >= tau
    alpha: float
    tau: float
    k_sel: int
    n_resample: int
    n_tiles: int
    tile_um: float
    n_null: int
    null_max_alone: np.ndarray    # per random placement: the best single gene on its own
    null_max_module: np.ndarray   # per random placement: the best module on its own
    null_max_freq: np.ndarray     # per random placement: the highest selection frequency of any gene
    expected_false_stable: float  # Meinshausen-Buhlmann bound on the expected number of false stable genes
    supported_fit: dict           # held-out best and explained per band of the supported genes together
    full_fit: dict                # the same for the full selected set
    band: str

    def summary(self):
        t = self.table
        sup = t[t.gene.isin(self.supported)]
        shown = sup.head(20)
        more = f" and {len(sup) - 20} more" if len(sup) > 20 else ""
        lines = [f"confidence: {self.n_null} random placements; {self.n_resample} half-samplings of {self.n_tiles} tiles ({self.tile_um:.0f} um), {self.k_sel} genes per run",
                 f"supported genes, {len(sup)} with p <= {self.alpha:.2f} (held-out 1-NMSE alone, freq, sign and band): "
                 + ", ".join(f"{g} ({a:.2f}, {f:.2f}, {s}{b})" for g, a, f, s, b in zip(shown.gene, shown.alone, shown.freq, shown.sign, shown.dominant)) + more]
        if len(self.null_max_alone):
            lines.append(f"random placements: best single gene {self.null_max_alone.mean():.3f} on average, {self.null_max_alone.max():.3f} at most; "
                         f"best module {self.null_max_module.mean():.3f} / {self.null_max_module.max():.3f}; most stable gene {self.null_max_freq.mean():.2f} / {self.null_max_freq.max():.2f}")
        m = self.modules
        ms = m[m.p <= self.alpha]
        if len(ms):
            lines.append(f"supported modules, {len(ms)} (members, held-out 1-NMSE together, p): "
                         + "; ".join(f"[{r.genes}] {r.alone:.2f}, {r.p:.3f}" for _, r in ms.head(8).iterrows()) + (f"; and {len(ms) - 8} more" if len(ms) > 8 else ""))
        lines.append(f"stable genes (freq >= {self.tau:.2f}): {len(self.stable)}, expected false stable <= {self.expected_false_stable:.2f}")
        sf, ff = self.supported_fit, self.full_fit
        lines.append(f"supported genes together ({sf['k']}), held-out: best {sf['best']:.3f}, explained " + ", ".join(f"{b} {sf['explained'][b]:.2f}" for b in BANDS)
                     + f" | full list ({ff['k']}): {ff['best']:.3f}, " + ", ".join(f"{b} {ff['explained'][b]:.2f}" for b in BANDS))
        return "\n".join(lines)


def make_tiles(coords, tile, min_tiles=30):
    """Square tiles of side ``tile`` (coordinate units); the side shrinks until there are at least ``min_tiles``."""
    coords = np.asarray(coords, float)
    lo = coords.min(0)
    while True:
        ij = np.floor((coords - lo) / tile).astype(int)
        key = ij[:, 0] * 100000 + ij[:, 1]
        uniq, inv = np.unique(key, return_inverse=True)
        if len(uniq) >= min_tiles:
            return inv, tile
        tile /= 1.5


def selected_groups(Xbands, S, cover=0.5, ridge=1e-6):
    """Modules within a selected set: each gene regressed on the others (bands stacked); a gene whose
    R² is at least ``cover`` is linked to its representative; connected components are the modules.
    Returns a module label per position in ``S``."""
    k = len(S)
    if k == 0:
        return np.zeros(0, int)
    cols = []
    for b in BANDS:
        Z = Xbands[b][:, S]
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) < 1e-10, 1.0, Z.std(0))
        cols.append(Z)
    Z = np.vstack(cols)
    n = Z.shape[0]
    G = Z.T @ Z
    links = []
    for i in range(k):
        others = np.array([j for j in range(k) if j != i])
        if len(others) == 0:
            break
        A = G[np.ix_(others, others)] + ridge * n * np.eye(len(others))
        coef = np.linalg.solve(A, G[others, i])
        r2 = 1 - (G[i, i] - 2 * coef @ G[others, i] + coef @ G[np.ix_(others, others)] @ coef) / max(G[i, i], 1e-12)
        if r2 >= cover:
            links.append((i, int(others[int(np.argmax(np.abs(coef)))])))
    if links:
        rows, cols_ = zip(*links)
        M = csr_matrix((np.ones(len(links)), (rows, cols_)), shape=(k, k))
        _, lab = connected_components(M, directed=False)
    else:
        lab = np.arange(k)
    return lab


def _runs(Xbands, ybands, tiles, k_sel, n_resample, rng):
    """Forward selection on complementary halves of the tiles. Returns selected (runs x P) and the final coefficients."""
    P = Xbands[BANDS[0]].shape[1]
    T = int(tiles.max()) + 1
    sel = np.zeros((n_resample, P), bool)
    beta = np.zeros((n_resample, len(BANDS), P))
    r = 0
    while r < n_resample:
        perm = rng.permutation(T)
        for half in (perm[: T // 2], perm[T // 2:]):
            if r >= n_resample:
                break
            rows = np.isin(tiles, half)
            Z, t = [], []
            for b in BANDS:
                B = Xbands[b][rows]
                mu, sd = B.mean(0), B.std(0)
                sd = np.where(sd < 1e-10, 1.0, sd)
                Z.append((B - mu) / sd)
                yb = ybands[b][rows]
                t.append(yb - yb.mean())
            S, betas = fit.forward_select(Z, t, k_sel)
            sel[r, S] = True
            for bi in range(len(BANDS)):
                beta[r, bi, S] = betas[-1][bi]
            r += 1
    return sel, beta


def _null_target(res, rng, pitch_units):
    if res.target_kind == "region":
        yn = nulls.rigid_mask(res.y > 0.5, res.coords, pitch_units, rng)
    else:
        yn = nulls.rigid_field(res.y, res.coords, pitch_units, rng)
    return None if yn is None else np.asarray(yn, float)


def _modules_of(Xbands, S, cover):
    """Module label per position in S, and the member columns of every module."""
    lab = selected_groups(Xbands, S, cover=cover)
    members = {m: [S[i] for i in range(len(S)) if lab[i] == m] for m in range(int(lab.max()) + 1)} if len(S) else {}
    return lab, members


def _module_alone(X, Xbands, y, ybands, block, members):
    """Held-out 1 - NMSE of every module fitted on its own."""
    return {m: (fit.evaluate_fixed(X, Xbands, y, ybands, block, cols)["best"] if len(cols) > 1
                else None) for m, cols in members.items()}


def confidence(res, *, n_null=20, n_resample=100, k_sel=15, tile_um=None, alpha=0.05, tau=0.8, cover=0.5,
               n_null_stability=5, n_resample_null=50, band="MID", seed=0, verbose=True):
    """Gene-level confidence for ``res`` (a Result from ``explain`` with its dictionary still in memory).

    n_null             random rigid placements of the target; the best single gene (and module) on its
                       own on each of them is the bar
    alpha              a gene is supported when the share of placements whose best gene reached its
                       held-out score alone is at or below alpha; the floor is 1/(n_null + 1)
    cover              R² at which a selected gene is linked to another selected gene (module building)
    n_resample, k_sel  half-samplings for the selection frequency, and genes chosen per half-sampling
                       (smaller than kmax on purpose: the stable set is the genes needed first)
    tile_um            tile side in micrometres (default 2 x sigma_L, shrunk until there are 30 tiles)
    tau                frequency above which a gene is called stable
    n_null_stability   placements on which the half-sampling is repeated (with n_resample_null runs)
    band               the band whose columns the summary reports
    """
    if res.Xbands is None:
        raise ValueError("this Result has no dictionary in memory; run explain() again in this session")
    rng = np.random.default_rng(seed)
    log = (lambda *a: print("[confidence]", *a, flush=True)) if verbose else (lambda *a: None)
    pitch_units = res.meta.get("pitch_units", 1.0)
    kmax = int(res.meta.get("kmax", 30))
    k_sel = int(min(k_sel, kmax))
    genes = np.asarray(res.genes).astype(str)
    P = len(genes)
    gi = {g: i for i, g in enumerate(genes)}
    full = res.gene_table.set_index("gene")
    S_full = [gi[g] for g in res.gene_table.gene]

    # 1. every gene on its own; the selected genes dropped one at a time; modules on their own
    al = fit.alone_scores(res.Xbands, res.y, res.ybands, res.block)
    alone, alone_b, slope_b = al["alone"], al["alone_band"], al["slope_band"]
    lab, members = _modules_of(res.Xbands, S_full, cover)
    groups = {j: members[int(lab[i])] for i, j in enumerate(S_full)}
    ev = fit.evaluate("perband", res.X, res.Xbands, res.y, res.ybands, res.block, kmax, per_gene=True, groups=groups)
    full_be = fit.band_explained(res.L, res.y, ev["oof"], res.ybands, res.pitch_um, res.sigmas)
    drop, drop_b, drop_m = ev["heldout"], ev["heldout_band"], ev["heldout_module"]
    mod_alone = _module_alone(res.X, res.Xbands, res.y, res.ybands, res.block, members)
    for m, cols in members.items():
        if mod_alone[m] is None:
            mod_alone[m] = float(alone[cols[0]])
    log(f"{len(members)} modules among the {len(S_full)} selected genes; best single gene alone {alone.max():.3f} ({genes[int(np.argmax(alone))]})")

    # 2. the same on random placements of the target
    null_a, null_m, null_ab = [], [], []
    for i in range(n_null):
        yn = _null_target(res, rng, pitch_units)
        if yn is None:
            continue
        ynb = bands.split(res.L, yn, res.pitch_um, res.sigmas)
        aln = fit.alone_scores(res.Xbands, yn, ynb, res.block)
        null_a.append(float(aln["alone"].max()))
        null_ab.append({b: float(aln["alone_band"][b].max()) for b in BANDS})
        Sn, _, _ = attribution.full_fit(res.Xbands, ynb, kmax)
        _, mem_n = _modules_of(res.Xbands, Sn, cover)
        ma = _module_alone(res.X, res.Xbands, yn, ynb, res.block, mem_n)
        null_m.append(max([v for v in ma.values() if v is not None] + [float(aln["alone"][Sn].max())]))
    null_a, null_m = np.asarray(null_a), np.asarray(null_m)
    p = np.array([nulls.empirical_p(null_a, alone[j]) for j in range(P)]) if len(null_a) else np.full(P, np.nan)
    pb = {b: (np.array([nulls.empirical_p([d[b] for d in null_ab], alone_b[b][j]) for j in range(P)]) if len(null_ab) else np.full(P, np.nan)) for b in BANDS}
    pm = {m: (nulls.empirical_p(null_m, mod_alone[m]) if len(null_m) else float("nan")) for m in members}
    if len(null_a):
        log(f"{len(null_a)} random placements: best single gene {null_a.mean():.3f} on average, {null_a.max():.3f} at most; "
            f"{int((p <= alpha).sum())} genes above that bar")

    # 3. selection frequency under half-sampling, real target and a few placements
    tile_um = 2 * res.sigmas[1] if tile_um is None else tile_um
    tiles, tile_units = make_tiles(res.coords, tile_um / res.pitch_um * pitch_units)
    tile_um = tile_units * res.pitch_um / pitch_units
    n_tiles = int(tiles.max()) + 1
    sel, beta = _runs(res.Xbands, res.ybands, tiles, k_sel, n_resample, rng)
    freq = sel.mean(0)
    null_freq = []
    for i in range(n_null_stability):
        yn = _null_target(res, rng, pitch_units)
        if yn is None:
            continue
        ynb = bands.split(res.L, yn, res.pitch_um, res.sigmas)
        s_null, _ = _runs(res.Xbands, ynb, tiles, k_sel, n_resample_null, rng)
        null_freq.append(float(s_null.mean(0).max()))
    null_freq = np.asarray(null_freq)
    log(f"{n_resample} half-samplings of {n_tiles} tiles ({tile_um:.0f} um): {int((freq >= tau).sum())} genes with freq >= {tau}"
        + (f"; random placements reach {null_freq.max():.2f} at most" if len(null_freq) else ""))

    # 4. tables
    mod_of = {j: int(lab[i]) for i, j in enumerate(S_full)}
    rows = []
    for g in genes[np.argsort(-alone)]:
        j = gi[g]
        if not (g in full.index or p[j] <= alpha or freq[j] >= 0.1):
            continue
        dom = full.dominant[g] if g in full.index else BANDS[int(np.argmax([alone_b[b][j] for b in BANDS]))]
        bi = BANDS.index(dom)
        picked = sel[:, j]
        b_sel = beta[picked, bi, j]
        ref = full[f"coef_{dom}"][g] if g in full.index else (np.median(b_sel) if picked.any() else 0.0)
        sign_cons = float(np.mean(np.sign(b_sel) == np.sign(ref))) if picked.any() and ref != 0 else float("nan")
        m = mod_of.get(j, -1)
        mem = members[m] if m >= 0 else [j]
        rows.append(dict(gene=g, order=int(full.order[g]) if g in full.index else -1, module=m, dominant=dom,
                         sign="+" if slope_b[dom][j] > 0 else "-", sign_fit=("+" if ref > 0 else "-") if g in full.index else "",
                         alone=float(alone[j]), **{f"alone_{b}": float(alone_b[b][j]) for b in BANDS},
                         p=float(p[j]), **{f"p_{b}": float(pb[b][j]) for b in BANDS},
                         drop=float(drop[j]) if g in full.index else float("nan"), **{f"drop_{b}": float(drop_b[b][j]) if g in full.index else float("nan") for b in BANDS},
                         freq=float(freq[j]), freq_module=float(sel[:, mem].any(1).mean()), n_module=len(mem), sign_consistency=sign_cons))
    table = pd.DataFrame(rows)
    mrows = []
    for m, cols in members.items():
        rows_m = table[table.gene.isin(genes[cols])]
        mrows.append(dict(module=m, n=len(cols), genes=", ".join(f"{g}{s}" for g, s in zip(rows_m.gene, rows_m.sign)),
                          alone=float(mod_alone[m]), p=float(pm[m]), drop=float(drop_m[cols[0]]),
                          dominant=rows_m.dominant.mode().iloc[0] if len(rows_m) else "",
                          freq_module=float(rows_m.freq_module.iloc[0]) if len(rows_m) else float("nan")))
    modules = pd.DataFrame(mrows).sort_values(["p", "alone"], ascending=[True, False]).reset_index(drop=True) if mrows else pd.DataFrame()
    supported = table[table.p <= alpha].gene.tolist()
    stable = table[table.freq >= tau].gene.tolist()

    # 5. the supported genes together
    S = [gi[g] for g in supported]
    if S:
        evs = fit.evaluate_fixed(res.X, res.Xbands, res.y, res.ybands, res.block, S)
        bes = fit.band_explained(res.L, res.y, evs["oof"], res.ybands, res.pitch_um, res.sigmas)
        supported_fit = dict(k=len(S), best=float(evs["best"]), explained=bes)
    else:
        supported_fit = dict(k=0, best=0.0, explained={b: 0.0 for b in BANDS})
    full_fit = dict(k=kmax, best=float(ev["best"]), explained=full_be)
    q = float(sel.sum(1).mean())
    bound = q ** 2 / ((2 * tau - 1) * P) if tau > 0.5 else float("inf")
    conf = Confidence(table=table.sort_values(["p", "alone"], ascending=[True, False]).reset_index(drop=True), modules=modules,
                      supported=supported, stable=stable, alpha=alpha, tau=tau, k_sel=k_sel, n_resample=n_resample,
                      n_tiles=n_tiles, tile_um=float(tile_um), n_null=int(len(null_a)), null_max_alone=null_a, null_max_module=null_m,
                      null_max_freq=null_freq, expected_false_stable=float(bound), supported_fit=supported_fit, full_fit=full_fit, band=band)
    res.confidence = conf
    log(f"supported: {len(supported)} genes; together they reach {supported_fit['best']:.3f} held-out")
    return conf
