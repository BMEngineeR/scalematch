"""One call: explain a region or a continuous field on a tissue section by its genes, scale by scale."""
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import attribution, bands, dictionary, fit, graph, nulls


@dataclass
class Result:
    """Everything ``explain`` returns. ``save(dir)`` writes the tables and arrays to disk."""
    target_kind: str
    pitch_um: float
    sigmas: tuple
    n_units: int
    genes: np.ndarray
    coords: np.ndarray
    keep: np.ndarray
    y: np.ndarray
    ybands: dict
    band_shares: dict
    scores: dict                     # per design: A, best, at_kmax, k_best, nmse curve
    band_explained: dict             # per design: LOW/MID/HIGH held-out explained variance
    oof: dict                        # per design: out-of-fold prediction at kmax
    null_scores: pd.DataFrame        # one row per null draw and design
    p_values: dict                   # per design and statistic
    gene_table: pd.DataFrame         # selected genes with per-band contributions
    contributions: dict              # band -> (N x k) contribution vectors
    modules: pd.DataFrame            # every dictionary gene, status against the selected set
    meta: dict = field(default_factory=dict)
    # kept in memory for ``confidence`` and further fits; not written by ``save``
    X: np.ndarray = None             # dictionary, log CP10K, units x genes
    Xbands: dict = None              # band -> units x genes
    L: object = None                 # normalised Laplacian (sparse)
    block: np.ndarray = None         # held-out strip label per unit
    confidence: object = None        # filled by ``scalematch.confidence``

    def summary(self, design="perband"):
        s = self.scores[design]
        be = self.band_explained[design]
        lines = [f"{self.target_kind} target on {self.n_units} units, pitch {self.pitch_um:g} um, bands {self.sigmas[0]:g}/{self.sigmas[1]:g} um",
                 f"held-out: A = {s['A']:.3f}, best 1-NMSE = {s['best']:.3f} at k = {s['k_best']}",
                 "explained per band (held-out): " + ", ".join(f"{b} {be[b]:.2f}" for b in bands.BANDS),
                 "target band shares: " + ", ".join(f"{b} {self.band_shares[b]:.2f}" for b in bands.BANDS)]
        for key, p in self.p_values.items():
            if key.startswith(design):
                lines.append(f"{key}: p = {p['p']:.3f}, excess over null mean = {p['excess']:+.3f} (n = {p['n']})")
        top = self.gene_table.head(10)
        lines.append("top genes (order, dominant band): " + ", ".join(f"{g} ({d})" for g, d in zip(top.gene, top.dominant)))
        if self.confidence is not None:
            lines.append(self.confidence.summary())
        return "\n".join(lines)

    def save(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        self.gene_table.to_csv(os.path.join(outdir, "genes.csv"), index=False)
        self.modules.to_csv(os.path.join(outdir, "modules.csv"), index=False)
        self.null_scores.to_csv(os.path.join(outdir, "nulls.csv"), index=False)
        np.savez_compressed(os.path.join(outdir, "arrays.npz"), coords=self.coords, keep=self.keep, y=self.y,
                            **{f"y_{b}": v for b, v in self.ybands.items()},
                            **{f"contrib_{b}": v for b, v in self.contributions.items()},
                            **{f"oof_{d}": v for d, v in self.oof.items()},
                            **{f"nmse_{d}": v["nmse"] for d, v in self.scores.items()}, genes=self.genes)
        meta = dict(target_kind=self.target_kind, pitch_um=self.pitch_um, sigmas=list(self.sigmas), n_units=self.n_units,
                    band_shares=self.band_shares, p_values=self.p_values,
                    scores={d: {k: v for k, v in s.items() if k != "nmse" and k != "oof"} for d, s in self.scores.items()},
                    band_explained=self.band_explained, **self.meta)
        json.dump(meta, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
        if self.confidence is not None:
            c = self.confidence
            c.table.to_csv(os.path.join(outdir, "confidence.csv"), index=False)
            c.modules.to_csv(os.path.join(outdir, "confidence_modules.csv"), index=False)
            meta["confidence"] = dict(supported=c.supported, stable=c.stable, alpha=c.alpha, tau=c.tau, k_sel=c.k_sel,
                                      n_resample=c.n_resample, n_tiles=c.n_tiles, tile_um=c.tile_um, n_null=c.n_null,
                                      null_max_alone=c.null_max_alone.tolist(), null_max_module=c.null_max_module.tolist(),
                                      null_max_freq=c.null_max_freq.tolist(), expected_false_stable=c.expected_false_stable,
                                      supported_fit=c.supported_fit, full_fit=c.full_fit)
            json.dump(meta, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
        open(os.path.join(outdir, "summary.txt"), "w").write(self.summary() + "\n")


def explain(counts, genes, coords, target, pitch_um, *, exclude_genes=(), include_genes=(), sigmas=(40.0, 200.0), n_genes=500,
            kmax=30, n_strips=5, strip_axis="target", designs=("perband", "raw"), null_kinds=("rigid",), n_null=40,
            graph_kind="auto", min_detect=0.05, seed=0, verbose=True):
    """Explain ``target`` (a 0/1 region or a continuous field over the units) by a gene dictionary.

    counts       units x genes count matrix (dense or scipy sparse)
    genes        gene names, length = counts.shape[1]
    coords       units x 2 coordinates, any unit; ``pitch_um`` gives the centre-to-centre spacing in micrometres
    target       length-units array: bool/0-1 for a region, float for a field
    exclude_genes  genes used to define the target (they are removed from the dictionary)
    include_genes  genes added to the dictionary whatever their variance (for example the genes a
                 whole-matrix ``scan_alone`` found outside the most variable ones)
    sigmas       (sigma_S, sigma_L) in micrometres: HIGH < sigma_S < MID < sigma_L < LOW
    null_kinds   any of "rigid", "surrogate", "blob" (blob only for a region)
    strip_axis   "target": held-out strips cut across the region's own principal axis so every strip
                 holds part of it (regions only); "section": strips along the section's long axis
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    log = (lambda *a: print(f"[scalematch {time.time() - t0:5.0f}s]", *a, flush=True)) if verbose else (lambda *a: None)
    coords = np.asarray(coords, float)
    target = np.asarray(target)
    is_region = target.dtype == bool or set(np.unique(target)).issubset({0, 1})
    y_all = target.astype(float)

    # graph on the largest component, in coordinate units; physical scale through pitch_um
    A, L, keep, pitch_units = graph.build_graph(coords, kind=graph_kind)
    N = int(keep.sum())
    coords_k = coords[keep]
    y = y_all[keep]
    log(f"graph: {N} units kept of {len(coords)}, pitch {pitch_units:.2f} units = {pitch_um:g} um")

    # dictionary
    Xn, tot = dictionary.log_cp10k(counts[keep] if not hasattr(counts, "tocsr") else counts.tocsr()[keep])
    idx = dictionary.select_dictionary(Xn, genes, n=n_genes, exclude=exclude_genes, min_detect=min_detect, include=include_genes)
    X = dictionary.dense_columns(Xn, idx)
    gsel = np.asarray(genes).astype(str)[idx]
    log(f"dictionary: {X.shape[1]} genes, {len(exclude_genes)} excluded by name" + (f", {len(include_genes)} included by name" if len(include_genes) else ""))

    # bands
    ybands = {b: v for b, v in bands.split(L, y, pitch_um, sigmas).items()}
    Xbands = bands.split(L, X, pitch_um, sigmas)
    shares = {b: float(v[0]) for b, v in bands.energy_shares(ybands, y).items()}
    log("bands done; target shares " + ", ".join(f"{b} {shares[b]:.2f}" for b in bands.BANDS))

    # held-out fits
    axis = fit.region_axis(coords_k, y > 0.5) if (is_region and strip_axis == "target") else None
    block = fit.make_strips(coords_k, n_strips, along=axis)
    if is_region:
        per = np.bincount(block[y > 0.5], minlength=n_strips)
        log(f"strips: {np.bincount(block).tolist()} units, region units per strip {per.tolist()}")
    scores, bexpl, oof = {}, {}, {}
    for d in designs:
        ev = fit.evaluate(d, X, Xbands, y, ybands, block, kmax)
        scores[d] = {k: v for k, v in ev.items() if k != "oof"}
        oof[d] = ev["oof"]
        bexpl[d] = fit.band_explained(L, y, ev["oof"], ybands, pitch_um, sigmas)
        log(f"{d}: A {ev['A']:.3f} best {ev['best']:.3f} | explained " + ", ".join(f"{b} {bexpl[d][b]:.2f}" for b in bands.BANDS))

    # nulls
    nrows = []
    nbrs = graph.neighbours(A) if "blob" in null_kinds else None
    for kind in null_kinds:
        for i in range(n_null):
            if kind == "rigid":
                yn = (nulls.rigid_mask(y > 0.5, coords_k, pitch_units, rng) if is_region
                      else nulls.rigid_field(y, coords_k, pitch_units, rng))
                yn = None if yn is None else yn.astype(float)
            elif kind == "surrogate":
                yn, _ = nulls.surrogate_field(ybands, L, pitch_um, sigmas, rng)
            elif kind == "blob":
                if not is_region:
                    continue
                yn = nulls.random_blob(int((y > 0.5).sum()), nbrs, rng).astype(float)
            else:
                raise ValueError(f"unknown null kind {kind!r}")
            if yn is None:
                continue
            ynb = bands.split(L, yn, pitch_um, sigmas)
            for d in designs:
                ev = fit.evaluate(d, X, Xbands, yn, ynb, block, kmax)
                be = fit.band_explained(L, yn, ev["oof"], ynb, pitch_um, sigmas)
                nrows.append(dict(kind=kind, draw=i, design=d, A=ev["A"], best=ev["best"], **{f"explained_{b}": be[b] for b in bands.BANDS}))
        log(f"null {kind}: {sum(r['kind'] == kind for r in nrows) // max(len(designs), 1)} draws scored")
    null_df = pd.DataFrame(nrows)
    pvals = {}
    for d in designs:
        for kind in null_kinds:
            sub = null_df[(null_df.kind == kind) & (null_df.design == d)]
            if len(sub) == 0:
                continue
            for stat in ("A", "best", "explained_MID"):
                obs = scores[d][stat] if stat in scores[d] else bexpl[d]["MID"]
                pvals[f"{d}/{kind}/{stat}"] = dict(p=nulls.empirical_p(sub[stat], obs), excess=float(obs - sub[stat].mean()),
                                                   null_mean=float(sub[stat].mean()), null_sd=float(sub[stat].std(ddof=1)) if len(sub) > 1 else float("nan"), n=int(len(sub)))

    # attribution and modules
    S, Z, beta = attribution.full_fit(Xbands, ybands, kmax)
    contrib, gtab = attribution.contributions(S, Z, beta, gsel)
    mods = pd.concat([attribution.modules(Xbands, Z, S, ybands, gsel, band=b) for b in bands.BANDS], ignore_index=True)
    log("attribution done: " + ", ".join(gtab.gene.head(8)))

    return Result(target_kind="region" if is_region else "field", pitch_um=float(pitch_um), sigmas=tuple(sigmas), n_units=N,
                  genes=gsel, coords=coords_k, keep=keep, y=y, ybands=ybands, band_shares=shares, scores=scores,
                  band_explained=bexpl, oof=oof, null_scores=null_df, p_values=pvals, gene_table=gtab,
                  contributions=contrib, modules=mods,
                  meta=dict(kmax=kmax, n_strips=n_strips, n_null=n_null, null_kinds=list(null_kinds), seed=seed,
                            pitch_units=float(pitch_units), runtime_s=time.time() - t0, counts_per_unit_median=float(np.median(tot))),
                  X=X, Xbands=Xbands, L=L, block=block)
