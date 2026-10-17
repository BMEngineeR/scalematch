"""Every gene in the matrix on its own, against random placements of the target.

``explain`` works on a dictionary of the most variable genes; a gene outside it cannot be
selected or scored. ``scan_alone`` goes through the whole matrix in chunks, splits each chunk
into the bands, scores every gene alone on the held-out strips (as ``confidence`` does for
the dictionary) and compares it with the best gene of the same chunked scan on random rigid
placements of the target. Nothing larger than one chunk is kept in memory, so the cost is
time: roughly the band split of ``explain`` times (genes / chunk).
"""
import time

import numpy as np
import pandas as pd

from . import bands, dictionary, fit, graph, nulls
from .bands import BANDS


def scan_alone(counts, genes, coords, target, pitch_um, *, exclude_genes=(), sigmas=(40.0, 200.0), min_detect=0.01,
               n_null=20, n_strips=5, strip_axis="target", chunk=1000, graph_kind="auto", seed=0, verbose=True):
    """Held-out score of every gene on its own, and its p against random placements of the target.

    Returns a DataFrame with one row per gene detected in at least ``min_detect`` of the units
    (``exclude_genes`` left out): ``alone`` and ``alone_<band>`` (held-out 1 - NMSE of the gene
    alone, overall and within each band), ``sign`` (the sign of its slope in the band where it
    does best), ``detected`` (fraction of units with a count), ``p`` and ``p_<band>`` (share of
    the ``n_null`` placements whose best gene reached the same value), sorted by ``alone``.
    ``df.attrs`` holds the null bars (``null_max_alone``, per placement) and the settings.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    log = (lambda *a: print(f"[scan {time.time() - t0:5.0f}s]", *a, flush=True)) if verbose else (lambda *a: None)
    coords = np.asarray(coords, float)
    target = np.asarray(target)
    is_region = target.dtype == bool or set(np.unique(target)).issubset({0, 1})
    y_all = target.astype(float)

    A, L, keep, pitch_units = graph.build_graph(coords, kind=graph_kind)
    coords_k = coords[keep]
    y = y_all[keep]
    ybands = bands.split(L, y, pitch_um, sigmas)
    axis = fit.region_axis(coords_k, y > 0.5) if (is_region and strip_axis == "target") else None
    block = fit.make_strips(coords_k, n_strips, along=axis)

    # random placements, split once
    targets = [(y, ybands)]
    for i in range(n_null):
        yn = (nulls.rigid_mask(y > 0.5, coords_k, pitch_units, rng) if is_region else nulls.rigid_field(y, coords_k, pitch_units, rng))
        if yn is None:
            continue
        yn = np.asarray(yn, float)
        targets.append((yn, bands.split(L, yn, pitch_um, sigmas)))
    n_placed = len(targets) - 1
    log(f"{int(keep.sum())} units, {n_placed} random placements split")

    # the genes: detected often enough, not in the exclusion list
    Xn, _ = dictionary.log_cp10k(counts[keep] if not hasattr(counts, "tocsr") else counts.tocsr()[keep])
    gnames = np.asarray(genes).astype(str)
    det = np.asarray((Xn > 0).mean(axis=0)).ravel()
    excl = set(map(str, exclude_genes))
    idx = np.where((det >= min_detect) & ~np.isin(gnames, list(excl)))[0]
    log(f"{len(idx)} genes detected in >= {min_detect:.0%} of units ({len(gnames) - len(idx)} left out), chunks of {chunk}")

    alone = np.zeros(len(idx))
    alone_b = {b: np.zeros(len(idx)) for b in BANDS}
    sign_b = {b: np.zeros(len(idx)) for b in BANDS}
    null_max = np.full(n_placed, -np.inf)
    null_max_b = {b: np.full(n_placed, -np.inf) for b in BANDS}
    for s in range(0, len(idx), chunk):
        cols = idx[s:s + chunk]
        X = dictionary.dense_columns(Xn, cols)
        Xb = bands.split(L, X, pitch_um, sigmas)
        for t, (yt, ytb) in enumerate(targets):
            al = fit.alone_scores(Xb, yt, ytb, block)
            if t == 0:
                alone[s:s + chunk] = al["alone"]
                for b in BANDS:
                    alone_b[b][s:s + chunk] = al["alone_band"][b]
                    sign_b[b][s:s + chunk] = al["slope_band"][b]
            else:
                null_max[t - 1] = max(null_max[t - 1], float(al["alone"].max()))
                for b in BANDS:
                    null_max_b[b][t - 1] = max(null_max_b[b][t - 1], float(al["alone_band"][b].max()))
        log(f"genes {s + len(cols)}/{len(idx)}: best so far {gnames[idx[int(np.argmax(alone[:s + len(cols)]))]]} {alone[:s + len(cols)].max():.3f}")

    best_band = [BANDS[int(np.argmax([alone_b[b][i] for b in BANDS]))] for i in range(len(idx))]
    df = pd.DataFrame(dict(gene=gnames[idx], detected=det[idx], alone=alone, **{f"alone_{b}": alone_b[b] for b in BANDS},
                           best_band=best_band, sign=["+" if sign_b[bb][i] > 0 else "-" for i, bb in enumerate(best_band)]))
    if n_placed:
        df["p"] = [nulls.empirical_p(null_max, v) for v in alone]
        for b in BANDS:
            df[f"p_{b}"] = [nulls.empirical_p(null_max_b[b], v) for v in alone_b[b]]
    df = df.sort_values("alone", ascending=False).reset_index(drop=True)
    df.attrs.update(dict(null_max_alone=null_max.tolist(), null_max_band={b: null_max_b[b].tolist() for b in BANDS},
                         n_null=n_placed, pitch_um=float(pitch_um), sigmas=list(sigmas), min_detect=min_detect,
                         n_units=int(keep.sum()), runtime_s=time.time() - t0))
    log(f"done; random placements: best gene {null_max.mean():.3f} on average, {null_max.max():.3f} at most")
    return df
