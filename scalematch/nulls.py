"""Chance references: random versions of the target that keep what should not matter.

``rigid``      the same shape or field, turned, flipped and put down elsewhere on the section
``surrogate``  a random field with the same energy in every band as the target
``blob``       a random connected region with as many units as the target region
"""
import numpy as np
from scipy.spatial import cKDTree
from .bands import BANDS, split


def _place(coords, pts, pitch, tree):
    d, idx = tree.query(pts)
    ok = d <= 0.6 * pitch
    return ok, idx


def rigid_mask(mask, coords, pitch, rng, tries=400, min_mapped=0.8):
    """The 0/1 region as a cut-out: random rotation, a flip half the time, random centre."""
    coords = np.asarray(coords, float)
    tree = cKDTree(coords)
    src = coords[mask] - coords[mask].mean(0)
    n = int(mask.sum())
    for _ in range(tries):
        th = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        if rng.uniform() < 0.5:
            R = R @ np.diag([1.0, -1.0])
        pts = src @ R.T + coords[int(rng.integers(len(coords)))]
        ok, idx = _place(coords, pts, pitch, tree)
        if ok.mean() < min_mapped:
            continue
        uniq = np.unique(idx[ok])
        if len(uniq) < min_mapped * n:
            continue
        out = np.zeros(len(coords), bool)
        out[uniq] = True
        return out
    return None


def rigid_field(y, coords, pitch, rng, tries=400, min_mapped=0.8):
    """A continuous field moved as a whole: rotation, flip, translation; unmapped units get the median.

    Random rotations are tried first; on an elongated section they rarely fit, so the fallback
    is the set of lattice-preserving moves (identity, two mirrors, 180 degrees) with a translation
    along the long axis.
    """
    coords = np.asarray(coords, float)
    y = np.asarray(y, float)
    tree = cKDTree(coords)
    c = coords.mean(0)
    span = np.ptp(coords, axis=0)
    ax = int(np.argmax(span))
    for attempt in range(tries):
        if attempt < tries // 2:
            th = rng.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            if rng.uniform() < 0.5:
                R = R @ np.diag([1.0, -1.0])
            shift = coords[int(rng.integers(len(coords)))] - c
        else:
            # lattice-preserving moves with a translation of 2-12% of the long axis (at least 4 pitches)
            R = [np.diag([1.0, 1.0]), np.diag([-1.0, 1.0]), np.diag([1.0, -1.0]), np.diag([-1.0, -1.0])][int(rng.integers(4))]
            shift = np.zeros(2)
            shift[ax] = max(rng.uniform(0.02, 0.12) * span[ax], 4 * pitch) * (1 if rng.uniform() < 0.5 else -1)
            shift[1 - ax] = rng.uniform(-0.02, 0.02) * span[1 - ax]
        src = (coords - c) @ R + c + shift
        ok, idx = _place(coords, src, pitch, tree)
        if ok.mean() < min_mapped or (attempt < tries // 2 and ok.mean() < 0.9):
            continue
        out = np.full(len(y), np.median(y))
        out[ok] = y[idx[ok]]
        return out
    return None


def surrogate_field(ybands, L, pitch_um, sigmas, rng):
    """White noise split into the same bands, each band rescaled to the target's band sd."""
    N = len(next(iter(ybands.values())))
    w = rng.standard_normal(N)
    wb = split(L, w, pitch_um, sigmas)
    parts = {}
    for b in BANDS:
        v = wb[b] - wb[b].mean()
        parts[b] = v / (v.std() + 1e-12) * ybands[b].std()
    return sum(parts.values()), parts


def random_blob(n, nbrs, rng):
    """A connected region grown from a random seed, one random frontier node at a time."""
    N = len(nbrs)
    seed = int(rng.integers(N))
    inset = np.zeros(N, bool)
    inset[seed] = True
    frontier = set(int(j) for j in nbrs[seed])
    while inset.sum() < n and frontier:
        j = int(rng.choice(sorted(frontier)))
        frontier.discard(j)
        inset[j] = True
        for k in nbrs[j]:
            if not inset[k]:
                frontier.add(int(k))
    return inset


def empirical_p(null_values, observed):
    """(number of null values >= observed + 1) / (n + 1)."""
    null_values = np.asarray(null_values, float)
    return float((np.sum(null_values >= observed) + 1) / (len(null_values) + 1))
