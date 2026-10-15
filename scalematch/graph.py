"""Spatial graph and normalised Laplacian for a set of spots, bins or cells."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def estimate_pitch(coords):
    """Median nearest-neighbour distance, in the units of ``coords``."""
    coords = np.asarray(coords, float)
    d, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(d[:, 1]))


def build_graph(coords, kind="auto", k=6, radius_factor=1.15, pitch=None):
    """Adjacency and symmetric normalised Laplacian of the largest connected component.

    kind
        ``"radius"``: neighbours closer than ``radius_factor * pitch`` (4 neighbours on a
        square lattice, 6 on a hexagonal one); ``"knn"``: ``k`` nearest neighbours, symmetrised;
        ``"auto"``: radius, falling back to knn when the lattice is irregular (mean degree < 3).

    Returns
        ``A`` (csr adjacency), ``L`` (csr normalised Laplacian), ``keep`` (bool mask of the
        points kept, the largest component), ``pitch`` (in coordinate units).
    """
    coords = np.asarray(coords, float)
    n = len(coords)
    pitch = estimate_pitch(coords) if pitch is None else float(pitch)
    tree = cKDTree(coords)

    def radius_graph():
        pairs = tree.query_pairs(radius_factor * pitch, output_type="ndarray")
        if len(pairs) == 0:
            return sp.csr_matrix((n, n))
        rows = np.r_[pairs[:, 0], pairs[:, 1]]
        cols = np.r_[pairs[:, 1], pairs[:, 0]]
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    def knn_graph():
        _, idx = tree.query(coords, k=k + 1)
        rows = np.repeat(np.arange(n), k)
        cols = idx[:, 1:].ravel()
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        A = A.maximum(A.T)
        A.data[:] = 1.0
        return A

    if kind == "radius":
        A = radius_graph()
    elif kind == "knn":
        A = knn_graph()
    elif kind == "auto":
        A = radius_graph()
        if A.nnz == 0 or np.asarray(A.sum(axis=1)).mean() < 3:
            A = knn_graph()
    else:
        raise ValueError("kind must be 'auto', 'radius' or 'knn'")

    ncomp, lab = connected_components(A, directed=False)
    keep = lab == np.argmax(np.bincount(lab))
    A = A[keep][:, keep].tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(deg))
    L = (sp.eye(A.shape[0]) - Dm @ A @ Dm).tocsr()
    return A, L, keep, pitch


def neighbours(A):
    """List of neighbour index arrays, one per node."""
    A = A.tocsr()
    return [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(A.shape[0])]


def clean_region(mask, coords, min_units=20, kind="auto"):
    """Drop the connected pieces of a 0/1 region that have fewer than ``min_units`` units.

    Returns the cleaned mask and, for the kept units, a component label (-1 outside the region).
    """
    coords = np.asarray(coords, float)
    mask = np.asarray(mask, bool)
    A, _, keep, _ = build_graph(coords, kind=kind)
    idx = np.where(keep)[0]
    m = mask[keep]
    sub = A[m][:, m]
    _, lab = connected_components(sub, directed=False)
    sizes = np.bincount(lab)
    good = sizes[lab] >= min_units
    out = np.zeros(len(mask), bool)
    out[idx[np.where(m)[0][good]]] = True
    comp = np.full(len(mask), -1)
    comp[idx[np.where(m)[0][good]]] = np.unique(lab[good], return_inverse=True)[1]
    return out, comp
