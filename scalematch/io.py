"""Loaders: 10x h5 matrices, Visium positions, Visium HD bin aggregation."""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_10x_h5(path):
    """Counts (barcodes x genes, csr), barcodes and gene names from a 10x ``filtered_feature_bc_matrix.h5``."""
    import h5py
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        shape = tuple(m["shape"][:])  # (genes, barcodes)
        M = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
        barcodes = m["barcodes"][:].astype(str)
        feats = m["features"]
        genes = feats["name"][:].astype(str)
    return M.T.tocsr(), barcodes, genes


def read_visium(h5_path, positions_csv, barcode_col="barcode", in_tissue_col="in_tissue",
                x_col="pixel_col", y_col="pixel_row"):
    """Counts, gene names, coordinates and the positions table, restricted to barcodes present in both files."""
    counts, barcodes, genes = read_10x_h5(h5_path)
    pos = pd.read_csv(positions_csv)
    if in_tissue_col in pos:
        pos = pos[pos[in_tissue_col] == 1]
    pos = pos.set_index(barcode_col).loc[[b for b in barcodes if b in set(pos[barcode_col])]]
    order = {b: i for i, b in enumerate(barcodes)}
    rows = np.array([order[b] for b in pos.index])
    return counts[rows], genes, pos[[x_col, y_col]].to_numpy(float), pos.reset_index()


def aggregate_bins(counts, row, col, factor):
    """Sum ``factor x factor`` blocks of a binned array (for example 8 um Visium HD bins to 16 um).

    Returns the aggregated counts (csr), the bin row and column indices, and the number of
    source bins in each aggregated bin.
    """
    row = np.asarray(row).astype(int) // factor
    col = np.asarray(col).astype(int) // factor
    key = row * 100000 + col
    uniq, inv = np.unique(key, return_inverse=True)
    M = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
    agg = (M @ sp.csr_matrix(counts)).tocsr()
    members = np.asarray(M.sum(axis=1)).ravel()
    return agg, uniq // 100000, uniq % 100000, members


def aggregate_labels(labels, row, col, factor):
    """Majority label of the source bins in every ``factor x factor`` block, in the order of ``aggregate_bins``."""
    import pandas as pd
    row = np.asarray(row).astype(int) // factor
    col = np.asarray(col).astype(int) // factor
    key = row * 100000 + col
    uniq, inv = np.unique(key, return_inverse=True)
    maj = pd.DataFrame(dict(b=inv, lab=np.asarray(labels))).groupby("b").lab.agg(lambda s: s.value_counts().index[0])
    return maj.reindex(np.arange(len(uniq))).to_numpy()


def programme_score(counts, genes, signature):
    """Mean log CP10K of a gene signature per unit: a continuous target from a gene list."""
    from .dictionary import log_cp10k
    Xn, _ = log_cp10k(counts)
    gi = {g: i for i, g in enumerate(np.asarray(genes).astype(str))}
    idx = [gi[g] for g in signature if g in gi]
    if not idx:
        raise ValueError("none of the signature genes is present")
    sub = Xn[:, idx]
    return np.asarray(sub.mean(axis=1)).ravel(), [g for g in signature if g in gi]
