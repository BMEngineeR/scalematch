"""Normalisation and label-free gene dictionary."""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def log_cp10k(counts):
    """log1p of counts per 10,000 per spot. Accepts a dense or sparse (spots x genes) matrix."""
    if sp.issparse(counts):
        counts = counts.tocsr().astype(float)
        tot = np.asarray(counts.sum(axis=1)).ravel()
        X = sp.diags(1e4 / np.maximum(tot, 1)) @ counts
        X.data = np.log1p(X.data)
        return X.tocsc(), tot
    counts = np.asarray(counts, float)
    tot = counts.sum(1)
    return np.log1p(counts * (1e4 / np.maximum(tot, 1))[:, None]), tot


def select_dictionary(Xn, genes, n=500, exclude=(), min_detect=0.05, drop_prefixes=("MT-", "RPL", "RPS")):
    """Indices of the ``n`` most variable genes, excluding named genes and housekeeping prefixes.

    ``Xn`` is the normalised matrix (spots x genes); ``exclude`` should contain every gene that
    was used to define the target, so that the explanation cannot be circular.
    """
    genes = np.asarray(genes).astype(str)
    N = Xn.shape[0]
    if sp.issparse(Xn):
        mean = np.asarray(Xn.mean(axis=0)).ravel()
        sq = np.asarray(Xn.multiply(Xn).mean(axis=0)).ravel()
        detected = np.asarray((Xn > 0).sum(axis=0)).ravel()
    else:
        mean = Xn.mean(0)
        sq = (Xn ** 2).mean(0)
        detected = (Xn > 0).sum(0)
    var = sq - mean ** 2
    ok = detected >= min_detect * N
    ok &= ~pd.Series(genes).str.startswith(tuple(drop_prefixes)).to_numpy()
    ok &= ~np.isin(genes, list(exclude))
    order = np.argsort(-np.where(ok, var, -np.inf))[:n]
    order = order[np.isfinite(np.where(ok, var, -np.inf)[order])]
    return order


def dense_columns(Xn, idx):
    """Dense array of the selected columns."""
    return Xn[:, idx].toarray() if sp.issparse(Xn) else np.asarray(Xn[:, idx], float)
