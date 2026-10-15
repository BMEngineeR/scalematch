"""Per-gene, per-band contributions and the module table for the whole dictionary."""
import numpy as np
import pandas as pd
from .bands import BANDS
from .fit import forward_select, _zstats


def full_fit(Xbands, ybands, kmax=30):
    """Band-matched fit on all units. Returns the selected columns, the standardised blocks and the coefficients."""
    N = next(iter(Xbands.values())).shape[0]
    allm = np.ones(N, bool)
    Z = {}
    for b in BANDS:
        mu, sd = _zstats(Xbands[b], allm)
        Z[b] = (Xbands[b] - mu) / sd
    targets = [ybands[b] - ybands[b].mean() for b in BANDS]
    S, betas = forward_select([Z[b] for b in BANDS], targets, kmax)
    beta = {b: betas[-1][i] for i, b in enumerate(BANDS)}
    return S, Z, beta


def contributions(S, Z, beta, genes):
    """Contribution vectors ``c[b][:, i] = z_{b, S_i} * beta_{b, i}`` and a per-gene summary table."""
    c = {b: Z[b][:, S] * beta[b][None, :] for b in BANDS}
    rows = []
    for i, j in enumerate(S):
        rms = {b: float(np.sqrt(np.mean(c[b][:, i] ** 2))) for b in BANDS}
        tot = sum(rms.values()) + 1e-12
        rows.append(dict(order=i + 1, gene=str(genes[j]), **{f"coef_{b}": float(beta[b][i]) for b in BANDS},
                         **{f"rms_{b}": rms[b] for b in BANDS}, **{f"share_{b}": rms[b] / tot for b in BANDS},
                         dominant=max(BANDS, key=lambda b: rms[b])))
    return c, pd.DataFrame(rows)


def modules(Xbands, Z, S, ybands, genes, band="MID", cover=0.5, relevance=0.3):
    """Every gene in the dictionary, in one band, classified against the selected set.

    relevance   |correlation| of the gene's band with the target's band
    coverage    R^2 of the gene's band on the selected genes' bands
    status      selected / covered (redundant with a representative) / relevant-uncovered / unrelated
    """
    N = Z[band].shape[0]
    y = ybands[band] - ybands[band].mean()
    Zb = Z[band]
    Zs = Zb[:, S]
    G = Zs.T @ Zs + 1e-6 * N * np.eye(len(S))
    rel = np.abs(Zb.T @ y / N) / (y.std() + 1e-12)
    Bcoef = np.linalg.solve(G, Zs.T @ Zb)  # coefficients of every gene on the selected set
    fitted = Zs @ Bcoef
    cov = 1 - ((Zb - fitted) ** 2).sum(0) / np.maximum((Zb ** 2).sum(0), 1e-12)
    rep = np.array([str(genes[S[int(np.argmax(np.abs(Bcoef[:, j])))]]) for j in range(Zb.shape[1])])
    sel = np.zeros(Zb.shape[1], bool)
    sel[S] = True
    status = np.where(sel, "selected", np.where((rel >= relevance) & (cov >= cover), "covered",
                      np.where(rel >= relevance, "relevant-uncovered", "unrelated")))
    df = pd.DataFrame(dict(gene=np.asarray(genes).astype(str), band=band, relevance=rel, coverage=cov,
                           representative=np.where(sel, "", rep), status=status))
    return df.sort_values(["status", "relevance"], ascending=[True, False]).reset_index(drop=True)
