"""Heat-kernel bands calibrated in physical units.

On a lattice with the symmetric normalised Laplacian ``L``, ``exp(-tL)`` is a continuous-time
random walk with unit jump rate, so a Gaussian of width ``sigma`` corresponds to
``t = 2 (sigma / pitch)^2``. Two kernels give three bands that add up to the identity::

    LOW  = exp(-t_L L)                   structure wider than sigma_L
    MID  = exp(-t_S L) - exp(-t_L L)     structure between sigma_S and sigma_L
    HIGH = I - exp(-t_S L)               structure finer than sigma_S

The bands overlap (they are not orthogonal), so their energy shares need not add to one.
"""
import numpy as np
from scipy.sparse.linalg import expm_multiply

BANDS = ("LOW", "MID", "HIGH")


def t_of(sigma_um, pitch_um):
    """Heat-kernel time for a Gaussian width ``sigma_um`` on a lattice of pitch ``pitch_um``."""
    return 2.0 * (float(sigma_um) / float(pitch_um)) ** 2


def heat(L, S, t):
    """``exp(-tL) S`` for a vector or matrix ``S`` (columns are signals)."""
    if t <= 0:
        return np.array(S, float, copy=True)
    return expm_multiply(-t * L, np.asarray(S, float))


def split(L, S, pitch_um, sigmas=(40.0, 200.0)):
    """Split the columns of ``S`` into LOW, MID and HIGH bands. Returns a dict of arrays."""
    sig_s, sig_l = sigmas
    Es = heat(L, S, t_of(sig_s, pitch_um))
    El = heat(L, S, t_of(sig_l, pitch_um))
    return {"LOW": El, "MID": Es - El, "HIGH": np.asarray(S, float) - Es}


def energy_shares(bands, S):
    """Share of the centred variance of each column of ``S`` carried by each band."""
    S = np.asarray(S, float)
    if S.ndim == 1:
        S = S[:, None]
        bands = {b: v[:, None] for b, v in bands.items()}
    tot = ((S - S.mean(0)) ** 2).sum(0)
    return {b: ((v - v.mean(0)) ** 2).sum(0) / tot for b, v in bands.items()}


def scale_spectrum(L, y, pitch_um, sigmas_um=None):
    """Fraction of the variance of ``y`` at scales finer than ``sigma``, for a grid of sigmas.

    Returns ``(sigmas, residual_fraction)``; the difference between consecutive entries is
    the band-pass spectrum.
    """
    if sigmas_um is None:
        sigmas_um = [8 * 2 ** (i / 2) for i in range(13)]  # 8 ... 512 um
    y = np.asarray(y, float)
    yc = y - y.mean()
    tot = float(yc @ yc)
    out = []
    for s in sigmas_um:
        Sy = heat(L, y, t_of(s, pitch_um))
        out.append(float(((yc - (Sy - Sy.mean())) ** 2).sum() / tot))
    return np.array(sigmas_um, float), np.array(out)
