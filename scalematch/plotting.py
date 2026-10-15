"""Matplotlib views of a Result: target bands, held-out curves, contribution maps."""
import numpy as np
from .bands import BANDS


def _scatter(ax, coords, v, s=6, cmap="RdBu_r", lim=None, title=""):
    v = np.asarray(v, float)
    lim = np.percentile(np.abs(v - v.mean()), 99.5) if lim is None else lim
    ax.scatter(coords[:, 0], -coords[:, 1], c=v - v.mean(), s=s, lw=0, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9, loc="left")


def plot_target_bands(res, s=6, figsize=(14, 3.2)):
    """Y = LOW + MID + HIGH on the tissue. Y and LOW share a colour scale; MID and HIGH use their own."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    yc = res.y - res.y.mean()
    lim = np.percentile(np.abs(yc), 99.5)
    _scatter(axes[0], res.coords, res.y, s, lim=lim, title="target")
    _scatter(axes[1], res.coords, res.ybands["LOW"], s, lim=lim, title=f"= LOW (> {res.sigmas[1]:g} um), share {res.band_shares['LOW']:.2f}")
    for ax, b, lab in [(axes[2], "MID", f"+ MID ({res.sigmas[0]:g}-{res.sigmas[1]:g} um)"), (axes[3], "HIGH", f"+ HIGH (< {res.sigmas[0]:g} um)")]:
        _scatter(ax, res.coords, res.ybands[b], s, title=f"{lab}, share {res.band_shares[b]:.2f}, own scale")
    fig.tight_layout()
    return fig


def plot_curves(res, figsize=(5.5, 3.6)):
    """Held-out 1 - NMSE against the number of genes, one line per design, with the null band."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    for d, s in res.scores.items():
        k = np.arange(1, len(s["nmse"]))
        ax.plot(k, 1 - s["nmse"][1:], lw=2, label=f"{d} (A = {s['A']:.2f})")
    if len(res.null_scores):
        for kind, sub in res.null_scores.groupby("kind"):
            ax.axhline(sub[sub.design == list(res.scores)[0]].best.mean(), color="#898781", lw=1, ls=":")
            ax.text(len(k), sub[sub.design == list(res.scores)[0]].best.mean(), f" null {kind}", fontsize=8, color="#898781", va="center")
    ax.set_xlabel("genes (k)")
    ax.set_ylabel("held-out 1 - NMSE")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_contributions(res, genes=None, band="MID", n=4, s=6, figsize=None):
    """The target's band next to the contribution maps of the leading genes in that band."""
    import matplotlib.pyplot as plt
    tab = res.gene_table
    if genes is None:
        genes = tab.sort_values(f"rms_{band}", ascending=False).gene.head(n).tolist()
    fig, axes = plt.subplots(1, len(genes) + 1, figsize=figsize or (3.3 * (len(genes) + 1), 3.0))
    _scatter(axes[0], res.coords, res.ybands[band], s, title=f"target {band} band")
    order = tab.gene.tolist()
    for ax, g in zip(axes[1:], genes):
        i = order.index(g)
        c = res.contributions[band][:, i]
        _scatter(ax, res.coords, c, s, title=f"{g}: {band} contribution (RMS {np.sqrt(np.mean(c ** 2)):.3f})")
    fig.tight_layout()
    return fig
