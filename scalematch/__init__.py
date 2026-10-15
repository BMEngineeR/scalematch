"""scalematch: explain a region or field on a tissue section by genes, scale by scale."""
from .api import explain, Result
from .bands import split, energy_shares, scale_spectrum, t_of
from .graph import build_graph, estimate_pitch, clean_region
from .fit import evaluate, make_strips, region_axis, band_explained
from .io import read_visium, read_10x_h5, aggregate_bins, aggregate_labels, programme_score

__version__ = "0.1.0"
__all__ = ["explain", "Result", "split", "energy_shares", "scale_spectrum", "t_of", "build_graph", "estimate_pitch", "clean_region",
           "evaluate", "make_strips", "region_axis", "band_explained", "read_visium", "read_10x_h5", "aggregate_bins", "aggregate_labels", "programme_score"]
