"""Reproduces the published kidney numbers (Visium HD iu04 tissue 3 at 16 um) when the data file is present."""
import os
import numpy as np
import pytest

H5AD = os.environ.get("SCALEMATCH_KIDNEY_H5AD", "/mnt/user-data/uploads/orchestra/data/Kidney/iu04_tissues/iu04_tissue_3.h5ad")
SIG = {
    "podocyte": ["NPHS1", "NPHS2", "PODXL", "PTPRO", "WT1", "SYNPO", "CLIC5", "NDNF", "DDN", "PTPRQ"],
    "TAL": ["UMOD", "SLC12A1", "CASR", "KCNJ1", "CLDN16", "ESRRB", "PAPPA2"],
    "VSMC": ["MYH11", "ACTA2", "CNN1", "TAGLN", "DES", "RERGL", "PLN", "LDB3"],
}
EXPECTED = {"podocyte": (0.417, 0.798), "TAL": (0.386, 0.655), "VSMC": (0.596, 0.921)}  # held-out best, MID explained


@pytest.mark.skipif(not os.path.exists(H5AD), reason="kidney data not available")
@pytest.mark.parametrize("target", list(SIG))
def test_kidney16(target):
    import anndata as ad
    from scalematch import explain, aggregate_bins, programme_score
    a = ad.read_h5ad(H5AD)
    counts, r, c, members = aggregate_bins(a.X, a.obs["grid_row"], a.obs["grid_col"], 2)
    coords = np.column_stack([c * 16.0, r * 16.0])
    y, used = programme_score(counts, a.var_names, SIG[target])
    excl = sum(SIG.values(), [])
    res = explain(counts, np.asarray(a.var_names), coords, y, pitch_um=16.0, exclude_genes=excl, sigmas=(40, 200),
                  designs=("perband",), null_kinds=(), n_null=0, verbose=False)
    best, mid = EXPECTED[target]
    assert abs(res.scores["perband"]["best"] - best) < 0.01
    assert abs(res.band_explained["perband"]["MID"] - mid) < 0.01
