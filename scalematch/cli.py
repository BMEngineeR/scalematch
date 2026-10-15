"""Command line: ``scalematch explain``."""
import argparse
import sys

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(prog="scalematch", description="Explain a tissue region or field by genes, scale by scale.")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("explain", help="run the band-matched explanation on an .h5ad file")
    e.add_argument("--h5ad", required=True, help="AnnData with counts in .X, coordinates in .obsm['spatial']")
    e.add_argument("--pitch-um", type=float, required=True, help="centre-to-centre spacing of the units in micrometres")
    e.add_argument("--target-obs", help=".obs column holding the target (0/1, bool, label or float)")
    e.add_argument("--positive", help="label in --target-obs that defines the region (for categorical columns)")
    e.add_argument("--signature", nargs="*", default=None, help="gene list: target = mean log CP10K of these genes (excluded from the dictionary)")
    e.add_argument("--sigmas", nargs=2, type=float, default=(40.0, 200.0), metavar=("SIGMA_S", "SIGMA_L"))
    e.add_argument("--n-genes", type=int, default=500)
    e.add_argument("--kmax", type=int, default=30)
    e.add_argument("--nulls", nargs="*", default=["rigid"], choices=["rigid", "surrogate", "blob"])
    e.add_argument("--n-null", type=int, default=40)
    e.add_argument("--designs", nargs="*", default=["perband", "raw"], choices=["perband", "raw", "concat"])
    e.add_argument("--out", required=True)
    e.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    import anndata as ad
    from .api import explain
    from .io import programme_score

    adata = ad.read_h5ad(a.h5ad)
    counts = adata.X
    genes = np.asarray(adata.var_names).astype(str)
    coords = np.asarray(adata.obsm["spatial"], float)
    exclude = ()
    if a.signature:
        target, used = programme_score(counts, genes, a.signature)
        exclude = tuple(used)
    elif a.target_obs:
        col = adata.obs[a.target_obs]
        if a.positive is not None:
            target = (col.astype(str) == a.positive).to_numpy()
        elif col.dtype == bool or set(np.unique(col)).issubset({0, 1}):
            target = col.to_numpy().astype(bool)
        else:
            target = col.to_numpy().astype(float)
    else:
        sys.exit("give --target-obs or --signature")
    res = explain(counts, genes, coords, target, a.pitch_um, exclude_genes=exclude, sigmas=tuple(a.sigmas),
                  n_genes=a.n_genes, kmax=a.kmax, designs=tuple(a.designs), null_kinds=tuple(a.nulls), n_null=a.n_null, seed=a.seed)
    res.save(a.out)
    print(res.summary())


if __name__ == "__main__":
    main()
