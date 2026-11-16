"""Command line for SSC pipeline V3.
Input: an .h5ad with raw counts in X (units x genes, e.g. 32 um bins), coordinates in obsm["spatial"] and the target in obs[<target-col>]
(0/1 region or a float field).
usage: python run_ssc_v3.py data.h5ad --target-col glom --out results/ [--pitch 32] [--sigma 40] [--variance v1|v2] [--exclude-signature]
       [--include-genes genes.txt] [--calib] [--name section]"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="SSC pipeline V3")
    ap.add_argument("h5ad"); ap.add_argument("--target-col", required=True); ap.add_argument("--out", required=True); ap.add_argument("--name", default="section")
    ap.add_argument("--pitch", type=float, default=32.0, help="unit pitch in micrometres (bin size)")
    ap.add_argument("--sigma", type=float, default=40.0, help="lower band edge sigma_S in um (sigma_L is 200)")
    ap.add_argument("--variance", default="v1", choices=["v1", "v2"]); ap.add_argument("--exclude-signature", action="store_true")
    ap.add_argument("--include-genes", help="text file, one gene per line (dictionary, in this order)")
    ap.add_argument("--calib", action="store_true", help="calibration gate with 200 exact random-sign surrogates (switches to V2 if V1 fails)")
    a = ap.parse_args()
    import anndata as ad
    from ssc_v3 import run_v3, write_outputs
    A = ad.read_h5ad(a.h5ad); inc = [l.strip() for l in open(a.include_genes) if l.strip()] if a.include_genes else None
    res = run_v3(A.X, list(A.var_names), np.asarray(A.obsm["spatial"], float), A.obs[a.target_col].to_numpy(float), pitch_um=a.pitch,
                 sigmas=(a.sigma, 200.0), include_genes=inc, exclude_signature=a.exclude_signature, variance=a.variance, calibrate=a.calib)
    write_outputs(res, a.out, a.name)
    r = res["run"]; print(f"S_orig {r['k_best']} genes; SSC {r['n_SSC']}; variance used {r['variance_used']}; outputs in {a.out}")


if __name__ == "__main__":
    main()
