# ssc_v3: SSC pipeline V3

V3 finds the genes that explain a target region at each spatial scale (scale-specific contributors, SSCs). It runs four steps:

1. **Bands:** split the target and every gene into heat-kernel bands.
2. **S_orig:** select a gene set by forward selection.
3. **Removal test:** test each selected gene by taking it out of the set.
4. **Substitution table:** describe the genes that were not selected.

V3 replaces the earlier v3d workflow (backward reduction to a minimal set, swap test, enumeration of combinations). Those steps are not run. The old backward reduction is kept as a stand-alone function, `ssc_v3.minimal_panel()`.

It reuses this repository's `scalematch` package for the graph, normalisation, strips and attribution, so it lives in a separate folder and does not change the package.

## Algorithm

| Step | What it does |
|---|---|
| 1. Bands | Heat-kernel filters on the unit graph's L_sym, expanded as degree-94 Chebyshev series (error < 1e-12 against `expm_multiply`). The bands sum to the signal: LOW = e^{−t_L L}, MID = e^{−t_S L} − e^{−t_L L}, HIGH = I − e^{−t_S L}, with t = 2(σ/pitch)². Default σ = 40/200 µm; the rule is that the lower edge is max(40 µm, 0.7 × pitch). |
| 2. S_orig | Band-matched forward selection up to 30 genes. The score is 1 − pooled held-out NMSE over 5 spatial strips. k_best is the size with the lowest cross-validated error, and S_orig is the first k_best genes. |
| 3. Removal test | For each g in S_orig, the context is C = S_orig − g. C is fitted band by band on all units, giving residuals r_b. The statistic is T_b = ⟨h_b(L) g, r_b⟩. |
| 3. Null | Random signs on the graph-Fourier coefficients of g (all modes except the null mode). The covariance Σ_bb′ = Σ_k h_b h_b′(λ_k) ĝ_k² r̂_bk r̂_b′k is approximated without eigenvectors. **V1** (default) smooths ĝ_k² over λ with Chebyshev/KPM moments; **V2** computes the lowest 100 eigenpairs exactly and smooths the rest. |
| 3. Call | Q = zᵀR⁺z against χ²(rank) gives p_joint; q_joint is BH over the S_orig genes only (no control genes). **is_SSC = q_joint ≤ 0.05.** The scale is the dominant band of a full refit of S_orig, and the sign is that band's coefficient. The output also gives per-band z and removal_cost = score(S_orig) − score(S_orig − g); neither is used for the call. |
| 4. Substitution | For every gene h not in S_orig: rel_b = corr(h_b, y_b); cover_b = R² of h_b on the S_orig band signals; b† = argmax\|rel_b\|; rep = the S_orig gene most correlated with h in band b†. The label is `covered by <rep>`, `relevant, not covered` or `unrelated`, using thresholds \|rel\| ≥ 0.15 and cover ≥ 0.5. |
| Calibration gate | Optional (`calibrate=True`). Builds 200 exact random-sign surrogate genes, tests each in S_orig − (a random member), and reports type I error at α = 0.05 and 0.01 with Clopper–Pearson CIs. If V1's interval excludes the nominal level, the calls are redone with V2. Exact surrogates need the eigenvectors. On a bipartite unit lattice they come from one SVD of an (n/2 × n/2) block; other graphs use dense eigh, so only small ones are practical. |
| Noise floor | Reference line only, not a gate: 2 × SD of score(S_orig) over 16 strip layouts (4/5/6/8 strips × 4 offsets). |

## Use

Requirements: numpy, scipy, pandas, **numba**, and this repository's `scalematch` package (`pip install -e .` from the repository root).

```python
import sys; sys.path.insert(0, "ssc_v3")
from ssc_v3 import run_v3, write_outputs

res = run_v3(counts, genes, coords, target,         # units x genes counts (e.g. 32 um bins), names, (units, 2) coords, 0/1 or float target
             pitch_um=32, sigmas=(40, 200),
             include_genes=my_dictionary,           # optional; default = every detected gene
             exclude_signature=False,               # True drops NPHS2, PODXL, WT1
             variance="v1", calibrate=True)
write_outputs(res, "out/", name="section1")
```

Command line, with an `.h5ad` that has counts in `X`, coordinates in `obsm["spatial"]` and the target in `obs`:

```bash
python ssc_v3/run_ssc_v3.py section.h5ad --target-col glom --out out/ --name section1 --sigma 40 --calib
```

## Outputs

| File | Columns |
|---|---|
| `ssc_<name>.csv` | gene, order, q_joint, p_joint, z_LOW, z_MID, z_HIGH, removal_cost, coef_*, share_*, dominant, is_SSC (plus sign, rank) |
| `substitution_<name>.csv` | gene, rel_LOW/MID/HIGH, cover_LOW/MID/HIGH, rep, label (plus band_dagger, max_abs_rel, cover_dagger) |
| `masking_<name>.csv` | non-SSC genes of S_orig with a partner in S_orig at \|corr\| ≥ 0.8 in their dominant band (description only) |
| `calib_<name>.csv` | per-surrogate p values (when `calibrate=True`) |
| `layouts_<name>.csv`, `run_<name>.json` | layout scores for the noise floor; S_orig, SSC list, calibration, band energy of the target, timing |

## Validation

- **Reference implementation.** On two kidney Xenium sections (32 µm bins) this folder reproduces the internal reference implementation exactly: S_orig, SSC calls, calibration and substitution labels are identical, with numerical differences ≤ 2e-15.
- **Forward selection.** On 8 sections, forward selection reproduces stored S_orig sets exactly at σ 70/200 and at 40/200.
- **Speed.** One section runs in seconds to about a minute on 16 cores, not counting the calibration surrogates.
- **Smoke test.** `ssc_v3/tests/test_smoke.py` runs on a synthetic lattice.

## Caveats

- **The removal test is liberal.** It uses an in-sample statistic. On real sections it calls most S_orig genes significant, including some whose held-out removal cost is ≤ 0, where a held-out permutation test of the same removal does not. Treat `is_SSC` together with `removal_cost`; the noise floor is the reference line for removal_cost.
- **p values below about 1e-10 are too small** under V1 and V2. Calls are unaffected.
