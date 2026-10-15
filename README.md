# scalematch

Explain a region or a field on a tissue section by genes, scale by scale.

Give it one section (coordinates and counts) and one target: a drawn region as 0/1, a probability map, a programme score. It splits the target and every gene into three bands measured in micrometres (coarse, medium, fine), fits each band of the target only with the same band of the genes using one shared set of up to 30 genes, grades the fit on tissue it was not shown, compares the score with randomly placed versions of the target, and returns a gene list in which every gene carries a scale label and a contribution map.

Two uses:

* **Is this region supported by the data?** A coherence score with a chance reference. The score falls when a region is moved off its structure (brain layer 1: 0.55 as drawn, 0.41 moved by one spot, 0.12 moved by three, 0.11 for a random region of the same size) and holds when the region is grown (0.49).
* **Which genes explain it, and at which scale?** For glomeruli in Visium HD kidney cortex the medium band (40–200 µm) is reproduced at 0.79 on unseen tissue, against a random field with the same spectrum at about −0.2; its carriers are NES, NPNT and RIPOR1 (podocyte and glomerular basement membrane genes), while C7 and PLPP1 carry the coarse band. The genes that define the target are withheld, so these are found, not planted. Drawn as a region (twelve glomeruli from the Space Ranger clustering), the target is MID-dominated (band shares 0.09 / 0.33 / 0.27) and its medium band is reproduced at 0.92, against −0.24 for the same outlines placed elsewhere.

## Install

```bash
pip install -e ".[io,plot]"
```

Requires Python 3.9+, numpy, scipy, pandas; `h5py`/`anndata` for the loaders, `matplotlib` for the plots.

## Ten lines

```python
import numpy as np, scalematch as sm

counts, genes, coords, pos = sm.read_visium("filtered_feature_bc_matrix.h5", "spots.csv")   # any (units x genes) counts, coordinates
region = (pos.layer == "L1").to_numpy()                                                        # 0/1 region, or a float field

res = sm.explain(counts, genes, coords, region, pitch_um=100, sigmas=(70, 200),
                 null_kinds=("rigid", "blob"), n_null=40)
print(res.summary())          # held-out score A, explained variance per band, p against chance, leading genes
res.gene_table                # selected genes: contribution RMS per band, dominant band
res.modules                   # every dictionary gene: selected / covered by a representative / relevant-uncovered / unrelated
res.save("out/L1")            # csv tables, npz arrays, summary.json
```

Command line, from an `.h5ad` with counts in `.X` and coordinates in `.obsm["spatial"]`:

```bash
scalematch explain --h5ad section.h5ad --pitch-um 16 --signature NPHS1 NPHS2 PODXL PTPRO WT1 --out out/podocyte
scalematch explain --h5ad section.h5ad --pitch-um 100 --target-obs layer --positive L1 --sigmas 70 200 --out out/L1
```

## Reading the output

`res.summary()` answers three questions; the rest of the output is supporting detail.

**1. Is the structure real?** `p` and `excess`. The target is placed elsewhere on the section (`rigid`), replaced by a random connected region of the same size (`blob`) or by a random field with the same spectrum (`surrogate`), and every copy is scored exactly as the target was. `excess` is the observed value minus the mean of the copies. `p` is the share of copies that did as well and cannot go below 1/(n + 1): 0.048 with 20 draws means none of the 20 came close, not a precise probability. Twelve glomeruli drawn as a region: MID band explained 0.92; the same twelve outlines placed elsewhere, −0.24; random regions of the same size, −0.42.

**2. How big is it?** `explained per band`. Each band of the target is scored on its own, and the band with the high number is the size of the structure; a negative value (worse than the band's mean) reads as zero. Glomeruli: LOW 0.97, MID 0.92, HIGH 0.22, a 40–200 µm structure whose rims are only partly predictable. Cortex layer 1 on Visium: LOW 0.74, MID 0.05, HIGH −0.11, a broad band with nothing finer in it (nothing finer than 200 µm is representable at a 100 µm pitch either).

**3. Which genes carry it?** `gene_table`, sorted by the `rms_` column of that band. Glomeruli, MID band: EMCN and IGFBP5 with a positive sign; RGS5, COL1A1 and C7 with a negative sign, so where they rise the target falls. `modules` is the bench: `covered` genes belong with a selected gene (ETS1, EGFL7 and ENG with the endothelial representatives), `relevant-uncovered` genes are related but the 30-gene budget left them out.

The supporting numbers: `A` and `best` are the overall held-out score from 0 to 1 (`best` the peak over 1–30 genes, `A` the mean, so `A` is high when a few genes already do well); `target band shares` are the target's own make-up, to be read with the explained values, since a band with a large share and a low explained value is noise (the podocyte score keeps 56% of its variance in HIGH and the fit explains 11% of it); `raw` is the genes fitted unsplit, which falls below `perband` whenever the scale separation is doing work.

Three targets, read this way:

| target | pitch | shares LOW / MID / HIGH | explained LOW / MID / HIGH | reading |
|---|---|---|---|---|
| cortex layer 1, drawn region (Visium) | 100 µm | 0.74 / 0.05 / 0.03 | 0.74 / 0.05 / −0.11 | a broad band; coarse structure only |
| twelve glomeruli, drawn region (Visium HD) | 16 µm | 0.09 / 0.33 / 0.27 | 0.97 / 0.92 / 0.22 | a MID-sized structure; the rims only partly predictable |
| podocyte score, field (Visium HD) | 16 µm | 0.08 / 0.16 / 0.56 | 0.92 / 0.79 / 0.11 | the same structure under bin-level grain; HIGH is the grain |

## Tutorials

* [`tutorials/01_region_visium_dlpfc.ipynb`](tutorials/01_region_visium_dlpfc.ipynb): a drawn region on Visium (human DLPFC 151673, public): score, chance reference, genes, and what happens when the region is moved or grown.
* [`tutorials/02_continuous_target_visium_hd.ipynb`](tutorials/02_continuous_target_visium_hd.ipynb): a programme score on Visium HD kidney at 16 µm bins: the glomeruli live in the medium band, which genes carry it, the module table, and the same glomeruli drawn as a region, a MID-dominated target.

Both notebooks are stored with their outputs.

## Method in one paragraph

On the spatial graph of the section (radius graph on a lattice, or k nearest neighbours) with symmetric normalised Laplacian `L`, the heat kernel `exp(-tL)` is a continuous-time random walk, so a Gaussian of width σ corresponds to `t = 2 (σ / pitch)²`. Two kernels give three bands that add up to the identity: LOW = `exp(-t_L L)` (wider than σ_L), MID = `exp(-t_S L) − exp(-t_L L)`, HIGH = `I − exp(-t_S L)` (finer than σ_S). The target `y` and the dictionary `X` (the 500 most variable genes after removing the ones that define the target) are split with the same bands. The model is `ŷ = Σ_b (α_b + X_b β_b)` with band `b` of `y` fitted by `X_b` only; one gene is one group across bands; group forward selection with an OLS refit at each step, k = 1…30 (a joint group lasso gives the same picture). Five spatial strips are held out in turn: `NMSE(k)` on the held-out units gives the score `A = mean_k max(0, 1 − NMSE(k))`, the best `1 − NMSE`, and the explained variance per band (the held-out residual re-split with the same bands). Chance references: the same mask or field moved rigidly (rotated, flipped, translated), random fields with the same band energies, random connected regions of the same size; `p = (#null ≥ observed + 1) / (n + 1)`. Attribution: in the full-data fit at k = 30 the contribution of gene `j` in band `b` is `z_{b,j} β_{b,j}`, a map in target units; the module table classifies every dictionary gene against the selected set by its correlation with the target band and its R² on the selected genes' bands.

See [`docs/method.md`](docs/method.md) for details, defaults and limitations.

## What it does not do yet

* Gene-level significance. Contributions are sizes, not tests; selection frequency under block bootstrap and contribution-vs-null p values are the planned additions.
* Boundary tightness. The score is a test of placement; a region drawn too large keeps most of its score.
* Non-linear explanations. The model is linear in the band-filtered genes.
* Pitch and structure size. Mid-band explanations need bins well below the structure size: on one kidney section re-sampled from 16 to 96 µm the MID-band explained variance fell from 0.80 to 0.46, and to 0.15 on a virtual Visium lattice. On Visium (100 µm pitch) anything under 200 µm is below the sampling limit.

## License

MIT.
