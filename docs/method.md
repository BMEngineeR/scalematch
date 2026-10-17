# Method notes

## Inputs

| Argument | Meaning | Default |
|---|---|---|
| `counts` | units x genes count matrix, dense or scipy sparse | required |
| `genes` | gene names | required |
| `coords` | units x 2 coordinates in any unit | required |
| `target` | bool / 0-1 (region) or float (field), one value per unit | required |
| `pitch_um` | centre-to-centre spacing of the units in micrometres | required |
| `exclude_genes` | genes used to define the target; removed from the dictionary | `()` |
| `sigmas` | `(σ_S, σ_L)` in µm: HIGH < σ_S < MID < σ_L < LOW | `(40, 200)` |
| `n_genes` | dictionary size (most variable genes after exclusions) | 500 |
| `kmax` | gene budget of the sparse fit | 30 |
| `n_strips` | held-out spatial strips | 5 |
| `strip_axis` | `"target"` (strips across the region's own axis) or `"section"` | `"target"` |
| `designs` | `"perband"`, `"raw"`, `"concat"` | `("perband", "raw")` |
| `null_kinds` | `"rigid"`, `"surrogate"`, `"blob"` | `("rigid",)` |
| `n_null` | draws per null kind | 40 |

Helpers for building a target: `programme_score` (mean log CP10K of a gene list), `aggregate_bins` and `aggregate_labels` (counts summed and majority labels over `factor x factor` blocks of a binned array), `clean_region` (drops the connected pieces of a 0/1 region under a chosen size). Any 0/1 or float vector over the units is a valid target.

Choosing `sigmas`: the bands are physical, so the same values compare across platforms. The default 40/200 µm puts a glomerulus (about 150 µm) in the MID band on Visium HD or imaging data. On Visium (100 µm pitch) the shortest representable wavelength is 200 µm, so 40 µm is below the lattice; use `(70, 200)` and read the MID band as the spot-to-spot band.

## Graph

`build_graph` keeps neighbours closer than 1.15 x pitch (4 on a square lattice, 6 on a hexagonal one) and falls back to 6 nearest neighbours on an irregular layout. Only the largest connected component is kept. `L` is the symmetric normalised Laplacian.

## Bands

`t = 2 (σ / pitch)²`, `exp(-tL)` by `scipy.sparse.linalg.expm_multiply`. The three bands add up to the identity; their energy shares overlap and need not add to one. At small pitches and large σ the kernel is expensive (t grows with (σ/pitch)²); 16 µm bins with σ_L = 200 µm take a few minutes for 500 genes on 12,000 units.

## Fit

Design `perband`: `ŷ = Σ_b (α_b + X_b β_b)` with band b of y fitted by X_b only; one gene = one group. Group forward selection: at each step the gene with the largest sum over bands of squared partial correlation with the current residuals enters; every band then refits its coefficients by OLS on the selected set (ridge 1e-8 N for numerical stability). `raw` fits the unsplit genes on y; `concat` fits the three bands of every gene as one 3P-column design on y (the control that shows why the bands are kept apart).

Held-out: `n_strips` strips of equal count cut perpendicular to an axis; for a region, the principal axis of the region's own coordinates so that every strip holds part of it. Pooled held-out `NMSE(k)`; `A = mean_k max(0, 1 − NMSE(k))`; `best = max_k (1 − NMSE(k))`; explained variance per band from the held-out residual re-split with the same bands.

## Chance references

* `rigid`: for a region, the mask as a cut-out, rotated by a random angle, flipped half the time, centred on a random unit (kept if at least 80% lands on tissue); for a field, the whole field moved rigidly. In both cases random rotations are tried first and, when they do not land on tissue (elongated sections, regions scattered along them), the lattice-preserving moves about the section centre with a translation of 2–12% of the long axis are used instead. A draw that finds no placement is skipped.
* `surrogate`: white noise split into the bands, each band rescaled to the target's band standard deviation, summed: same spectrum, no anatomy.
* `blob`: a connected region grown from a random seed one random neighbour at a time, to the size of the target region.

Every null is scored with the identical procedure (same strips, same dictionary). `p = (#null ≥ observed + 1)/(n + 1)`; `excess` is the observed value minus the null mean. The band-explained values of nulls are computed on the re-split null field, exactly as for the real target; for a random field they are typically at or below zero.

## Attribution

Full-data fit at `kmax`. Contribution of gene j in band b: `c_{b,j} = z_{b,j} β_{b,j}` (standardised band-filtered gene times its coefficient), a vector in target units; `rms_b` is its root mean square; `share_b = rms_b / Σ_b rms_b` is the gene's own split across bands; `dominant` is the band with the largest `rms`.

`modules`: for every dictionary gene and band, `relevance` = |correlation| of the gene's band with the target's band; `coverage` = R² of the gene's band on the selected genes' bands; `representative` = the selected gene with the largest coefficient in that regression; `status` = selected / covered (relevance ≥ 0.3 and coverage ≥ 0.5) / relevant-uncovered / unrelated.

## Gene-level confidence

`confidence(res)` works on the band-split dictionary kept in the Result (so it runs in the same session as `explain`). It reports three readings per gene, and the first carries the p value.

* `alone`: the gene fitted on its own, one slope per band on the training strips, scored on the held-out strips (1 − NMSE); `alone_<band>` the same within one band against the band's own variance. Every dictionary gene gets this number. The same is computed for `n_null` random rigid placements of the target; `p = (#placements whose best single gene reached the gene's value + 1)/(n_null + 1)`. Because the bar is the best of all dictionary genes on each placement, it already accounts for the search over genes. A gene with `p <= alpha` is *supported*.
* `drop`: the held-out error rises by this fraction of the target's variance when the gene is removed from the fold's selected set and the remaining genes are refitted (selection is not repeated). This is necessity; genes that stand in for one another have a large `alone` and a small `drop`.
* `freq`: selection frequency among the first `k_sel` genes over `n_resample` half-samplings of the section (complementary halves of square tiles of side `tile_um`, default 2 σ_L, shrunk until there are at least 30 tiles); `sign_consistency` is how often the coefficient in the gene's dominant band keeps its sign when the gene is selected. `expected_false_stable` is the Meinshausen–Bühlmann bound `q² / ((2 τ − 1) p)` with `q` the mean number of genes selected per run and `p` the dictionary size. The half-sampling is repeated on `n_null_stability` placements; `null_max_freq` is the highest frequency any gene reached on them. It is often high: a randomly placed target also has a gene that fits it best in every half, so frequency on its own is not evidence.

Modules: within the selected set, every gene is regressed on the other selected genes (bands stacked); a gene with R² ≥ `cover` is linked to the gene with the largest coefficient; connected components are the modules. `conf.modules` gives each module's members with their signs, `alone` (the members fitted together), `p` (against the best module of each placement) and `drop` (the whole module removed). The supported genes fitted together on the held-out strips (`supported_fit`) show what the trusted genes reproduce without the rest of the list.

Everything is computed with the same strips as the main fit. The p values are one-sided and test placement only: they say that the gene's pattern follows the target better than chance, not that its sign or its band label is right.

## Known limits

1. Gene-level p values are against random placements of the target and are one-sided. Collinear genes share credit with the first one selected in the full fit; `confidence` reports each of them on its own and groups them, but it does not say which member of a module is the cause.
2. The score does not penalise a region drawn too large.
3. Everything is linear.
4. Mid-band explanations need a pitch well below the structure size.
5. The energy shares of overlapping bands are descriptive, not a partition.
