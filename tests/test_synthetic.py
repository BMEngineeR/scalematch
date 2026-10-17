"""Fast checks on a synthetic lattice: bands add up, a planted region is recovered, nulls score lower."""
import numpy as np
from scalematch import explain, confidence, build_graph, split, make_strips


def make_section(seed=0, n=40):
    rng = np.random.default_rng(seed)
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    coords = np.column_stack([c.ravel() * 10.0, r.ravel() * 10.0])  # 10 um pitch
    N = len(coords)
    region = ((coords[:, 0] - 200) ** 2 + (coords[:, 1] - 200) ** 2) < 120 ** 2
    A, L, keep, pitch = build_graph(coords)
    smooth = split(L, rng.standard_normal((N, 60)), pitch, (30, 120))["LOW"]
    genes = ["G%03d" % i for i in range(120)]
    base = rng.poisson(1.0, (N, 120)).astype(float)
    # genes 0-4 follow the region, 5-64 are smooth random fields, the rest is noise
    for j in range(5):
        base[:, j] += rng.poisson(6.0 * region + 0.5)
    base[:, 5:65] += rng.poisson(np.exp(2 * (smooth - smooth.mean(0)) / smooth.std(0)))
    return base, genes, coords, region, A, L, pitch


def test_bands_add_up():
    counts, genes, coords, region, A, L, pitch = make_section()
    y = region.astype(float)
    b = split(L, y, pitch, (30, 120))
    assert np.allclose(b["LOW"] + b["MID"] + b["HIGH"], y, atol=1e-8)


def test_region_recovered_and_beats_nulls():
    counts, genes, coords, region, A, L, pitch = make_section()
    res = explain(counts, genes, coords, region, pitch_um=10.0, sigmas=(30, 120), n_genes=100, kmax=10,
                  designs=("perband",), null_kinds=("rigid", "blob"), n_null=8, verbose=False)
    assert res.scores["perband"]["best"] > 0.6
    top = set(res.gene_table.gene.head(5))
    assert len(top & {"G000", "G001", "G002", "G003", "G004"}) >= 3
    for key, p in res.p_values.items():
        if key.endswith("/best"):
            assert p["excess"] > 0.1
    assert set(res.modules.status) <= {"selected", "covered", "relevant-uncovered", "unrelated"}


def test_strips_balanced():
    coords = np.random.default_rng(1).uniform(size=(1000, 2))
    block = make_strips(coords, 5)
    counts = np.bincount(block)
    assert len(counts) == 5 and counts.min() > 150


def test_confidence_supports_planted_genes():
    counts, genes, coords, region, A, L, pitch = make_section()
    res = explain(counts, genes, coords, region, pitch_um=10.0, sigmas=(30, 120), n_genes=100, kmax=10,
                  designs=("perband",), null_kinds=(), n_null=0, verbose=False)
    conf = confidence(res, n_null=20, n_resample=20, k_sel=6, n_null_stability=2, n_resample_null=10, verbose=False)
    planted = {"G000", "G001", "G002", "G003", "G004"}
    assert len(planted & set(conf.supported)) >= 4
    assert conf.supported_fit["best"] > 0.6
    assert conf.null_max_alone.max() < conf.table.alone.max()
    assert {"alone", "p", "drop", "freq", "sign_consistency", "module"} <= set(conf.table.columns)
