"""SSC pipeline V3, end to end: Bands -> S_orig -> per-gene removal test on S_orig -> substitution table.
No backward reduction, no swap test, no combination enumeration (see ../README.md for the specification)."""
import time, json, os, numpy as np, pandas as pd
from scalematch import graph, dictionary, fit, attribution
from . import engine as E, specnull as CN, v3lib as V
from .surrogates import random_sign_surrogates

SIGNATURE = ("NPHS2", "PODXL", "WT1")
NULL_KW = dict(K=200, n_probe=50, eps=1e-3, K_low=100)


def select_dictionary(counts, genes, include_genes=None, exclude_genes=(), min_detect_frac=0.0):
    """Dictionary = include_genes (in that order) or every gene, restricted to genes detected in > min_detect_frac of units (and at least one),
    minus exclude_genes."""
    gi = {g: i for i, g in enumerate(genes)}; det = np.asarray((counts > 0).sum(0)).ravel() / counts.shape[0]
    base = list(include_genes) if include_genes is not None else list(genes)
    ex = set(exclude_genes)
    return [g for g in base if g in gi and det[gi[g]] > max(min_detect_frac, 0.0) and g not in ex]


def run_v3(counts, genes, coords, target, pitch_um=32.0, sigmas=(40.0, 200.0), include_genes=None, exclude_genes=(), exclude_signature=False,
           min_detect_frac=0.0, kmax=30, n_strips=5, variance="v1", null_kw=None, calibrate=False, n_surrogates=200, surrogates=None, auto_v2=True,
           noise_floor=True, rel_thr=0.15, cover_thr=0.5, q_thr=0.05, log=print):
    """counts: (units x genes) sparse or dense counts; genes: names; coords: (units, 2) in micrometres-scaled coordinates as scalematch expects;
    target: (units,) 0/1 mask or field. Returns dict of DataFrames ('ssc', 'substitution', 'masking', 'calibration', 'layouts') and 'run' (dict)."""
    import scipy.sparse as sp
    T00 = time.time(); TIM = {}; null_kw = dict(NULL_KW, **(null_kw or {})); SIG = tuple(float(s) for s in sigmas); genes = [str(g) for g in genes]
    # ---- graph and dictionary ----
    t = time.time(); A, Lsym, keep, pu = graph.build_graph(np.asarray(coords, float), kind="auto"); Lsym = sp.csr_matrix(Lsym); deg = np.asarray(A.sum(1)).ravel()
    ck = np.asarray(coords, float)[keep]; N = len(ck); C = sp.csr_matrix(counts)[keep]
    dic = select_dictionary(C, genes, include_genes, tuple(exclude_genes) + (SIGNATURE if exclude_signature else ()), min_detect_frac)
    P = len(dic); gidx = {g: i for i, g in enumerate(genes)}
    Xn, _ = dictionary.log_cp10k(C); X = np.asarray(dictionary.dense_columns(Xn, np.array([gidx[g] for g in dic])), float)
    y = np.asarray(target, float)[keep]; block = fit.make_strips(ck, n_strips, along=fit.region_axis(ck, y > 0.5))
    NS, SRC = 0, []
    if calibrate:
        if surrogates is None:
            pool = [g for g in dic if g not in SIGNATURE]; SRC = sorted(np.random.default_rng(20261116).choice(pool, min(n_surrogates, len(pool)), replace=False).tolist())
            XS, sinfo = random_sign_surrogates(Lsym, X[:, [dic.index(g) for g in SRC]]); log("surrogates", sinfo)
        else:
            XS, SRC = surrogates; XS = np.asarray(XS, float); SRC = [str(s) for s in SRC]; assert XS.shape[0] == N
        NS = XS.shape[1]; X = np.hstack([X, XS])
    TIM["inputs"] = time.time() - t
    # ---- Step 1: bands ----
    t = time.time(); op = CN.Op(Lsym); PX = X.shape[1]
    Xb_all, Fq_all, KH = CN.band_features(op, np.ascontiguousarray(np.column_stack([X, y, np.ones(N)])), SIG, pitch_um, squared=True)
    Xb = {b: np.ascontiguousarray(v[:, :PX]) for b, v in Xb_all.items()}; Fq = {b: np.ascontiguousarray(v[:, :PX]) for b, v in Fq_all.items()}
    yb = {b: v[:, PX].copy() for b, v in Xb_all.items()}; Fy = {b: v[:, PX].copy() for b, v in Fq_all.items()}; H1 = {b: v[:, PX + 1].copy() for b, v in Xb_all.items()}
    del Xb_all, Fq_all; TIM["bands"] = time.time() - t
    en = {b: float(((yb[b] - yb[b].mean()) ** 2).sum()) for b in V.BANDS}; tot = sum(en.values()); en = {b: v / tot for b, v in en.items()}
    # ---- Step 2: S_orig ----
    t = time.time(); XbD = {b: Xb[b][:, :P] for b in V.BANDS}
    T = E.Target(E.build_cache(XbD, block, y[:, None], {b: yb[b][:, None] for b in V.BANDS}), 0)
    nm, _ = T.cv_curve(kmax); kb = int(np.argmin(nm) + 1); Sf, _ = T.fs(T.F, kmax); S = list(Sf[:kb]); S_orig = [dic[j] for j in S]
    score = lambda sets: E.score_any(T, [list(x) for x in sets]); score_ref = float(score([S])[0]); TIM["forward_selection"] = time.time() - t
    log(f"S_orig: k_best {kb}, score {score_ref:.4f}")
    # ---- Step 3: per-gene removal test ----
    ctx = [[x for x in S if x != g] for g in S]

    def removal_test(method):
        sn = CN.SpecNull(method, Lsym=Lsym, deg=deg, Xraw=X, Xb=Xb, Fq=Fq, yb=yb, Fy=Fy, H1=H1, sig=SIG, pitch_um=pitch_um, op=op, **null_kw)
        sn.add_contexts(ctx); rows = []
        for g, c in zip(S, ctx):
            o = sn.test(c, [g]); rows.append(dict(gene=dic[g], p_joint=float(o["p_joint"][0]), rank=int(o["rank"][0]), **{f"z_{b}": float(o["z"][0, i]) for i, b in enumerate(V.BANDS)}))
        R = pd.DataFrame(rows); R["q_joint"] = V.bh(R.p_joint.values); return sn, R

    def calib(sn):
        pick = np.random.default_rng(20261118).integers(0, len(S), NS); rows = []
        for j in range(NS):
            o = sn.test(ctx[pick[j]], [P + j]); rows.append(dict(surrogate=j, source_gene=SRC[j], removed=dic[S[pick[j]]], p_joint=float(o["p_joint"][0])))
        Rc = pd.DataFrame(rows); out = {}
        for al in (0.05, 0.01):
            k = int((Rc.p_joint < al).sum()); lo, hi = V.clopper_pearson(k, NS); out[str(al)] = dict(type1=k / NS, lo=lo, hi=hi, k=k, n=NS, excludes_nominal=not (lo <= al <= hi))
        return Rc, out

    t = time.time(); sn, R = removal_test(variance); TIM["removal_test"] = time.time() - t; used = variance; CAL = {}; CALT = []
    if calibrate:
        t = time.time(); Rc, CAL[variance] = calib(sn); CALT.append(Rc.assign(variance=variance)); TIM["calibration"] = time.time() - t
        if any(v["excludes_nominal"] for v in CAL[variance].values()) and variance == "v1" and auto_v2:
            log("V1 fails calibration; rerunning with V2"); sn2, R2 = removal_test("v2"); Rc2, CAL["v2"] = calib(sn2); CALT.append(Rc2.assign(variance="v2")); R, used = R2, "v2"
    R["removal_cost"] = score([S])[0] - score(ctx)
    desc = V.describe(XbD, yb, S, dic, CN, attribution).set_index("gene")
    for c in ["order"] + [f"{p}_{b}" for p in ("coef", "share") for b in V.BANDS] + ["dominant"]: R[c] = R.gene.map(desc[c])
    R["is_SSC"] = R.q_joint <= q_thr; R["sign"] = [np.sign(desc.loc[g, f"coef_{desc.loc[g, 'dominant']}"]) for g in R.gene]
    COLS = ["gene", "order", "q_joint", "p_joint", "z_LOW", "z_MID", "z_HIGH", "removal_cost", "coef_LOW", "coef_MID", "coef_HIGH",
            "share_LOW", "share_MID", "share_HIGH", "dominant", "is_SSC", "sign", "rank"]
    R = R[COLS]
    # ---- Step 4: substitution table ----
    t = time.time(); SUB = V.substitution_table(XbD, yb, S, [j for j in range(P) if j not in S], dic, rel_thr, cover_thr); TIM["substitution"] = time.time() - t
    TIM["v3_total_excl_calibration"] = time.time() - T00 - TIM.get("calibration", 0)
    # ---- auxiliary ----
    LAY = pd.DataFrame(); NF = None
    if noise_floor:
        lay = []
        for (n_, d_), blk in V.strip_layouts(ck, y, fit):
            Tl = E.Target(E.build_cache(XbD, blk, y[:, None], {b: yb[b][:, None] for b in V.BANDS}), 0); lay.append(dict(n_strips=n_, offset=d_, score=float(E.score(Tl, [S])[0])))
        LAY = pd.DataFrame(lay); NF = 2 * float(LAY.score.std(ddof=1))
    MSK = V.masking_partners(XbD, S, dic, dict(zip(R.gene, R.dominant)), R.gene[~R.is_SSC].tolist())
    run = dict(n_units=N, n_dictionary=P, dictionary=dic, sigmas=SIG, pitch_um=pitch_um, variance_requested=variance, variance_used=used, null_settings=null_kw,
               k_best=kb, S_orig=S_orig, score_S_orig=score_ref, n_SSC=int(R.is_SSC.sum()), SSC=R.gene[R.is_SSC].tolist(),
               SSC_neg_MID=R.gene[R.is_SSC & (R.coef_MID < 0)].tolist(), noise_floor=NF, n_layouts=len(LAY), target_band_energy=en,
               calibration=CAL, timing=TIM)
    return dict(ssc=R, substitution=SUB, masking=MSK, calibration=pd.concat(CALT) if CALT else pd.DataFrame(), layouts=LAY, run=run)


def write_outputs(res, outdir, name="section"):
    os.makedirs(outdir, exist_ok=True)
    res["ssc"].to_csv(f"{outdir}/ssc_{name}.csv", index=False)
    res["substitution"].to_csv(f"{outdir}/substitution_{name}.csv", index=False)
    res["masking"].to_csv(f"{outdir}/masking_{name}.csv", index=False)
    if len(res["calibration"]): res["calibration"].to_csv(f"{outdir}/calib_{name}.csv", index=False)
    if len(res["layouts"]): res["layouts"].to_csv(f"{outdir}/layouts_{name}.csv", index=False)
    json.dump(res["run"], open(f"{outdir}/run_{name}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
