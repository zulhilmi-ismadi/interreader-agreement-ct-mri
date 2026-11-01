"""Additional analyses reported in the manuscript.

Empty-pair views of the prostate floors with between-cohort contrasts and of every reportable floor; interval
sources and resampling units; intervals for rank correlations; the CURVAS worked example and consensus comparator;
paired contrasts (CryoNuSeg, RIGA) and C3RO expert-minus-non-expert differences; identical-mask counts; QUBIQ tasks.

Reads results/ and writes results/manuscript/ (CSV files and values_analyses.md). Bootstrap conventions are
those of the layer (percentile intervals, B = 1000, seed 20260903).

    python code/manuscript/supplementary_analyses.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = R / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)
SEED, B = 20260903, 1000
rows = []


def put(key, value, source, note=""):
    rows.append((key, value, source, note))


def f3(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def ci(a):
    return np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)


pairs = pd.read_csv(R / "layer" / "pairs.csv",
                    usecols=["dataset", "task", "image_id", "rater_a", "rater_b", "pair_class",
                             "cluster_id", "dice", "hd95"], low_memory=False)
pairs["cluster_id"] = pairs["cluster_id"].fillna(pairs["image_id"]).astype(str)
pairs["both_empty"] = (pairs.dice == 1) & (pairs.hd95 == 0)
pairs["one_empty"] = (pairs.dice == 0) & pairs.hd95.isna()
tf = pd.read_csv(R / "layer" / "task_floor.csv", low_memory=False)

VARIANTS = {
    "primary": lambda d: d,
    "both_empty_excluded": lambda d: d[~d.both_empty],                        # pairs in which both masks are empty excluded
    "conditional_delineation": lambda d: d[~d.both_empty & ~d.one_empty],  # both masks non-empty
}


def image_means(d):
    return d.groupby("image_id").agg(v=("dice", "mean"), c=("cluster_id", "first"))


def cluster_boot(im, rng, b=B):
    groups = {c: g.v.values for c, g in im.groupby("c")}
    keys = list(groups)
    out = np.empty(b)
    for i in range(b):
        draw = rng.choice(len(keys), size=len(keys), replace=True)
        out[i] = np.concatenate([groups[keys[k]] for k in draw]).mean()
    return out


# ---------------------------------------------------------------- empty-pair views of the prostate floors
rng = np.random.default_rng(SEED)
s4_rows, boots = [], {}
for ds in ["arvaniti", "gleason2019"]:
    for t in ["gleason3", "gleason4", "gleason5"]:
        base = pairs[(pairs.dataset == ds) & (pairs.task == t) & (pairs.pair_class == "P1")]
        for name, fn in VARIANTS.items():
            sub = fn(base)
            im = image_means(sub)
            bt = cluster_boot(im, rng)
            boots[(ds, t, name)] = bt
            lo, hi = ci(bt)
            s4_rows.append(dict(dataset=ds, task=t, variant=name, floor=im.v.mean(), ci_low=lo, ci_high=hi,
                                n_pairs=len(sub), n_images=len(im), n_clusters=im.c.nunique()))
            put(f"S4.{ds}/{t}/{name}", f"{f3(im.v.mean())} [{f3(lo)}, {f3(hi)}] pairs={len(sub)} images={len(im)} "
                f"clusters={im.c.nunique()}", "results/layer/pairs.csv", "cluster bootstrap, B=1000, seed 20260903")
for t in ["gleason3", "gleason4", "gleason5"]:
    for name in VARIANTS:
        d = boots[("arvaniti", t, name)] - boots[("gleason2019", t, name)]
        pa = [r for r in s4_rows if r["dataset"] == "arvaniti" and r["task"] == t and r["variant"] == name][0]["floor"]
        pg = [r for r in s4_rows if r["dataset"] == "gleason2019" and r["task"] == t and r["variant"] == name][0]["floor"]
        lo, hi = ci(d)
        put(f"S4.contrast.arvaniti-gleason2019/{t}/{name}", f"{pa - pg:+.3f} [{lo:+.3f}, {hi:+.3f}] "
            f"({'excludes 0' if lo > 0 or hi < 0 else 'includes 0'})", "results/layer/pairs.csv",
            "independent cluster bootstraps; Arvaniti has 3 clusters, so its interval is not reliable")
pd.DataFrame(s4_rows).to_csv(OUT / "s4_prostate_variants.csv", index=False)

# point estimates of all three views for every reportable Dice floor
all_rows = []
rep = tf[(tf.floor_reportable == True) & (tf.metric == "dice")]  # noqa: E712
for r in rep.itertuples():
    base = pairs[(pairs.dataset == r.dataset) & (pairs.task == r.task) & (pairs.pair_class == r.pair_class)]
    row = dict(dataset=r.dataset, task=r.task, pair_class=r.pair_class, annotation_type=r.annotation_type,
               frac_both_empty=base.both_empty.mean(), frac_one_empty=base.one_empty.mean(),
               frac_both_nonempty=(~base.both_empty & ~base.one_empty).mean(),
               frac_at_least_one_nonempty=1 - base.both_empty.mean())
    for name, fn in VARIANTS.items():
        sub = fn(base)
        row[name] = image_means(sub).v.mean() if len(sub) else np.nan
    all_rows.append(row)
s4_all = pd.DataFrame(all_rows)
s4_all.to_csv(OUT / "s4_all_floors.csv", index=False)
put("S4.all.n_floors_with_both_empty", int((s4_all.frac_both_empty > 0).sum()), "s4_all_floors.csv")
put("S4.all.max_change_both_empty_excluded", f3((s4_all.both_empty_excluded - s4_all.primary).abs().max()),
    "s4_all_floors.csv")

# ---------------------------------------------------------------- reproduction: C3RO by tier
e5 = pd.read_csv(R / "e5_published_vs_recomputed.csv")
for tier, sub in e5[e5.dataset == "c3ro"].groupby("pair_class"):
    put(f"E5.c3ro.{tier}.common@0.01", f"{int(sub.reproduces_common.sum())}/{len(sub)}",
        "results/e5_published_vs_recomputed.csv")

# ---------------------------------------------------------------- interval sources, units, ranges
unit_rows = []
for r in rep.itertuples():
    sub = pairs[(pairs.dataset == r.dataset) & (pairs.task == r.task) & (pairs.pair_class == r.pair_class)]
    if r.ci_source == "cluster_bootstrap":
        units = sub.cluster_id.nunique()
    elif r.ci_source == "rater_bootstrap":
        units = int(r.n_raters)
    else:
        units = int(r.n_images)
    unit_rows.append(dict(dataset=r.dataset, task=r.task, pair_class=r.pair_class,
                          annotation_type=r.annotation_type, floor=r.mean, ci_low=r.ci_low, ci_high=r.ci_high,
                          ci_source=r.ci_source, resampling_units=units, n_images=int(r.n_images),
                          n_pairs=int(r.n_pairs)))
un = pd.DataFrame(unit_rows)
un.to_csv(OUT / "floor_interval_units.csv", index=False)
put("R5.ci_source_counts", un.ci_source.value_counts().to_dict(), "results/layer/task_floor.csv")
put("R5.single_image_floors", int((un.n_images == 1).sum()), "results/layer/task_floor.csv")
put("R5.floors_rater_bootstrap_or_le10_units",
    int(((un.ci_source == "rater_bootstrap") | (un.resampling_units <= 10)).sum()), "results/layer/task_floor.csv")
put("R5.floors_gt10_units_not_rater", int(((un.ci_source != "rater_bootstrap") & (un.resampling_units > 10)).sum()),
    "results/layer/task_floor.csv")
pix = un[(un.annotation_type != "box") & (un.pair_class == "P1")]
for label, sub in [("all P1 pixel/instance", pix), ("excluding single-image", pix[pix.n_images > 1]),
                   ("excluding single-image and single-pair", pix[(pix.n_images > 1) & (pix.n_pairs > 1)]),
                   ("excluding single-image, single-pair and <=10 units",
                    pix[(pix.n_images > 1) & (pix.n_pairs > 1) & (pix.resampling_units > 10)])]:
    lo_r, hi_r = sub.loc[sub.floor.idxmin()], sub.loc[sub.floor.idxmax()]
    put(f"R5.range[{label}]", f"{f3(lo_r.floor)} ({lo_r.dataset}/{lo_r.task} [{f3(lo_r.ci_low)}, {f3(lo_r.ci_high)}]) "
        f"to {f3(hi_r.floor)} ({hi_r.dataset}/{hi_r.task} [{f3(hi_r.ci_low)}, {f3(hi_r.ci_high)}]); n={len(sub)}",
        "results/layer/task_floor.csv")

# ---------------------------------------------------------------- Kendall tau with bootstrap over tasks
r3 = pd.read_csv(R / "r3_metric_reorder.csv")
rng_t = np.random.default_rng(SEED)
for grid, sub in r3.groupby("grid"):
    for label, s in [("all", sub), ("excluding C3RO", sub[sub.dataset != "c3ro"])]:
        if len(s) < 4:
            continue
        tau = kendalltau(s.dice, s["nsd@2"])[0]
        bt = []
        for _ in range(2000):
            k = s.sample(len(s), replace=True, random_state=rng_t.integers(1 << 31))
            bt.append(kendalltau(k.dice, k["nsd@2"])[0])
        lo, hi = ci(np.array(bt))
        put(f"S9.tau[{grid}][{label}]", f"{tau:.2f} [{lo:.2f}, {hi:.2f}] n={len(s)}", "results/r3_metric_reorder.csv",
            "bootstrap over tasks, B=2000; tasks not fully exchangeable, so indicative")

# ---------------------------------------------------------------- CURVAS worked example and consensus comparator
s1 = pd.read_csv(R / "s1_e4_view_a_vs_b.csv")
for r in s1[s1.dataset == "curvas"].itertuples():
    put(f"R1.curvas/{r.task}", f"pairwise (same images) {f3(r.view_a_same_images_dice)}; leave-one-out vs STAPLE "
        f"{f3(r.view_b_staple_dice)}; vs majority {f3(r.view_b_majority_dice)}; images {r.n_images_view_b}",
        "results/s1_e4_view_a_vs_b.csv")
for r in tf[(tf.dataset == "curvas") & (tf.metric == "dice") & (tf.pair_class == "P1")].itertuples():
    put(f"R1.curvas/{r.task}.rss", f3(r.rss), "results/layer/task_floor.csv", "reference substitution spread (range)")
cp = pairs[(pairs.dataset == "curvas") & (pairs.task == "pancreas") & (pairs.pair_class == "P1")].copy()
cp["pair"] = cp[["rater_a", "rater_b"]].astype(str).agg("-".join, axis=1)
per_pair = cp.groupby(["pair", "image_id"]).dice.mean().groupby("pair").mean()
put("R1.curvas/pancreas.rater_pair_floors", "; ".join(f"{k} {f3(v)}" for k, v in per_pair.items()),
    "results/layer/pairs.csv", "fixed-panel dependence of the floor")
put("R1.curvas/pancreas.mean_within_image_pair_sd", f3(cp.groupby("image_id").dice.std().mean()),
    "results/layer/pairs.csv", "between-rater dispersion per image (SD of the three pair scores)")


# ---------------------------------------------------------------- paired contrasts
def paired_boot(da, db, rng):
    m = image_means(da).join(image_means(db), lsuffix="_a", rsuffix="_b", how="inner")
    m["d"] = m.v_a - m.v_b
    im = m[["d"]].rename(columns={"d": "v"}).assign(c=m.c_a)
    bt = cluster_boot(im, rng)
    lo, hi = ci(bt)
    return m.d.mean(), lo, hi, len(m), im.c.nunique()


rng_p = np.random.default_rng(SEED)
cr = pairs[pairs.dataset == "cryonuseg"]
d, lo, hi, n, g = paired_boot(cr[(cr.task == "nuclei_intra")], cr[(cr.task == "nuclei") & (cr.pair_class == "P1")], rng_p)
put("S8.cryonuseg.intra-minus-inter", f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}] images={n} clusters={g}",
    "results/layer/pairs.csv", "paired cluster bootstrap")
rg = pairs[(pairs.dataset == "riga") & (pairs.pair_class == "P1")]
d, lo, hi, n, g = paired_boot(rg[rg.task == "disc"], rg[rg.task == "cup"], rng_p)
put("S8.riga.disc-minus-cup", f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}] images={n} clusters={g}", "results/layer/pairs.csv",
    "paired cluster bootstrap")


def rater_boot_floor(sub, rng, b=B):
    raters = sorted(set(sub.rater_a) | set(sub.rater_b))
    idx = {r: i for i, r in enumerate(raters)}
    m = np.full((len(raters), len(raters)), np.nan)
    for r in sub.itertuples():
        i, j = idx[r.rater_a], idx[r.rater_b]
        m[i, j] = m[j, i] = r.dice
    iu = np.triu_indices(len(raters), 1)
    out = np.empty(b)
    for k in range(b):
        draw = rng.integers(0, len(raters), len(raters))
        sm = m[np.ix_(draw, draw)]
        out[k] = np.nanmean(sm[iu])
    return np.nanmean(m[iu]), out


c3p = pairs[pairs.dataset == "c3ro"]
res = []
for t in sorted(c3p.task.unique()):
    e = c3p[(c3p.task == t) & (c3p.pair_class == "P1")]
    ne = c3p[(c3p.task == t) & (c3p.pair_class == "P2")]
    if e.empty or ne.empty:
        continue
    pe, be = rater_boot_floor(e, rng_p)
    pn, bn = rater_boot_floor(ne, rng_p)
    lo, hi = ci(be - bn)
    res.append(dict(task=t, expert=pe, nonexpert=pn, diff=pe - pn, ci_low=lo, ci_high=hi))
cres = pd.DataFrame(res)
cres.to_csv(OUT / "c3ro_expert_minus_nonexpert.csv", index=False)
put("S8.c3ro.expert-minus-nonexpert", f"{len(cres)} structures; positive with CI excluding 0: "
    f"{int((cres.ci_low > 0).sum())}; CI includes 0: {int(((cres.ci_low <= 0) & (cres.ci_high >= 0)).sum())}; "
    f"negative with CI excluding 0: {int((cres.ci_high < 0).sum())}; mean diff {f3(cres['diff'].mean())}",
    "results/layer/pairs.csv", "independent rater bootstraps per tier (single case per structure)")

# ---------------------------------------------------------------- identical masks
for ds in ["curvas", "pdacvi", "multiorg", "cryonuseg", "riga", "ima", "gleason2019", "arvaniti", "liverhccseg"]:
    for (t, pc), sub in pairs[pairs.dataset == ds].groupby(["task", "pair_class"]):
        ne = sub[~sub.one_empty]
        ident = ((ne.dice >= 0.999999) & (ne.hd95 == 0))
        put(f"R10.identical_pairs.{ds}/{t}/{pc}", f"{ident.mean() * 100:.1f}% of {len(ne)} pairs with Dice=1 and HD95=0 "
            f"(both-empty pairs included: {int(sub.both_empty.sum())})", "results/layer/pairs.csv",
            "pair-level identity")

# ---------------------------------------------------------------- QUBIQ tasks and eligibility
q = tf[(tf.dataset == "qubiq21") & (tf.metric == "dice")][["task", "pair_class", "floor_eligible", "floor_reportable",
                                                          "mean", "n_images", "n_raters"]]
for r in q.itertuples():
    put(f"S15.qubiq21/{r.task}/{r.pair_class}", f"eligible={r.floor_eligible} reportable={r.floor_reportable} "
        f"floor={f3(r.mean)} images={r.n_images} raters={r.n_raters}", "results/layer/task_floor.csv")

lines = ["# Values of the additional analyses (generated by code/manuscript/supplementary_analyses.py)", "",
         "| key | value | source | note |", "|---|---|---|---|"]
lines += [f"| {k} | {str(v).replace('|', '/')} | {s} | {n} |" for k, v, s, n in rows]
(OUT / "values_analyses.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT / 'values_analyses.md'} ({len(rows)} entries)")
