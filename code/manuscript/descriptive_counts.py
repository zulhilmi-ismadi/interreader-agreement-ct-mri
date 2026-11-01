"""Descriptive counts reported in the manuscript.

Comparisons against distinct rater pairs; the Dice-versus-NSD ranking among pixel and instance floors; tolerance
rank correlations and the tolerance at which surface agreement reaches 0.95; rater-count sub-sampling summaries and
panel completeness; reproduction counts with one inconsistent LiverHccSeg scan unresolved; RIGA mask-recovery
exclusions; the modality breakdown; Arvaniti clusters; QUBIQ leave-one-out; consensus-comparator coverage; the
CURVAS leaderboard interval.

Reads results/ and results/manuscript/ (from supplementary_analyses.py); writes results/manuscript/:
e5_tolerance_by_source.csv, modality_breakdown.csv and values_counts.md. Stops if a value the manuscript reports
is not reproduced.

    python code/manuscript/descriptive_counts.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
RM = R / "manuscript"
KEY = ["dataset", "task", "pair_class"]
rows = []


def put(key, value, source, note=""):
    rows.append((key, value, source, note))
    print(f"{key}: {value}")


#: Expected values are the manuscript's, with the QUBIQ 2021 training split and MSSEG-2016 in the layer.
def check(label, got, expected):
    """Stop if a value the manuscript reports is not reproduced exactly."""
    if got != expected:
        raise SystemExit(f"CHECK FAILED {label}: got {got!r}, manuscript reports {expected!r}")
    print(f"  check ok: {label} = {got}")


tf = pd.read_csv(R / "layer" / "task_floor.csv")
rep = tf[(tf.metric == "dice") & (tf.floor_reportable == True)]  # noqa: E712
check("reportable Dice floors", len(rep), 88)

# ---------------------------------------------------------------- comparisons vs distinct rater pairs
pairs = pd.read_csv(R / "layer" / "pairs.csv", usecols=KEY + ["image_id", "rater_a", "rater_b"], dtype=str)
p = pairs.merge(rep[KEY].drop_duplicates(), on=KEY)
check("pairwise comparisons in reportable floors", len(p), 112363)
check("images in reportable floors", p[["dataset", "image_id"]].drop_duplicates().shape[0], 3149)
p["lo"] = np.where(p.rater_a < p.rater_b, p.rater_a, p.rater_b)
p["hi"] = np.where(p.rater_a < p.rater_b, p.rater_b, p.rater_a)
distinct = p.drop_duplicates(["dataset", "image_id", "lo", "hi"])
put("A1.distinct_rater_pairs_on_an_image", len(distinct), "results/layer/pairs.csv",
    "unique (dataset, image, unordered rater pair) over reportable floors; comparisons count each task separately")

# ---------------------------------------------------------------- Dice-versus-NSD ranking (pixel and instance floors)
typ = tf[(tf.metric == "dice") & (tf.pair_class == "P1")][["dataset", "task", "annotation_type"]].drop_duplicates()
r3 = pd.read_csv(R / "r3_metric_reorder.csv").merge(typ, on=["dataset", "task"], how="left")
check("box rows in the ranking", int((r3.annotation_type == "box").sum()), 0)
check("2D grid size", int((r3.grid == "2D (px)").sum()), 13)
check("volumetric grid size", int((r3.grid == "volumetric (mm)").sum()), 43)
two = r3[r3.grid == "2D (px)"].sort_values("rank_overlap")
put("A2.2D_ranks", two[["dataset", "task", "rank_overlap", "rank_boundary"]].apply(
    lambda r: f"{r.dataset}/{r.task} {r.rank_overlap}->{r.rank_boundary}", axis=1).tolist(),
    "results/r3_metric_reorder.csv", "Dice rank -> NSD@2 rank among 2D expert pixel/instance floors")
vol = r3[r3.grid == "volumetric (mm)"]
lv = vol[(vol.dataset == "liverhccseg") & (vol.task == "liver")].iloc[0]
put("A2.liverhccseg_liver_vol_rank", f"{lv.rank_overlap}->{lv.rank_boundary} of {len(vol)}", "results/r3_metric_reorder.csv")

s2 = pd.read_csv(R / "s2_tolerance.csv").merge(
    tf[tf.metric == "dice"][KEY + ["annotation_type"]].drop_duplicates(), on=KEY, how="left")
v2, t2 = s2[s2["nsd@3"].notna()], s2[s2["nsd@3"].isna()]
check("tolerance tau, volumetric group", round(kendalltau(v2["nsd@1"], v2["nsd@3"])[0], 3), 0.757)
t2p = t2[t2.annotation_type != "box"]
put("A2.tolerance_tau_2D", f"{kendalltau(t2p['nsd@1'], t2p['nsd@4'])[0]:.3f} n={len(t2p)}", "results/s2_tolerance.csv",
    "pixel and instance floors")
t95 = pd.read_csv(R / "tau95.csv")
reached = t95[t95.tau95.notna()]
put("A2.tau95_reached", f"{len(reached)} of {len(t95)} floors; of {int((t95.annotation_type != 'box').sum())} "
    f"pixel/instance floors: {int((reached.annotation_type != 'box').sum())}", "results/tau95.csv")

# ---------------------------------------------------------------- rater-count sub-sampling
s8 = pd.read_csv(R / "s8_rater_subsampling.csv")
for ds in ["riga", "c3ro", "ima", "nucls"]:
    sub = s8[s8.dataset == ds]
    for pc, g in sub.groupby("pair_class"):
        k2 = g[g.k == 2]
        full = g[g.k == g.n_raters_panel]
        put(f"A3.{ds}/{pc}", f"k=2 floor {k2.floor_mean.mean():.3f} on {k2.mean_images_floor.mean():.1f} images; "
            f"full panel {full.floor_mean.mean():.3f} on {full.mean_images_floor.mean():.1f} images; "
            f"tasks {g.task.nunique()}; panel {int(g.n_raters_panel.max())}; "
            f"k=2 central 95% {k2.floor_p2_5.mean():.3f}-{k2.floor_p97_5.mean():.3f}",
            "results/s8_rater_subsampling.csv", "means over tasks")
completeness = tf[(tf.dataset.isin(["riga", "c3ro", "ima", "nucls"])) & (tf.metric.isin(["dice", "instance_f1@0.5"]))
                  & tf.floor_reportable][KEY + ["n_raters", "n_images", "n_pairs"]]
completeness["complete_pairs"] = completeness.n_images * completeness.n_raters * (completeness.n_raters - 1) / 2
completeness["share_of_complete"] = (completeness.n_pairs / completeness.complete_pairs).round(3)
put("A3.panel_completeness", completeness.groupby(["dataset", "pair_class"]).share_of_complete.mean().round(3).to_dict(),
    "results/layer/task_floor.csv", "observed pairs / pairs if every rater annotated every image")

# ---------------------------------------------------------------- reproduction with the inconsistent scan unresolved
e5 = pd.read_csv(R / "e5_published_vs_recomputed.csv")
check("published statistics", len(e5), 98)
e5["unresolved"] = e5.task.str.contains("TCGA-DD-A4NH")  # published liver row duplicates another patient's
check("unresolved rows", int(e5.unresolved.sum()), 2)
out = []
for tol in (0.005, 0.01, 0.02):
    own_ok = (e5.recomputed_source_definition - e5.published).abs() <= tol + 1e-9
    common_ok = (e5.recomputed_common - e5.published).abs() <= tol + 1e-9
    scopes = [("all", e5.index == e5.index), ("excluding C3RO", e5.dataset != "c3ro")]
    scopes += [(d, e5.dataset == d) for d in sorted(e5.dataset.unique())]
    for name, m in scopes:
        t = m & ~e5.unresolved
        out.append(dict(tolerance=tol, scope=name, n=int(m.sum()), unresolved=int((m & e5.unresolved).sum()),
                        tested=int(t.sum()), own=int((own_ok & t).sum()), common=int((common_ok & t).sum())))
tab = pd.DataFrame(out)
tab.to_csv(RM / "e5_tolerance_by_source.csv", index=False)
at = tab[(tab.tolerance == 0.01)]
a = at[at.scope == "all"].iloc[0]
check("reproduced under own definitions at 0.01", int(a.own), 93)
check("matched under the common definition at 0.01", int(a.common), 52)
put("A4.at_0.01", f"tested {a.tested} of {a.n} (unresolved {a.unresolved}); own {a.own}; common {a.common}",
    "results/e5_published_vs_recomputed.csv")
c3 = at[at.scope == "c3ro"].iloc[0]
put("A4.common_misses_from_c3ro", f"{c3.tested - c3.common} of {a.tested - a.common}",
    "results/e5_published_vs_recomputed.csv")
x = at[at.scope == "excluding C3RO"].iloc[0]
put("A4.excluding_c3ro_at_0.01", f"own {x.own}/{x.tested}; common {x.common}/{x.tested}",
    "results/e5_published_vs_recomputed.csv")
for tol in (0.005, 0.02):
    b = tab[(tab.tolerance == tol) & (tab.scope == "all")].iloc[0]
    cc = tab[(tab.tolerance == tol) & (tab.scope == "c3ro")].iloc[0]
    put(f"A4.at_{tol}", f"own {b.own}/{b.tested}; common {b.common}/{b.tested}; C3RO own {cc.own}/{cc.tested}",
        "results/e5_published_vs_recomputed.csv")
put("A4.statistics_by_source", e5.dataset.value_counts().to_dict(), "results/e5_published_vs_recomputed.csv")

# ---------------------------------------------------------------- RIGA mask-recovery exclusions
rk = pd.read_csv(R / "riga_prepare_skipped.csv")
contours = rk[rk.rater.notna()]
put("A5.riga_skipped", f"{len(contours)} rater-image annotations on {contours[['subset', 'image']].drop_duplicates().shape[0]}"
    f" images; {int(rk.rater.isna().sum())} image with no unmarked copy", "results/riga_prepare_skipped.csv")

# ---------------------------------------------------------------- modality breakdown
mod = rep[KEY + ["modality", "annotation_type"]]
pm = p.merge(mod, on=KEY)
brk = pd.DataFrame({
    "floors": mod.groupby("modality").size(),
    "images": pm.drop_duplicates(["dataset", "image_id"]).groupby("modality").size(),
    "comparisons": pm.groupby("modality").size(),
    "sources": mod.groupby("modality").dataset.nunique(),
}).fillna(0).astype(int).sort_values("images", ascending=False)
brk.to_csv(RM / "modality_breakdown.csv")
put("A6.modality_breakdown", brk.to_dict("index"), "results/layer/pairs.csv + task_floor.csv")

# ---------------------------------------------------------------- Arvaniti clusters behind each empty-pair view
s4 = pd.read_csv(RM / "s4_prostate_variants.csv")
put("A7.arvaniti_clusters", s4[s4.dataset == "arvaniti"][["task", "variant", "n_clusters", "n_pairs"]].values.tolist(),
    "results/manuscript/s4_prostate_variants.csv")

# ---------------------------------------------------------------- QUBIQ brain-tumour leave-one-out
s5 = pd.read_csv(R / "s5_drop_one_rater.csv")
qb = s5[(s5.dataset == "qubiq21") & s5.task.str.startswith("brain-tumor") & (s5.metric == "dice")]
put("A8.qubiq_brain_drop_one", qb.groupby("task").apply(
    lambda g: f"all {g.floor_all_raters.iloc[0]:.3f}; without each {', '.join(f'{v:.3f}' for v in g.floor_without)}; "
              f"max |change| {g.delta.abs().max():.3f}", include_groups=False).to_dict(), "results/s5_drop_one_rater.csv")

# ---------------------------------------------------------------- consensus comparator coverage
s1 = pd.read_csv(R / "s1_e4_view_a_vs_b.csv")
check("floors with three or more raters", len(rep[rep.n_raters >= 3]), 75)
check("floors with a consensus comparator", int(s1.view_b_staple_dice.notna().sum()), 47)
put("A9.floors_without_comparator", s1[s1.view_b_staple_dice.isna()].groupby(["dataset", "pair_class"]).size().to_dict(),
    "results/s1_e4_view_a_vs_b.csv")

# ---------------------------------------------------------------- CURVAS leaderboard interval
cs = pd.read_csv(R / "wp_l_curvas_summary.csv").iloc[0]
put("A10.curvas_G_interval", f"{cs.e2_adjacent_inside:.3f} [{cs.e2_ci_low:.3f}, {cs.e2_ci_high:.3f}] from "
    f"{int(cs.n_adjacent_gaps)} gaps", "results/wp_l_curvas_summary.csv")

pix = rep[rep.annotation_type != "box"]
put("A11.pixel_instance_floors", f"{len(pix)} floors from {pix.dataset.nunique()} sources; box floors "
    f"{int((rep.annotation_type == 'box').sum())}", "results/layer/task_floor.csv")

with (RM / "values_counts.md").open("w", encoding="utf-8") as f:
    f.write("# Descriptive counts (generated by code/manuscript/descriptive_counts.py)\n\n")
    f.write("| key | value | source | note |\n|---|---|---|---|\n")
    for k, v, s, n in rows:
        f.write(f"| {k} | {str(v).replace('|', '/')} | {s} | {n} |\n")
print(f"\n{len(rows)} entries -> {RM / 'values_counts.md'}")
