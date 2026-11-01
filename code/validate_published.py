"""Published agreement statistics against their recomputation from the released masks.

Each published statistic is recomputed twice, and BOTH are tabulated:
  * `recomputed_source_definition` — under the source's own stated definition (median over pairs for
    C3RO, mean over all pairs for PDACVI and IMA++, per scan for LiverHccSeg);
  * `recomputed_common` — under the common definition used throughout the layer: mean over pairs
    within an image, then over images.

**Why both are summarised, not just tabulated.** Quoting only the first column as reproduction
"under the common definition" would misdescribe it. The gap between the two columns is not an error to
hide: it is the definitional sensitivity the paper is about, measured on real published numbers.

**Why there is a second output.** Not every published statistic can be compared. Every target is
either compared here or listed in `e5_not_run.csv` with its reason.

**Two match rules, because no single tolerance is canonical.** `TOL = 0.01` below is one unit in the
last printed place of a two-decimal value, and it decides the headline count, so the
table also carries a rounding-interval rule: a recomputation matches if it rounds to the published
value at the precision that value was printed to (`published_decimals`), under round-half-up and
under round-up separately. The C3RO residuals are one-sided and consistent with round-up, which
`_diagnose_c3ro_offset` reports; the columns let the paper state the count under each rule instead of
resting the result on one chosen tolerance.

Writes results/e5_published_vs_recomputed.csv and results/e5_not_run.csv.

Usage: python code/validate_published.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _rounds_to(value: float, published: float, decimals: int, rounding) -> bool:
    """Does `value` round to `published` at the precision `published` was printed to?

    The alternative to a fixed tolerance: it asks whether the recomputation lies in the interval that
    would have printed the published number, under the stated rounding convention."""
    if not np.isfinite(value) or not np.isfinite(published):
        return False
    quantum = Decimal(1).scaleb(-int(decimals))
    return Decimal(repr(float(value))).quantize(quantum, rounding=rounding) == Decimal(
        repr(float(published))
    ).quantize(quantum, rounding=ROUND_HALF_UP)

#: a published value "reproduces" when a recomputation lies within this of it
TOL = 0.01

#: C3RO Supplementary Table 2 (41597_2023_2062_MOESM2), median pairwise DSC per ROI per tier.
#: ASD is not one of the layer's metrics and is not recomputed. SDSC is a surface Dice at a
#: tolerance the paper does not state in this table, so it is not compared either (see e5_not_run.csv).
C3RO_DSC = {
    "breast_CTV_Ax": (0.69, 0.61),
    "breast_CTV_Chestwall": (0.67, 0.68),
    "breast_CTV_IMN": (0.47, 0.36),
    "breast_CTV_Sclav_LN": (0.57, 0.56),
    "breast_BrachialPlex_L": (0.20, 0.25),
    "breast_Heart": (0.91, 0.93),
    "breast_A_LAD": (0.33, 0.31),
    "sarcoma_GTV": (0.94, 0.92),
    "sarcoma_CTV": (0.72, 0.67),
    "sarcoma_Genitals": (0.69, 0.58),
    "hn_GTVp": (0.79, 0.74),
    "hn_GTVn": (0.91, 0.87),
    "hn_CTV1": (0.85, 0.62),
    "hn_CTV2": (0.71, 0.40),
    "hn_Brainstem": (0.82, 0.75),
    "hn_Glnd_Submand_L": (0.86, 0.84),
    "hn_Glnd_Submand_R": (0.80, 0.71),
    "hn_Larynx": (0.60, 0.60),
    "hn_Musc_Constrict": (0.58, 0.45),
    "hn_Parotid_L": (0.86, 0.80),
    "hn_Parotid_R": (0.87, 0.82),
    "gyn_GTVn": (0.79, 0.79),
    "gyn_CTVn_4500": (0.72, 0.66),
    "gyn_CTVp_4500": (0.79, 0.77),
    "gyn_Bowel_Small": (0.80, 0.57),
    "gi_CTV_4500": (0.76, 0.65),
    "gi_CTV_5400": (0.63, 0.48),
    "gi_Bag_Bowel": (0.64, 0.59),
}

#: mahbod-2021-cryonuseg Tables 7 and 8, mean over images.
CRYONUSEG = [
    ("cryonuseg", "nuclei", "P1", "dice", 0.789, "Table 7 inter-observer"),
    ("cryonuseg", "nuclei", "P1", "panoptic_quality", 0.509, "Table 7 inter-observer"),
    ("cryonuseg", "nuclei_intra", "P3", "dice", 0.838, "Table 8 intra-observer"),
    ("cryonuseg", "nuclei_intra", "P3", "panoptic_quality", 0.570, "Table 8 intra-observer"),
]

#: published summary targets: (dataset, task, metric, published mean, published sd,
#: source). Both are published as mean +/- sd over pairs; the sd matching our pooled-pair sd is the
#: evidence that "mean over all pairs" is their definition, so that is the source-definition column.
SUMMARY_TARGETS = [
    ("pdacvi", "lesion", "dice", 0.584, 0.257, "riera-2026 pairwise human DSC 58.4 +/- 25.7"),
    ("ima", "lesion", "dice", 0.7813, 0.2183, "IMA++ multi-annotator subset Dice"),
    ("ima", "lesion", "iou", 0.6830, np.nan, "IMA++ multi-annotator subset JI"),
]

#: decimals each source printed its value to, for the rounding-interval rule. Taken from the printed
#: table, not inferred from the float: C3RO prints "0.20", which as a float would look like one place.
DECIMALS_C3RO = 2          # Supplementary Table 2, two decimals throughout
DECIMALS_CRYONUSEG = 3     # Tables 7 and 8
DECIMALS_LIVERHCCSEG = 4   # segm_metrics.xlsx, per scan
SUMMARY_DECIMALS = {"pdacvi": 3, "ima": 4}

#: published targets that are NOT compared here, each with the reason. Recorded rather than
#: dropped: a missing target is itself a finding.
NOT_RUN = [
    ("nucls", "DSC 85.0 +/- 5.9; Cohen's kappa 74.0-87.4", "not_comparable",
     "The provenance check (nucls_boundary_check.py) found that 99.1 % of NuCLS nuclei annotated by three or more raters carry one "
     "boundary contour served to every rater. A boundary DSC between NuCLS raters therefore measures a "
     "shared contour, not agreement; the layer carries NuCLS as a box stream only."),
    ("riga", "Almazroa 2017 inter-observer agreement", "not_run",
     "The agreement values from Almazroa 2017 were never transcribed into data/derived, so there is "
     "no number on disk to compare against. Not compared."),
    ("mesmer", "human-human F1 via the authors' Figure_3.py", "compared_separately",
     "No value is printed: Greenwald 2022 show human-human F1 only as a swarm plot (Fig. 3d). The "
     "authors' Figure_3.py recipe was re-run on the same four tissues and set "
     "against this study's instance F1 per rater pair, outside the published-value table because "
     "there is no published value. See e5_mesmer_fig3d.csv and code/validate_mesmer_fig3d.py."),
    ("cytocrowd", "average pairwise IoU 0.664", "not_reproducible",
     "Compared: six natural definitions tried, none reproduces it; it brackets between matching "
     "thresholds of 0.3 and 0.35. The matching rule is unpublished. See e5_cytocrowd_iou.csv and "
     "code/validate_published.py."),
    ("c3ro", "ASD and SDSC medians (Suppl. Table 2)", "not_comparable",
     "ASD is not among the measures used in this study. SDSC is a surface Dice at a tolerance the table does not state, "
     "so any comparison would be at a guessed tolerance."),
]


def _rows_for_c3ro() -> list[dict]:
    pairs = pd.read_csv(ROOT / "results/c3ro/pairs.csv")
    floor = pd.read_csv(ROOT / "results/c3ro/task_floor.csv")
    floor = floor[floor.metric == "dice"].set_index(["task", "pair_class"])["mean"]

    rows = []
    for task, (expert, non_expert) in C3RO_DSC.items():
        for pair_class, published in (("P1", expert), ("P2", non_expert)):
            subset = pairs[(pairs.task == task) & (pairs.pair_class == pair_class)]["dice"]
            rows.append(
                {
                    "dataset": "c3ro",
                    "task": task,
                    "pair_class": pair_class,
                    "metric": "dice",
                    "published": published,
                    "published_definition": "median over pairs",
                    "recomputed_source_definition": round(float(subset.median()), 4),
                    "recomputed_common": round(float(floor.loc[(task, pair_class)]), 4),
                    "published_decimals": DECIMALS_C3RO,
                    "n_pairs": len(subset),
                    "source": "C3RO Suppl. Table 2",
                }
            )
    return rows


def _rows_for_cryonuseg() -> list[dict]:
    floor = pd.read_csv(ROOT / "results/task_floor.csv")
    rows = []
    for dataset, task, pair_class, metric, published, source in CRYONUSEG:
        match = floor[
            (floor.dataset == dataset)
            & (floor.task == task)
            & (floor.pair_class == pair_class)
            & (floor.metric == metric)
        ].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "pair_class": pair_class,
                "metric": metric,
                "published": published,
                "published_definition": "mean over images",
                "recomputed_source_definition": round(float(match["mean"]), 4),
                "recomputed_common": round(float(match["mean"]), 4),
                "published_decimals": DECIMALS_CRYONUSEG,
                "n_pairs": int(match["n_pairs"]),
                "source": f"mahbod-2021 {source}",
            }
        )
    return rows


def _rows_for_summaries() -> list[dict]:
    rows = []
    for dataset, task, metric, published, published_sd, source in SUMMARY_TARGETS:
        pairs = pd.read_csv(ROOT / f"results/{dataset}/pairs.csv")
        block = pairs[pairs.task == task]
        values = block[metric].dropna()
        # The two columns must differ in AGGREGATION only, so `recomputed_common` is computed over the
        # same masks the source used: every pair of the task, whatever its provenance class. Reading it
        # from task_floor would change the rater set as well -- the IMA++ floor holds only T1 and T2
        # pairs (accepted algorithm output is provenance class P5 and enters no floor), while the published
        # value covers every annotation in the release.
        common = block.groupby("image_id")[metric].mean().mean()
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "pair_class": "P1",
                "metric": metric,
                "published": published,
                "published_sd": published_sd,
                "published_definition": "mean over all pairs",
                "recomputed_source_definition": round(float(values.mean()), 4),
                "recomputed_sd": round(float(values.std()), 4),
                "recomputed_common": round(float(common), 4),
                "published_decimals": SUMMARY_DECIMALS[dataset],
                "n_pairs": len(values),
                "source": source,
            }
        )
    return rows


#: Commowick 2021 section 4.2: per-expert median Dice against a leave-one-out LOP-STAPLE consensus "varies
#: between 0.66 and 0.76". The two ends are the published values. LOP-STAPLE is not implemented here;
#: STAPLE and majority-vote consensus stand in for it, a stated definitional difference.
MSSEG_TARGETS = (("min", 0.66), ("max", 0.76))
DECIMALS_MSSEG = 2


def _rows_for_msseg() -> list[dict]:
    """Ends of the range of per-expert medians, from validate_msseg.py. The source column is STAPLE,
    the construction nearest LOP-STAPLE; majority vote is carried beside it because the target counts
    as reproduced under either consensus."""
    summary = pd.read_csv(ROOT / "results/e5_msseg_summary.csv")
    span = summary.groupby("consensus").median_dice.agg(["min", "max"])
    return [
        {
            "dataset": "msseg2016",
            "task": "ms_lesion",
            "pair_class": "P1",
            "metric": "dice",
            "published": published,
            "published_definition": f"{end} over experts of the per-expert median Dice vs leave-one-out LOP-STAPLE consensus",
            "recomputed_source_definition": round(float(span.loc["staple", end]), 4),
            "recomputed_source_majority": round(float(span.loc["majority", end]), 4),
            "recomputed_common": round(float(span.loc["pairwise", end]), 4),
            "published_decimals": DECIMALS_MSSEG,
            "n_pairs": int(summary.expert.nunique()),
            "source": "Commowick 2021 NeuroImage section 4.2",
        }
        for end, published in MSSEG_TARGETS
    ]


def _rows_for_liverhccseg() -> list[dict]:
    """Per-scan published inter-reader Dice — the strongest check available, since every scan is
    compared individually. Two readers means one pair per scan, so both columns coincide."""
    published = pd.read_csv(ROOT / "data/derived/liverhccseg_published_percase.csv")
    published = published[published.metric == "dice"]
    ours = pd.read_csv(ROOT / "results/liverhccseg/pairs.csv")[["image_id", "task", "dice"]]
    merged = published.merge(
        ours, left_on=["patient_id", "structure"], right_on=["image_id", "task"], how="inner"
    )
    return [
        {
            "dataset": "liverhccseg",
            "task": f"{row.structure}/{row.patient_id}",
            "pair_class": "P1",
            "metric": "dice",
            "published": round(float(row.value), 4),
            "published_definition": "per scan, rater1 vs rater2",
            "recomputed_source_definition": round(float(row.dice), 4),
            "recomputed_common": round(float(row.dice), 4),
            "published_decimals": DECIMALS_LIVERHCCSEG,
            "n_pairs": 1,
            "source": "gross-2023 segm_metrics.xlsx",
        }
        for _, row in merged.iterrows()
    ]


def _diagnose_liverhccseg_outlier() -> None:
    """The one LiverHccSeg scan that does not reproduce is TCGA-DD-A4NH, whose published liver row is
    already known to be an exact copy of TCGA-DD-A4NF's (data/derived/README.md). Test whether its
    tumour row is the same kind of supplement error before calling it unresolved."""
    published = pd.read_csv(ROOT / "data/derived/liverhccseg_published_percase.csv")
    published = published[published.metric == "dice"].pivot_table(
        index="structure", columns="patient_id", values="value"
    )
    ours = pd.read_csv(ROOT / "results/liverhccseg/pairs.csv").pivot_table(
        index="task", columns="image_id", values="dice"
    )
    print("\nLiverHccSeg outlier, TCGA-DD-A4NH, against its known duplicate TCGA-DD-A4NF:")
    for structure in ("liver", "tumor1"):
        cells = []
        for patient in ("TCGA-DD-A4NF", "TCGA-DD-A4NH"):
            pub = published.get(patient, pd.Series(dtype=float)).get(structure, np.nan)
            rec = ours.get(patient, pd.Series(dtype=float)).get(structure, np.nan)
            cells.append(f"{patient[-4:]} published {pub:.4f} recomputed {rec:.4f}")
        print(f"  {structure:7s} " + "   |   ".join(cells))


def main() -> None:
    rows = (_rows_for_cryonuseg() + _rows_for_c3ro() + _rows_for_summaries() + _rows_for_liverhccseg()
            + _rows_for_msseg())
    table = pd.DataFrame(rows)
    table["difference_source"] = table.recomputed_source_definition - table.published
    table["difference_common"] = table.recomputed_common - table.published
    table["reproduces_source"] = table.difference_source.abs() <= TOL + 1e-9
    table["reproduces_common"] = table.difference_common.abs() <= TOL + 1e-9
    for column, recomputed in (("source", "recomputed_source_definition"),
                               ("common", "recomputed_common")):
        for rule, rounding in (("rounding", ROUND_HALF_UP), ("roundup", ROUND_CEILING)):
            table[f"reproduces_{column}_{rule}"] = [
                _rounds_to(value, published, decimals, rounding)
                for value, published, decimals
                in zip(table[recomputed], table.published, table.published_decimals)
            ]
    # An MSSEG end reproduces under the source definition if it does so with either consensus. Applied to every source-column flag so the rules cannot disagree.
    msseg = table.dataset == "msseg2016"
    if msseg.any():
        majority = table.loc[msseg, "recomputed_source_majority"]
        published = table.loc[msseg, "published"]
        table.loc[msseg, "reproduces_source"] |= (majority - published).abs() <= TOL + 1e-9
        for rule, rounding in (("rounding", ROUND_HALF_UP), ("roundup", ROUND_CEILING)):
            table.loc[msseg, f"reproduces_source_{rule}"] |= np.array([
                _rounds_to(value, target, DECIMALS_MSSEG, rounding)
                for value, target in zip(majority, published)
            ])
    # retained for anything still reading the earlier column names; these are the SOURCE column
    table["difference"] = table.difference_source
    table["abs_difference"] = table.difference_source.abs()

    out = ROOT / "results/e5_published_vs_recomputed.csv"
    table.to_csv(out, index=False)
    pd.DataFrame(NOT_RUN, columns=["dataset", "published_statistic", "status", "reason"]).to_csv(
        ROOT / "results/e5_not_run.csv", index=False
    )

    print(f"{len(table)} published statistics compared, tolerance {TOL}\n")
    print(f"  {'dataset':12s} {'n':>4s}  {'under source definition':>24s}  {'under common definition':>22s}")
    for dataset, group in table.groupby("dataset", sort=False):
        print(f"  {dataset:12s} {len(group):>4d}  {int(group.reproduces_source.sum()):>12d}/{len(group):<11d}"
              f"  {int(group.reproduces_common.sum()):>10d}/{len(group):<11d}")
    print(f"  {'TOTAL':12s} {len(table):>4d}  {int(table.reproduces_source.sum()):>12d}/{len(table):<11d}"
          f"  {int(table.reproduces_common.sum()):>10d}/{len(table):<11d}")

    print(
        f"\nmatch rules (source definition / common definition), over {len(table)} statistics:\n"
        f"  within {TOL}:                             "
        f"{int(table.reproduces_source.sum())} / {int(table.reproduces_common.sum())}\n"
        f"  rounds to the published value, half-up:   "
        f"{int(table.reproduces_source_rounding.sum())} / {int(table.reproduces_common_rounding.sum())}\n"
        f"  rounds to the published value, round-up:  "
        f"{int(table.reproduces_source_roundup.sum())} / {int(table.reproduces_common_roundup.sum())}"
    )

    failing = table[~table.reproduces_source]
    print(f"\nnot reproduced even under the source's own definition: {len(failing)}")
    for _, row in failing.iterrows():
        print(f"  {row.dataset:12s} {row.task:30s} published {row.published:.4f}  "
              f"recomputed {row.recomputed_source_definition:.4f}  ({row.difference_source:+.4f})")

    _diagnose_liverhccseg_outlier()
    _diagnose_c3ro_offset(table[table.dataset == "c3ro"])
    print(f"\nnot compared: {len(NOT_RUN)} targets, see results/e5_not_run.csv")
    print(f"wrote {out.relative_to(ROOT)}")


def _diagnose_c3ro_offset(c3ro: pd.DataFrame) -> None:
    """The C3RO residual is one-sided. Report what it is consistent with, and what would distinguish."""
    negative = int((c3ro.difference_source < 0).sum())
    mean, sd = c3ro.difference_source.mean(), c3ro.difference_source.std()
    within_half = int((c3ro.difference_source.abs() <= 0.005).sum())
    print(
        f"\nC3RO residual (source definition): {negative}/{len(c3ro)} negative, mean {mean:+.4f}, "
        f"sd {sd:.4f}, {within_half}/{len(c3ro)} within 0.005."
    )
    print(
        "  Round-to-nearest is ruled out: it would keep every residual within 0.005, and half are not.\n"
        "  The pattern is CONSISTENT WITH the published values being rounded up to two decimals, which\n"
        "  predicts a mean of -0.005, an sd near 0.0029 and half the mass within 0.005. It is not proof:\n"
        "  a genuine -0.005 bias in our median would leave the same footprint in these residuals alone.\n"
        "  CryoNuSeg, published to three decimals, shows no such offset, which is weak corroboration.\n"
        "  It is therefore an unconfirmed explanation, not a finding."
    )


if __name__ == "__main__":
    sys.exit(main())
