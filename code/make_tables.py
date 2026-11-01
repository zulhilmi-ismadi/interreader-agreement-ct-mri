"""Summary tables of the layer: the benchmarks examined against the selection criteria, the agreement
per source, the two leaderboard comparisons and the table of recomputed sources.

Every number is regenerated from `results/` by this script, nothing hand-entered. Run after
`census.py` and `merge_layer.py`:

    python code/make_tables.py

Writes, for each table, a CSV to reuse and a Markdown rendering:

  results/ed1_census.{csv,md}          every benchmark examined, and why it did or did not qualify
  results/table1_layer.{csv,md}        one row per source, on the headline metric
  results/ed3_layer_full.csv           every (dataset, task, pair_class, metric)
  results/table2_leaderboard.{csv,md}  the two leaderboard comparisons
  results/table_sources.{csv,md}       one row per recomputed source (read by the dataset flowchart)

`ed1_census` carries what a flowchart cannot: rater and image counts, provenance class, licence,
and the one-line reason a benchmark failed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LAYER = RESULTS / "layer"

#: the compact table reports the floor on these; the full table carries every agreement metric
MAIN_METRICS = ["dice", "iou", "nsd@2", "hd95"]
HEADLINE = "dice"

STATUS_LABEL = {
    "recomputed": "recomputed floor",
    "published_band": "published band only",
    "pending_access": "access granted, not ingested",
    "census_only": "cannot support a floor",
}


def markdown(frame: pd.DataFrame) -> str:
    """A GitHub-flavoured table. Written out rather than imported so the script has no dependency
    on `tabulate` (the environment is pinned in requirements.txt; do not add to it for a formatting
    nicety)."""
    columns = list(frame.columns)
    widths = [
        max(len(str(c)), *(len(str(v)) for v in frame[c].astype(str))) if len(frame) else len(str(c))
        for c in columns
    ]
    header = "| " + " | ".join(str(c).ljust(w) for c, w in zip(columns, widths)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    lines = [header, rule]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |")
    return "\n".join(lines)


def _fmt(value: float, places: int = 3) -> str:
    return "" if pd.isna(value) else f"{value:.{places}f}"


def table1() -> pd.DataFrame:
    census = pd.read_csv(RESULTS / "census.csv")
    rows = []
    for _, row in census.iterrows():
        if row.status == "recomputed":
            outcome = "floor" if row.c3_provenance_ok else "census only"
            if row.c3_provenance_ok and not row.c4_pixel_floor:
                outcome = "floor (box stream)"
            if row.c3_provenance_ok and not row.c5_releasable:
                outcome += ", withheld"
        else:
            outcome = STATUS_LABEL.get(row.status, row.status)
        rows.append(
            {
                "benchmark": row.display_name,
                "modality": row.modality,
                "raters": "" if pd.isna(row.n_raters) else int(row.n_raters),
                "images": "" if pd.isna(row.n_images) else int(row.n_images),
                "annotation": row.annotation_type if pd.notna(row.annotation_type) else "",
                "provenance": row.provenance if pd.notna(row.provenance) else "",
                "outcome": outcome,
                "licence": row.licence if pd.notna(row.licence) and row.licence else "not stated",
                "note": (row.reason[:88] + "…") if isinstance(row.reason, str) and len(row.reason) > 88
                else (row.reason if isinstance(row.reason, str) else ""),
            }
        )
    return pd.DataFrame(rows)


def table2() -> tuple[pd.DataFrame, pd.DataFrame]:
    floors = pd.read_csv(LAYER / "task_floor.csv")
    # Report only floors with at least one informative pair.
    column = "floor_reportable" if "floor_reportable" in floors else "floor_eligible"
    full = floors[floors[column]].copy()

    # A floor computed entirely from both-empty pairs is arithmetic, not agreement: a both-empty pair
    # scores 1.0, so a task where no rater ever marked the structure reports a perfect
    # 1.000. QUBIQ's pancreas_task02 is exactly that. Carry the informative fraction beside every
    # floor so the reader can see it, rather than only in the sensitivity tables.
    # `frac_informative` arrives on the layer itself (merge_layer). Merging it in again from S4 would
    # produce frac_informative_x / _y and silently drop the flag, so it is not.

    # --- the full table: every task x metric, with its interval and provenance ------------------
    keep = [
        "dataset", "task", "pair_class", "metric", "modality", "annotation_type", "licence",
        "mean", "median", "ci_low", "ci_high", "ci_source", "rss",
        "n_raters", "n_images", "n_pairs", "n_scored", "n_excluded_nan", "frac_informative",
    ]
    full_table = full[[c for c in keep if c in full.columns]].sort_values(
        ["modality", "dataset", "task", "metric"]
    )

    # --- the compact table: one row per source, on the headline metric -------------------------
    # `n_images` in task_floor is per task, and C3RO's tasks are one image each — so taking its max
    # reports "1" for a source covering 28 ROIs. Count distinct images from per_image instead.
    # Count per ROW, not per dataset: QUBIQ spans two modalities (one row each, so a dataset-wide
    # count double-reports it) and NuCLS has tasks outside the floor whose images must not be added.
    per_image = pd.read_csv(LAYER / "per_image.csv")

    def images_for(dataset: str, tasks: set[str], pair_classes: set[str] | None = None) -> int:
        # Count only the images behind the rows this table reports. Without the pair-class filter a
        # source whose images carry raters of more than one class reports every image it has: the
        # IMA++ P5 and mixed rows would make an 800-image floor read as 2,250 images, beside a
        # resampling-unit count of 800 in the same row.
        block = per_image[(per_image.dataset == dataset) & (per_image.task.isin(tasks))]
        if pair_classes is not None:
            block = block[block.pair_class.isin(pair_classes)]
        return int(block.image_id.nunique())

    # P1 only: expert and non-expert tiers are never merged, and a per-source median across both is
    # exactly that merge. Box sources report box F1 at 0.5 IoU, the box stream's own metric, so no box
    # row is ever read as though it were a pixel Dice.
    p1 = full[full.pair_class == "P1"]
    head = pd.concat([
        p1[(p1.annotation_type != "box") & (p1.metric == HEADLINE)],
        p1[(p1.annotation_type == "box") & (p1.metric == "instance_f1@0.5")],
    ])
    rows = []
    for (dataset, modality), block in head.groupby(["dataset", "modality"]):
        floor_min, floor_max = block["mean"].min(), block["mean"].max()
        widest = block.loc[block["mean"].idxmin()]
        rows.append(
            {
                "source": dataset,
                "modality": modality,
                "annotation": "/".join(sorted(block.annotation_type.unique())),
                "raters": int(block.n_raters.max()),
                "images": images_for(dataset, set(block.task), set(block.pair_class)),
                "tasks": block.task.nunique(),
                "metric": "box F1@0.5" if (block.annotation_type == "box").all() else "Dice",
                "floor (median task)": _fmt(block["mean"].median()),
                "range across tasks": f"{_fmt(floor_min)} to {_fmt(floor_max)}",
                "lowest task": widget_name(widest.task),
                "95% CI of lowest": f"{_fmt(widest.ci_low)} to {_fmt(widest.ci_high)}",
                # the number of rater pairs behind every floor quoted, so a
                # one-case or single-pair floor cannot pass for a precise one
                "pairs (lowest task)": int(widest.n_pairs),
                "pairs (all tasks)": int(block.n_pairs.sum()),
                "min informative fraction": _fmt(block["frac_informative"].min(), 2)
                if "frac_informative" in block else "",
                "single-pair tasks": int((block.n_pairs <= 1).sum()),
                "licence": widest.licence,
            }
        )
    main = pd.DataFrame(rows).sort_values(["modality", "source"])
    return main, full_table


def widget_name(task: str) -> str:
    """Full task name, readable: the table wraps long names rather than truncating them."""
    return task.replace("_", " ")


#: publication names for the sources table
DISPLAY = {
    "c3ro": "C3RO", "nucls": "NuCLS", "ima": "IMA++", "qubiq21": "QUBIQ 2021",
    "gleason2019": "Gleason 2019", "pdacvi": "PDACVI", "riga": "RIGA",
    # not "Mesmer": that is the model in Greenwald et al. This row is their separate
    # interannotator experiment (code/loaders/mesmer.py: the "TissueNet side study"), and the
    # model's own prediction is excluded from it.
    "mesmer": "TissueNet (human agreement)",
    "cytocrowd": "CytoCrowd", "lidc": "LIDC-IDRI", "multiorg": "MultiOrg", "curvas": "CURVAS",
    "arvaniti": "Arvaniti", "cryonuseg": "CryoNuSeg", "liverhccseg": "LiverHccSeg", "stare": "STARE",
    "chasedb1": "CHASE_DB1", "msseg2016": "MSSEG-2016",
}
MODALITY = {"ct": "CT", "mri": "MRI", "ct/mri": "CT, MRI", "fundus": "Fundus", "dermoscopy": "Dermoscopy",
            "histology": "Histology", "cytology": "Cytology", "microscopy": "Microscopy"}


def _licence(value) -> str:
    if not isinstance(value, str) or not value or value == "unstated":
        return "Not stated"
    if "OFSEP" in value:  # a signed data use agreement, not a public licence
        return "OFSEP data access agreement"
    return (value.replace("-", " ").replace("CC BY NC ND", "CC BY-NC-ND").replace("CC BY NC SA", "CC BY-NC-SA")
            .replace("CC BY NC", "CC BY-NC"))


def _criteria(provenance: str, annotation: str, releasable: bool) -> str:
    """Which of the manuscript's five selection criteria this source meets.

    Every source in the layer meets (i) and (ii) by construction: it would not be recomputable
    otherwise. (iii) fails where no rater class defines a floor, (iv) sends boxes to their own
    stream, and (v) decides whether the per-image rows can be redistributed.
    """
    independent = any(p in provenance for p in ("P1", "P2"))
    if not independent:
        return f"i, ii, iv, v; fails iii ({provenance})"
    notes = []
    if annotation == "box":
        notes.append("iv box stream")
    if not releasable:
        notes.append("v task level only")
    if not notes:
        return "i-v"
    met = "i-iii" if annotation == "box" else "i-iv"
    return f"{met}; " + ", ".join(notes)


def seeded_sources() -> set[str]:
    """Datasets whose raters corrected an algorithmic pre-segmentation, from the layer's own record.

    `results/s6_seeded.csv` is where the flag lives; the tables and figures read the same set. PDACVI is
    deliberately absent: its documentation reports a pre-segmentation for one of five raters, the release
    does not say which, so none is flagged.
    """
    s6 = pd.read_csv(RESULTS / "s6_seeded.csv")
    return set(s6.loc[s6.seeded.astype(bool), "dataset"])


def table_sources() -> pd.DataFrame:
    """The sources table: one row per recomputed source, counted from the layer itself.

    `census.csv` carries per-task counts that understate multi-task sources (C3RO reads as one
    image), so rater identifiers, images and tasks are counted here from `pairs` and `per_image`.
    Provenance, licence and release status come from `census.csv`, which is their record of origin.
    """
    pairs = pd.read_parquet(LAYER / "pairs.parquet")
    per_image = pd.read_parquet(LAYER / "per_image.parquet")
    census = pd.read_csv(RESULTS / "census.csv").set_index("source")
    seeded = seeded_sources()
    rows = []
    # Intra-observer repeats (class P3) are a second annotation by the same person under a new
    # identifier (MultiOrg A_t0/A_t1; CryoNuSeg annotator1_round1/round2). Counting them as raters
    # would overstate the readers, so raters, tasks and per-image counts come from inter-rater pairs.
    inter = pairs[pairs.pair_class != "P3"]
    for dataset, block in inter.groupby("dataset"):
        images = per_image[(per_image.dataset == dataset) & per_image.task.isin(set(block.task))]
        entry = census.loc[dataset]
        per_image_raters = (
            pd.concat([block[["image_id", "task", "rater_a"]].rename(columns={"rater_a": "r"}),
                       block[["image_id", "task", "rater_b"]].rename(columns={"rater_b": "r"})])
            .groupby(["image_id", "task"]).r.nunique()
        )
        rows.append({
            "Source": DISPLAY.get(dataset, dataset),
            "Modality": MODALITY.get(entry.modality, entry.modality),
            "Annotation": entry.annotation_type,
            "Raters": len(set(block.rater_a) | set(block.rater_b)),
            "Most raters on one image": int(per_image_raters.max()),
            "Images": int(images.image_id.nunique()),
            "Tasks": int(block.task.nunique()),
            # the provenance class, and whether the raters started from an algorithm's output, which
            # the figures mark with a dagger.
            "Provenance": (entry.provenance.replace("/mixed", "")
                           + (", seeded" if dataset in seeded else "")),
            "Licence": _licence(entry.licence),
            "Rows released": "Yes" if bool(entry.releasable) else "No",
            "Criteria": _criteria(entry.provenance.replace("/mixed", ""),
                                  entry.annotation_type, bool(entry.releasable)),
        })
    frame = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(["CT", "CT, MRI", "MRI", "Histology", "Microscopy", "Cytology",
                                         "Dermoscopy", "Fundus"])}
    return frame.sort_values(["Modality", "Source"], key=lambda s: s.map(order) if s.name == "Modality" else s)


def table3() -> pd.DataFrame:
    """The two leaderboard comparisons side by side. One table.

    The two are deliberately of opposite kinds — CTC's band is transcribed from published annotator
    values because its masks were never released, CURVAS's is recomputed from masks. Reporting them
    together is the point: the layer works either way, and they agree.
    """
    ctc = pd.read_csv(RESULTS / "wp_l_ctc_summary.csv")
    curvas = pd.read_csv(RESULTS / "wp_l_curvas_summary.csv").iloc[0]
    ctc_per_dataset = pd.read_csv(RESULTS / "wp_l_ctc_per_dataset.csv")
    null = pd.read_csv(RESULTS / "wp_l_null_summary.csv").set_index(["leaderboard", "measure"])

    def pct(value: float, low: float, high: float) -> str:
        return f"{value:.1%}  [{low:.1%}, {high:.1%}]"

    rows = []
    for _, row in ctc.sort_values("measure", ascending=False).iterrows():
        block = ctc_per_dataset[ctc_per_dataset.measure == row.measure]
        rows.append(
            {
                "Leaderboard": f"Cell Tracking Challenge, {row.measure}",
                "Spread from": "Published per-annotator values",
                "Units": f"{row.n_datasets} datasets",
                "Methods": f"Up to {int(block.n_methods.max())}",
                "Adjacent gaps": int(block.n_adjacent_gaps.sum()),
                "Mean spread": _fmt(row.mean_rss, 4),
                "Adjacent gaps inside spread": pct(row.e2_pooled, row.e2_ci_low, row.e2_ci_high),
                "All method pairs inside spread": pct(row.e6_pooled, row.e6_ci_low, row.e6_ci_high),
                # without this column the share cannot be read at all: it rises with method count
                "Expected from density alone": f"{null.loc[('CTC-CSB', row.measure), 'e2_density_null']:.1%}",
                "Spread in median gaps": f"{null.loc[('CTC-CSB', row.measure), 'median_band_over_median_gap']:.1f}",
                "Resampling unit": "Dataset",
                "Principal caveat": "Annotators scored against a majority gold standard voted from the "
                                    "same annotators, so the spread is optimistic",
            }
        )
    rows.append(
        {
            "Leaderboard": "CURVAS, Dice",
            "Spread from": "Recomputed from masks",
            "Units": f"{int(curvas.n_annotated_cases)} cases",
            "Methods": f"{int(curvas.n_teams)} teams",
            "Adjacent gaps": int(curvas.n_adjacent_gaps),
            "Mean spread": _fmt(curvas.rss_band, 4),
            "Adjacent gaps inside spread": pct(
                curvas.e2_adjacent_inside, curvas.e2_ci_low, curvas.e2_ci_high
            ),
            "All method pairs inside spread": pct(
                curvas.e6_all_pairs_inside, curvas.e6_ci_low, curvas.e6_ci_high
            ),
            "Expected from density alone": f"{null.loc[('CURVAS', 'DSC'), 'e2_density_null']:.1%}",
            "Spread in median gaps": f"{null.loc[('CURVAS', 'DSC'), 'median_band_over_median_gap']:.1f}",
            "Resampling unit": "Image (spread only)",
            "Principal caveat": "Spread computed on the 90 public cases, which include the 60 the "
                                "leaderboard scores, averaged over organs to match the leaderboard",
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    one = table1()
    one.to_csv(RESULTS / "ed1_census.csv", index=False)
    (RESULTS / "ed1_census.md").write_text(markdown(one), encoding="utf-8")

    main_table, full_table = table2()
    main_table.to_csv(RESULTS / "table1_layer.csv", index=False)
    (RESULTS / "table1_layer.md").write_text(markdown(main_table), encoding="utf-8")
    full_table.to_csv(RESULTS / "ed3_layer_full.csv", index=False)

    three = table3()
    three.to_csv(RESULTS / "table2_leaderboard.csv", index=False)
    (RESULTS / "table2_leaderboard.md").write_text(markdown(three), encoding="utf-8")

    sources = table_sources()
    sources.to_csv(RESULTS / "table_sources.csv", index=False)
    (RESULTS / "table_sources.md").write_text(markdown(sources), encoding="utf-8")
    print(f"Sources table: {len(sources)} sources")

    print(f"ed1_census — benchmarks examined: {len(one)}")
    print(f"  outcomes: " + ", ".join(f"{k} {v}" for k, v in one.outcome.value_counts().items()))
    print(f"\ntable1_layer — agreement: {len(main_table)} sources in the compact table, "
          f"{len(full_table)} rows in the full table")
    print(f"  metrics per task: {sorted(full_table.metric.unique())}")
    print(f"\ntable2_leaderboard — leaderboard pairings: {len(three)} rows")
    for _, row in three.iterrows():
        print(f"  {row['Leaderboard']:44s} adjacent gaps inside {row['Adjacent gaps inside spread']}")

    print("\nwrote results/ed1_census.{csv,md}, results/table1_layer.{csv,md}, "
          "results/ed3_layer_full.csv, results/table2_leaderboard.{csv,md}")


if __name__ == "__main__":
    main()
