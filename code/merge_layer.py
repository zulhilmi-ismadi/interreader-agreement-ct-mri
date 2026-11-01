"""Merge the per-source outputs of `compute_layer.py` into one GTU layer.

`compute_layer.py` writes one directory per run — `results/<source>/{pairs,per_image,per_rater,
task_floor,view_b,view_b_skipped}` — because the sources were computed on different days and the
volumetric ones take hours. This assembles them into `results/layer/`, which is the layer itself: the
tables the paper's agreement values are drawn from and the tables `make_release.py` releases.

Merging is a concatenation, not a recomputation, and that is only sound because the groupby keys of
every table start with `dataset`: no dataset spans two runs, so no group is split across inputs and
no bootstrap needs redoing. `--check` asserts exactly that before writing anything.

**The unit trap this script exists to close.** `task_floor` is long-form (one row per metric) and
merges cleanly. `pairs`, `per_image` and `per_rater` are wide, one column per metric, and the NSD
tolerance grid is set per modality: 1/2/4 **px** for 2D microscopy, histology, cytology, fundus
and dermoscopy; 1/2/3 **mm** for CT and MRI volumes. So `nsd@2` is 2 px in the RIGA rows and 2 mm in
the PDACVI rows of the same merged column, `nsd@4` is 2D-only, `nsd@3` volumetric-only, and `hd95`
carries whichever unit its row's modality implies. Concatenating without saying so would put two
different physical quantities in one column under one name. Every merged wide table therefore gains a
`distance_unit` column (`px` or `mm`), derived from each row's modality, and no distance metric may
be pooled across units without grouping on it.

Usage:
    python code/merge_layer.py                 # discover, check, write results/layer/
    python code/merge_layer.py --check         # validate only, write nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gtu.schema import VOLUMETRIC_MODALITIES, AnnotationType, Modality

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_OUT = RESULTS / "layer"

TABLES = (
    "pairs",
    "per_image",
    "per_rater",
    "task_floor",
    "view_b",
    "view_b_skipped",
    "unpaired",  # images with <2 raters: excluded from the floor and counted, not failures
)

#: wide tables: one column per metric, so they need the unit column (see module docstring)
WIDE_TABLES = ("pairs", "per_image", "per_rater")

#: `results/` itself holds the cryonuseg+mesmer run, which predates the per-source directories.
ROOT_RUN_NAME = "root"

#: never treated as a source directory
NOT_SOURCES = {"checkpoints", "layer"}


def _informative_fraction(pairs):
    """Share of pairs per task that are neither both-empty nor one-empty.

    Both-empty pairs are scored 1 and one-empty pairs 0 by the empty-mask policy, so a task made
    entirely of them reports a floor that no rater's judgement contributed to."""
    import numpy as np

    if pairs.empty or not {"dice", "iou", "hd95"} <= set(pairs.columns):
        return None
    both = (pairs.dice == 1.0) & (pairs.iou == 1.0) & (pairs.hd95 == 0.0)
    one = (pairs.dice == 0.0) & (pairs.hd95.isna())
    frame = pairs.assign(_informative=~(both | one))
    return (
        frame.groupby(["dataset", "task", "pair_class"])["_informative"]
        .mean()
        .reset_index()
        .rename(columns={"_informative": "frac_informative"})
    )


def discover(results: Path, out: Path) -> dict[str, Path]:
    """Input runs, by name. Every directory holding a task_floor.csv, plus `results/` itself."""
    runs: dict[str, Path] = {}
    if (results / "task_floor.csv").is_file():
        runs[ROOT_RUN_NAME] = results
    for path in sorted(p for p in results.iterdir() if p.is_dir()):
        if path.name in NOT_SOURCES or path.resolve() == out.resolve():
            continue
        if (path / "task_floor.csv").is_file():
            runs[path.name] = path
    return runs


def read_table(run: Path, table: str) -> pd.DataFrame:
    """Prefer parquet (dtypes survive the round trip); fall back to csv; empty frame if neither.

    `compute_layer.py` skips the parquet for an empty frame, so a source with no skipped View B
    panels has a 2-byte csv and no parquet at all."""
    parquet, csv = run / f"{table}.parquet", run / f"{table}.csv"
    if parquet.is_file():
        return pd.read_parquet(parquet)
    if csv.is_file():
        try:
            return pd.read_csv(csv)
        except pd.errors.EmptyDataError:
            # a source with nothing to report writes a headerless 2-byte csv and no parquet
            return pd.DataFrame()
    return pd.DataFrame()


def modality_units(floors: pd.DataFrame) -> dict[tuple[str, str], str]:
    """(dataset, task) -> `mm` or `px`, from the modality recorded in task_floor: CT and MRI in mm."""
    units: dict[tuple[str, str], str] = {}
    for row in floors.itertuples():
        modality = Modality(row.modality)
        units[(row.dataset, row.task)] = (
            "mm" if modality in VOLUMETRIC_MODALITIES else "px"
        )
    return units


def add_units(frame: pd.DataFrame, units: dict[tuple[str, str], str]) -> pd.DataFrame:
    """Stamp the distance unit on a wide table. Unknown keys become NaN rather than a guess."""
    if frame.empty:
        return frame
    keys = list(zip(frame["dataset"], frame["task"]))
    frame = frame.copy()
    frame["distance_unit"] = [units.get(key) for key in keys]
    return frame


def check(runs: dict[str, Path]) -> tuple[pd.DataFrame, list[str]]:
    """Validate that the runs can be concatenated, and return the merged task_floor plus problems."""
    problems: list[str] = []
    floors = []
    owner: dict[tuple[str, str, str, str], str] = {}

    for name, run in runs.items():
        frame = read_table(run, "task_floor")
        if frame.empty:
            problems.append(f"{name}: task_floor is empty")
            continue
        frame = frame.copy()
        frame.insert(0, "source_run", name)
        floors.append(frame)
        for row in frame.itertuples():
            key = (row.dataset, row.task, row.pair_class, row.metric)
            if key in owner:
                problems.append(
                    f"duplicate floor {key} in both {owner[key]!r} and {name!r} — "
                    f"the runs overlap and must not be concatenated"
                )
            owner[key] = name

    merged = pd.concat(floors, ignore_index=True) if floors else pd.DataFrame()

    # every dataset appearing in one table must appear in the other; a source that half-wrote its
    # outputs would otherwise merge silently
    for name, run in runs.items():
        floor_datasets = set(read_table(run, "task_floor").get("dataset", []))
        image_datasets = set(read_table(run, "per_image").get("dataset", []))
        missing = floor_datasets ^ image_datasets
        if missing:
            problems.append(f"{name}: datasets in task_floor xor per_image: {sorted(missing)}")

    return merged, problems


def merge(runs: dict[str, Path], out: Path, floors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    units = modality_units(floors)
    tables: dict[str, pd.DataFrame] = {}

    for table in TABLES:
        frames = []
        for name, run in runs.items():
            frame = read_table(run, table)
            if frame.empty:
                continue
            frame = frame.copy()
            frame.insert(0, "source_run", name)
            if table in WIDE_TABLES:
                frame = add_units(frame, units)
            frames.append(frame)
        tables[table] = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )

    # The informative fraction and `floor_reportable`, applied before the tables are written so the
    # columns reach disk. A task in which every pair is both-empty scores 1.000 without any rater's
    # judgement contributing, so it is kept in the layer but marked not reportable.
    informative = _informative_fraction(tables.get("pairs", pd.DataFrame()))
    if informative is not None and not tables["task_floor"].empty:
        floor_table = tables["task_floor"].merge(
            informative, on=["dataset", "task", "pair_class"], how="left"
        )
        floor_table["floor_reportable"] = floor_table.floor_eligible & (
            floor_table.frac_informative.fillna(1) > 0
        )
        tables["task_floor"] = floor_table

    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
        if not frame.empty:
            frame.to_parquet(out / f"{name}.parquet", index=False)
    return tables


def summarise(tables: dict[str, pd.DataFrame]) -> None:
    floors = tables["task_floor"]
    if floors.empty:
        print("nothing merged")
        return

    print("\nsources merged:")
    for name, group in floors.groupby("source_run", sort=True):
        datasets = sorted(group.dataset.unique())
        print(f"  {name}: {', '.join(datasets)}  ({group.task.nunique()} tasks)")

    print("\nlayer coverage, per dataset:")
    header = f"  {'dataset':<14}{'modality':<11}{'tasks':>6}{'images':>8}{'raters':>8}  licence"
    print(header)
    for dataset, group in floors.groupby("dataset", sort=True):
        eligible = group[group.floor_eligible]
        images = int(eligible.n_images.max()) if not eligible.empty else 0
        raters = int(group.n_raters.max())
        print(
            f"  {dataset:<14}{group.modality.iloc[0]:<11}{group.task.nunique():>6}"
            f"{images:>8}{raters:>8}  {group.licence.iloc[0]}"
        )

    # Count the recomputed microscopy/histology/cytology sources with a floor-eligible PIXEL floor,
    # rather than leaving it to be recounted by hand. Box sources are a separate stream, never pooled
    # with pixel floors, and never count: NuCLS is
    # histology, floor-eligible and box-only, and would inflate this number by one if the
    # annotation_type were ignored.
    microscopy = {Modality.MICROSCOPY.value, Modality.HISTOLOGY.value, Modality.CYTOLOGY.value}
    pixel_types = {AnnotationType.PIXEL.value, AnnotationType.INSTANCE.value}
    eligible = floors[floors.floor_eligible & floors.modality.isin(microscopy)]

    if "annotation_type" in eligible:
        typed = eligible[eligible.annotation_type.notna()]
        counted = sorted(set(typed[typed.annotation_type.isin(pixel_types)].dataset))
        boxes = sorted(set(typed[~typed.annotation_type.isin(pixel_types)].dataset))
        untyped = sorted(set(eligible[eligible.annotation_type.isna()].dataset))
    else:
        counted, boxes, untyped = [], [], sorted(set(eligible.dataset))

    print(
        f"\nrecomputed microscopy/histology/cytology PIXEL floors: {len(counted)}  {counted}"
    )
    if boxes:
        print(f"  excluded, box stream: {boxes}")
    if untyped:
        print(
            f"  NOT COUNTED, annotation_type missing from task_floor: {untyped}\n"
            f"    these runs predate the column; recompute them or confirm the type by hand "
            f"before quoting the count"
        )

    # A task where no rater ever marked the structure has a floor of 1.000 under the empty-mask
    # policy (both-empty scores 1), computed on zero informative pairs. That is arithmetic, not
    # agreement, so it is marked not reportable here and excluded from figures and tables. It is kept
    # in the layer rather than deleted, because "the released masks are empty" is itself a fact about
    # the source.
    if "floor_reportable" in floors.columns:
        dropped = floors[floors.floor_eligible & ~floors.floor_reportable]
        if not dropped.empty:
            names = sorted({f"{r.dataset}/{r.task}" for _, r in dropped.iterrows()})
            print(
                f"not reportable, no informative pairs: "
                f"{len(names)}  {names}"
            )

    unstated = sorted(set(floors[floors.licence == "unstated"].dataset))
    if unstated:
        print(
            f"held from release (licence unstated): "
            f"{len(unstated)}  {unstated}"
        )

    # Rows whose licence is unstated are withheld. A licence can also be confirmed and still forbid
    # redistributing derived rows: CC BY-NC-ND's NoDerivatives term plausibly covers per-image
    # agreement statistics. Naming such a licence accurately must not move the source from
    # "withheld" to "releasable", so it is a second withholding category.
    no_derivatives = sorted(
        set(floors[floors.licence.fillna("").str.contains("-ND-", case=False)].dataset)
    )
    if no_derivatives:
        print(
            f"held from release, licence CONFIRMED but forbids derivatives: "
            f"{len(no_derivatives)}  {no_derivatives}"
        )

    # A third category: a signed data use agreement that allows group-level derivatives only, so
    # task-level rows ship and per-image rows do not.
    agreements = sorted(
        set(floors[floors.licence.fillna("").str.contains("-DUA", case=False)].dataset)
    )
    if agreements:
        print(
            f"per-image rows held from release, data use agreement allows group level only"
            f": {len(agreements)}  {agreements}"
        )

    for table in WIDE_TABLES:
        frame = tables[table]
        if not frame.empty and "distance_unit" in frame:
            counts = frame.distance_unit.value_counts(dropna=False).to_dict()
            print(f"{table}: {len(frame)} rows, distance_unit {counts}")

    print("\nrows written:")
    for name, frame in tables.items():
        print(f"  {name}: {len(frame)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--check", action="store_true", help="validate the runs and write nothing"
    )
    args = parser.parse_args()

    results, out = Path(args.results), Path(args.out)
    runs = discover(results, out)
    if not runs:
        raise SystemExit(f"no runs with a task_floor.csv under {results}")
    print(f"{len(runs)} run(s): {', '.join(runs)}")

    floors, problems = check(runs)
    for problem in problems:
        print(f"  PROBLEM  {problem}")
    if problems:
        raise SystemExit("refusing to merge: fix the problems above")
    print("  checks passed: no overlapping (dataset, task, pair_class, metric)")

    if args.check:
        return

    tables = merge(runs, out, floors)
    summarise(tables)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
