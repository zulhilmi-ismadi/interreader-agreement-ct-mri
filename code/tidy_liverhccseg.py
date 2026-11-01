"""Tidy LiverHccSeg's published inter-reader metrics into data/derived/ (input to the reproduction of
published agreement values).

gross-2023-liverhccseg ships `segm_metrics.xlsx` with three sheets:

  Liver  17 scans x 12 metrics - inter-reader agreement for the liver contour, PER SCAN
  HCC    16 rows  x 12 metrics - same for each tumour; TCGA-BC-A10Y carries tumour1/2/3
  Stats  published mean, SD, median and quartiles over each of the two sheets

Per-case published agreement is rare: every other reproduction target publishes only a
summary (PDACVI 58.4 +/- 25.7, NuCLS 85.0 +/- 5.9, CryoNuSeg's Tables 7-8). Here the recompute can
be checked scan by scan, which is a far stronger reproduction claim, so the per-case table is kept
as the primary artefact and the Stats sheet is kept as the summary to reproduce.

Two things found while tidying, both of which travel with the numbers (see the printed report):

1. **TCGA-DD-A4NF and TCGA-DD-A4NH carry identical liver rows** - all 12 metrics agree to every
   published decimal. Two different patients scanned 16 months apart cannot agree to 4 dp on
   Hausdorff distance; one row is duplicated in their supplement. Which of the two is the real
   measurement is not recoverable from the file, so both are flagged and neither is silently used.
   Our own recompute will say which, and that is a reproduction result rather than a reproduction failure.

2. **HCC TCGA-DD-A4NH tumour1 has Dice 0.281** against a mean of 0.85 - recall 0.988 with precision
   0.164, i.e. one reader outlined something roughly six times larger than the other. It is the sole
   reason the published HCC SD (0.16) is an order above the liver SD (0.01). Keep it; it is a real
   disagreement, and dropping it would be the kind of quiet filtering the layer never applies: no pair is dropped for its
   score.

Usage: python code/tidy_liverhccseg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/LiverHccSeg/segm_metrics.xlsx"
DERIVED = ROOT / "data/derived"

#: Only these are metrics of the agreement layer; the rest are carried through but marked, so a later
#: join cannot accidentally compare our Dice with their false-positive rate.
LAYER_METRICS = {"dice", "jaccard", "hd95"}

#: `jaccard` is Metrics Reloaded's IoU; the rest of the names already match ours.
METRIC_ALIAS = {"jaccard": "iou"}


def _scan_id(raw: str) -> tuple[str, str]:
    """`nifti_and_segms/TCGA-BC-4073/02-21-2000/` -> ('TCGA-BC-4073', '02-21-2000').

    The patient id is the release key (the original dataset's identifier, verbatim);
    the study date distinguishes scans if a patient ever gets a second one (none does in this release)."""
    parts = [p for p in raw.strip().strip("/").split("/") if p]
    if len(parts) != 3 or parts[0] != "nifti_and_segms":
        raise ValueError(f"unexpected LiverHccSeg scan path: {raw!r}")
    return parts[1], parts[2]


def _long_rows(sheet, structure_column: bool) -> list[dict]:
    rows_iter = sheet.iter_rows(values_only=True)
    header = [h for h in next(rows_iter) if h is not None]
    metrics = header[2:] if structure_column else header[1:]

    out = []
    for row in rows_iter:
        if row is None or row[0] is None:
            continue
        patient, study_date = _scan_id(row[0])
        structure = row[1] if structure_column else "liver"
        values = row[2:] if structure_column else row[1:]
        for metric, value in zip(metrics, values):
            if value is None:
                continue
            out.append(
                {
                    "patient_id": patient,
                    "study_date": study_date,
                    "structure": structure,
                    "metric": METRIC_ALIAS.get(metric, metric),
                    "value": float(value),
                    "is_layer_metric": metric in LAYER_METRICS,
                }
            )
    return out


def _summary_rows(sheet) -> list[dict]:
    """The Stats sheet stacks two blocks, each headed by its structure name."""
    out: list[dict] = []
    structure = None
    metrics: list[str] = []
    for row in sheet.iter_rows(values_only=True):
        if row is None or all(v is None for v in row):
            continue
        first, rest = row[0], row[1:]
        if first in ("Liver", "HCC"):
            structure = first.lower()
            continue
        if first is None:
            metrics = [m for m in rest if m is not None]
            continue
        statistic = {0.25: "q25", 0.75: "q75"}.get(first, first)
        for metric, value in zip(metrics, rest):
            if value is None:
                continue
            out.append(
                {
                    "structure": structure,
                    "statistic": str(statistic),
                    "metric": METRIC_ALIAS.get(metric, metric),
                    "value": float(value),
                    "is_layer_metric": metric in LAYER_METRICS,
                }
            )
    return out


def _report_duplicates(per_case: pd.DataFrame) -> pd.DataFrame:
    """Find scans whose whole metric vector is identical to another scan's (see docstring note 1)."""
    wide = per_case.pivot_table(
        index=["patient_id", "study_date", "structure"], columns="metric", values="value"
    )
    signature = wide.round(6).astype(str).agg("|".join, axis=1)
    duplicated = signature[signature.duplicated(keep=False)]

    flags = []
    for group in duplicated.unique():
        members = duplicated[duplicated == group].index.tolist()
        for patient, date, structure in members:
            others = [f"{p}/{d}" for p, d, s in members if p != patient]
            flags.append(
                {
                    "patient_id": patient,
                    "study_date": date,
                    "structure": structure,
                    "identical_to": ";".join(others),
                    "n_metrics": int(wide.shape[1]),
                }
            )
    return pd.DataFrame(flags)


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    workbook = openpyxl.load_workbook(SOURCE, read_only=True, data_only=True)

    per_case = pd.DataFrame(
        _long_rows(workbook["Liver"], structure_column=False)
        + _long_rows(workbook["HCC"], structure_column=True)
    )
    summary = pd.DataFrame(_summary_rows(workbook["Stats"]))
    duplicates = _report_duplicates(per_case)

    DERIVED.mkdir(parents=True, exist_ok=True)
    targets = {
        "liverhccseg_published_percase.csv": per_case,
        "liverhccseg_published_summary.csv": summary,
        "liverhccseg_published_duplicate_rows.csv": duplicates,
    }
    for name, table in targets.items():
        table.to_csv(DERIVED / name, index=False)

    print(f"per-case:  {len(per_case)} rows, "
          f"{per_case.patient_id.nunique()} patients, "
          f"{per_case.groupby(['patient_id', 'structure']).ngroups} scan-structures, "
          f"{per_case.metric.nunique()} metrics")
    for structure, group in per_case[per_case.is_layer_metric].groupby("structure"):
        dice = group[group.metric == "dice"].value
        print(f"  {structure:8s} n={len(dice):2d}  dice mean {dice.mean():.3f}  "
              f"median {dice.median():.3f}  min {dice.min():.3f}  max {dice.max():.3f}")

    print(f"\nsummary:   {len(summary)} published statistics "
          f"({sorted(summary.statistic.unique())})")

    if len(duplicates):
        print(f"\n*** {len(duplicates)} scans share an identical metric vector with another scan:")
        for _, row in duplicates.iterrows():
            print(f"      {row.patient_id}/{row.study_date} {row.structure} "
                  f"== {row.identical_to} on all {row.n_metrics} metrics")
        print("    Treat as a transcription duplicate in the supplement, not as data (see docstring).")

    outlier = per_case[(per_case.metric == "dice") & (per_case.value < 0.5)]
    for _, row in outlier.iterrows():
        print(f"\n    low-agreement scan kept: {row.patient_id} {row.structure} dice {row.value:.3f}")

    for name in targets:
        print(f"wrote data/derived/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
