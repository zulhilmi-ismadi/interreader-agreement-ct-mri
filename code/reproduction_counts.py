"""Reproduction of published agreement values at the printed precision and at fixed tolerances.

    python code/reproduction_counts.py

A published value counts as reproduced when the recomputed value lies within, or equal to, half of the
last digit printed in its source: |recomputed - published| <= 0.5 x 10^(-d), where d is the number of
decimals the source prints (`published_decimals` in e5_published_vs_recomputed.csv, transcribed from
each publication, not inferred from the stored float). Counts at the fixed tolerances 0.005, 0.01 and
0.02 are given beside it. Each count is made twice: with the value recomputed under the dataset's own
definition (`difference_source`) and under the common definition (`difference_common`, the mean over
images of the mean pairwise value).

Definitions used here:
  - the two LiverHccSeg values of scan TCGA-DD-A4NH are not tested (the published liver row of this
    scan duplicates another patient's, so neither value can be matched to a released structure);
  - CT and MRI datasets: C3RO, CURVAS, PDACVI, LiverHccSeg, MSSEG-2016 and QUBIQ 2021; every other
    dataset is the comparison set;
  - a comparison at the boundary counts as reproduced (a tolerance of 1e-9 absorbs float error);
  - for C3RO, which prints two decimals, two rounding conventions are tested on the value recomputed
    under its own definition: round half up, and round up (ceiling).

This rule differs from the rounding-interval columns of `validate_published.py`
(`reproduces_*_rounding`) only for a value lying exactly on the half-digit boundary: one CryoNuSeg
value does, so over all 96 tested values the rounding-interval rule gives 61 / 38 and this rule
62 / 39; the CT and MRI counts (60 / 37) are the same under both.

Reads results/e5_published_vs_recomputed.csv.
Writes results/manuscript/reproduction_counts.csv and results/manuscript/c3ro_rounding.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results" / "e5_published_vs_recomputed.csv"
OUT = ROOT / "results" / "manuscript"

CT_MRI = ("c3ro", "curvas", "pdacvi", "liverhccseg", "msseg2016", "qubiq21")
UNTESTED_SCAN = "TCGA-DD-A4NH"
TOLERANCES = (0.005, 0.01, 0.02)
EPS = 1e-9


def load() -> pd.DataFrame:
    e5 = pd.read_csv(TABLE)
    e5 = e5[~e5.task.str.contains(UNTESTED_SCAN)].copy()
    e5["half_last_printed_digit"] = 0.5 * 10.0 ** (-e5.published_decimals.astype(float))
    return e5


def count(df: pd.DataFrame, column: str, tolerance) -> int:
    tol = df.half_last_printed_digit if tolerance == "printed precision" else tolerance
    return int((df[column].abs() <= tol + EPS).sum())


def main() -> None:
    e5 = load()
    ct_mri = e5.dataset.isin(CT_MRI)
    groups = {
        "CT and MRI": e5[ct_mri],
        "CT and MRI excluding C3RO": e5[ct_mri & (e5.dataset != "c3ro")],
        "C3RO": e5[e5.dataset == "c3ro"],
        "comparison set": e5[~ct_mri],
        "CryoNuSeg": e5[e5.dataset == "cryonuseg"],
        "all": e5,
    }
    rows = []
    for name, df in groups.items():
        for tol in ["printed precision", *TOLERANCES]:
            rows.append({"group": name,
                         "rule": "within half of the last printed digit" if tol == "printed precision"
                         else f"within {tol}",
                         "n_values": len(df),
                         "own_definition": count(df, "difference_source", tol),
                         "common_definition": count(df, "difference_common", tol)})
            r = rows[-1]
            print(f"{name:27s} {r['rule']:40s} own {r['own_definition']:3d}  common {r['common_definition']:3d}"
                  f"  of {len(df)}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "reproduction_counts.csv", index=False)

    c3ro = e5[e5.dataset == "c3ro"].copy()
    scale = 10.0 ** c3ro.published_decimals.astype(float)
    c3ro["offset"] = c3ro.difference_source
    c3ro["round_up_matches"] = (np.ceil(c3ro.recomputed_source_definition * scale - EPS) / scale
                                - c3ro.published).abs() < EPS
    c3ro["round_half_up_matches"] = (np.floor(c3ro.recomputed_source_definition * scale + 0.5) / scale
                                     - c3ro.published).abs() < EPS
    keep = ["task", "pair_class", "published", "published_decimals", "recomputed_source_definition", "offset",
            "round_up_matches", "round_half_up_matches", "recomputed_common", "difference_common"]
    c3ro[keep].sort_values(["offset", "task", "pair_class"]).to_csv(OUT / "c3ro_rounding.csv", index=False)
    print(f"C3RO: {len(c3ro)} values, printed decimals {c3ro.published_decimals.value_counts().to_dict()}; "
          f"offsets below / at / above zero {int((c3ro.offset < -EPS).sum())} / "
          f"{int((c3ro.offset.abs() <= EPS).sum())} / {int((c3ro.offset > EPS).sum())} "
          f"(mean {c3ro.offset.mean():.4f}, min {c3ro.offset.min():.4f}); "
          f"round up {int(c3ro.round_up_matches.sum())}, round half up {int(c3ro.round_half_up_matches.sum())}")
    print(f"wrote {(OUT / 'reproduction_counts.csv').relative_to(ROOT)} and {(OUT / 'c3ro_rounding.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
