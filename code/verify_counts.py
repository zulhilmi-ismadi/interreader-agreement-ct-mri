"""Independent check of the reproduction counts (Table 4, Tables S5 and S6, Figure 3).

    python code/verify_counts.py

Recounts, with plain pandas and without importing the code that wrote them, how many published
agreement values are reproduced within half of the last digit printed in their source, under the
dataset's own definition and under the common definition, and compares the counts with
`reproduction_counts.csv` and with the values printed in the article: 60 / 37 of 90 for the CT and MRI
datasets, 28 / 5 of 56 for C3RO and 62 / 39 of all 96 tested values. Prints PASS or FAIL per check and
exits non-zero on any failure.

Definitions: half of the last printed digit = 0.5 x 10^(-published_decimals); a difference equal to it
counts as reproduced (1e-9 absorbs float error); the two LiverHccSeg values of scan TCGA-DD-A4NH are not
tested.

Reads results/e5_published_vs_recomputed.csv and results/manuscript/reproduction_counts.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results" / "e5_published_vs_recomputed.csv"
COUNTS = ROOT / "results" / "manuscript" / "reproduction_counts.csv"

CT_MRI = {"c3ro", "curvas", "pdacvi", "liverhccseg", "msseg2016", "qubiq21"}
EXPECTED = {"CT and MRI": (90, 60, 37), "C3RO": (56, 28, 5), "all": (96, 62, 39),
            "CT and MRI excluding C3RO": (34, 32, 32), "comparison set": (6, 2, 2)}
fails, checks = [], 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fails.append(label)


e5 = pd.read_csv(TABLE)
e5 = e5[~e5.task.str.contains("TCGA-DD-A4NH")]
half = 0.5 * 10.0 ** (-e5.published_decimals.astype(float))
own = (e5.recomputed_source_definition - e5.published).abs() <= half + 1e-9
common = (e5.recomputed_common - e5.published).abs() <= half + 1e-9
check("stored differences equal recomputed minus published",
      bool(((e5.recomputed_common - e5.published - e5.difference_common).abs() < 1e-9).all()))
groups = {"CT and MRI": e5.dataset.isin(CT_MRI), "C3RO": e5.dataset == "c3ro", "all": e5.dataset.notna(),
          "CT and MRI excluding C3RO": e5.dataset.isin(CT_MRI) & (e5.dataset != "c3ro"),
          "comparison set": ~e5.dataset.isin(CT_MRI)}
counts = pd.read_csv(COUNTS)
counts = counts[counts.rule == "within half of the last printed digit"].set_index("group")
for name, mask in groups.items():
    n, o, c = int(mask.sum()), int((own & mask).sum()), int((common & mask).sum())
    row = counts.loc[name]
    check(f"{name}: recount equals reproduction_counts.csv", (n, o, c) == (int(row.n_values), int(row.own_definition),
                                                                          int(row.common_definition)), f"{o} / {c} of {n}")
    check(f"{name}: equals the value printed in the article", (n, o, c) == EXPECTED[name],
          "{1} / {2} of {0}".format(*EXPECTED[name]))
c3 = e5[e5.dataset == "c3ro"]
check("every C3RO value is printed to two decimals", bool((c3.published_decimals == 2).all()))

print(f"\n{checks} checks, {len(fails)} failed")
for f in fails:
    print(f"  FAILED: {f}")
sys.exit(1 if fails else 0)
