"""Reproduction of published agreement values by tolerance, for the CT/MRI datasets and for all datasets.

    python code/report_reproduction_tolerance.py

The reproduction test compares each published inter-reader value with the value recomputed from the
released masks. The count depends on the tolerance, so the counts at 0.005, 0.01 and 0.02 are shown
together, beside the counts under a rule that needs no tolerance at all: the precision interval of the
printed value.

Precision-interval rules. A value printed to d decimals reproduces if the recomputed value falls in
the interval that would print as it:
  - half-up: rounding the recomputed value half-up to d decimals gives the printed value;
  - round-up: the printed value is the recomputed value rounded up (ceiling) to d decimals.
Both are stored per row in `results/e5_published_vs_recomputed.csv`
(`reproduces_{source,common}_{rounding,roundup}`).

Definitions. "Own" scores each value under the definition its source paper used; "common" under the
study's single common definition (mean over images of the mean pairwise value). The two TCGA-DD-A4NH LiverHccSeg rows are excluded from every count
because the published row duplicates another patient's values and cannot be resolved.

Scope. CT/MRI = C3RO, LiverHccSeg, MSSEG-2016 and PDACVI, the radiology datasets with a published value.
Within CT/MRI only the C3RO values, printed to 2 decimals, change outcome with the tolerance.

Writes results/manuscript/e5_tolerance_radiology.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "manuscript" / "e5_tolerance_radiology.csv"

CT_MRI = ("c3ro", "liverhccseg", "msseg2016", "pdacvi")
TOLERANCES = (0.005, 0.01, 0.02)
UNRESOLVED = "TCGA-DD-A4NH"


def main() -> None:
    e5 = pd.read_csv(RESULTS / "e5_published_vs_recomputed.csv")
    e5["unresolved"] = e5.task.str.contains(UNRESOLVED, regex=False)

    rules: dict[str, tuple[pd.Series, pd.Series]] = {}
    for tol in TOLERANCES:
        own = (e5.recomputed_source_definition - e5.published).abs() <= tol + 1e-9
        common = (e5.recomputed_common - e5.published).abs() <= tol + 1e-9
        rules[f"tolerance {tol}"] = (own, common)
    rules["precision interval, half-up"] = (e5.reproduces_source_rounding.astype(bool),
                                            e5.reproduces_common_rounding.astype(bool))
    rules["precision interval, round-up"] = (e5.reproduces_source_roundup.astype(bool),
                                             e5.reproduces_common_roundup.astype(bool))

    scopes = {"CT/MRI": e5.dataset.isin(CT_MRI), "all": pd.Series(True, index=e5.index)}
    scopes.update({name: e5.dataset == name for name in CT_MRI})

    rows = []
    for scope, in_scope in scopes.items():
        tested = in_scope & ~e5.unresolved
        decimals = e5.loc[tested, "published_decimals"].value_counts().sort_index()
        for rule, (own, common) in rules.items():
            rows.append(dict(
                scope=scope,
                rule=rule,
                n_rows=int(in_scope.sum()),
                unresolved_excluded=int((in_scope & e5.unresolved).sum()),
                tested=int(tested.sum()),
                reproduced_own=int((own & tested).sum()),
                reproduced_common=int((common & tested).sum()),
                published_decimals="; ".join(f"{int(d)} dp: {int(n)}" for d, n in decimals.items()),
            ))
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)

    for scope in ("CT/MRI", "all"):
        print(f"{scope}:")
        for _, r in table[table.scope == scope].iterrows():
            print(f"  {r.rule:30s} own {r.reproduced_own}/{r.tested}   common {r.reproduced_common}/{r.tested}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
