"""Log of the datasets identified during selection, with the conditions each met or failed (Table S1).

    python code/candidate_log.py

Datasets were identified from dataset descriptions, challenge reports and multireader studies, and the
list is a log of the datasets examined, with no claim of completeness. This script assembles the log
from the two files that recorded the selection.

Definitions used here (the selection conditions of Appendix S1):
  (i)   two or more readers outlined each image;
  (ii)  each reader's outlines were released separately;
  (iii) no source or released file showed that a reader had seen another reader's outlines;
  (iv)  the outlines are pixel masks (box datasets form a separate stream);
  (v)   the licence allows derived results to be shared (otherwise dataset-level results only).
For a dataset that was not recomputed, only conditions (i) and (ii) are recorded.

Reads results/ed1_census.csv (every dataset examined, with its outcome) and results/census.csv
(conditions (i) to (v) for the recomputed datasets).
Writes results/manuscript/candidate_log.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "manuscript" / "candidate_log.csv"

DISPLAY = {
    "c3ro": "C3RO", "nucls": "NuCLS", "ima": "IMA++", "qubiq21": "QUBIQ 2021", "gleason2019": "Gleason 2019",
    "pdacvi": "PDACVI", "riga": "RIGA", "mesmer": "TissueNet human-agreement set", "cytocrowd": "CytoCrowd",
    "lidc": "LIDC-IDRI", "multiorg": "MultiOrg", "curvas": "CURVAS", "arvaniti": "Arvaniti", "cryonuseg": "CryoNuSeg",
    "liverhccseg": "LiverHccSeg", "stare": "STARE", "chasedb1": "CHASE_DB1", "msseg2016": "MSSEG-2016",
    "CTC (Maska 2023)": "Cell Tracking Challenge", "SCGM": "Spinal Cord Grey Matter challenge",
    "Jungo 2018": "Jungo 2018 glioblastoma", "VinDr-CXR": "VinDr-CXR", "DRIVE": "DRIVE",
    "Kromp S-BSST265": "Kromp S-BSST265", "LIVECell": "LIVECell", "DSB2018": "DSB 2018",
    "TissueNet (main)": "TissueNet (main release)", "nnU-Net's 23 datasets": "nnU-Net's 23 training datasets",
}
FIELD = {"ct": "CT", "mri": "MRI", "ct/mri": "CT and MRI", "cxr": "Chest radiography", "histology": "Histology",
         "microscopy": "Microscopy", "cytology": "Cytology", "fundus": "Fundus photography",
         "dermoscopy": "Dermoscopy", "mixed": "Mixed"}
CONDITIONS = {"c1_multirater": "(i) two or more readers per image",
              "c2_individually_released": "(ii) each reader's outlines released",
              "c3_provenance_ok": "(iii) no shared outlines shown",
              "c4_pixel_floor": "(iv) pixel masks",
              "c5_releasable": "(v) licence allows derived results to be shared"}
CT_MRI = {"c3ro", "curvas", "pdacvi", "liverhccseg", "msseg2016", "qubiq21", "lidc"}


def outcome(key: str, recorded: str) -> str:
    comparison = "" if key in CT_MRI else " (comparison set)"
    if recorded.startswith("floor") and "withheld" in recorded:
        return f"included{comparison}; dataset-level results only"
    if recorded.startswith("floor"):
        return f"included{comparison}"
    if recorded.startswith("census only"):
        return "independence check only"
    if recorded.startswith("published band only"):
        return "not included: per-reader outlines not released"
    if recorded.startswith("access"):
        return "not included: access not completed"
    return "not included: " + recorded


def main() -> None:
    examined = pd.read_csv(RESULTS / "ed1_census.csv")
    census = pd.read_csv(RESULTS / "census.csv").set_index("source")
    rows = []
    for row in examined.itertuples(index=False):
        key = row.benchmark
        met, failed = [], []
        if key in census.index:
            c = census.loc[key]
            for column, label in CONDITIONS.items():
                (met if bool(c[column]) else failed).append(label)
            if key == "lidc" and CONDITIONS["c3_provenance_ok"] in met:
                met.remove(CONDITIONS["c3_provenance_ok"])
                failed.insert(0, CONDITIONS["c3_provenance_ok"])
        else:
            single = str(row.raters) in ("1", "1.0")
            (failed if single else met).append(CONDITIONS["c1_multirater"])
            if single or "cannot support" in str(row.outcome) or "published band" in str(row.outcome):
                failed.append(CONDITIONS["c2_individually_released"])
        rows.append({"dataset": DISPLAY.get(key, key), "field": FIELD.get(str(row.modality), str(row.modality)),
                     "readers": row.raters, "images": row.images, "outline_type": row.annotation,
                     "conditions_met": "; ".join(met), "conditions_failed": "; ".join(failed) if failed else "none",
                     "licence": row.licence if isinstance(row.licence, str) else "not stated",
                     "outcome": outcome(key, str(row.outcome))})
    log = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(OUT, index=False)
    print(log[["dataset", "field", "conditions_failed", "outcome"]].to_string(index=False))
    print(f"{len(log)} datasets logged, {int(log.outcome.str.startswith('included').sum())} included at some level")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
