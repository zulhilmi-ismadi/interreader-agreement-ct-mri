"""QUBIQ 2021: prostate pair counts, the brain-tumour label sets, and the dimensionality of every structure.

    python code/qubiq_audit.py

Per prostate task: readers and pairs per image and the task total. Per brain-tumour task: mean Dice of each
pair of label-set indices and the leave-one-out agreement, which shows which label set is the outlier. Two of
the three brain-tumour label sets are fusions of algorithm outputs, so these tasks are set aside from
agreement. Dimensionality comes from the loader's inspection of every released mask (`loaders/qubiq21.py`,
note 6): brain growth, brain tumour and kidney are rank-2 (2D); pancreas, pancreatic lesion and prostate
rank-3. The QUBIQ paper (arXiv 2405.18435, Table 2) lists prostate as 2D, and the released prostate files are
volumes one slice thick.

Reads results/qubiq21/pairs.csv.
Writes results/manuscript/qubiq_prostate_audit.csv, qubiq_braintumour_pairs.csv and qubiq_dimensions.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "manuscript"

DIMENSIONS = {  # rank of every released mask (loaders/qubiq21.py, note 6)
    "brain-growth": ("2D", "2D"), "brain-tumor": ("2D", "2D"), "kidney": ("2D", "2D"),
    "pancreas": ("3D", "3D"), "pancreatic-lesion": ("3D", "3D"), "prostate": ("2D", "3D"),
}


def say(label: str, value: object) -> None:
    print(f"  {label}: {value}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(RESULTS / "qubiq21" / "pairs.csv")  # brain tumour is class P5 (census only), the rest P1

    rows = []
    for (task, image), block in pairs[pairs.task.str.startswith("prostate")].groupby(["task", "image_id"]):
        rows.append({"task": task, "image_id": image, "n_readers": len(set(block.rater_a) | set(block.rater_b)),
                     "n_pairs": len(block)})
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "qubiq_prostate_audit.csv", index=False)
    for task, block in audit.groupby("task"):
        say(f"{task}: images / pairs", f"{len(block)} / {int(block.n_pairs.sum())}")
        say(f"{task}: images by reader count", block.n_readers.value_counts().sort_index().to_dict())

    rows = []
    for task, block in pairs[pairs.task.str.startswith("brain-tumor")].groupby("task"):
        rows.append({"task": task, "quantity": "all label sets", "indices": "all", "n_images": block.image_id.nunique(),
                     "dice": block.groupby("image_id").dice.mean().mean()})
        for (a, b), pb in block.groupby(["rater_a", "rater_b"]):
            rows.append({"task": task, "quantity": "pair", "indices": f"{a}-{b}", "n_images": pb.image_id.nunique(),
                         "dice": pb.dice.mean()})
        for r in sorted(set(block.rater_a) | set(block.rater_b)):
            rest = block[(block.rater_a != r) & (block.rater_b != r)]
            rows.append({"task": task, "quantity": "leave one out", "indices": f"without {r}",
                         "n_images": rest.image_id.nunique(), "dice": rest.groupby("image_id").dice.mean().mean()})
    bt = pd.DataFrame(rows)
    bt.to_csv(OUT / "qubiq_braintumour_pairs.csv", index=False)
    print(bt.round(4).to_string(index=False))

    dims = pd.DataFrame([{"structure": s, "paper_table2": p, "released_files": f,
                          "n_images": pairs[pairs.task.str.startswith(s)].image_id.nunique()}
                         for s, (p, f) in DIMENSIONS.items()])
    dims.to_csv(OUT / "qubiq_dimensions.csv", index=False)
    print(dims.to_string(index=False))
    say("wrote", OUT / "qubiq_prostate_audit.csv")


if __name__ == "__main__":
    main()
