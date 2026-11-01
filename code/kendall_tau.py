"""Rank correlation between the Dice and NSD at 2 mm rankings of the CT and MRI structures.

    python code/kendall_tau.py

Definitions used here:
  - structures: every reportable independent-reader (P1) structure of the six CT and MRI datasets;
  - Kendall's tau between their Dice values and their NSD values at a tolerance of 2 (millimetres, or
    voxels for QUBIQ 2021, which releases no voxel size), so the correlation is also given without
    QUBIQ 2021 and, separately, without C3RO;
  - 95% interval: 1000 resamples of structures with replacement (seed 20260903), percentile interval,
    skipping a resample in which either measure takes one value only;
  - rank 1 is the highest value.

Reads results/layer/task_floor.csv.
Writes results/manuscript/kendall_tau.csv and results/manuscript/kendall_tau_ranks.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
FLOORS = ROOT / "results" / "layer" / "task_floor.csv"
OUT = ROOT / "results" / "manuscript" / "kendall_tau.csv"

CT_MRI = ("c3ro", "curvas", "pdacvi", "liverhccseg", "msseg2016", "qubiq21")
DRAWS, SEED = 1000, 20260903


def tau_interval(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    tau = float(kendalltau(x, y).statistic)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(DRAWS):
        pick = rng.integers(0, x.size, x.size)
        if np.unique(x[pick]).size > 1 and np.unique(y[pick]).size > 1:
            draws.append(kendalltau(x[pick], y[pick]).statistic)
    return tau, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    floors = pd.read_csv(FLOORS, low_memory=False)
    floors = floors[(floors.pair_class == "P1") & floors.dataset.isin(CT_MRI) & floors.floor_reportable]
    dice = floors[floors.metric == "dice"].set_index(["dataset", "task"])["mean"]
    nsd = floors[floors.metric == "nsd@2"].set_index(["dataset", "task"])["mean"]
    table = pd.concat([dice.rename("dice"), nsd.rename("nsd2")], axis=1).dropna().reset_index()
    groups = {"all CT and MRI structures": table,
              "excluding C3RO": table[table.dataset != "c3ro"],
              "excluding QUBIQ 2021 (voxel tolerance)": table[table.dataset != "qubiq21"],
              "excluding QUBIQ 2021 and C3RO": table[~table.dataset.isin(["qubiq21", "c3ro"])]}
    rows = []
    for name, df in groups.items():
        tau, lo, hi = tau_interval(df.dice.to_numpy(float), df.nsd2.to_numpy(float))
        rows.append({"group": name, "n_structures": len(df), "kendall_tau": tau, "ci_low": lo, "ci_high": hi})
        print(f"{name} (n={len(df)}): tau {tau:.2f} (95% CI: {lo:.2f}, {hi:.2f})")
    ranks = table.assign(rank_dice=table.dice.rank(ascending=False), rank_nsd=table.nsd2.rank(ascending=False))
    rank_rows = []
    for ds, task in (("msseg2016", "ms_lesion"), ("liverhccseg", "liver")):
        r = ranks[(ranks.dataset == ds) & (ranks.task == task)].iloc[0]
        rank_rows.append({"dataset": ds, "task": task, "n_structures": len(table),
                          "rank_by_dice": int(r.rank_dice), "rank_by_nsd2": int(r.rank_nsd)})
        print(f"{ds} {task}: rank {int(r.rank_dice)} by Dice and {int(r.rank_nsd)} by NSD of {len(table)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(rank_rows).to_csv(OUT.with_name("kendall_tau_ranks.csv"), index=False)
    print(f"wrote {OUT.relative_to(ROOT)} and kendall_tau_ranks.csv")


if __name__ == "__main__":
    main()
