"""Fixed-reference control for the CURVAS per-radiologist model comparison.

    python code/curvas_fixed_reference_control.py

The per-radiologist comparison counts how often two DIFFERENT reference radiologists disagree about
which model wins across bootstrap resamples of the scans. This script reports the companion rate: holding
the reference radiologist FIXED, how often the winner on a resample differs from the winner on all scans.
The two rates describe different events and are reported side by side.

For each task and each reference radiologist, the full-sample winner is the model with the highest mean
Dice against that radiologist; the control rate is the share of resamples in which a different model
has the highest mean. Scans are resampled with replacement, with the seed and draw count of the
per-radiologist experiment (20260917, B = 2000). Each task uses its own generator started from that
seed, so the liver and pancreas results do not depend on each other.

Reads results/per_rater/curvas_model_vs_rater_{liver,pancreas}.csv, which are regenerated whenever the
models are re-scored, so this script is re-run after them.

Writes results/per_rater/curvas_fixed_reference_control.csv.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PER_RATER = ROOT / "results" / "per_rater"
OUT = PER_RATER / "curvas_fixed_reference_control.csv"

DRAWS, SEED = 2000, 20260917
TASKS = ("liver", "pancreas")


def resample_means(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per draw, the mean of each column over a with-replacement sample of rows (NaN-aware)."""
    resampled = np.empty((DRAWS, values.shape[1]))
    present, filled = ~np.isnan(values), np.nan_to_num(values)
    for d in range(DRAWS):
        pick = rng.integers(0, values.shape[0], values.shape[0])
        totals, counts = filled[pick].sum(axis=0), present[pick].sum(axis=0)
        resampled[d] = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return resampled


def main() -> None:
    rows = []
    for task in TASKS:
        scores = pd.read_csv(PER_RATER / f"curvas_model_vs_rater_{task}.csv")
        wide = scores.pivot_table(index="image_id", columns=["model", "reference_rater"], values="dice")
        values, cols = wide.to_numpy(float), list(wide.columns)
        pos = {c: i for i, c in enumerate(cols)}
        models = sorted(scores.model.unique())
        raters = sorted(scores.reference_rater.unique())

        resampled = resample_means(values, np.random.default_rng(SEED))
        print(f"CURVAS {task}: {values.shape[0]} cases, {len(models)} models, {len(raters)} radiologists")

        for a, b in combinations(raters, 2):
            first = resampled[:, [pos[(m, a)] for m in models]]
            second = resampled[:, [pos[(m, b)] for m in models]]
            agree = float((first.argmax(axis=1) == second.argmax(axis=1)).mean())
            print(f"  different references {a} vs {b}: same winner in {agree:.4f} of resamples")

        for r in raters:
            col = [pos[(m, r)] for m in models]
            full_winner = int(np.nanmean(values[:, col], axis=0).argmax())
            changes = int((resampled[:, col].argmax(axis=1) != full_winner).sum())
            rows.append(dict(task=task, reference_radiologist=r, full_sample_winner=models[full_winner],
                             n_cases=values.shape[0], n_models=len(models), winner_changes=changes,
                             draws=DRAWS, rate=changes / DRAWS, seed=SEED))
            print(f"  fixed reference {r}: winner changes in {changes}/{DRAWS} = {changes / DRAWS:.1%}")

    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
