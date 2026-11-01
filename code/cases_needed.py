"""Cases needed to order two models whose margin is m: n = (z s / m)^2, for the CURVAS liver arm.

    python code/cases_needed.py

If two models differ by a true mean margin m and their case-by-case difference has standard deviation
s, the margin estimated on n cases has standard error s / sqrt(n). Requiring the margin to be z = 1.96
standard errors from zero gives n = (z s / m)^2. Both quantities can be measured on any benchmark
before a ranking is claimed, which is what makes the rule transferable. A second target, a 95% CI of
the margin that excludes zero with 80% power, uses z = 1.96 + 0.8416 (the 97.5th and 80th percentiles
of the standard normal distribution, 2.8016) in the same formula.

Definitions used here, per reference radiologist:
  - the model pair is the two models scored on the liver; the difference is the model with the higher
    mean Dice minus the other, per case;
  - m is the mean of that per-case difference, unrounded;
  - s is the sample standard deviation of that per-case difference (ddof = 1);
  - z = 1.96 for the first target, and z_80_power = Phi^-1(0.975) + Phi^-1(0.80) = 2.8016 for the second.
s is the spread of the per-case difference, not the bootstrap standard error of the margin rescaled by
sqrt(n); the two routes give similar but not identical n.

Reads results/per_rater/curvas_model_vs_rater_liver.csv, which is regenerated whenever the models are
re-scored, so this script is re-run after it.

Writes results/per_rater/cases_needed.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
PER_RATER = ROOT / "results" / "per_rater"
OUT = PER_RATER / "cases_needed.csv"

TASK = "liver"
Z = 1.96
Z_80_POWER = float(norm.ppf(0.975) + norm.ppf(0.80))


def main() -> None:
    scores = pd.read_csv(PER_RATER / f"curvas_model_vs_rater_{TASK}.csv")
    models = sorted(scores.model.unique())
    if len(models) != 2:
        raise SystemExit(f"expected two models on {TASK}, found {models}")

    rows = []
    for rater, block in scores.groupby("reference_rater"):
        wide = block.pivot_table(index="image_id", columns="model", values="dice").dropna()
        means = wide.mean()
        better, other = means.sort_values(ascending=False).index[:2]
        diff = wide[better] - wide[other]
        m = float(diff.mean())
        s = float(diff.std(ddof=1))
        n_needed = (Z * s / m) ** 2
        rows.append(dict(task=TASK, reference_radiologist=rater, model_pair=f"{better} - {other}",
                         n_cases=len(diff), margin_m=m, per_case_sd_s=s, z=Z,
                         n_needed=n_needed, n_needed_rounded=int(round(n_needed)),
                         z_80_power=Z_80_POWER, n_ci_80_power=(Z_80_POWER * s / m) ** 2))

    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    for _, r in table.iterrows():
        print(f"{r.reference_radiologist}: {r.model_pair}, m {r.margin_m:.5f}, s {r.per_case_sd_s:.5f}, "
              f"n = {r.n_needed:.1f}; 95% CI excluding zero with 80% power: n = {r.n_ci_80_power:.1f}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
