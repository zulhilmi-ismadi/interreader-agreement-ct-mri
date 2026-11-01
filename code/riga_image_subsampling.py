"""RIGA image sub-sampling: how many images are needed before a model margin of a given size is ordered?

    python code/riga_image_subsampling.py

The per-radiologist arms contain either many images with wide margins (RIGA, 749 images, margins
0.032-0.159 Dice) or few cases with a narrow margin (CURVAS liver), never many cases with a narrow
margin. Sub-sampling RIGA supplies the missing combination: drawing small samples from a wide-margin
arm shows how ordering stability grows with the number of images at a known margin.

Method. For each arm (disc, cup) and each reference rater, the ordering compared is SegFormer against
U-Net (MedSAM sits 0.5-0.6 Dice below both and never competes for the win). For each subset size n,
B samples of n images are drawn with replacement, both models' mean Dice is recomputed on each sample,
and the share of samples whose winner matches the full-sample winner is recorded. That share is the
`same_winner_rate` used elsewhere, with the reference rater held fixed, so it is directly comparable
with `curvas_fixed_reference_control.py`. Seed and draw count follow the per-rater experiment
(20260917, B = 2000); each (arm, rater) starts its own generator from that seed.

The margin and the rate are stored rounded to 4 decimals. A rate reported in percent to one decimal is
rounded half-up from this column (for example 0.8005 is 80.05%, reported as 80.1%).

Writes results/per_rater/riga_image_subsampling.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PER_RATER = ROOT / "results" / "per_rater"
OUT = PER_RATER / "riga_image_subsampling.csv"

DRAWS, SEED = 2000, 20260917
SIZES = [20, 40, 80, 160, 320, 749]
PAIR = ("segformer_odoc", "unet_drishti")


def main() -> None:
    rows = []
    for arm in ("disc", "cup"):
        df = pd.read_csv(PER_RATER / f"model_vs_rater_{arm}.csv")
        for rater in sorted(df.reference_rater.unique()):
            sub = df[df.reference_rater == rater]
            wide = sub.pivot_table(index="image_id", columns="model", values="dice")[list(PAIR)]
            vals = wide.to_numpy(float)                      # (images, 2)
            n_img = vals.shape[0]
            present, filled = ~np.isnan(vals), np.nan_to_num(vals)

            full = np.nanmean(vals, axis=0)
            full_winner = int(full.argmax())
            margin = float(abs(full[0] - full[1]))

            rng = np.random.default_rng(SEED)
            for n in SIZES:
                pick = rng.integers(0, n_img, (DRAWS, n))
                totals = filled[pick].sum(axis=1)             # (DRAWS, 2)
                counts = present[pick].sum(axis=1)
                means = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
                stability = float((means.argmax(axis=1) == full_winner).mean())
                rows.append(dict(arm=arm, rater=rater, n=n, margin=round(margin, 4),
                                 same_winner_rate=round(stability, 4)))

    cal = pd.DataFrame(rows)
    cal.to_csv(OUT, index=False)

    print(f"RIGA image sub-sampling: {PAIR[0]} vs {PAIR[1]}, B={DRAWS}, seed={SEED}")
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    for arm in ("disc", "cup"):
        a = cal[cal.arm == arm]
        print(f"--- {arm}: margins {a.margin.min():.4f} to {a.margin.max():.4f} ---")
        piv = a.pivot_table(index="rater", columns="n", values="same_winner_rate")
        piv.insert(0, "margin", a.groupby("rater").margin.first())
        print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
        print()

    print("Smallest tested n reaching the given stability, per (arm, rater):")
    print(f"{'arm':6} {'rater':6} {'margin':>8} {'n@0.95':>8} {'n@0.99':>8}")
    for (arm, rater), g in cal.groupby(["arm", "rater"]):
        g = g.sort_values("n")

        def first(threshold: float) -> str:
            ok = g[g.same_winner_rate >= threshold]
            return str(int(ok.n.iloc[0])) if len(ok) else f">{max(SIZES)}"

        print(f"{arm:6} {rater:6} {g.margin.iloc[0]:8.4f} {first(0.95):>8} {first(0.99):>8}")


if __name__ == "__main__":
    main()
