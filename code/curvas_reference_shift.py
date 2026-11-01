"""Gaps between adjacent CURVAS leaderboard teams against the reference-radiologist shift.

    python code/curvas_reference_shift.py

Each CURVAS radiologist's outlines are scored, as if they were a model's output, against each of the
other two radiologists' outlines on every scan. Averaged over scans this gives two dataset means per
scored radiologist, one for each possible reference radiologist.

Definitions used here:
  - shift of radiologist p: |mean Dice of p against reference q - mean Dice of p against reference r|,
    where q and r are the other two radiologists. Like a leaderboard gap, it is a difference between
    two means over scans. The organ-averaged basis (the mean of the liver, kidney and pancreas Dice on
    each scan, as the leaderboard averages organs) is primary; each organ is also given;
  - mean shift and largest shift: the mean and maximum of the three radiologists' shifts;
  - per-scan spread: on each scan, the range of a radiologist's score over the two references,
    averaged over radiologists and then over scans. It is a within-scan quantity, not on the scale of
    a difference between means;
  - a gap lies inside a band when gap < band;
  - interval of the mean shift and of the share of gaps inside it: 2000 resamples of scans with
    replacement (seed 20260917; a fresh generator for each row), 95% percentile interval. The shift is
    a mean of absolute differences, so its resampled values are biased upward;
  - randomly placed team scores: seven scores drawn uniformly at random over the leaderboard's own
    range (lowest to highest published team score), sorted; the share of the six adjacent gaps inside
    each band, averaged over 10 000 draws (seed 20260903);
  - testing split: the 65 scans of the challenge's testing split, from the case splits recorded with
    the model runs (the challenge scored 60 of them; which 5 were not scored is not published).

Reads results/curvas/pairs.csv, results/wp_l_curvas_gaps.csv and the case splits recorded in
results/per_rater/predictions_curvas_lps/<model>/_model.json.
Writes results/per_rater/curvas_reference_shift.csv, results/per_rater/curvas_gap_spacing_null.csv and
results/per_rater/curvas_reader_pairs.csv (the CURVAS reader-pair Dice rows behind the shift, so the
release can recompute it).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PER_RATER = RESULTS / "per_rater"
SPLITS = PER_RATER / "predictions_curvas_lps"

RATERS = ("r1", "r2", "r3")
ORGANS = ("liver", "kidney", "pancreas")
DRAWS, SEED = 2000, 20260917
NULL_DRAWS, NULL_SEED = 10_000, 20260903


def case_splits() -> dict[str, str]:
    record = sorted(SPLITS.glob("*/_model.json"))[0]
    return json.loads(record.read_text(encoding="utf-8"))["case_splits"]


def directed_wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per scan, one column per (scored radiologist, reference radiologist)."""
    both = pd.concat([df, df.rename(columns={"rater_a": "rater_b", "rater_b": "rater_a"})])
    return both.pivot_table(index="image_id", columns=["rater_a", "rater_b"], values="dice")


def shifts(wide: pd.DataFrame) -> list[float]:
    means = wide.mean()
    out = []
    for p in RATERS:
        q, r = [x for x in RATERS if x != p]
        out.append(float(abs(means[(p, q)] - means[(p, r)])))
    return out


def per_scan_spread(wide: pd.DataFrame) -> float:
    ranges = []
    for p in RATERS:
        q, r = [x for x in RATERS if x != p]
        ranges.append((wide[(p, q)] - wide[(p, r)]).abs())
    return float(pd.concat(ranges, axis=1).mean(axis=1).mean())


def resampled_interval(wide: pd.DataFrame, gaps: np.ndarray) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(SEED)
    n = len(wide)
    mean_shift, share = [], []
    for _ in range(DRAWS):
        sample = wide.iloc[rng.integers(0, n, n)]
        m = float(np.mean(shifts(sample)))
        mean_shift.append(m)
        share.append(float((gaps < m).mean()))
    return (*np.percentile(mean_shift, [2.5, 97.5]), *np.percentile(share, [2.5, 97.5]))


def random_placement(n_teams: int, low: float, high: float, band: float) -> float:
    rng = np.random.default_rng(NULL_SEED)
    draws = np.sort(rng.uniform(low, high, size=(NULL_DRAWS, n_teams)), axis=1)
    return float((np.diff(draws, axis=1) < band).mean())


def main() -> None:
    pairs = pd.read_csv(RESULTS / "curvas" / "pairs.csv")
    pairs = pairs[pairs.pair_class == "P1"]
    board = pd.read_csv(RESULTS / "wp_l_curvas_gaps.csv")
    gaps = board.gap.to_numpy(float)
    splits = case_splits()
    testing = {case for case, split in splits.items() if split == "testing_set"}

    released = pairs[["dataset", "task", "image_id", "rater_a", "rater_b", "dice"]].copy()
    released["split"] = released.image_id.map(splits)
    released.sort_values(["task", "image_id", "rater_a", "rater_b"]).to_csv(
        PER_RATER / "curvas_reader_pairs.csv", index=False)

    rows = []
    for subset, cases in (("all 90 scans", None), ("65 testing-split scans", testing)):
        p = pairs if cases is None else pairs[pairs.image_id.isin(cases)]
        for basis in ("organ-averaged", *ORGANS):
            if basis == "organ-averaged":
                df = p.groupby(["image_id", "rater_a", "rater_b"]).dice.mean().reset_index()
            else:
                df = p[p.task == basis][["image_id", "rater_a", "rater_b", "dice"]]
            wide = directed_wide(df)
            s = shifts(wide)
            mean_s, max_s = float(np.mean(s)), float(max(s))
            spread = per_scan_spread(wide)
            lo, hi, share_lo, share_hi = resampled_interval(wide, gaps)
            rows.append({"basis": basis, "subset": subset, "n_scans": len(wide),
                         "shift_r1": s[0], "shift_r2": s[1], "shift_r3": s[2],
                         "shift_mean": mean_s, "shift_max": max_s,
                         "shift_mean_ci_low": lo, "shift_mean_ci_high": hi,
                         "n_gaps": len(gaps),
                         "gaps_inside_shift_mean": int((gaps < mean_s).sum()),
                         "gaps_inside_shift_max": int((gaps < max_s).sum()),
                         "share_inside_shift_mean_ci_low": share_lo, "share_inside_shift_mean_ci_high": share_hi,
                         "per_scan_spread": spread, "gaps_inside_per_scan_spread": int((gaps < spread).sum()),
                         "draws": DRAWS, "seed": SEED})
            r = rows[-1]
            print(f"{basis:14s} {subset:22s} n={r['n_scans']:3d} shifts {np.round(s, 4)} mean {mean_s:.4f} "
                  f"({lo:.4f}, {hi:.4f}) max {max_s:.4f}; gaps inside mean/max {r['gaps_inside_shift_mean']}/"
                  f"{r['gaps_inside_shift_max']}; per-scan spread {spread:.4f} ({r['gaps_inside_per_scan_spread']})")
    table = pd.DataFrame(rows)
    table.to_csv(PER_RATER / "curvas_reference_shift.csv", index=False)

    scores = np.concatenate([board.dsc_above.to_numpy(float), board.dsc_below.to_numpy(float)[-1:]])
    low, high = float(scores.min()), float(scores.max())
    primary = table[(table.basis == "organ-averaged") & (table.subset == "all 90 scans")].iloc[0]
    null_rows = []
    for label, band in (("mean shift", primary.shift_mean), ("largest shift", primary.shift_max),
                        ("per-scan spread", primary.per_scan_spread)):
        expected = random_placement(len(scores), low, high, band)
        null_rows.append({"band": label, "band_value": band, "n_teams": len(scores), "score_low": low,
                          "score_high": high, "share_inside_observed": float((gaps < band).mean()),
                          "share_inside_random_placement": expected, "draws": NULL_DRAWS, "seed": NULL_SEED})
        print(f"  {label:16s} band {band:.4f}: observed {null_rows[-1]['share_inside_observed']:.3f}, "
              f"randomly placed scores {expected:.3f}")
    pd.DataFrame(null_rows).to_csv(PER_RATER / "curvas_gap_spacing_null.csv", index=False)
    print(f"median adjacent gap {np.median(gaps):.4f}; gaps {np.round(np.sort(gaps), 4).tolist()}")
    print("wrote results/per_rater/curvas_reference_shift.csv, curvas_gap_spacing_null.csv, curvas_reader_pairs.csv")


if __name__ == "__main__":
    main()
