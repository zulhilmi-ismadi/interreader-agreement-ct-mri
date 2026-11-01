"""Step 2 of the seed-distance screen for CURVAS (the seed is regenerated; see below).

For every case and organ, measure how closely each radiologist agrees with the regenerated
TotalSegmentator starting outline, and how closely the radiologists agree with one another, and summarise
the paired difference (starting outline minus radiologists) per organ with a bootstrap over cases.
Agreement with the starting outline close to agreement between radiologists means the CURVAS reader
agreement describes radiologists who each corrected the same algorithm outline.

The seeded flag on CURVAS is a provenance record and is kept whatever this measurement shows.

The seed is the one regenerated on correctly oriented images by `curvas_seed_inference.py`. An earlier
seed, made on images read with the default orientation, saw the body mirrored front to back and is not
used.

Dice comes from the project's own `code/gtu/metrics.py`, so the numbers are on the same definition as
every floor in the layer (both-empty pairs score 1.0).

    python code/curvas_seed_screen.py

Writes results/per_rater/curvas_seed_distance.csv (every comparison) and
results/per_rater/curvas_seed_distance_summary.csv (per organ, with the paired bootstrap interval).
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from gtu.metrics import dice  # noqa: E402  the layer's own definition

RAW = ROOT / "data" / "gtu-raw" / "curvas"
SEED = ROOT / "data" / "derived" / "curvas_totalsegmentator_lps"
OUT = ROOT / "results" / "per_rater" / "curvas_seed_distance.csv"
SUMMARY = OUT.with_name("curvas_seed_distance_summary.csv")

ORGANS = {1: "pancreas", 2: "kidney", 3: "liver"}
RATERS = (1, 2, 3)
DRAWS, BOOT_SEED = 2000, 20260917


def load(path: Path) -> np.ndarray:
    """Label maps need rounding, not casting (code/loaders/curvas.py note 3)."""
    return np.rint(nib.load(path).get_fdata()).astype(np.int16)


def main() -> int:
    rows = []
    case_dirs = [d for split in ("training_set", "validation_set", "testing_set")
                 for d in sorted((RAW / split).iterdir()) if d.is_dir()]

    for n, case_dir in enumerate(case_dirs, 1):
        case = case_dir.name
        seed_path = SEED / f"{case}.nii.gz"
        if not seed_path.exists():
            continue
        ann = {}
        for r in RATERS:
            p = case_dir / f"annotation_{r}.nii.gz"
            if p.exists():
                ann[r] = load(p)
        if len(ann) < 2:
            continue
        seed = load(seed_path)

        for label, organ in ORGANS.items():
            seed_m = seed == label
            rater_m = {r: (a == label) for r, a in ann.items()}
            for r, m in rater_m.items():
                if m.shape != seed_m.shape:
                    print(f"  {case}/{organ}: shape mismatch, skipped", flush=True)
                    continue
                rows.append(dict(case=case, organ=organ, kind="rater_vs_seed",
                                 a=f"r{r}", b="seed", dice=dice(m, seed_m)))
            for r1, r2 in combinations(sorted(rater_m), 2):
                rows.append(dict(case=case, organ=organ, kind="rater_vs_rater",
                                 a=f"r{r1}", b=f"r{r2}", dice=dice(rater_m[r1], rater_m[r2])))
        if n % 10 == 0:
            print(f"  {n}/{len(case_dirs)} cases", flush=True)

    if not rows:
        print("No seeded cases found. Run curvas_seed_inference.py first.")
        return 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(df):,} comparisons, "
          f"{df.case.nunique()} cases)\n")

    # Per organ: the two means, and a paired bootstrap over cases of their difference.
    print(f"{'organ':10} {'cases':>6} {'rater-seed':>11} {'rater-rater':>12} "
          f"{'difference':>11} {'95% interval':>22}")
    verdict = {}
    summary = []
    for organ in ORGANS.values():
        g = df[df.organ == organ]
        per_case = g.pivot_table(index="case", columns="kind", values="dice", aggfunc="mean")
        per_case = per_case.dropna(subset=["rater_vs_seed", "rater_vs_rater"])
        if per_case.empty:
            continue
        s = per_case.rater_vs_seed.to_numpy()
        rr = per_case.rater_vs_rater.to_numpy()
        diff = s - rr                                   # positive => closer to the seed

        rng = np.random.default_rng(BOOT_SEED)
        idx = rng.integers(0, len(diff), (DRAWS, len(diff)))
        boot = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict[organ] = (diff.mean(), lo, hi)
        summary.append(dict(organ=organ, cases=len(diff), rater_vs_seed=round(s.mean(), 4),
                            rater_vs_rater=round(rr.mean(), 4), difference=round(diff.mean(), 4),
                            ci_low=round(lo, 4), ci_high=round(hi, 4), draws=DRAWS, seed=BOOT_SEED))
        print(f"{organ:10} {len(diff):6d} {s.mean():11.4f} {rr.mean():12.4f} "
              f"{diff.mean():+11.4f}   [{lo:+.4f}, {hi:+.4f}]")

    pd.DataFrame(summary).to_csv(SUMMARY, index=False)
    print(f"wrote {SUMMARY.relative_to(ROOT)}")

    print("\n" + "=" * 78)
    print("Decision rule applied")
    print("=" * 78)
    for organ, (d, lo, hi) in verdict.items():
        if lo > 0:
            call = ("CLOSER TO THE SEED than to one another (interval excludes zero): CURVAS "
                    "agreement reflects the shared starting outline under light correction")
        elif hi < 0:
            call = ("FURTHER FROM THE SEED than from one another (interval excludes zero): the "
                    "correction was substantive; the seeded flag still applies")
        else:
            call = "INDETERMINATE (interval includes zero): neither reading is supported"
        print(f"  {organ:9} {d:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {call}")
    print("\nThe seeded flag stays on CURVAS whatever this says.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
