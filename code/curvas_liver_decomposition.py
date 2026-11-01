"""Reference radiologist against scan sampling in the CURVAS model comparison.

    python code/curvas_liver_decomposition.py

Three public models were scored against each CURVAS radiologist on the 90 scans
(`per_rater_curvas.py`). For the liver, where two models are admissible, this script puts the two
sources of uncertainty in the paired margin on one scale; for every organ it tabulates each model's
shortfall against agreement between radiologists; and it sets the fixed-reference control beside the
probability that a margin reverses by chance.

Definitions used here:
  - paired margin under reference radiologist r: per scan, Dice(Swin UNETR, r) - Dice(DiNTS
    multiorgan, r), averaged over scans;
  - scan resamples: 2000 resamples of the 90 scans with replacement (seed 20260917; one generator per
    organ, started from the seed), shared by all models and radiologists;
  - variance across scan resamples: for each radiologist, the sample variance (ddof = 1) of the margin
    over the 2000 resamples, then the mean over the three radiologists;
  - variance across reference radiologists: within each resample, the variance of the three margins,
    then the mean over resamples. Given with the sample divisor (ddof = 1, primary: the three
    radiologists are a sample of possible reference readers, and it gives the smaller ratio) and with
    the population divisor (ddof = 0);
  - analytic reversal probability: Phi(-m sqrt(n) / s), for margin m between the two highest-scoring
    models, scan-to-scan standard deviation s of their per-scan difference (ddof = 1) and n scans; it
    approximates how often the first-ranked model changes when only the scans are resampled;
  - shortfall: interreader agreement (mean pairwise Dice, Table 3) minus a model's mean Dice over the
    three radiologists, or minus its Dice against its most favourable radiologist.

Reads results/per_rater/curvas_model_vs_rater_{pancreas,liver,kidney}.csv,
results/per_rater/curvas_fixed_reference_control.csv and results/curvas/task_floor.csv.
Writes results/per_rater/curvas_liver_margin_decomposition.csv,
results/per_rater/curvas_model_shortfall.csv and results/per_rater/curvas_fixed_reference_check.csv.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
PER_RATER = ROOT / "results" / "per_rater"
FLOORS = ROOT / "results" / "curvas" / "task_floor.csv"

ORGANS = ("pancreas", "liver", "kidney")
LIVER_PAIR = ("swin_unetr_btcv", "dints_multi_organ")
DRAWS, SEED = 2000, 20260917


def resample_means(wide: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    values = wide.to_numpy(float)
    pos = {c: i for i, c in enumerate(wide.columns)}
    rng = np.random.default_rng(SEED)
    res = np.empty((DRAWS, values.shape[1]))
    present, filled = ~np.isnan(values), np.nan_to_num(values)
    for k in range(DRAWS):
        pick = rng.integers(0, values.shape[0], values.shape[0])
        total, n = filled[pick].sum(0), present[pick].sum(0)
        res[k] = np.where(n > 0, total / np.maximum(n, 1), np.nan)
    return values, res, pos


def main() -> None:
    floors = pd.read_csv(FLOORS)
    floors = floors[(floors.metric == "dice") & (floors.pair_class == "P1")].set_index("task")["mean"]
    control = pd.read_csv(PER_RATER / "curvas_fixed_reference_control.csv")

    shortfall_rows, check_rows, decomposition = [], [], None
    for organ in ORGANS:
        s = pd.read_csv(PER_RATER / f"curvas_model_vs_rater_{organ}.csv")
        matrix = s.groupby(["model", "reference_rater"]).dice.mean().unstack()
        agreement = float(floors[organ])
        for model, row in matrix.iterrows():
            shortfall_rows.append({"organ": organ, "model": model, **{f"vs_{r}": v for r, v in row.items()},
                                   "mean_over_radiologists": row.mean(), "best_radiologist": row.max(),
                                   "shift_max_minus_min": row.max() - row.min(), "interreader_agreement": agreement,
                                   "shortfall_vs_mean": agreement - row.mean(),
                                   "shortfall_vs_best": agreement - row.max()})
        if s.model.nunique() < 2:
            continue
        wide = s.pivot_table(index="image_id", columns=["model", "reference_rater"], values="dice")
        models, raters = sorted(s.model.unique()), sorted(s.reference_rater.unique())
        values, res, pos = resample_means(wide)
        n = values.shape[0]
        for p, q in combinations(raters, 2):
            a = res[:, [pos[(m, p)] for m in models]].argmax(1)
            b = res[:, [pos[(m, q)] for m in models]].argmax(1)
            check_rows.append({"organ": organ, "quantity": "two reference radiologists disagree on the first-ranked model",
                               "radiologists": f"{p} vs {q}", "share_of_resamples": float((a != b).mean()),
                               "analytic_reversal_probability": np.nan, "margin_m": np.nan, "sd_s": np.nan,
                               "n_scans": n})
        for r in raters:
            cols = [pos[(m, r)] for m in models]
            means = {m: float(np.nanmean(values[:, pos[(m, r)]])) for m in models}
            top = sorted(means, key=means.get, reverse=True)[:2]
            d = wide[(top[0], r)] - wide[(top[1], r)]
            m, sd = float(d.mean()), float(d.std(ddof=1))
            full = int(np.nanmean(values[:, cols], 0).argmax())
            observed = float((res[:, cols].argmax(1) != full).mean())
            logged = control[(control.task == organ) & (control.reference_radiologist == r)]
            check_rows.append({"organ": organ, "quantity": "first-ranked model changes with the radiologist fixed",
                               "radiologists": r, "share_of_resamples": observed,
                               "analytic_reversal_probability": float(norm.cdf(-m * np.sqrt(n) / sd)),
                               "margin_m": m, "sd_s": sd, "n_scans": n,
                               "matches_fixed_reference_control": bool(
                                   len(logged) == 1 and abs(float(logged.winner_changes.iloc[0]) / DRAWS - observed) < 1e-12)})
        if organ == "liver":
            first, second = LIVER_PAIR
            margins = np.stack([res[:, pos[(first, r)]] - res[:, pos[(second, r)]] for r in raters], 1)
            per_scan = pd.DataFrame({r: wide[(first, r)] - wide[(second, r)] for r in raters})
            corr = per_scan.corr().to_numpy()
            off = corr[~np.eye(len(raters), dtype=bool)]
            var_scans = margins.var(axis=0, ddof=1)
            var_refs_sample = float(margins.var(axis=1, ddof=1).mean())
            var_refs_population = float(margins.var(axis=1, ddof=0).mean())
            decomposition = pd.DataFrame({
                "reference_radiologist": raters,
                "first": first, "second": second,
                "margin_point": [float(per_scan[r].mean()) for r in raters],
                "margin_ci_low": np.percentile(margins, 2.5, axis=0),
                "margin_ci_high": np.percentile(margins, 97.5, axis=0),
                "variance_across_scan_resamples": var_scans,
                "variance_across_scan_resamples_mean": var_scans.mean(),
                "variance_across_radiologists_sample": var_refs_sample,
                "variance_across_radiologists_population": var_refs_population,
                "variance_ratio_sample": var_scans.mean() / var_refs_sample,
                "variance_ratio_population": var_scans.mean() / var_refs_population,
                "corr_per_scan_margin_min": float(off.min()),
                "corr_per_scan_margin_max": float(off.max()),
                "n_scans": n, "draws": DRAWS, "seed": SEED,
            })
            print("liver margins", np.round(decomposition.margin_point, 5).tolist(),
                  "CIs", list(zip(np.round(decomposition.margin_ci_low, 4), np.round(decomposition.margin_ci_high, 4))))
            print(f"variance across scan resamples {var_scans.mean():.3e}; across radiologists "
                  f"{var_refs_sample:.3e} (sample, ratio {var_scans.mean() / var_refs_sample:.1f}) and "
                  f"{var_refs_population:.3e} (population, ratio {var_scans.mean() / var_refs_population:.1f}); "
                  f"per-scan margin correlations {off.min():.4f} to {off.max():.4f}")

    decomposition.to_csv(PER_RATER / "curvas_liver_margin_decomposition.csv", index=False)
    shortfall = pd.DataFrame(shortfall_rows)
    shortfall.to_csv(PER_RATER / "curvas_model_shortfall.csv", index=False)
    checks = pd.DataFrame(check_rows)
    checks.to_csv(PER_RATER / "curvas_fixed_reference_check.csv", index=False)
    print(shortfall.round(4).to_string(index=False))
    print(checks.round(4).to_string(index=False))
    print("wrote results/per_rater/curvas_liver_margin_decomposition.csv, curvas_model_shortfall.csv, "
          "curvas_fixed_reference_check.csv")


if __name__ == "__main__":
    main()
