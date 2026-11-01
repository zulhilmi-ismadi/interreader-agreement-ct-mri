"""Two sensitivity analyses and one descriptor: the reference-substitution spread by range against
s.d. (`s7_*`), rater-count sub-sampling (`s8_*`), and tau95.

    python code/sensitivity_s7_s8.py

Writes
  results/s7_rss_range_vs_sd.csv   every recomputed task with >= 3 raters: RSS by range and by s.d.
  results/s7_leaderboards.csv      share of adjacent gaps (e2) and of all method pairs (e6) inside the
                                   band, on CTC (SEG, DET) and CURVAS, with the range and s.d. bands
  results/s8_rater_subsampling.csv floor and RSS against k raters (C3RO, NuCLS, RIGA, IMA++)
  results/tau95.csv                smallest grid tolerance at which a task's NSD floor reaches 0.95

The s.d. is the sample s.d. (ddof = 1) of one prediction rater's scores across the reference raters;
the task value is the mean over prediction raters and images, as for the range. Sub-sampling draws, for
k = 2 ... n raters, 200 subsets uniformly without replacement (or every distinct subset when there are
at most 200); each subset's floor uses images where at least two subset raters annotated, and its RSS
images where at least three did. tau95 is the smallest tolerance on the grid ({1, 2, 4} px; {1, 2, 3} mm)
at which a task's NSD floor reaches 0.95, with no interpolation.

Everything is computed from `results/layer/pairs`; no mask is re-read. Two guards stop the script
from reporting numbers that are not like for like with the layer: the range RSS rebuilt here must
match the layer's stored `rss`, and the sub-sampled floor at k = n must match the layer's floor.
"""
from __future__ import annotations

import itertools
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gtu import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAYER = ROOT / "results" / "layer"
RESULTS = ROOT / "results"

SEED = 20260903
S8_SOURCES = ("c3ro", "nucls", "riga", "ima")
S8_SUBSETS = 200
WITHIN_CLASS = ("P1", "P2")  # the cross-tier "mixed" pairs are never used
TOLERANCE = 1e-9


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_parquet(LAYER / "pairs.parquet")
    floors = pd.read_csv(LAYER / "task_floor.csv")
    return pairs, floors


def _primary_metric(floors: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Dice for pixel and instance sources; instance F1 at IoU 0.5 for box streams."""
    kinds = floors.drop_duplicates(["dataset", "task"]).set_index(["dataset", "task"]).annotation_type
    return {key: ("instance_f1@0.5" if kind == "box" else "dice") for key, kind in kinds.items()}


def _reportable(floors: pd.DataFrame) -> pd.DataFrame:
    column = "floor_reportable" if "floor_reportable" in floors else "floor_eligible"
    return floors[floors[column]]


# ------------------------------------------------------------------------ RSS by range and by s.d.


def _rss_for_block(block: pd.DataFrame, metric: str, statistic: str) -> float:
    """Task RSS as the layer builds it: per image, the spread for each prediction rater across the
    reference raters, averaged over raters (gtu.metrics), then the mean over images."""
    values = []
    for _, image in block.groupby("image_id"):
        scores: dict[tuple[str, str], float] = {}
        for a, b, v in zip(image.rater_a, image.rater_b, image[metric]):
            scores[(a, b)] = v
            scores[(b, a)] = v
        values.append(metrics.reference_substitution_spread(scores, statistic=statistic))
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def s7_tasks(pairs: pd.DataFrame, floors: pd.DataFrame) -> pd.DataFrame:
    primary = _primary_metric(floors)
    stored = _reportable(floors).set_index(["dataset", "task", "pair_class", "metric"]).rss
    rows, mismatches = [], []
    for (dataset, task, pair_class), block in pairs.groupby(["dataset", "task", "pair_class"]):
        if pair_class not in WITHIN_CLASS:
            continue
        metric = primary.get((dataset, task))
        key = (dataset, task, pair_class, metric)
        if metric is None or key not in stored.index or not np.isfinite(stored[key]):
            continue  # not reportable, or fewer than three raters (RSS undefined)
        rng_value = _rss_for_block(block, metric, "range")
        sd_value = _rss_for_block(block, metric, "sd")
        if not math.isclose(rng_value, float(stored[key]), abs_tol=1e-6):
            mismatches.append((key, rng_value, float(stored[key])))
        rows.append(
            {
                "dataset": dataset, "task": task, "pair_class": pair_class, "metric": metric,
                "n_raters": len(set(block.rater_a) | set(block.rater_b)),
                "n_images": block.image_id.nunique(),
                "rss_range": rng_value, "rss_sd": sd_value,
                "sd_over_range": sd_value / rng_value if rng_value else float("nan"),
                "rss_layer": float(stored[key]),
            }
        )
    frame = pd.DataFrame(rows)
    print(f"range-vs-s.d. guard: range RSS rebuilt from pairs matches the layer on "
          f"{len(frame) - len(mismatches)}/{len(frame)} tasks")
    for key, mine, layer in mismatches[:10]:
        print(f"    MISMATCH {key}: rebuilt {mine:.6f} vs layer {layer:.6f}")
    if mismatches:
        raise SystemExit("range-vs-s.d. guard failed: the rebuilt range RSS does not match the layer")
    return frame


def s7_leaderboards() -> pd.DataFrame:
    import leaderboard_ctc as ctc
    import leaderboard_curvas as curvas

    rows = []

    # CTC: the spread across the three published annotator values per dataset and video
    table = pd.read_csv(ctc.DERIVED / "ctc_maska2023_annotator_measures.csv")
    table = table[table.rater.isin(ctc.ANNOTATORS)].dropna(subset=["value"])
    per_video = table.groupby(["measure", "dataset", "video"]).value.agg(
        spread_range=lambda v: v.max() - v.min(), spread_sd=lambda v: v.std(ddof=1), n="size"
    ).reset_index()
    per_video = per_video[per_video.n == len(ctc.ANNOTATORS)]
    bands = per_video.groupby(["measure", "dataset"])[["spread_range", "spread_sd"]].mean()
    board = ctc.leaderboard()
    for measure, annotator_measure in ctc.MEASURE_PAIRS.items():
        for statistic in ("range", "sd"):
            per_dataset = []
            for dataset, group in board[board.measure == measure].groupby("dataset"):
                if (annotator_measure, dataset) not in bands.index:
                    continue
                band = float(bands.loc[(annotator_measure, dataset), f"spread_{statistic}"])
                if band <= 0:
                    continue  # degenerate band, excluded as in leaderboard_ctc.py
                e2, e6, _, _ = ctc._gap_fractions(group.value.to_numpy(float), band)
                per_dataset.append((e2, e6, band))
            e2s, e6s, bs = (np.array(x) for x in zip(*per_dataset))
            low, high = metrics.bootstrap_ci(e2s)
            rows.append({"leaderboard": "CTC-CSB", "measure": measure, "statistic": statistic,
                         "n_units": len(e2s), "mean_band": bs.mean(), "e2": e2s.mean(),
                         "e2_ci_low": low, "e2_ci_high": high, "e6": e6s.mean()})

    # CURVAS: organ-averaged, recomputed from masks, as leaderboard_curvas.py
    averaged = curvas.organ_averaged_pairs()
    scores = pd.read_csv(curvas.LEADERBOARD).dsc_percent.to_numpy(float) / 100.0
    adjacent = np.abs(np.diff(np.sort(scores)[::-1]))
    all_pairs = np.array([abs(a - b) for a, b in itertools.combinations(scores, 2)])
    for statistic in ("range", "sd"):
        per_image = []
        for _, block in averaged.groupby("image_id"):
            scores_ = {}
            for a, b, v in zip(block.rater_a, block.rater_b, block.dice_organ_mean):
                scores_[(a, b)] = v
                scores_[(b, a)] = v
            per_image.append(metrics.reference_substitution_spread(scores_, statistic=statistic))
        band = float(np.nanmean(per_image))
        rows.append({"leaderboard": "CURVAS", "measure": "DSC", "statistic": statistic,
                     "n_units": len(per_image), "mean_band": band,
                     "e2": float((adjacent < band).mean()), "e2_ci_low": np.nan,
                     "e2_ci_high": np.nan, "e6": float((all_pairs < band).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ rater-count sub-sampling


def _panel_stack(block: pd.DataFrame, metric: str) -> tuple[list[str], np.ndarray]:
    """Dense (images x raters x raters) array of the pairwise metric, NaN where a pair is absent."""
    raters = sorted(set(block.rater_a) | set(block.rater_b))
    index = {r: i for i, r in enumerate(raters)}
    images = {img: i for i, img in enumerate(sorted(block.image_id.unique()))}
    stack = np.full((len(images), len(raters), len(raters)), np.nan)
    ii = block.image_id.map(images).to_numpy()
    aa = block.rater_a.map(index).to_numpy()
    bb = block.rater_b.map(index).to_numpy()
    vv = block[metric].to_numpy(float)
    stack[ii, aa, bb] = vv
    stack[ii, bb, aa] = vv
    return raters, stack


def _subset_floor_and_rss(stack: np.ndarray, subset: np.ndarray) -> tuple[float, float, int, int]:
    sub = stack[:, subset][:, :, subset]  # images x k x k
    k = len(subset)
    upper = np.triu_indices(k, 1)
    pair_values = sub[:, upper[0], upper[1]]  # images x pairs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        image_floor = np.nanmean(pair_values, axis=1)
        floor = float(np.nanmean(image_floor)) if np.isfinite(image_floor).any() else np.nan
        # RSS: for each prediction rater, the range across its (finite) references, if >= 2
        finite = np.isfinite(sub)
        count = finite.sum(axis=2)  # images x k
        spread = np.nanmax(sub, axis=2) - np.nanmin(sub, axis=2)
        spread[count < 2] = np.nan
        image_rss = np.nanmean(spread, axis=1)
        rss = float(np.nanmean(image_rss)) if np.isfinite(image_rss).any() else np.nan
    return floor, rss, int(np.isfinite(image_floor).sum()), int(np.isfinite(image_rss).sum())


def s8(pairs: pd.DataFrame, floors: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    primary = _primary_metric(floors)
    layer = _reportable(floors).set_index(["dataset", "task", "pair_class", "metric"])["mean"]
    rows, checks = [], []
    for dataset in S8_SOURCES:
        part = pairs[(pairs.dataset == dataset) & pairs.pair_class.isin(WITHIN_CLASS)]
        for (task, pair_class), block in part.groupby(["task", "pair_class"]):
            metric = primary[(dataset, task)]
            if (dataset, task, pair_class, metric) not in layer.index:
                continue
            raters, stack = _panel_stack(block, metric)
            n = len(raters)
            for k in range(2, n + 1):
                if math.comb(n, k) <= S8_SUBSETS:
                    subsets = [np.array(c) for c in itertools.combinations(range(n), k)]
                else:
                    subsets = [np.sort(rng.choice(n, size=k, replace=False)) for _ in range(S8_SUBSETS)]
                results = np.array([_subset_floor_and_rss(stack, s) for s in subsets], dtype=float)
                floors_k, rss_k = results[:, 0], results[:, 1]
                ok_f, ok_r = np.isfinite(floors_k), np.isfinite(rss_k)
                rows.append(
                    {
                        "dataset": dataset, "task": task, "pair_class": pair_class, "metric": metric,
                        "n_raters_panel": n, "k": k, "n_subsets": len(subsets),
                        "floor_mean": floors_k[ok_f].mean() if ok_f.any() else np.nan,
                        "floor_p2_5": np.percentile(floors_k[ok_f], 2.5) if ok_f.any() else np.nan,
                        "floor_p97_5": np.percentile(floors_k[ok_f], 97.5) if ok_f.any() else np.nan,
                        "rss_mean": rss_k[ok_r].mean() if ok_r.any() else np.nan,
                        "rss_p2_5": np.percentile(rss_k[ok_r], 2.5) if ok_r.any() else np.nan,
                        "rss_p97_5": np.percentile(rss_k[ok_r], 97.5) if ok_r.any() else np.nan,
                        "mean_images_floor": results[:, 2].mean(),
                        "mean_images_rss": results[:, 3].mean(),
                    }
                )
                if k == n:
                    checks.append(((dataset, task, pair_class), floors_k[0],
                                   float(layer[(dataset, task, pair_class, metric)])))
    bad = [c for c in checks if not math.isclose(c[1], c[2], abs_tol=1e-6)]
    print(f"sub-sampling guard: floor at k = n matches the layer on {len(checks) - len(bad)}/{len(checks)} panels")
    for key, mine, lay in bad[:10]:
        print(f"    MISMATCH {key}: {mine:.6f} vs layer {lay:.6f}")
    if bad:
        raise SystemExit("sub-sampling guard failed: the full-panel floor does not match the layer")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- tau95


def tau95(floors: pd.DataFrame) -> pd.DataFrame:
    nsd = _reportable(floors)
    nsd = nsd[nsd.metric.str.startswith("nsd@")]
    rows = []
    for (dataset, task, pair_class), block in nsd.groupby(["dataset", "task", "pair_class"]):
        grid = sorted((float(m.split("@")[1]), v) for m, v in zip(block.metric, block["mean"]) if np.isfinite(v))
        if not grid:
            continue
        taus = [t for t, _ in grid]
        unit = "mm" if taus[-1] == 3.0 else "px"
        reached = [t for t, v in grid if v >= 0.95]
        value = reached[0] if reached else np.nan
        label = f"{value:g} {unit}" if reached else f"> {taus[-1]:g} {unit}"
        row = {"dataset": dataset, "task": task, "pair_class": pair_class,
               "annotation_type": block.annotation_type.iloc[0], "unit": unit,
               "tau95": value, "tau95_label": label}
        row.update({f"nsd@{t:g}": v for t, v in grid})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    pairs, floors = _load()

    seven = s7_tasks(pairs, floors)
    seven.to_csv(RESULTS / "s7_rss_range_vs_sd.csv", index=False)
    print(f"range-vs-s.d. tasks: {len(seven)}; s.d. / range median {seven.sd_over_range.median():.3f} "
          f"(IQR {seven.sd_over_range.quantile(.25):.3f} to {seven.sd_over_range.quantile(.75):.3f})")

    boards = s7_leaderboards()
    boards.to_csv(RESULTS / "s7_leaderboards.csv", index=False)
    print("range-vs-s.d. leaderboards:")
    for _, r in boards.iterrows():
        ci = "" if not np.isfinite(r.e2_ci_low) else f" [{r.e2_ci_low:.1%}, {r.e2_ci_high:.1%}]"
        print(f"    {r.leaderboard:8s} {r.measure:4s} {r.statistic:5s} band {r.mean_band:.4f}  "
              f"adjacent {r.e2:.1%}{ci}  all pairs {r.e6:.1%}")

    eight = s8(pairs, floors)
    eight.to_csv(RESULTS / "s8_rater_subsampling.csv", index=False)
    print(f"sub-sampling: {len(eight)} (panel, k) rows over "
          f"{eight.groupby(['dataset', 'task', 'pair_class']).ngroups} panels")
    for (dataset, pair_class), part in eight.groupby(["dataset", "pair_class"]):
        first = part[part.k == 2].floor_mean.mean()
        last = part.groupby(["task"]).apply(lambda g: g[g.k == g.k.max()].floor_mean.iloc[0]).mean()
        print(f"    {dataset:6s} {pair_class}: mean floor at k=2 {first:.3f}, at k=n {last:.3f}")

    tau = tau95(floors)
    tau.to_csv(RESULTS / "tau95.csv", index=False)
    reached = tau.tau95.notna()
    print(f"tau95: {len(tau)} tasks; {reached.sum()} reach NSD 0.95 on the grid, "
          f"{(~reached).sum()} do not")
    print("\nwrote results/s7_rss_range_vs_sd.csv, s7_leaderboards.csv, s8_rater_subsampling.csv, tau95.csv")


if __name__ == "__main__":
    main()
