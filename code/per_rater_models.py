"""Score off-the-shelf segmentation models against EACH INDIVIDUAL RATER.

The layer says how far two raters of a task sit apart. It does not say whether that distance is large
enough to reorder methods, and a public leaderboard cannot answer it, because no
leaderboard publishes per-rater scores. Where every rater's own annotation is released, the question is
answerable directly: run models, score each against each rater, and see whether the ordering holds.

This module does the data-side half, which needs no model and no GPU:

  * `riga_index()` pairs each prepared RIGA base with its unmarked `<base>prime` image. RIGA ships the
    raters' marks drawn ON the picture, so the prime is the only clean input a model may see; the
    masks in `_prepared/<base>/<rater>__{disc,cup}.tif` were recovered by differencing against it.
  * `score_masks()` scores one predicted mask against one rater's mask with the SAME metric code that
    built the layer (`gtu.metrics`), so a model-versus-rater number and a rater-versus-rater number are
    on one scale and can be printed in the same table.
  * `rank_table()` turns per-rater scores into the ranking measures: Kendall's
    tau between the orderings induced by different reference raters, top-1 stability, and the margin
    between adjacent models beside the floor and the pair dispersion.

Model inference lives in `per_rater_infer.py`, which writes `results/per_rater/predictions/`; this file
reads those predictions. Keeping them apart means the scoring can be checked without a GPU, and a model
can be swapped without touching the analysis.

    python code/per_rater_models.py --index          # check the image/mask pairing only
    python code/per_rater_models.py --score          # score whatever predictions exist

Writes results/per_rater/<name>_<task>.csv: the per-image scores (`model_vs_rater`), the per-rater
ranking (`model_ranks`), its bootstrap intervals (`model_ranks_bootstrap`), the ordering agreement
between reference raters (`rank_stability`, `rank_stability_bootstrap`), the margin against the rater
spread (`model_margins`), and with `--consensus`, `model_vs_consensus_<task>.csv`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

RIGA_RAW = ROOT / "data/gtu-raw/riga"
RIGA_PREPARED = RIGA_RAW / "_prepared"
OUT = ROOT / "results" / "per_rater"
PREDICTIONS = OUT / "predictions"
#: RIGA is 2D, so the tolerance grid is in pixels; 2 px is the reported tolerance.
TAU = 2
_PRIME = re.compile(r"^(?P<base>.+)prime$", re.IGNORECASE)


def riga_index() -> pd.DataFrame:
    """One row per prepared RIGA image: the unmarked image a model sees, and the raters available.

    The key is `<collection folder>_<image stem>`, which is how `_prepared` names its folders. The 50
    BinRushed1 bases ship no prime in `BinRushed/`; their primes live in `BinRushedcorrected/`, and the
    key resolves them without a special case.
    """
    primes: dict[str, Path] = {}
    for path in RIGA_RAW.rglob("*"):
        if not path.is_file():
            continue
        match = _PRIME.match(path.stem)
        if match:
            primes.setdefault(f"{path.parent.name}_{match.group('base')}", path)

    rows = []
    for folder in sorted(p for p in RIGA_PREPARED.iterdir() if p.is_dir()):
        raters = sorted({f.name.split("__")[0] for f in folder.glob("*__disc.tif")})
        prime = primes.get(folder.name)
        rows.append(
            {
                "dataset": "riga",
                "image_id": folder.name,
                "image_path": str(prime) if prime else "",
                "n_raters": len(raters),
                "raters": ";".join(raters),
                "has_image": prime is not None,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.has_image.all():
        missing = frame[~frame.has_image].image_id.tolist()
        raise FileNotFoundError(f"{len(missing)} RIGA bases have no unmarked image: {missing[:5]}")
    return frame


def rater_mask(image_id: str, rater: str, task: str) -> np.ndarray:
    from gtu.io import load_mask

    return load_mask(RIGA_PREPARED / image_id / f"{rater}__{task}.tif") != 0


def score_masks(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """The layer's own metrics, so model-vs-rater and rater-vs-rater numbers share one scale."""
    from gtu import metrics
    from gtu.io import WindowedMask, align

    a, b = align(WindowedMask.from_array(prediction), WindowedMask.from_array(reference))
    return metrics.pairwise_scores(a, b, instance=False, spacing=None)


def score_predictions(index: pd.DataFrame, task: str = "disc") -> pd.DataFrame:
    """Score every available prediction against every rater of the same image."""
    from gtu.io import load_mask

    rows = []
    for model_dir in sorted(p for p in PREDICTIONS.glob("*") if p.is_dir()):
        model = model_dir.name
        for _, row in index.iterrows():
            path = model_dir / f"{row.image_id}__{task}.tif"
            if not path.is_file():
                continue
            prediction = load_mask(path) != 0
            for rater in row.raters.split(";"):
                scores = score_masks(prediction, rater_mask(row.image_id, rater, task))
                rows.append(
                    {
                        "dataset": "riga",
                        "task": task,
                        "model": model,
                        "image_id": row.image_id,
                        "reference_rater": rater,
                        "dice": round(float(scores["dice"]), 4),
                        f"nsd@{TAU}": round(float(scores[f"nsd@{TAU}"]), 4),
                    }
                )
    return pd.DataFrame(rows)


def consensus_scores(index: pd.DataFrame, task: str = "disc") -> pd.DataFrame:
    """Score every model against the panel's STAPLE and majority-vote consensus.

    A model that wins against a consensus may still lose against some
    individual rater, and the paper may only say so if both were measured — a leaderboard built on a
    consensus reference is exactly the object this experiment is about. The consensus is built with the
    layer's own code (`gtu.consensus`), never a second implementation, so a model-versus-consensus
    number sits on the same scale as the model-versus-rater and rater-versus-rater numbers.

    STAPLE does not always terminate: on one Gleason 2019 panel it
    ran for hours on panels that were almost entirely empty. RIGA is the benign case — six raters, one
    2D image, a structure present in every panel — but the cost is real, so this runs behind its own
    flag rather than inside the default scoring pass.
    """
    from gtu import consensus as gtu_consensus
    from gtu.io import load_mask

    model_dirs = sorted(p for p in PREDICTIONS.glob("*") if p.is_dir())
    rows = []
    for _, row in index.iterrows():
        raters = row.raters.split(";")
        if len(raters) < 2:
            continue  # no panel, no consensus
        masks = [rater_mask(row.image_id, rater, task) for rater in raters]
        built = {
            "majority": gtu_consensus.majority_vote(masks),
            "staple": gtu_consensus.staple_consensus(masks),
        }
        for model_dir in model_dirs:
            path = model_dir / f"{row.image_id}__{task}.tif"
            if not path.is_file():
                continue
            prediction = load_mask(path) != 0
            for method, reference in built.items():
                scores = score_masks(prediction, reference)
                rows.append(
                    {
                        "dataset": "riga", "task": task, "model": model_dir.name,
                        "image_id": row.image_id, "comparator": method,
                        "n_raters": len(raters),
                        "dice": round(float(scores["dice"]), 4),
                        f"nsd@{TAU}": round(float(scores[f"nsd@{TAU}"]), 4),
                    }
                )
    return pd.DataFrame(rows)


def rank_table(scores: pd.DataFrame, metric: str = "dice") -> pd.DataFrame:
    """Per reference rater: each model's mean score and its rank. The endpoint of the experiment."""
    per_model = (
        scores.groupby(["dataset", "task", "reference_rater", "model"])[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "mean_score"})
    )
    per_model["rank"] = (
        per_model.groupby(["dataset", "task", "reference_rater"])["mean_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return per_model.sort_values(["dataset", "task", "reference_rater", "rank"])


def stability(ranks: pd.DataFrame) -> pd.DataFrame:
    """Does the ordering survive a change of reference rater? Kendall tau per rater pair, and top-1."""
    from itertools import combinations

    from scipy.stats import kendalltau

    rows = []
    for (dataset, task), block in ranks.groupby(["dataset", "task"]):
        wide = block.pivot(index="model", columns="reference_rater", values="mean_score").dropna()
        raters = list(wide.columns)
        for a, b in combinations(raters, 2):
            tau = kendalltau(wide[a], wide[b]).statistic if len(wide) > 1 else np.nan
            rows.append({"dataset": dataset, "task": task, "rater_a": a, "rater_b": b,
                         "kendall_tau": round(float(tau), 4) if np.isfinite(tau) else np.nan,
                         "same_winner": bool(wide[a].idxmax() == wide[b].idxmax())})
    return pd.DataFrame(rows)


#: Intervals, not point estimates, decide each ranking measure: a
#: rehearsal on rater masks reordered "models" on differences of 0.005-0.013 Dice, which is inside the
#: noise of 749 images. Images are resampled independently — RIGA ships no patient identifier, and the
#: loader records `cluster_id=None` because images do not nest — so there is no cluster to resample.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260917


def bootstrap(scores: pd.DataFrame, metric: str = "dice", draws: int = BOOTSTRAP_DRAWS,
              seed: int = BOOTSTRAP_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample images to put an interval on every ranking measure.

    Returns two frames:

      * per (task, reference rater, model): the mean score and its 95% interval;
      * per (task, rater pair): Kendall's tau with an interval, and the share of resamples in which
        the two raters disagree about which model is best.

    That last column is the endpoint that matters. A single ordering flip on one resample says
    nothing; a flip in most resamples says the ranking genuinely depends on who drew the reference.
    """
    from itertools import combinations

    from scipy.stats import kendalltau

    wide = scores.pivot_table(
        index="image_id", columns=["task", "model", "reference_rater"], values=metric
    )
    values = wide.to_numpy(dtype=float)
    columns = list(wide.columns)
    rng = np.random.default_rng(seed)

    # a resample is a draw of images with replacement; a column's score is its mean over that draw,
    # ignoring images where the rater is absent (13 of 749 RIGA images carry fewer than six)
    present = ~np.isnan(values)
    filled = np.nan_to_num(values)
    resampled = np.empty((draws, values.shape[1]))
    for draw in range(draws):
        pick = rng.integers(0, values.shape[0], values.shape[0])
        totals = filled[pick].sum(axis=0)
        counts = present[pick].sum(axis=0)
        resampled[draw] = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)

    intervals = pd.DataFrame(
        [
            {
                "task": task, "model": model, "reference_rater": rater,
                "mean_score": round(float(np.nanmean(values[:, j])), 4),
                "ci_lo": round(float(np.nanpercentile(resampled[:, j], 2.5)), 4),
                "ci_hi": round(float(np.nanpercentile(resampled[:, j], 97.5)), 4),
                "images": int(present[:, j].sum()),
            }
            for j, (task, model, rater) in enumerate(columns)
        ]
    )

    position = {key: j for j, key in enumerate(columns)}
    rows = []
    for task in sorted({t for t, _, _ in columns}):
        models = sorted({m for t, m, _ in columns if t == task})
        raters = sorted({r for t, _, r in columns if t == task})
        if len(models) < 2:
            continue
        for first, second in combinations(raters, 2):
            a = resampled[:, [position[(task, m, first)] for m in models]]
            b = resampled[:, [position[(task, m, second)] for m in models]]
            taus = np.array([kendalltau(a[d], b[d]).statistic for d in range(draws)])
            same_winner = a.argmax(axis=1) == b.argmax(axis=1)
            rows.append(
                {
                    "task": task, "rater_a": first, "rater_b": second,
                    "kendall_tau": round(float(np.nanmean(taus)), 4),
                    "tau_lo": round(float(np.nanpercentile(taus, 2.5)), 4),
                    "tau_hi": round(float(np.nanpercentile(taus, 97.5)), 4),
                    "same_winner_rate": round(float(same_winner.mean()), 4),
                }
            )

    # The margin's own sampling SD, paired across draws. The two models are scored on the same resampled
    # images in every draw, so their errors move together; taking the two models' independent intervals
    # instead would overstate the spread of their difference. This is the scale a margin is read
    # against, and it is not the rater-pair dispersion: the RIGA cup and the CURVAS liver sit at the same
    # margin-to-dispersion ratio and behave differently, because 749 images pin a mean that 20 cases do not.
    # Descriptive.
    margins = []
    for task in sorted({t for t, _, _ in columns}):
        models = sorted({m for t, m, _ in columns if t == task})
        if len(models) < 2:
            continue
        for rater in sorted({r for t, _, r in columns if t == task}):
            means = {m: resampled[:, position[(task, m, rater)]] for m in models}
            order = sorted(means, key=lambda m: float(np.nanmean(means[m])), reverse=True)
            paired = means[order[0]] - means[order[1]]
            margins.append({
                "task": task, "reference_rater": rater,
                "best_model": order[0], "runner_up": order[1],
                "margin_sd": round(float(np.nanstd(paired, ddof=1)), 5),
            })
    return intervals, pd.DataFrame(rows), pd.DataFrame(margins)


#: Written by `pair_dispersion.py`: the within-image spread across rater pairs, averaged over images.
DISPERSION = ROOT / "results" / "pair_dispersion.csv"


def dispersion_for(task: str, dataset: str = "riga", metric: str = "dice") -> float | None:
    """The spread between this task's own raters, which is the yardstick a model margin is read against."""
    if not DISPERSION.is_file():
        return None
    frame = pd.read_csv(DISPERSION)
    rows = frame[(frame.dataset == dataset) & (frame.task == task) & (frame.metric == metric)]
    rows = rows[rows.within_image_sd.notna()]
    if rows.empty:
        return None
    # RIGA's six ophthalmologists are all P1; prefer that class explicitly rather than whichever row
    # happens to sort first
    preferred = rows[rows.pair_class == "P1"]
    return float((preferred if not preferred.empty else rows).iloc[0].within_image_sd)


def margin_table(ranks: pd.DataFrame, metric_name: str = "dice",
                 margin_sd: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per reference rater: the gap between the best and second-best model, beside the rater spread.

    This is the decision rule: an ordering that changes on a margin smaller than the within-image dispersion of
    rater pairs is not evidence that rankings are rater-dependent in any practical sense. Putting the
    two numbers in one row forces that comparison to be made in the open instead of asserted.
    """
    rows = []
    for (dataset, task, rater), block in ranks.groupby(["dataset", "task", "reference_rater"]):
        ordered = block.sort_values("mean_score", ascending=False)
        if len(ordered) < 2:
            continue
        best, runner_up = ordered.iloc[0], ordered.iloc[1]
        margin = float(best.mean_score - runner_up.mean_score)
        spread = dispersion_for(task, dataset=dataset)
        rows.append(
            {
                "dataset": dataset, "task": task, "reference_rater": rater,
                "best_model": best.model, "best_score": round(float(best.mean_score), 4),
                "runner_up": runner_up.model, "runner_up_score": round(float(runner_up.mean_score), 4),
                "margin": round(margin, 4),
                "rater_pair_dispersion": spread,
                "margin_exceeds_dispersion": None if spread is None else bool(margin > spread),
            }
        )
    table = pd.DataFrame(rows)
    if margin_sd is not None and not margin_sd.empty and not table.empty:
        table = table.merge(margin_sd[["task", "reference_rater", "margin_sd"]],
                            on=["task", "reference_rater"], how="left")
        table["margin_over_sd"] = (table.margin / table.margin_sd).round(2)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="check the image/mask pairing and stop")
    parser.add_argument("--score", action="store_true", help="score the predictions on disk")
    parser.add_argument("--consensus", action="store_true",
                        help="also score every model against the STAPLE and majority-vote consensus")
    parser.add_argument("--task", default="disc", choices=("disc", "cup"))
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    index = riga_index()
    index.to_csv(OUT / "riga_index.csv", index=False)
    print(f"RIGA: {len(index)} images, all with an unmarked image; "
          f"rater counts {index.n_raters.value_counts().to_dict()}")

    if args.index or not args.score:
        return

    if not PREDICTIONS.exists():
        print(f"no predictions yet under {PREDICTIONS.relative_to(ROOT)}; run per_rater_infer.py first")
        return
    scores = score_predictions(index, task=args.task)
    if scores.empty:
        print("no predictions matched the index")
        return
    # One invocation scores one task, so every output carries the task in its name. Without that the
    # cup pass silently overwrites the disc results half an hour after they were computed, and the
    # loss is invisible: the files are all still there, with the right columns and the wrong contents.
    suffix = args.task
    scores.to_csv(OUT / f"model_vs_rater_{suffix}.csv", index=False)
    ranks = rank_table(scores)
    stable = stability(ranks)
    ranks.to_csv(OUT / f"model_ranks_{suffix}.csv", index=False)
    stable.to_csv(OUT / f"rank_stability_{suffix}.csv", index=False)
    intervals, resampled, margin_sd = bootstrap(scores)
    intervals.to_csv(OUT / f"model_ranks_bootstrap_{suffix}.csv", index=False)
    resampled.to_csv(OUT / f"rank_stability_bootstrap_{suffix}.csv", index=False)
    margins = margin_table(ranks, margin_sd=margin_sd)
    margins.to_csv(OUT / f"model_margins_{suffix}.csv", index=False)

    if args.consensus:
        against = consensus_scores(index, task=args.task)
        path = OUT / f"model_vs_consensus_{args.task}.csv"
        against.to_csv(path, index=False)
        summary = against.groupby(["model", "comparator"])["dice"].mean().round(4)
        print(f"\nagainst the panel consensus ({args.task}), mean dice over "
              f"{against.image_id.nunique()} images:")
        print(summary.to_string())
    print(f"\n{scores.model.nunique()} models x {scores.reference_rater.nunique()} raters "
          f"on {scores.image_id.nunique()} images")
    print(ranks.to_string(index=False))
    print(f"\nrank agreement between reference raters (Kendall tau), {args.task}:")
    print(stable.to_string(index=False))
    print(f"\nsame winner under every rater pair: {stable.same_winner.all()}")


if __name__ == "__main__":
    main()
