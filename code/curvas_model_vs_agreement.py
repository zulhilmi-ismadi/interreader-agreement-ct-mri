"""CURVAS: each public model's score against the radiologists, beside the radiologists' agreement.

    python code/curvas_model_vs_agreement.py            # reuses the consensus-region scores if present
    python code/curvas_model_vs_agreement.py --rescore  # recomputes them from the masks (about 20 min)

The question is whether a model agrees with a radiologist as closely as the radiologists agree with one
another, on the same scans and on the same measure. Three comparisons are made.

1. **Per organ, against each radiologist.** For each model, organ and radiologist k, the model's mean
   Dice (and NSD at 2 mm) against radiologist k is set beside the agreement that the other two
   radiologists reach with radiologist k: the mean of the two reader pairs that include k. Both numbers
   then use radiologist k as the reference, so the comparison is like for like.

2. **Per organ, averaged over radiologists.** The model's score averaged over the three references is set
   beside the reader agreement for that organ: the mean over the three reader pairs, which is the value the
   manuscript reports (0.974 liver, 0.955 kidney, 0.901 pancreas on 90 scans). The difference is the
   model's score minus the agreement; a negative value is a shortfall.

3. **Organ-averaged, beside the CURVAS leaderboard.** The leaderboard reports one Dice per team, the
   mean over pancreas, kidney and liver; the top team scored 0.9457 (Riera-Marin et al. 2025, read from
   data/derived/curvas_riera2025_leaderboard.csv). The leaderboard publishes no per-case or per-organ
   scores, so the team value is a fixed number with no interval.

What is and is not like for like in comparison 3:
  - **Reference and scoring region: not like for like.** The challenge computes Dice only inside the
    consensus regions, the voxels that all three radiologists labelled as the organ or all three labelled
    as background; voxels on which they disagree are ignored, so a false positive can occur only in the
    consensus background and a false negative only in the consensus foreground (Riera-Marin et al. 2025,
    arXiv:2505.08685, section on evaluation metrics). Pairwise reader agreement counts every voxel,
    including the disputed boundary, so the leaderboard value is measured on an easier target. A reader
    scored the challenge's way against a consensus that includes that reader scores 1 by construction,
    so there is no reader value on the leaderboard's own scale. To place the public models on that scale,
    this script also scores each model's prediction inside the three-reader consensus regions, and, as an
    indicative human comparator, each radiologist inside the consensus regions of the other two (a
    two-reader consensus region is larger than a three-reader one, so this comparator is close to, but not
    exactly, the challenge's construction).
  - **Cases: close, not identical.** The leaderboard scored 60 cases: 20 of group A, 17 of group B and
    23 of group C. The released testing split holds 65 (20, 22 and 23); which 5 group B cases were not
    scored is not published, and the released files do not record the group. The testing-split rows here
    therefore use all 65 testing-split cases, read from the case_splits field of each model's _model.json.
    The all-case rows use the 90 released cases.
  - **Organ averaging: like for like.** Model and reader values are averaged over the three organs per
    case, as the challenge does. Only Swin UNETR outlines all three organs; the composite row takes, for
    each organ, the admissible model with the highest mean Dice on the 90 cases against the three
    radiologists (DiNTS pancreas for the pancreas, Swin UNETR for the liver and kidney).
  - **Training: not like for like.** The challenge teams trained on the CURVAS training data; the three
    MONAI models were run as released, without retraining, and were never trained on CURVAS.

Uncertainty. Every interval is a 95% percentile interval from B = 2000 resamples of cases with
replacement (seed 20260917). Within a case set one set of resampled case indices is shared by every model,
organ, measure and reference, so each difference is paired: the model score and the agreement are
recomputed on the same resampled cases. The case sets use separate generators started from the same seed.

Reads results/per_rater/curvas_model_vs_rater_{pancreas,kidney,liver}.csv (model against each
radiologist, all 90 cases), results/curvas/pairs.csv (reader pairs, the values behind the reader agreement
in results/layer/task_floor.csv), the case_splits field of
results/per_rater/predictions_curvas_lps/<model>/_model.json, and data/derived/curvas_riera2025_leaderboard.csv.
For the consensus-region scores it reads the predictions in results/per_rater/predictions_curvas_lps/ and
the radiologists' masks in data/gtu-raw/curvas/.

Writes
  results/per_rater/curvas_model_vs_agreement.csv          comparisons 1 and 2, every model, both case sets
  results/per_rater/curvas_model_vs_agreement_summary.csv  per organ: reader agreement, best model, shortfall
  results/per_rater/curvas_leaderboard_vs_agreement.csv    comparison 3
  results/per_rater/curvas_model_consensus_region.csv      per-case consensus-region Dice (cache)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PER_RATER = ROOT / "results" / "per_rater"
PAIRS = ROOT / "results" / "curvas" / "pairs.csv"
PREDICTIONS = PER_RATER / "predictions_curvas_lps"
CURVAS = ROOT / "data" / "gtu-raw" / "curvas"
LEADERBOARD = ROOT / "data" / "derived" / "curvas_riera2025_leaderboard.csv"

OUT = PER_RATER / "curvas_model_vs_agreement.csv"
OUT_SUMMARY = PER_RATER / "curvas_model_vs_agreement_summary.csv"
OUT_LEADERBOARD = PER_RATER / "curvas_leaderboard_vs_agreement.csv"
CONSENSUS_CACHE = PER_RATER / "curvas_model_consensus_region.csv"

DRAWS, SEED = 2000, 20260917
ORGANS = ("pancreas", "kidney", "liver")
LABELS = {"pancreas": 1, "kidney": 2, "liver": 3}
METRICS = ("dice", "nsd@2")
RATERS = ("annotation_1", "annotation_2", "annotation_3")
PAIR_RATER = {"r1": "annotation_1", "r2": "annotation_2", "r3": "annotation_3"}
SPLITS = ("training_set", "validation_set", "testing_set")
LEADERBOARD_CASES = 60


# ------------------------------------------------------------------------------------------- inputs

def model_scores() -> pd.DataFrame:
    frames = [pd.read_csv(PER_RATER / f"curvas_model_vs_rater_{organ}.csv") for organ in ORGANS]
    return pd.concat(frames, ignore_index=True)


def reader_agreement() -> pd.DataFrame:
    """Per case and organ: the mean over the three reader pairs, and over the two pairs that include k."""
    pairs = pd.read_csv(PAIRS)
    pairs["a"], pairs["b"] = pairs.rater_a.map(PAIR_RATER), pairs.rater_b.map(PAIR_RATER)
    rows = []
    for (case, organ), block in pairs.groupby(["image_id", "task"]):
        if len(block) != 3:
            raise SystemExit(f"{case} {organ}: expected 3 reader pairs, found {len(block)}")
        for metric in METRICS:
            rows.append(dict(image_id=case, organ=organ, metric=metric, reference="mean_of_three",
                             agreement=float(block[metric].mean())))
            for rater in RATERS:
                involved = block[(block.a == rater) | (block.b == rater)]
                rows.append(dict(image_id=case, organ=organ, metric=metric, reference=rater,
                                 agreement=float(involved[metric].mean())))
    return pd.DataFrame(rows)


def case_splits() -> dict[str, str]:
    """The split of every case, read from the model run records; the three records must agree."""
    found: dict[str, str] | None = None
    for record in sorted(PREDICTIONS.glob("*/_model.json")):
        splits = json.loads(record.read_text(encoding="utf-8"))["case_splits"]
        if found is not None and splits != found:
            raise SystemExit(f"{record.parent.name}: case_splits differ from the other models")
        found = splits
    if not found:
        raise SystemExit(f"no _model.json under {PREDICTIONS}")
    return found


# ------------------------------------------------------------------------------------------ bootstrap

def interval(values: np.ndarray, picks: np.ndarray) -> tuple[float, float, float]:
    """Mean over cases and its 95% percentile interval; `values` is per case, `picks` is draws x cases."""
    draws = values[picks].mean(axis=1)
    return float(values.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def compare(model: np.ndarray, agreement: np.ndarray, picks: np.ndarray) -> dict[str, float]:
    m, m_lo, m_hi = interval(model, picks)
    a, a_lo, a_hi = interval(agreement, picks)
    d, d_lo, d_hi = interval(model - agreement, picks)
    return dict(model_mean=m, model_ci_low=m_lo, model_ci_high=m_hi,
                agreement=a, agreement_ci_low=a_lo, agreement_ci_high=a_hi,
                difference=d, difference_ci_low=d_lo, difference_ci_high=d_hi)


# --------------------------------------------------------------------------- consensus-region scores

def consensus_region_scores(rescore: bool) -> pd.DataFrame:
    """Dice inside the three-reader consensus regions, the challenge's construction, per case and organ.

    For each model prediction P: TP = P inside the consensus foreground, FN = consensus foreground not in P,
    FP = P inside the consensus background; Dice = 2TP / (2TP + FP + FN). For each radiologist k the same
    is computed with k's mask as P and the consensus regions of the other two radiologists.
    """
    if CONSENSUS_CACHE.is_file() and not rescore:
        return pd.read_csv(CONSENSUS_CACHE)
    import nibabel

    def dice(pred: np.ndarray, fg: np.ndarray, bg: np.ndarray) -> float:
        tp = int(np.count_nonzero(pred & fg))
        fn = int(np.count_nonzero(fg)) - tp
        fp = int(np.count_nonzero(pred & bg))
        denominator = 2 * tp + fp + fn
        return 1.0 if denominator == 0 else 2 * tp / denominator

    def label_planes(image) -> dict[str, np.ndarray]:
        """One boolean mask per organ from a CURVAS label map.

        The maps are int16 scaled across the whole dtype range, so the stored code of each label is found
        from the header's slope and intercept and compared with the unscaled array. This avoids converting
        a 200-million-voxel volume to floating point and gives the same masks as rounding the scaled values.
        """
        raw = np.asanyarray(image.dataobj.get_unscaled())
        slope, inter = image.dataobj.slope, image.dataobj.inter
        if not np.isfinite(slope) or slope == 0:
            scaled = np.rint(np.asarray(image.dataobj)).astype(np.int16)
            return {organ: scaled == value for organ, value in LABELS.items()}
        planes = {}
        for organ, value in LABELS.items():
            code = int(np.rint((value - inter) / slope))
            if int(np.rint(code * slope + inter)) != value:
                raise SystemExit(f"label {value}: no stored code found from slope {slope}, intercept {inter}")
            planes[organ] = raw == code
        return planes

    models = sorted(p.name for p in PREDICTIONS.iterdir() if p.is_dir())
    cases = sorted(p for split in SPLITS for p in (CURVAS / split).iterdir() if p.is_dir())
    rows = []
    for number, case in enumerate(cases, 1):
        maps = [label_planes(nibabel.load(case / f"{r}.nii.gz")) for r in RATERS]
        for organ in ORGANS:
            masks = [m[organ] for m in maps]
            votes = masks[0].astype(np.uint8) + masks[1] + masks[2]
            fg, bg = votes == 3, votes == 0
            for model in models:
                path = PREDICTIONS / model / f"{case.name}__{organ}.nii.gz"
                if path.is_file():
                    pred = np.asarray(nibabel.load(path).dataobj) != 0
                    if pred.shape != fg.shape:
                        raise SystemExit(f"{path.name}: grid {pred.shape} differs from masks {fg.shape}")
                    rows.append(dict(image_id=case.name, organ=organ, scorer=model,
                                     kind="model", consensus="three_readers",
                                     dice_consensus_region=dice(pred, fg, bg)))
            for k, rater in enumerate(RATERS):
                others = [masks[j] for j in range(3) if j != k]
                fg2, bg2 = others[0] & others[1], ~(others[0] | others[1])
                rows.append(dict(image_id=case.name, organ=organ, scorer=rater, kind="radiologist",
                                 consensus="other_two_readers",
                                 dice_consensus_region=dice(masks[k], fg2, bg2)))
        print(f"  consensus regions {number}/{len(cases)} {case.name}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(CONSENSUS_CACHE, index=False)
    return table


# ------------------------------------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore", action="store_true",
                        help="recompute the consensus-region scores from the masks")
    args = parser.parse_args()

    scores = model_scores()
    agreement = reader_agreement()
    splits = case_splits()
    all_cases = sorted(scores.image_id.unique())
    if sorted(agreement.image_id.unique()) != all_cases or sorted(splits) != all_cases:
        raise SystemExit("model scores, reader pairs and case splits do not cover the same cases")
    case_sets = {"all_90": all_cases,
                 "testing_split_65": sorted(c for c in all_cases if splits[c] == "testing_set")}

    # per case, model score averaged over the three references
    mean_ref = (scores.groupby(["task", "model", "image_id"])[list(METRICS)]
                .mean().reset_index().assign(reference_rater="mean_of_three"))
    long = pd.concat([scores, mean_ref], ignore_index=True).rename(columns={"task": "organ"})

    # composite: for each organ the model with the highest mean Dice on all cases, references averaged
    best = (long[long.reference_rater == "mean_of_three"].groupby(["organ", "model"]).dice.mean()
            .reset_index().sort_values("dice", ascending=False).groupby("organ").head(1)
            .set_index("organ").model.to_dict())

    region = consensus_region_scores(args.rescore)
    rows, summary, board_rows = [], [], []
    for set_name, cases in case_sets.items():
        index = {c: i for i, c in enumerate(cases)}
        picks = np.random.default_rng(SEED).integers(0, len(cases), (DRAWS, len(cases)))

        def vector(frame: pd.DataFrame, column: str) -> np.ndarray:
            series = frame.set_index("image_id")[column].reindex(cases)
            if series.isna().any():
                raise SystemExit(f"{set_name}: missing cases in {column}")
            return series.to_numpy(float)

        organ_vectors: dict[tuple[str, str, str], np.ndarray] = {}
        for organ in ORGANS:
            for metric in METRICS:
                for reference in RATERS + ("mean_of_three",):
                    agree = vector(agreement[(agreement.organ == organ) & (agreement.metric == metric)
                                             & (agreement.reference == reference)], "agreement")
                    if reference == "mean_of_three":
                        organ_vectors[("agreement", organ, metric)] = agree
                    for model in sorted(long[long.organ == organ].model.unique()):
                        block = long[(long.organ == organ) & (long.model == model)
                                     & (long.reference_rater == reference)]
                        values = vector(block, metric)
                        if reference == "mean_of_three":
                            organ_vectors[(model, organ, metric)] = values
                        rows.append(dict(case_set=set_name, organ=organ, metric=metric, model=model,
                                         reference=reference, n_cases=len(cases),
                                         agreement_definition=("mean of the 3 reader pairs"
                                                               if reference == "mean_of_three" else
                                                               f"mean of the 2 reader pairs that include "
                                                               f"{reference}"),
                                         **compare(values, agree, picks)))
                # summary: the best model for this organ, against the three-pair agreement
                model = best[organ]
                result = compare(organ_vectors[(model, organ, metric)],
                                 organ_vectors[("agreement", organ, metric)], picks)
                summary.append(dict(case_set=set_name, organ=organ, metric=metric, n_cases=len(cases),
                                    reader_agreement=result["agreement"],
                                    reader_agreement_ci_low=result["agreement_ci_low"],
                                    reader_agreement_ci_high=result["agreement_ci_high"],
                                    best_model=model, best_model_score=result["model_mean"],
                                    best_model_ci_low=result["model_ci_low"],
                                    best_model_ci_high=result["model_ci_high"],
                                    difference=result["difference"],
                                    difference_ci_low=result["difference_ci_low"],
                                    difference_ci_high=result["difference_ci_high"]))

        # organ-averaged rows: Swin UNETR (the one model with all three organs) and the composite
        for metric in METRICS:
            agree = np.mean([organ_vectors[("agreement", o, metric)] for o in ORGANS], axis=0)
            candidates = {"swin_unetr_btcv": {o: "swin_unetr_btcv" for o in ORGANS},
                          "best_model_per_organ": best}
            for label, chosen in candidates.items():
                values = np.mean([organ_vectors[(chosen[o], o, metric)] for o in ORGANS], axis=0)
                result = compare(values, agree, picks)
                rows.append(dict(case_set=set_name, organ="organ_average", metric=metric, model=label,
                                 reference="mean_of_three", n_cases=len(cases),
                                 agreement_definition="mean of the 3 reader pairs, averaged over organs",
                                 **result))
                if label == "best_model_per_organ":
                    summary.append(dict(case_set=set_name, organ="organ_average", metric=metric,
                                        n_cases=len(cases), reader_agreement=result["agreement"],
                                        reader_agreement_ci_low=result["agreement_ci_low"],
                                        reader_agreement_ci_high=result["agreement_ci_high"],
                                        best_model="best_model_per_organ",
                                        best_model_score=result["model_mean"],
                                        best_model_ci_low=result["model_ci_low"],
                                        best_model_ci_high=result["model_ci_high"],
                                        difference=result["difference"],
                                        difference_ci_low=result["difference_ci_low"],
                                        difference_ci_high=result["difference_ci_high"]))

        # comparison 3: the leaderboard's top team beside organ-averaged agreement on this case set
        board = pd.read_csv(LEADERBOARD).sort_values("dsc_percent", ascending=False)
        top_team, top_dsc = board.iloc[0].team, float(board.iloc[0].dsc_percent) / 100
        agree = np.mean([organ_vectors[("agreement", o, "dice")] for o in ORGANS], axis=0)
        a, a_lo, a_hi = interval(agree, picks)
        board_rows.append(dict(case_set=set_name, n_cases=len(cases), quantity="pairwise reader agreement",
                               scorer="radiologists (3 reader pairs)", scoring="every voxel",
                               value=a, ci_low=a_lo, ci_high=a_hi,
                               top_team=top_team, top_team_dsc=top_dsc, top_team_cases=LEADERBOARD_CASES,
                               top_team_minus_value=top_dsc - a,
                               top_team_minus_value_ci_low=top_dsc - a_hi,
                               top_team_minus_value_ci_high=top_dsc - a_lo))
        for kind, scorers in (("model", list(candidates)), ("radiologist", list(RATERS) + ["mean_of_three"])):
            for scorer in scorers:
                per_organ = []
                for organ in ORGANS:
                    if kind == "model":
                        name = "swin_unetr_btcv" if scorer == "swin_unetr_btcv" else best[organ]
                        block = region[(region.kind == "model") & (region.scorer == name)
                                       & (region.organ == organ)]
                    elif scorer == "mean_of_three":
                        block = (region[(region.kind == "radiologist") & (region.organ == organ)]
                                 .groupby("image_id").dice_consensus_region.mean().reset_index())
                    else:
                        block = region[(region.scorer == scorer) & (region.organ == organ)]
                    per_organ.append(vector(block, "dice_consensus_region"))
                values = np.mean(per_organ, axis=0)
                v, v_lo, v_hi = interval(values, picks)
                board_rows.append(dict(
                    case_set=set_name, n_cases=len(cases),
                    quantity=("model, challenge scoring" if kind == "model"
                              else "radiologist against the other two, challenge-style scoring"),
                    scorer=scorer,
                    scoring=("inside the three-reader consensus regions" if kind == "model"
                             else "inside the consensus regions of the other two readers"),
                    value=v, ci_low=v_lo, ci_high=v_hi,
                    top_team=top_team, top_team_dsc=top_dsc, top_team_cases=LEADERBOARD_CASES,
                    top_team_minus_value=top_dsc - v, top_team_minus_value_ci_low=top_dsc - v_hi,
                    top_team_minus_value_ci_high=top_dsc - v_lo))

    for frame, path in ((pd.DataFrame(rows), OUT), (pd.DataFrame(summary), OUT_SUMMARY),
                        (pd.DataFrame(board_rows), OUT_LEADERBOARD)):
        frame.assign(draws=DRAWS, seed=SEED).round(4).to_csv(path, index=False)
        print(f"wrote {path.relative_to(ROOT)}")

    show = pd.DataFrame(summary)
    for _, r in show.iterrows():
        print(f"{r.case_set:17s} {r.organ:13s} {r.metric:6s} readers {r.reader_agreement:.3f} "
              f"({r.reader_agreement_ci_low:.3f}, {r.reader_agreement_ci_high:.3f})  {r.best_model} "
              f"{r.best_model_score:.3f}  difference {r.difference:+.3f} "
              f"({r.difference_ci_low:+.3f}, {r.difference_ci_high:+.3f})")
    print(pd.DataFrame(board_rows)[["case_set", "scorer", "scoring", "value", "ci_low", "ci_high",
                                    "top_team_minus_value"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
