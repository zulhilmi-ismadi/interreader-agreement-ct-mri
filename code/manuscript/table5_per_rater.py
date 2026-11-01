"""Per-rater model table: each model read against every individual rater, the floor and the consensus comparators.

One row per (source, task, model). The columns that carry the comparison are the best and worst individual
rater, because a model can sit above every consensus comparator and below the floor against one person.

The RIGA arm has model-versus-consensus scores; the CURVAS arm was scored against individual raters only,
so its consensus columns are empty rather than filled with a substitute.

The last two columns are the **matched** comparison, which is the one a reader should
perform: each model's margin against the floor set by the same rater, at its minimum and maximum across
raters. They are deliberately not keyed to the best- and worst-*scoring* raters of the columns before them,
because in 4 of the 12 rows the lowest-scoring rater is not the rater whose floor is hardest to clear, so
such columns would report a margin that is not the worst case. See `matched_margins`.

    python code/manuscript/table5_per_rater.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PER_RATER = ROOT / "results" / "per_rater"
#: per rater pair and image, the source of each rater's own floor
PAIRS = ROOT / "results" / "layer" / "pairs.parquet"
OUT = ROOT / "results" / "manuscript" / "tables" / "table5_per_rater.csv"
MODEL_NAME = {"segformer_odoc": "SegFormer", "unet_drishti": "U-Net", "medsam": "MedSAM",
              "swin_unetr_btcv": "Swin UNETR", "dints_multi_organ": "DiNTS multi-organ",
              "dints_pancreas": "DiNTS pancreas"}
SOURCE_NAME = {"riga": "RIGA", "curvas": "CURVAS"}
CURVAS_TASKS = ["pancreas", "liver", "kidney"]


def _round(value) -> str:
    return "" if value is None or pd.isna(value) else f"{float(value):.4f}"


def _signed(value, rater: str) -> str:
    """A margin with its sign and the rater it occurs at, e.g. "-0.0026 (r6)"."""
    return "" if value is None or pd.isna(value) else f"{float(value):+.4f} ({rater})"


def model_scores(dataset: str, task: str) -> pd.DataFrame | None:
    """One model-versus-rater table, with rater ids put into the layer's vocabulary.

    The per-rater experiment keeps CURVAS's own file stems (`annotation_1`); the layer renames them (`r1`).
    They are the same rater by construction: `code/loaders/curvas.py` derives its id from the identical
    filename with `.replace("annotation_", "r")`. A rename, not a join on an assumption.
    """
    prefix = "curvas_" if dataset == "curvas" else ""
    path = PER_RATER / f"{prefix}model_vs_rater_{task}.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    frame["reference_rater"] = frame.reference_rater.str.replace("annotation_", "r", regex=False)
    return frame


def matched_margins() -> dict[tuple[str, str, str], dict]:
    """Each model's margin against the floor set by the same rater, per (dataset, task, model).

    A rater's own floor is their agreement with each of the other raters, averaged per image - the same
    quantity, computed the same way, as the per-rater floor interval. The margin is then the mean over the
    images that rater and the model share, so for CURVAS it lands on the cases the models were actually run
    on.

    Returns, for each row of the per-rater model table, the smallest and largest margin and the rater each occurs at, plus
    how many of that model's raters it clears.
    """
    pairs = pd.read_parquet(PAIRS, columns=["dataset", "task", "pair_class", "image_id",
                                            "rater_a", "rater_b", "dice"])
    pairs = pairs[pairs.pair_class == "P1"]
    out: dict[tuple[str, str, str], dict] = {}
    for (dataset, task), arm in pairs.groupby(["dataset", "task"]):
        scores = model_scores(dataset, task)
        if scores is None:
            continue
        own_floor = {rater: arm[(arm.rater_a == rater) | (arm.rater_b == rater)]
                     .groupby("image_id").dice.mean()
                     for rater in sorted(set(arm.rater_a) | set(arm.rater_b))}
        for model, block in scores.groupby("model"):
            margins = {}
            for rater, rows in block.groupby("reference_rater"):
                floor = own_floor.get(rater)
                if floor is None:
                    continue
                model_dice = rows.set_index("image_id").dice
                common = model_dice.index.intersection(floor.index)
                difference = (model_dice.loc[common] - floor.loc[common]).dropna()
                if difference.empty:
                    continue
                margins[rater] = float(difference.mean())
            if not margins:
                continue
            ordered = sorted(margins.items(), key=lambda kv: kv[1])
            out[(dataset, task, model)] = {
                "min": ordered[0][1], "min_rater": ordered[0][0],
                "max": ordered[-1][1], "max_rater": ordered[-1][0],
                "clears": sum(v > 0 for v in margins.values()), "raters": len(margins),
            }
    return out


MATCHED = matched_margins()


def matched_cells(dataset: str, task: str, model: str) -> dict[str, str]:
    """The two matched columns for one row, empty if the arm has no per-case scores."""
    m = MATCHED.get((dataset, task, model))
    if m is None:
        return {"Margin vs own floor, lowest": "", "Margin vs own floor, highest": ""}
    return {"Margin vs own floor, lowest": _signed(m["min"], m["min_rater"]),
            "Margin vs own floor, highest": _signed(m["max"], m["max_rater"])}


def rows_riga() -> list[dict]:
    h = pd.read_csv(PER_RATER / "model_vs_human_level.csv")
    out = []
    for _, r in h.iterrows():
        out.append({
            "Source": SOURCE_NAME.get(r.dataset, r.dataset), "Task": r.task,
            "Model": MODEL_NAME.get(r.model, r.model),
            "Against best rater": _round(r.best_individual),
            "Against worst rater": _round(r.worst_individual),
            "Against STAPLE": _round(r.vs_staple),
            "Against majority": _round(r.vs_majority),
            "Floor": _round(r.pairwise_floor),
            "Rater vs STAPLE": _round(r.rater_vs_staple),
            "Beats floor against every rater": "yes" if bool(r.beats_floor_against_every_rater) else "no",
            **matched_cells(r.dataset, r.task, r.model),
        })
    return out


def rows_curvas() -> list[dict]:
    floors = pd.read_csv(ROOT / "results" / "manuscript" / "s4_all_floors.csv")
    views = pd.read_csv(ROOT / "results" / "s1_e4_view_a_vs_b.csv")
    out = []
    for task in CURVAS_TASKS:
        path = PER_RATER / f"curvas_model_vs_rater_{task}.csv"
        if not path.is_file():
            continue
        scores = pd.read_csv(path)
        floor_row = floors[(floors.dataset == "curvas") & (floors.task == task)]
        floor = float(floor_row.iloc[0].primary) if not floor_row.empty else None
        view = views[(views.dataset == "curvas") & (views.task == task)]
        rater_staple = float(view.iloc[0].view_b_staple_dice) if not view.empty else None
        for model, block in scores.groupby("model"):
            per_rater = block.groupby("reference_rater").dice.mean()
            beats = floor is not None and bool((per_rater > floor).all())
            out.append({
                "Source": "CURVAS", "Task": task, "Model": MODEL_NAME.get(model, model),
                "Against best rater": _round(per_rater.max()),
                "Against worst rater": _round(per_rater.min()),
                "Against STAPLE": "", "Against majority": "",
                "Floor": _round(floor), "Rater vs STAPLE": _round(rater_staple),
                "Beats floor against every rater": "yes" if beats else "no",
                **matched_cells("curvas", task, model),
            })
    return out


S2N = ROOT / "results" / "manuscript" / "tables" / "s2n_per_rater_means.csv"
ARMS = [("", "riga", "disc"), ("", "riga", "cup"),
        ("curvas_", "curvas", "pancreas"), ("curvas_", "curvas", "liver"),
        ("curvas_", "curvas", "kidney")]


def per_rater_means() -> pd.DataFrame:
    """Each model mean against each individual rater, with its bootstrap interval.

    Means only; the per-case rows are not published in this table.
    """
    rows = []
    for prefix, dataset, task in ARMS:
        boot = PER_RATER / f"{prefix}model_ranks_bootstrap_{task}.csv"
        # a task with one admissible model has no ordering and so no bootstrap table; its means still
        # belong here, from the plain rank table, with the interval columns left empty
        source = boot if boot.is_file() else PER_RATER / f"{prefix}model_ranks_{task}.csv"
        if not source.is_file():
            continue
        b = pd.read_csv(source)
        for _, r in b.iterrows():
            rows.append({
                "Source": SOURCE_NAME.get(dataset, dataset), "Task": task,
                "Model": MODEL_NAME.get(r.model, r.model),
                "Reference rater": str(r.reference_rater).replace("annotation_", "r"),
                "Mean Dice": _round(r.mean_score),
                "CI lower": _round(r.ci_lo) if "ci_lo" in b.columns else "",
                "CI upper": _round(r.ci_hi) if "ci_hi" in b.columns else "",
                "Images or cases": int(r.images) if "images" in b.columns else "",
            })
    return pd.DataFrame(rows)


def main() -> None:
    table = pd.DataFrame(rows_riga() + rows_curvas())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    means = per_rater_means()
    means.to_csv(S2N, index=False)
    print(f"wrote {S2N.relative_to(ROOT)} ({len(means)} rows)")
    print(table.to_string(index=False))
    print(f"\nwrote {OUT.relative_to(ROOT)} ({len(table)} rows)")
    clears_all = sum(1 for m in MATCHED.values() if m["clears"] == m["raters"])
    clears_any = sum(1 for m in MATCHED.values() if m["clears"] > 0)
    print(f"\nmatched comparison: {clears_all} of {len(MATCHED)} rows clear their own rater's floor "
          f"against every rater; {clears_any} clear it against any")
    for key, m in sorted(MATCHED.items()):
        if m["clears"]:
            print(f"  {key[0]} {key[1]} {key[2]}: clears {m['clears']} of {m['raters']}")


if __name__ == "__main__":
    main()
