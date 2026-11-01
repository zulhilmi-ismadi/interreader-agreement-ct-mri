"""Compute the GTU layer from the loaded sources.

Writes to results/ as parquet + CSV, so that nothing in a figure or table is hand-entered:
  pairs.*       one row per (dataset, task, image, rater pair) - View A, every overlap, boundary and
                instance metric
  per_image.*   mean over pairs per image and pair class, with the reference-substitution spread (RSS)
  per_rater.*   each rater's mean against all others (description only)
  task_floor.*  the layer: per (dataset, task, pair class, metric) mean over images, median, IQR and
                bootstrap CI
  view_b.*      leave-one-rater-out vs STAPLE and majority consensus

Pair classes: P1 (independent expert) and P2 (independent non-expert) panels are kept strictly
separate, and P3 (intra-observer repeats) never enters a pooled floor, so every aggregate here is
grouped by the class of the rater pair, never
pooled across classes. A pair whose two raters differ in class is labelled `mixed` and enters nothing.

Usage: python code/compute_layer.py [--sources cryonuseg,mesmer,c3ro] [--max-view-b-raters N]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gtu import consensus, metrics
from gtu.io import WindowedMask, align, load_mask
from gtu.schema import (
    FLOOR_ELIGIBLE,
    Annotation,
    AnnotationType,
    Provenance,
    VOLUMETRIC_MODALITIES,
    group_by_image,
)
from loaders import (
    arvaniti,
    c3ro,
    chasedb1,
    cryonuseg,
    curvas,
    cytocrowd,
    gleason2019,
    ima,
    lidc,
    liverhccseg,
    mesmer,
    msseg2016,
    multiorg,
    nucls,
    pdacvi,
    qubiq21,
    riga,
    stare,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SOURCES = {
    "cryonuseg": (cryonuseg.load, ROOT / "data/gtu-raw/cryonuseg"),
    "mesmer": (mesmer.load, ROOT / "data/gtu-raw/mesmer"),
    "c3ro": (c3ro.load, ROOT / "data/gtu-raw/c3ro/Organized_files_v4"),
    "curvas": (curvas.load, ROOT / "data/gtu-raw/curvas"),
    "pdacvi": (pdacvi.load, ROOT / "data/gtu-raw/pdacvi"),
    "nucls": (nucls.load, ROOT / "data/gtu-raw/nucls"),
    "riga": (riga.load, ROOT / "data/gtu-raw/riga"),
    "gleason2019": (gleason2019.load, ROOT / "data/gtu-raw/gleason2019"),
    "liverhccseg": (liverhccseg.load, ROOT / "data/gtu-raw/liverhccseg"),
    "qubiq21": (qubiq21.load, ROOT / "data/gtu-raw/qubiq21"),
    "multiorg": (multiorg.load, ROOT / "data/gtu-raw/multiorg"),
    "cytocrowd": (cytocrowd.load, ROOT / "data/gtu-raw/cytocrowd"),
    "arvaniti": (arvaniti.load, ROOT / "data/gtu-raw/arvaniti"),
    "stare": (stare.load, ROOT / "data/gtu-raw/stare"),
    # second two-observer vessel source; clustered by child
    "chasedb1": (chasedb1.load, ROOT / "data/gtu-raw/chasedb1"),
    "ima": (ima.load, ROOT / "data/gtu-raw/ima"),
    # seven-rater MS lesion MRI; masks only, OFSEP data access agreement
    "msseg2016": (msseg2016.load, ROOT / "data/gtu-raw/msseg2016"),
    # read in place: 64,080 PNGs, nothing to reshape, so no copy into gtu-raw
    "lidc": (lidc.load, ROOT / "data/LIDC-IDRI-Kohl"),
}

#: Below this many images an image bootstrap is uninformative, so a rater bootstrap (resampling raters
#: with replacement) is computed too, and the wider of the two intervals is reported.
FEW_IMAGE_THRESHOLD = 10

#: View B holds out every rater and builds a STAPLE consensus of the rest, so its cost grows with the
#: square of the panel. Above this it is skipped, and the skip is recorded with its reason in
#: view_b_skipped.csv rather than silently dropped.
DEFAULT_MAX_VIEW_B_RATERS = 12


#: Per-group tables written to the checkpoint directory as each task finishes.
CHECKPOINT_TABLES = ("pairs", "per_image", "view_b", "view_b_skipped", "unpaired")


def _checkpoint_key(dataset: str, task: str, image_id: str) -> str:
    return f"{dataset}__{task}__{image_id}"


def _write_checkpoint(directory: Path, key: str, tables: dict[str, list[dict]]) -> None:
    """Write one finished group. `pairs` is written last: its presence is the done-marker, so a run
    killed mid-write is re-done rather than resumed from a half-written group."""
    for name in CHECKPOINT_TABLES:
        if name == "pairs":
            continue
        folder = directory / name
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tables[name]).to_csv(folder / f"{key}.csv", index=False)
    folder = directory / "pairs"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tables["pairs"]).to_csv(folder / f"{key}.csv", index=False)


def _collect_checkpoints(directory: Path, name: str) -> pd.DataFrame:
    files = sorted((directory / name).glob("*.csv")) if (directory / name).is_dir() else []
    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue  # a group with nothing for this table, e.g. no View B skips
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _worker_init() -> None:
    """Keep each worker to one thread. The heavy calls here are scipy.ndimage and SimpleITK, which
    would otherwise each try to use the whole machine and oversubscribe it N times over."""
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"


def compute(
    annotations: list[Annotation],
    max_view_b_raters: int = DEFAULT_MAX_VIEW_B_RATERS,
    checkpoint_dir: Path | None = None,
    workers: int = 1,
) -> dict[str, pd.DataFrame]:
    """Compute every group, then aggregate.

    `workers` > 1 runs the group loop in processes. This cannot change a number: each group writes
    its own checkpoint file and reads nothing from any other group, the collection step re-reads
    those files in sorted order, and the only RNG in the pipeline (the bootstrap) lives in
    `_task_floor`, which runs afterwards in this process. Parallelism moves wall-clock only."""
    groups = sorted(group_by_image(annotations).items())
    checkpoint_dir = checkpoint_dir or (RESULTS / "checkpoints")

    pending = [
        (key, group)
        for key, group in groups
        if not (checkpoint_dir / "pairs" / f"{_checkpoint_key(*key)}.csv").exists()
    ]
    done = len(groups) - len(pending)
    if done:
        print(f"  {done}/{len(groups)} groups already checkpointed, skipping", flush=True)

    if workers > 1 and pending:
        _compute_parallel(pending, max_view_b_raters, checkpoint_dir, workers, len(groups), done)
    else:
        for index, (key, group) in enumerate(pending, start=done + 1):
            print(f"  [{index}/{len(groups)}] {_process_group((key, group, max_view_b_raters, checkpoint_dir))}",
                  flush=True)

    pairs = _collect_checkpoints(checkpoint_dir, "pairs")
    per_image = _collect_checkpoints(checkpoint_dir, "per_image")
    return {
        "pairs": pairs,
        "per_image": per_image,
        "per_rater": _per_rater(pairs),
        "task_floor": _task_floor(pairs, per_image, annotations),
        "view_b": _collect_checkpoints(checkpoint_dir, "view_b"),
        "view_b_skipped": _collect_checkpoints(checkpoint_dir, "view_b_skipped"),
        "unpaired": _collect_checkpoints(checkpoint_dir, "unpaired"),
    }


def _compute_parallel(pending, max_view_b_raters, checkpoint_dir, workers, total, done) -> None:
    """Run the group loop across processes. Groups pickle to ~1 KB and workers return only a log
    line, so nothing large crosses a process boundary; the results travel as checkpoint files."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = [(key, group, max_view_b_raters, checkpoint_dir) for key, group in pending]
    print(f"  {len(jobs)} groups over {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_process_group, job): job[0] for job in jobs}
        for index, future in enumerate(as_completed(futures), start=done + 1):
            key = futures[future]
            try:
                print(f"  [{index}/{total}] {future.result()}", flush=True)
            except Exception as error:  # a failed group must name itself, not vanish into the pool
                print(f"  [{index}/{total}] FAILED {'/'.join(key)}: {type(error).__name__}: {error}",
                      flush=True)
                raise


def _process_group(job) -> str:
    """Compute and checkpoint exactly one group. Module level so the pool can pickle it."""
    (dataset, task, image_id), group, max_view_b_raters, checkpoint_dir = job
    if True:  # always runs; the block holds the computation for one group
        started = time.time()
        key = _checkpoint_key(dataset, task, image_id)
        pair_rows, image_rows, view_b_rows, skipped_rows = [], [], [], []
        by_rater = {a.rater_id: a for a in group}
        if len(by_rater) != len(group):
            # A dict keyed by rater silently keeps the last row, so a source that ships two masks
            # for one rater on one image would lose one of them without a word (IMA++ ships such
            # duplicates on 218 images). A loader must resolve that itself and say how.
            repeated = sorted({a.rater_id for a in group if sum(
                b.rater_id == a.rater_id for b in group) > 1})
            raise ValueError(
                f"{dataset}/{task}/{image_id}: {len(group)} annotations for {len(by_rater)} raters; "
                f"more than one mask per rater on one image ({', '.join(repeated)}). "
                "Resolve it in the loader: a pair is two different raters on the same image."
            )

        if len(by_rater) < 2:
            # An image with fewer than two raters forms no pair, so it is excluded from the floor
            # and recorded as unpaired. That is a property of the source, not a failure —
            # Gleason 2019's slide006_core137 carries one pathologist — but every aggregation below
            # assumes at least one pair (`next(iter(scored.values()))` raises StopIteration on an
            # empty dict), so the group is recorded and returned here instead. The masks are never
            # loaded: there is nothing to compare them with.
            _write_checkpoint(
                checkpoint_dir,
                key,
                {
                    "pairs": [],
                    "per_image": [],
                    "view_b": [],
                    "view_b_skipped": [],
                    "unpaired": [
                        {
                            "dataset": dataset,
                            "task": task,
                            "image_id": image_id,
                            "n_raters": len(by_rater),
                            "raters": ";".join(sorted(by_rater)),
                            "reason": "fewer than 2 readers on this image",
                        }
                    ],
                },
            )
            return (
                f"{dataset}/{task}: {len(by_rater)} rater, no pair - recorded as unpaired, "
                f"{time.time() - started:.1f}s"
            )

        masks = {
            rater: WindowedMask.from_array(_task_mask(a)) for rater, a in by_rater.items()
        }
        sample = group[0]
        # BOX joins INSTANCE here because the box stream is scored by box IoU and box F1@0.5, and
        # box F1@0.5 IS instance F1@0.5 evaluated on box masks - the same one-to-one Hungarian
        # matching on IoU, over rectangles instead of contours. No metric definition changes; the
        # separation of the box stream happens at reporting time (a box task is never pooled with
        # a pixel floor), not here.
        instance = sample.annotation_type in (AnnotationType.INSTANCE, AnnotationType.BOX)
        volumetric = sample.modality in VOLUMETRIC_MODALITIES

        scored: dict[frozenset[str], dict[str, float]] = {}
        classes: dict[frozenset[str], str] = {}
        for first, second in itertools.combinations(sorted(masks), 2):
            pair = frozenset((first, second))
            scored[pair] = metrics.pairwise_scores(
                *align(masks[first], masks[second]),
                instance=instance,
                spacing=sample.spacing,
                volumetric=volumetric,
            )
            both = {by_rater[first].rater_provenance, by_rater[second].rater_provenance}
            classes[pair] = next(iter(both)).value if len(both) == 1 else "mixed"
            pair_rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "image_id": image_id,
                    "rater_a": first,
                    "rater_b": second,
                    "pair_class": classes[pair],
                    "cluster_id": sample.cluster_id,
                    **scored[pair],
                }
            )

        metric_names = list(next(iter(scored.values())))
        for pair_class in sorted(set(classes.values())):
            members = [p for p, c in classes.items() if c == pair_class]
            raters = sorted({r for p in members for r in p})
            row = {
                "dataset": dataset,
                "task": task,
                "image_id": image_id,
                "pair_class": pair_class,
                "cluster_id": sample.cluster_id,
                "n_raters": len(raters),
                "n_pairs": len(members),
            }
            for metric_name in metric_names:
                row[metric_name] = _nanmean([scored[p][metric_name] for p in members])
                row[f"rss_{metric_name}"] = metrics.reference_substitution_spread(
                    {
                        (a, b): scored[frozenset((a, b))][metric_name]
                        for a, b in itertools.permutations(raters, 2)
                        if frozenset((a, b)) in scored
                    }
                )
            image_rows.append(row)

        _view_b(
            dataset, task, image_id, by_rater, masks, sample, volumetric,
            max_view_b_raters, view_b_rows, skipped_rows,
        )
        _write_checkpoint(
            checkpoint_dir,
            key,
            {
                "pairs": pair_rows,
                "per_image": image_rows,
                "view_b": view_b_rows,
                "view_b_skipped": skipped_rows,
                "unpaired": [],
            },
        )
        return (
            f"{dataset}/{task}: {len(masks)} raters, {len(scored)} pairs, "
            f"{time.time() - started:.1f}s"
        )


def _view_b(
    dataset, task, image_id, by_rater, masks, sample, volumetric,
    max_raters, view_b_rows, skipped_rows,
) -> None:
    """View B (each rater held out against a consensus of the others; three or more raters), run per
    pair class so a consensus never mixes P1 and P2 raters."""
    for pair_class in sorted({a.rater_provenance.value for a in by_rater.values()}):
        members = [r for r, a in by_rater.items() if a.rater_provenance.value == pair_class]
        if len(members) < 3:
            continue
        if len(members) > max_raters:
            skipped_rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "image_id": image_id,
                    "pair_class": pair_class,
                    "n_raters": len(members),
                    "reason": f"panel exceeds --max-view-b-raters={max_raters}",
                }
            )
            continue
        full = {r: masks[r].to_array() for r in members}
        for method, scores in consensus.view_b_scores(
            full, spacing=sample.spacing, volumetric=volumetric
        ).items():
            view_b_rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "image_id": image_id,
                    "pair_class": pair_class,
                    "consensus": method,
                    "n_raters": len(members),
                    **scores,
                }
            )


def _per_rater(pairs: pd.DataFrame) -> pd.DataFrame:
    """Each rater's mean against all others — description only; no rater is ever excluded on the basis
    of their scores."""
    metric_names = _metric_columns(pairs)
    long = pd.concat(
        [
            pairs.rename(columns={"rater_a": "rater_id"}),
            pairs.rename(columns={"rater_b": "rater_id"}),
        ]
    )
    return (
        long.groupby(["dataset", "task", "pair_class", "rater_id"])[metric_names]
        .mean()
        .reset_index()
    )


def _task_floor(
    pairs: pd.DataFrame, per_image: pd.DataFrame, annotations: list[Annotation]
) -> pd.DataFrame:
    """The layer: per task, the mean over images (with median and IQR) and its bootstrap interval."""
    licences = {(a.dataset, a.task): a.licence for a in annotations}
    modality = {(a.dataset, a.task): a.modality.value for a in annotations}
    # Box sources form a separate stream that is never pooled with pixel floors,
    # so the floor table has to carry the distinction; without it a downstream count of "recomputed
    # microscopy floors" silently includes NuCLS's boxes.
    annotation_type = {(a.dataset, a.task): a.annotation_type.value for a in annotations}

    rows = []
    for (dataset, task, pair_class), image_group in per_image.groupby(
        ["dataset", "task", "pair_class"]
    ):
        pair_group = pairs[
            (pairs.dataset == dataset)
            & (pairs.task == task)
            & (pairs.pair_class == pair_class)
        ]
        raters = sorted(set(pair_group.rater_a) | set(pair_group.rater_b))
        clusters = image_group.cluster_id
        use_clusters = clusters.notna().all() and clusters.nunique() < len(image_group)
        eligible = pair_class in {p.value for p in FLOOR_ELIGIBLE}

        for metric_name in _metric_columns(pairs):
            values = image_group[metric_name].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            low, high = metrics.bootstrap_ci(
                values, clusters=clusters.to_numpy() if use_clusters else None
            )
            row = {
                "dataset": dataset,
                "task": task,
                "pair_class": pair_class,
                "metric": metric_name,
                "modality": modality[(dataset, task)],
                "annotation_type": annotation_type[(dataset, task)],
                "floor_eligible": eligible,
                "licence": licences[(dataset, task)],
                "mean": finite.mean() if finite.size else np.nan,
                "median": np.median(finite) if finite.size else np.nan,
                "iqr_low": np.percentile(finite, 25) if finite.size else np.nan,
                "iqr_high": np.percentile(finite, 75) if finite.size else np.nan,
                "ci_low": low,
                "ci_high": high,
                "ci_source": "cluster_bootstrap" if use_clusters else "image_bootstrap",
                "n_raters": len(raters),
                "n_images": len(image_group),
                "n_pairs": int(pair_group.shape[0]),
                "n_scored": int(finite.size),
                "n_excluded_nan": int(values.size - finite.size),
                "rss": _nanmean(image_group[f"rss_{metric_name}"].tolist()),
            }
            if len(image_group) < FEW_IMAGE_THRESHOLD and len(raters) >= 3:
                pair_means = pair_group.groupby(["rater_a", "rater_b"])[metric_name].mean()
                r_low, r_high = metrics.rater_bootstrap_ci(dict(pair_means.items()), raters)
                row["rater_ci_low"], row["rater_ci_high"] = r_low, r_high
                if np.isfinite(r_low) and (r_high - r_low) > (high - low):
                    row["ci_low"], row["ci_high"] = r_low, r_high
                    row["ci_source"] = "rater_bootstrap"
            rows.append(row)
    return pd.DataFrame(rows)


def _task_mask(annotation: Annotation) -> np.ndarray:
    """The binary or instance mask for this task, selecting one class out of a multi-class label map
    where the source ships them together: each structure is its own binary task."""
    array = load_mask(annotation.mask_path)
    if annotation.label_value is None:
        return array
    return array == annotation.label_value


def _metric_columns(pairs: pd.DataFrame) -> list[str]:
    skip = {"dataset", "task", "image_id", "rater_a", "rater_b", "pair_class", "cluster_id"}
    return [c for c in pairs.columns if c not in skip]


def _nanmean(values) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="cryonuseg,mesmer")
    parser.add_argument("--max-view-b-raters", type=int, default=DEFAULT_MAX_VIEW_B_RATERS)
    parser.add_argument("--out", default=str(RESULTS))
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes for the group loop (default 1, serial). Results are "
             "identical either way; size it by MEMORY, not cores - a volumetric worker peaks near "
             "5 GB, a 2D one near 0.5 GB.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="per-task checkpoints; a rerun skips tasks already written there (default: <out>/checkpoints)",
    )
    args = parser.parse_args()

    annotations: list[Annotation] = []
    for name in args.sources.split(","):
        loader, root = SOURCES[name.strip()]
        loaded = loader(root)
        annotations += loaded
        print(f"{name}: {len(loaded)} rows")

    out = Path(args.out)
    checkpoints = Path(args.checkpoint_dir) if args.checkpoint_dir else out / "checkpoints"
    started = time.time()
    tables = compute(
        annotations,
        max_view_b_raters=args.max_view_b_raters,
        checkpoint_dir=checkpoints,
        workers=args.workers,
    )
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
        if not frame.empty:
            frame.to_parquet(out / f"{name}.parquet", index=False)
        print(f"wrote {out.name}/{name}  {frame.shape[0]} rows")
    print(f"total {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
