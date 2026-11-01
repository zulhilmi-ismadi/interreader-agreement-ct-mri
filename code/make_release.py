"""Build the release of the agreement layer.

    python code/make_release.py

Assembles `release/gtu-layer-v1.1/`:

  task_floor.csv            the layer, per (dataset, task, pair class, metric), for every source
  consensus_comparator.csv  each rater against a consensus of the others, beside the pairwise floor (3+ raters)
  per_image.csv             per-image agreement, for sources whose licence permits it
  per_rater.csv             per-rater means (description only; no rater is ever excluded on their scores),
                            same sources
  licences.tsv              what each source permits, where that was read, and what ships
  reporting_items.md        the article's eleven reporting items for a multireader dataset release (Table 2)
  candidate_log.csv         the datasets identified during selection and the conditions each met or failed
  published_values/         published agreement values beside their recomputation, the reproduction
                            counts and an independent check of them
  README.md                 how to look up a floor and read it
  per_rater_scores/         the per-rater model experiment: every model's score on every case against
                            every individual reference rater, with the registry and a verifier

What the release enforces:

  * **No images and no masks are redistributed.** Only derived statistics keyed by the original dataset's own
    `image_id`, so a user joins against data they obtain from the source themselves.
  * **Task-level rows of every source are released** as facts about the published data; **per-image and
    per-rater rows are withheld** for sources whose licence is unstated, forbids derivatives, or is a data use
    agreement that permits group-level derivatives only.
  * v1.x is the present set of definitions. Adding sources or derived tables without changing a metric,
    aggregation or spread definition is v1.x; changing one of those definitions is v2.
  * **The per-rater score tables ship** (`per_rater_scores/`). Few benchmark releases publish each model's
    score against each individual reference rater, and these tables do. Scores only: no predicted masks and no
    images.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LAYER = ROOT / "results" / "layer"
RESULTS = ROOT / "results"
RELEASE = ROOT / "release" / "gtu-layer-v1.1"
PER_RATER = RESULTS / "per_rater"
PER_RATER_RELEASE = RELEASE / "per_rater_scores"
#: The CURVAS predictions behind every released CURVAS model score: the run on correctly oriented input.
#: `predictions_curvas/`, where present, holds a run on images read with the default orientation (mirrored
#: front to back); it is never read here, and `build_model_registry` refuses it.
CURVAS_PREDICTIONS = PER_RATER / "predictions_curvas_lps"
assert CURVAS_PREDICTIONS.name.endswith("_lps"), "the registry must read the correctly oriented CURVAS run"
#: the independent recomputation of the CURVAS per-rater tables, shipped so a reader can check them
#: without trusting the code that wrote them
VERIFIER = ROOT / "code" / "verify_curvas.py"
#: the independent recount of the reproduction counts, shipped beside the table it reads
COUNTS_VERIFIER = ROOT / "code" / "verify_counts.py"
PUBLISHED_RELEASE = RELEASE / "published_values"
#: where `per_rater_infer.py` and `per_rater_curvas.py` cache downloaded weights. Only the bundles' own
#: version strings are read from it; no weights ship.
MODEL_CACHE = ROOT / ".model-cache"

VERSION = "1.1"

#: The per-rater experiment's bootstrap, set in `per_rater_models.py` and restated here so a reader of
#: the released tables need not open the code to learn how an interval was drawn.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260917

#: How MedSAM was pointed at the structure (`per_rater_infer.run_sam_box`). A prompt-driven model is
#: defined as much by its prompt rule as by its weights, so the rule ships beside the scores. The
#: README quotes the description of the box and the statement that it reads no rater mask;
#: `models.json` carries the full string unchanged.
MEDSAM_PROMPT_BOX = (
    "A square box covering 18% of the frame, centred on the brightest region of the green channel - the "
    "usual optic-nerve-head cue in a fundus photograph."
)
MEDSAM_PROMPT_RULE = (
    MEDSAM_PROMPT_BOX + " Set by a fixed rule and applied identically to every image; no reader mask is "
    "read."
)
MEDSAM_PROMPT_README = MEDSAM_PROMPT_BOX + " Applied identically to every image; no rater mask is read."

#: Sources whose per-rater model scores ship. Gated on the same licence test as `per_image.csv`, and
#: asserted in `build_per_rater` rather than assumed, since these rows stand as close to the annotations.
PER_RATER_SOURCES = ("curvas", "riga")

#: Source of record for each dataset, transcribed once here so `licences.tsv` can cite where the
#: licence was read rather than merely asserting it.
SOURCE_OF_RECORD = {
    "arvaniti": ("CC0-1.0", "Harvard Dataverse doi:10.7910/DVN/OCYCMP"),
    "c3ro": ("CC-BY-4.0", "Figshare (Wahid 2023, Sci Data)"),
    "chasedb1": ("CC-BY-4.0", "Kingston University dataset record (CHASE_DB1 retinal vessel reference dataset)"),
    "cryonuseg": ("MIT", "Kaggle ipateam/segmentation-of-nuclei-in-cryosectioned-he-images"),
    "curvas": ("CC-BY-4.0", "Zenodo (CURVAS 2024 challenge release)"),
    "cytocrowd": ("unstated", "github.com/YHSI5358/CytoCrowd-Datasets — no LICENSE file"),
    "gleason2019": ("unstated", "gleason2019.grand-challenge.org — citation request only"),
    "ima": ("CC-BY-NC-ND-4.0", "Zenodo doi:10.5281/zenodo.14201692 + ISIC Archive Terms"),
    "lidc": ("CC-BY-3.0", "Kohl 2018 LIDC crops, GCS bucket"),
    "liverhccseg": ("CC-BY-4.0", "Zenodo doi:10.5281/zenodo.7957516"),
    "mesmer": ("unstated", "publications-data GCS bundle — no licence file in the archive"),
    "msseg2016": ("LicenseRef-OFSEP-MSSEG2016-DUA", "OFSEP data access agreement, March 2021 (Shanoir)"),
    "multiorg": ("CC-BY-NC-SA-4.0", "Kaggle (Bukas 2024)"),
    "nucls": ("CC0-1.0", "github.com/PathologyDataScience/NuCLS"),
    "pdacvi": ("CC-BY-4.0", "Zenodo (CURVAS-PDACVI dataset)"),
    "qubiq21": ("unstated", "qubiq21.grand-challenge.org — participation terms only"),
    "riga": ("CC-BY-NC-4.0", "Deep Blue Data doi:10.7302/Z23R0R29"),
    "stare": ("unstated", "cecas.clemson.edu/~ahoover/stare — no terms stated"),
}

#: internal column labels replaced by descriptive names in the released comparator table
COMPARATOR_RENAME = {"n_images_view_b": "n_images_consensus"}
COMPARATOR_PARTS = [("view_a_", "pairwise_"), ("view_b_", "consensus_")]


def data_use_agreement(licence: str) -> bool:
    """A signed data use agreement that allows group-level derivatives only.

    Recorded as its own category, so that naming such a licence accurately can never make a source's
    per-image rows releasable.
    """
    return "-DUA" in licence.upper()


def per_image_released(licence: str) -> bool:
    if licence == "unstated" or data_use_agreement(licence):
        return False
    # a confirmed NoDerivatives licence still forbids redistributing derived per-image rows
    return "-ND-" not in licence.upper()


def readme_reason(reason: str) -> str:
    """The withholding reason as the README states it: the same wording as `licences.tsv`."""
    return reason.strip()


def licence_category(licence: str) -> str:
    if licence == "unstated":
        return "unstated"
    return "restrictive" if "-ND-" in licence.upper() or data_use_agreement(licence) else "permits"


def build_licences(datasets: list[str]) -> pd.DataFrame:
    rows = []
    for dataset in sorted(datasets):
        licence, where = SOURCE_OF_RECORD.get(dataset, ("unstated", "not recorded"))
        ships = per_image_released(licence)
        reason = ""
        if not ships:
            if data_use_agreement(licence):
                reason = ("data use agreement permits group-level results only; per-image and per-reader "
                          "rows not shared, task-level rows released")
            elif licence != "unstated":
                reason = ("the licence's NoDerivatives term does not permit derived per-image rows to be "
                          "redistributed")
            else:
                reason = "licence not stated; per-image rows not shared until it is confirmed"
        rows.append({
            "dataset": dataset,
            "licence": licence,
            "licence_category": licence_category(licence),
            "licence_source_of_record": where,
            "task_level_rows_released": "yes",
            "per_image_rows_released": "yes" if ships else "no",
            "withheld_reason": reason,
        })
    return pd.DataFrame(rows)


def consensus_table(floors: pd.DataFrame) -> pd.DataFrame:
    s1 = pd.read_csv(RESULTS / "s1_e4_view_a_vs_b.csv")
    types = floors[floors.metric == "dice"][["dataset", "task", "pair_class", "annotation_type"]].drop_duplicates()
    s1 = s1.merge(types, on=["dataset", "task", "pair_class"], how="left")
    columns = []
    for c in s1.columns:
        c = COMPARATOR_RENAME.get(c, c)
        for old, new in COMPARATOR_PARTS:
            c = c.replace(old, new)
        columns.append(c)
    s1.columns = columns
    s1["comparator_metric"] = "Dice and NSD@2 on the foreground (also for box and instance floors)"
    return s1


def per_rater_tables() -> list[Path]:
    """Every per-rater score table: every CSV directly under `results/per_rater`.

    Globbed rather than listed, so a table added to the experiment cannot be left out of the release by
    omission. The prediction folders (`predictions/`, `predictions_curvas_lps/`, and the default-orientation
    `predictions_curvas/`) are deliberately not globbed: the prediction masks are large, reproducible from
    the revisions recorded in `models.json`, and the layer redistributes no masks of any kind.
    """
    return sorted(PER_RATER.glob("*.csv"))


def shipped_frame(path: Path) -> pd.DataFrame:
    """Read a per-rater table and strip anything that is a fact about this machine, not about the data."""
    frame = pd.read_csv(path)
    if "image_path" in frame.columns:
        # `riga_index.csv` records where each photograph sat while inference ran. An absolute path on
        # a local disk is not a fact about RIGA, so only the part inside RIGA's own archive ships,
        # which is also the part a reader who downloads RIGA can act on.
        frame = frame.rename(columns={"image_path": "image_path_in_source"})
        frame["image_path_in_source"] = (
            frame["image_path_in_source"].astype(str)
            .str.replace(r"^.*gtu-raw[\\/]riga[\\/]", "", regex=True)
            .str.replace("\\", "/", regex=False)
        )
    return frame


def bundle_version(repo: str, revision: str | None) -> dict:
    """A MONAI bundle's own version string, read from the pinned snapshot that produced the scores.

    `_model.json` records the resolved snapshot hash, which is exact but unreadable; the bundle also
    carries a semantic version in `configs/metadata.json`, which is what the paper cites. Reading it
    out of the snapshot rather than restating it here means the two cannot disagree. Returns an empty dict
    for a model that is not a bundle, or whose snapshot is no longer cached.
    """
    import json

    if not revision:
        return {}
    snapshot = MODEL_CACHE / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    metadata = snapshot / "configs" / "metadata.json"
    if not metadata.is_file():
        return {}
    read = json.loads(metadata.read_text(encoding="utf-8"))
    found = {k: read[k] for k in ("version", "monai_version") if k in read}
    return {"bundle_version": found["version"]} | (
        {"monai_version": found["monai_version"]} if "monai_version" in found else {}
    ) if "version" in found else {}


def curvas_input_orientation(folder: Path) -> str:
    """How the CURVAS images were oriented before inference, from the reoriented inputs kept in the run.

    `per_rater_curvas.oriented_input` writes `_oriented/<case>_lps.nii.gz` for every image whose header
    carries no orientation codes; the count is read from that folder rather than restated.
    """
    n = len(list((folder / "_oriented").glob("*_lps.nii.gz")))
    assert n > 0, f"no reoriented inputs recorded in {folder}"
    return (f"LPS affine applied to the {n} images without orientation codes; outputs mapped back to the "
            f"original grid")


def build_model_registry() -> dict:
    """The record of what was run: repository, pinned revision, recipe, and what left the panel.

    Read out of the `_model.json` written beside each prediction folder, so the registry cannot drift
    from what actually produced a score. A model that was considered and not used is carried here too,
    with the reason, so that the full panel of candidate models is visible.
    """
    import json
    import sys

    sys.path.insert(0, str(ROOT / "code"))
    from per_rater_infer import MODELS as RIGA_REGISTRY

    models: dict[str, dict] = {}
    for folder in (PER_RATER / "predictions", CURVAS_PREDICTIONS):
        for record in sorted(folder.glob("*/_model.json")):
            spec = json.loads(record.read_text(encoding="utf-8"))
            spec["status"] = "included"
            spec.update(bundle_version(spec["repo"], spec.get("revision")))
            if folder == CURVAS_PREDICTIONS:
                spec["input_orientation"] = curvas_input_orientation(record.parent)
                spec["predictions_folder"] = f"results/per_rater/{CURVAS_PREDICTIONS.name}"
            models[spec["model"]] = spec
    curvas = [m for m in models.values() if m.get("predictions_folder")]
    assert len(curvas) == 3 and all(m["predictions_folder"].endswith("_lps") for m in curvas), (
        "the CURVAS registry entries must come from the correctly oriented run (predictions_curvas_lps)")
    for name, spec in RIGA_REGISTRY.items():
        if spec.get("dropped") and name not in models:
            models[name] = {"model": name, "repo": spec["repo"], "revision": None,
                            "note": spec["note"], "status": "dropped",
                            "dropped_reason": spec["dropped"],
                            "excluded_from_scoring": True}
    return {
        "release": f"per-rater model scores, v{VERSION}",
        "what_ships": "scores only - no predicted masks, no images",
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "note": "cases resampled, paired across models; the seed and draw count behind every "
                    "interval in these tables",
        },
        "medsam_prompt_rule": MEDSAM_PROMPT_RULE,
        "models": [models[name] for name in sorted(models)],
    }


#: `code/verify_curvas.py` runs inside the working tree. These substitutions repoint the shipped
#: copy at the release's own layout. Each one asserts, so if the working script moves on, the build fails
#: rather than quietly shipping a verifier that cannot run.
VERIFIER_REPOINT = [
    ('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parent.parent  # the release root'),
    ('OUT = ROOT / "results" / "per_rater"', 'OUT = ROOT / "per_rater_scores"'),
    ('floors = pd.read_csv(ROOT / "results" / "manuscript" / "s4_all_floors.csv")',
     '# the released layer, filtered to the Dice floor of the independent-rater class. `mean` here is the\n'
     '# same number the internal table calls `primary`: checked identical for all three CURVAS organs.\n'
     'floors = pd.read_csv(ROOT / "task_floor.csv")'),
    ('floors = floors[floors.dataset == "curvas"].set_index("task")',
     'floors = (floors[(floors.dataset == "curvas") & (floors.metric == "dice")\n'
     '                 & (floors.pair_class == "P1")]\n'
     '          .rename(columns={"mean": "primary"}).set_index("task"))'),
    ('disp = pd.read_csv(ROOT / "results" / "pair_dispersion.csv")',
     'disp = pd.read_csv(ROOT / "pair_dispersion.csv")'),
    ('per_image = pd.read_csv(ROOT / "results" / "layer" / "per_image.csv")',
     'per_image = pd.read_csv(ROOT / "per_image.csv")'),
]


def ship_verifier() -> None:
    """Copy `verify_curvas.py` into the release, repointed at the release's own layout."""
    lines = VERIFIER.read_text(encoding="utf-8").splitlines()
    for prefix, replacement in VERIFIER_REPOINT:
        hits = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        assert len(hits) == 1, f"verify_curvas.py: expected one line starting {prefix!r}, found {len(hits)}"
        lines[hits[0]] = replacement
    (PER_RATER_RELEASE / "verify_curvas.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


COUNTS_REPOINT = [
    ('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parent  # published_values/ in the release'),
    ('TABLE = ROOT / "results" / "e5_published_vs_recomputed.csv"', 'TABLE = ROOT / "e5_published_vs_recomputed.csv"'),
    ('COUNTS = ROOT / "results" / "manuscript" / "reproduction_counts.csv"', 'COUNTS = ROOT / "reproduction_counts.csv"'),
]


def build_published_values() -> None:
    """Write `published_values/`: the published-versus-recomputed table, the counts, and their check.

    Every row is a task-level value (a published statistic and its recomputation), so the table ships for
    every source under the same reading as `task_floor.csv`."""
    PUBLISHED_RELEASE.mkdir(parents=True)
    for source in (RESULTS / "e5_published_vs_recomputed.csv", RESULTS / "manuscript" / "reproduction_counts.csv",
                   RESULTS / "manuscript" / "c3ro_rounding.csv"):
        pd.read_csv(source).to_csv(PUBLISHED_RELEASE / source.name, index=False)
    lines = COUNTS_VERIFIER.read_text(encoding="utf-8").splitlines()
    for prefix, replacement in COUNTS_REPOINT:
        hits = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        assert len(hits) == 1, f"verify_counts.py: expected one line starting {prefix!r}, found {len(hits)}"
        lines[hits[0]] = replacement
    (PUBLISHED_RELEASE / "verify_counts.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  published_values/: e5_published_vs_recomputed.csv, reproduction_counts.csv, c3ro_rounding.csv, "
          "verify_counts.py")


def build_per_rater(ships: set[str]) -> int:
    """Write `per_rater_scores/`: the score tables, the model registry, the verifier and a README."""
    import json

    missing = [d for d in PER_RATER_SOURCES if d not in ships]
    assert not missing, (f"per-rater rows of {missing} would ship although the licence test withholds "
                         f"their per-image rows; settle the licence before releasing them")

    PER_RATER_RELEASE.mkdir(parents=True)
    tables = per_rater_tables()
    rows = 0
    for table in tables:
        frame = shipped_frame(table)
        frame.to_csv(PER_RATER_RELEASE / table.name, index=False)
        rows += len(frame)
    registry = build_model_registry()
    (PER_RATER_RELEASE / "models.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    ship_verifier()
    (PER_RATER_RELEASE / "README.md").write_text(
        PER_RATER_README.format(version=VERSION, draws=f"{BOOTSTRAP_DRAWS:,}", seed=BOOTSTRAP_SEED,
                                medsam_prompt_rule=MEDSAM_PROMPT_README, n_tables=len(tables)),
        encoding="utf-8",
    )
    print(f"  per_rater_scores/: {len(tables)} tables, {rows} rows, "
          f"{len(registry['models'])} models in the registry, plus verifier and README")
    return rows


PER_RATER_README = """# Per-reader model scores, v{version}

What three off-the-shelf segmentation models score on **every case against every individual reference
reader**, rather than against a consensus or a panel average. These are the tables behind the paper's
analysis in which public models were scored against each radiologist on the CURVAS CT scans, and against
each ophthalmologist on the RIGA fundus photographs of the comparison set.

**Scores only.** No predicted masks and no images are here, on the same rule as the rest of the layer.
Every prediction is reproducible from the repository and pinned revision recorded in `models.json`.

## Why against each reader separately

A model's score depends on which reader drew the reference. Comparing a model scored against one reader
with agreement averaged over a whole panel of readers compares unlike things, and can manufacture a
failure that disappears when like is compared with like. Averaging the readers away hides that. These
tables keep the reference reader as a column (`reference_rater`), so the comparison can be made properly.

**Read a model's margin against the agreement reached by the same reader**, not against the
panel-averaged reader agreement in `../task_floor.csv`.

## The tables

{n_tables} CSVs. In the column and file names, `rater` means reader and `floor` means the reader agreement
for the structure. The core tables:

* `model_vs_rater_<task>.csv`, `curvas_model_vs_rater_<task>.csv` — one row per
  (case, model, reference reader), with `dice` and `nsd@2`. Everything else is derived from these.
* `model_vs_consensus_<task>.csv` — the same cases scored against a STAPLE and a majority-vote
  consensus of the readers, for comparison with the per-reader columns.
* `model_ranks_*` and `rank_stability_*`, with their `_bootstrap_` counterparts — how the models order
  under each reference reader, and how often that order survives resampling.
* `model_margins_*` — each model's margin over the reader agreement, with its paired interval.
* `reference_sensitivity.csv` — how far a model's score moves when only the reference reader changes.
* `model_vs_human_level.csv` — each model beside the pairwise reader agreement and the
  reader-versus-consensus values on the same images.
* `riga_index.csv`, `riga_rater_areas.csv` — the 749 RIGA images each model saw, and each reader's marked
  area per image. `image_path_in_source` is the path **inside RIGA's own archive**.

The CURVAS analyses in the paper add these tables:

* `curvas_seed_distance.csv`, `curvas_seed_distance_summary.csv` — each radiologist's Dice against a
  regenerated TotalSegmentator starting outline and against each other radiologist, per scan and organ, and
  the paired difference per organ with its interval.
* `curvas_fixed_reference_control.csv` — with the reference radiologist held fixed and the scans resampled,
  how often the top-ranked model differs from the one on all 90 scans.
* `cases_needed.csv` — the number of scans needed to tell two models apart, n = (z s/m)², where m is the
  margin between the two models and s the standard deviation of their scan-by-scan difference, for z = 1.96 (the
  margin equals 1.96 standard errors) and z = 1.96 + 0.8416 (a 95% CI excluding zero with 80% power).
* `curvas_reader_pairs.csv` — the Dice of every CURVAS reader pair on every scan and organ, with the scan's split.
* `curvas_reference_shift.csv` — the shift in each radiologist's mean Dice when the reference radiologist changes,
  its mean and maximum, a 2000-scan interval for the mean, the per-scan spread, and how many adjacent leaderboard
  gaps lie inside each, organ-averaged and per organ, on all 90 scans and on the 65 testing-split scans.
* `curvas_gap_spacing_null.csv` — the share of adjacent gaps that seven team scores placed at random over the
  leaderboard's range would put inside each band, beside the observed share.
* `curvas_liver_margin_decomposition.csv` — for the two liver models, the paired margin per reference radiologist
  with its interval, and the variance of the margin across scan resamples against its variance across reference
  radiologists, with the sample (primary) and the population divisor.
* `curvas_model_shortfall.csv` — each model's mean Dice against each radiologist and its shortfall against reader
  agreement.
* `curvas_fixed_reference_check.csv` — how often the first-ranked model changes, beside the probability
  Φ(−m√n/s) that a margin m reverses on n scans.
* `curvas_model_vs_agreement.csv` — every model against each radiologist k, beside the mean agreement of the
  other two radiologists with k on the same scans, with paired intervals; also against the three
  radiologists averaged, and on the 65 scans of the testing split.
* `curvas_model_vs_agreement_summary.csv` — per organ: reader agreement, the best model and the difference.
* `curvas_model_consensus_region.csv` — each prediction, and each radiologist, scored as the challenge
  scores entries: Dice counted only on voxels that all readers of the consensus labelled the same way.
* `curvas_leaderboard_vs_agreement.csv` — the top leaderboard team beside pairwise reader agreement, the
  best public models scored the challenge's way, and each radiologist scored against the regions on which
  the other two agreed (an indicative comparator: a two-reader region is larger than the three-reader one).
* `riga_image_subsampling.csv` — for RIGA, how often subsets of 20 to 320 images order two models as the
  full set does, with the reference reader fixed.

## CURVAS image orientation

89 of the 90 CURVAS image files carry no orientation codes, and their arrays are stored in
left-posterior-superior order, as the one file with valid codes records. Software that falls back to a
default orientation sees the body mirrored from front to back. Every CURVAS model and starting-outline
result in this release was computed after giving these images that orientation, with outputs mapped back
to the original voxel grid; `models.json` records this for each CURVAS model (`input_orientation`).

## How to read an interval

Every interval in these tables comes from {draws} draws at seed {seed}, resampling cases and paired
across models, so two models are always compared on the same resample.

## The models

`models.json` carries, for each model, its repository, the **pinned revision** that produced the scores,
the inference recipe, the classes used per structure, and the date it ran. It also lists one model that
was considered and not used, with the reason, so that the full set of candidate models is visible.

One of the three is prompt-driven and so is defined as much by its prompt rule as by its weights. The
rule used:

> {medsam_prompt_rule}

## Check these tables yourself

`verify_curvas.py` recomputes every emitted CURVAS table from `curvas_model_vs_rater_<task>.csv`,
`curvas_reader_pairs.csv` and `curvas_model_consensus_region.csv` with plain pandas and numpy, without
importing the code that wrote them, and prints PASS or FAIL per claim:

```bash
python per_rater_scores/verify_curvas.py    # run from the release root
```

It reads `../task_floor.csv` and `../pair_dispersion.csv` for the published reader agreement and
dispersion, so it checks the released tables against the released layer.

## Licence

As for the rest of the layer: these rows are released CC BY 4.0, and each source's own licence continues
to govern anything derived from it. RIGA is CC BY-NC 4.0, so its rows may not be used commercially. See
`../licences.tsv`.
"""


REPORTING_ITEMS = """# Items to report when releasing a multireader segmentation dataset

The eleven reporting items of the article (Table 2), with the example from this study for each.

| Item | What to report | Example from this study |
|---|---|---|
| Readers | Number of readers per image, their training and experience, and a pseudonymous identifier for each reader that stays the same across images and carries no personal data | C3RO: up to 14 experts and 121 non-expert readers, reported as separate groups. PDACVI: four board-certified radiologists with 1, 3, 7 and 22 years' experience, beside the PANORAMA reference label |
| Independence | Whether any reader saw another reader's outlines before or while drawing | LIDC-IDRI: final marks were made after an unblinded review of the other readers' marks |
| Algorithm starting outline | Whether readers began from an algorithm's outline, which algorithm and version, whether the outline was shared, and how much the readers changed it | CURVAS: three radiologists corrected one shared TotalSegmentator outline per scan and agreed with a regenerated version of it nearly as closely as with one another |
| Annotation instructions | The written instructions or criteria the readers followed, and whether they were shared across readers | CURVAS: one radiologist defined the criteria and the other two followed them |
| Image and cohort description | Acquisition parameters, patient characteristics and the split each image belongs to | CURVAS: three scanner groups, with 20 training, 5 validation and 65 test scans |
| Release of each reader's outlines | Each reader's outline for every image, linked to the original image identifier, or a dataset-level summary where a licence does not allow more | Four of the six CT and MRI datasets allow per-image results to be shared |
| Definition of any agreement value | Measure, boundary tolerance, whether pairs or a consensus were compared, how values were averaged over readers and images, and whether images are volumes or single slices | C3RO reports the median over reader pairs; this study reports the mean over images; four QUBIQ 2021 structures are single 2D slices |
| Empty outlines | How pairs are scored when one or both readers drew nothing, and how often this happens | Both-empty pairs are scored as full agreement and counted separately (article, Appendix S3) |
| Uncertainty | Type of interval and what was resampled (images, patients or readers), with the number of each | CURVAS: 95% CIs from resampling the 90 scans, with the three radiologists fixed |
| Evaluation reference | For a challenge or benchmark, whether models are scored against a consensus or an individual reader, on which images, and with what restriction of the scored region | CURVAS: 60 test scans, Dice restricted to consensus regions |
| Licence | Whether derived results may be shared, at the level of each image as well as of the dataset | MSSEG-2016: dataset-level results only, under its data access agreement |

CI = confidence interval.
"""


README = """# The GTU layer, v{version}

Agreement between readers for public biomedical image segmentation datasets, measured from each reader's
released outlines, with the same definitions applied to every source.

**What this is for.** Before reading a leaderboard, look up how closely the readers who drew its reference agreed
with each other, and how closely each of them agreed with a consensus of the others. Pairwise reader agreement
describes how far the reference itself varies; it is not a ceiling for models, since a model close to the readers'
consensus is expected to exceed it. Judge a difference between two models with a paired interval on your own test
cases, and read it on the scale that reader agreement and the consensus comparator give.

**A note on names.** The tables keep their original column names. In them, `rater` means reader, and `floor`
means the reader agreement for a structure: for example, `floor_reportable` marks a reader-agreement value that can
be quoted, and `n_raters` is the number of readers.

## Look up a dataset's reader agreement

```python
import pandas as pd

floor = pd.read_csv("task_floor.csv")
floor.query("dataset == 'curvas' and metric == 'dice' and floor_reportable")[
    ["task", "mean", "ci_low", "ci_high", "n_raters", "n_images", "frac_informative"]
]
```

Each reader's agreement with a STAPLE or majority-vote consensus of the other readers, beside pairwise reader
agreement on the same images, is in `consensus_comparator.csv` for structures with three or more readers.
Per-image values are in `per_image.csv`, keyed by each source's **own** `image_id`, so you can join against the
subset you evaluate on.

## How agreement is measured

* **Pairs and aggregation.** For each image, every pair of readers who both outlined it is scored, and the image
  value is the mean over those pairs; the value for a structure is the mean over images, each image weighted
  equally. Where readers cover images unevenly, pairs are formed only among the readers present on that image.
* **Measures.** Dice and intersection over union (IoU) for overlap; the normalised surface Dice (NSD) at a
  tolerance and the symmetric 95th-percentile Hausdorff distance (HD95) for boundaries. NSD is reported at 2 px
  (with 1 and 4 px) for two-dimensional images and at 2 mm (with 1 and 3 mm) for CT and MRI volumes. Instance
  sources add instance F1 at an IoU of 0.5 (and 0.7), with one-to-one matching, and panoptic quality.
* **Empty masks.** A pair in which neither reader marked the structure scores 1 on Dice, IoU and NSD and 0 on HD95;
  a pair in which only one did scores 0, and its HD95 is undefined and left out of the HD95 mean. No pair is
  dropped for any other reason.
* **Intervals.** 95% percentile intervals from 1,000 bootstrap resamples of images (seed 20260903), resampling
  the higher unit (patient, child, video) where images are nested in one. Where a structure has fewer than ten
  images and three or more readers, readers are also resampled, and the wider interval is reported.

## Columns

Shared by the row tables:

* `dataset`, `task` — the source and the structure, in the source's own vocabulary.
* `image_id`, `rater_a`, `rater_b`, `rater_id` — the source's own image and reader identifiers, never renumbered.
* `pair_class` — how both readers of a pair worked, coded as follows:
  - `P1` readers who outlined independently, without seeing one another's outlines;
  - `P2` the same, for readers the source describes as non-experts;
  - `P3` the same reader outlining the same image twice;
  - `P4` readers who revised one another's outlines in turn;
  - `P5` outlines produced or fused by an algorithm, or algorithm outlines accepted without correction;
  - `P6` a single consensus reference;
  - `mixed` where the two readers of a pair differ.

  Only `P1` and `P2` give a reader-agreement value, and the two are never merged.
* `metric` and the metric columns — `dice`, `iou`, `hd95`, `nsd@<tolerance>`, `instance_f1@<threshold>`,
  `panoptic_quality`, as described above.
* `distance_unit` — `px` or `mm`: the unit of `hd95` and of the tolerance in every `nsd@` column for that row.
  Distances are in millimetres where the source ships physical spacing, otherwise in pixels; never pool the two.
* `cluster_id` — the unit images are nested in (patient, child, video, slide, tissue block), where the source
  documents one. Empty where images are independent.
* `source_run` — the computation run a row came from; `root` for sources computed in the first pass. It exists so
  a row can be traced to its run, and carries no meaning for analysis.

Task-level (`task_floor.csv`):

* `mean` — the reader agreement for the structure: per image the mean over reader pairs, then the mean over
  images.
* `median`, `iqr_low`, `iqr_high` — the same per-image values summarised by median and quartiles.
* `ci_low`, `ci_high`, `ci_source` — the 95% interval and how it was resampled: `image_bootstrap`,
  `cluster_bootstrap` where images are nested, or `rater_bootstrap` where too few images make an image bootstrap
  uninformative. A `rater_bootstrap` interval, and any interval over ten or fewer units, is descriptive.
* `rater_ci_low`, `rater_ci_high` — the reader-bootstrap interval where one was computed, whether or not it was
  adopted as the reported interval.
* `n_raters`, `n_images`, `n_pairs` — the readers, the images and the reader pairs behind the value.
* `n_scored`, `n_excluded_nan` — per-image values that were finite, and those that were not (a measure undefined on
  that image, such as a distance between two empty masks).
* `frac_informative` — the share of pairs in which both masks are non-empty. Pairs where neither reader marked the
  structure score 1 by definition and inflate agreement for a rare class, so read the value beside this number.
* `floor_eligible` — the pair class may give a reader-agreement value (`P1` or `P2`).
* `floor_reportable` — eligible, and at least one informative pair exists. Filter on this before quoting a value.
* `rss` — the per-scan spread: on each image, the range of a reader's score over the possible reference readers,
  averaged over readers and images. It is a within-image quantity, not on the scale of a difference between two
  means; for that, see the reference-reader shift below.
* `licence`, `modality`, `annotation_type` — the source's licence string, its imaging modality, and whether the
  annotations are pixel masks, instances or boxes. Box sources are never pooled with pixel masks.

`consensus_comparator.csv` adds each reader's agreement with a STAPLE or majority-vote consensus of the other
readers on the same images, beside pairwise reader agreement on those images, for structures with three or more
readers. Majority-vote ties are resolved to foreground, so for a three-reader panel the consensus of the two
remaining readers is their union.

**The reference-reader shift.** A leaderboard gap is a difference between two means over images. Its like-for-like
comparator is the shift in one reader's mean score when the reference reader changes: each reader's outlines are
scored against each other reader's on every image, averaged over images, and the two means for the same scored
reader are subtracted. For CURVAS it is in `per_rater_scores/curvas_reference_shift.csv` (organ-averaged and per
organ, on all 90 scans and on the 65 testing-split scans, with a 2000-scan interval), computed from the reader-pair
rows in `per_rater_scores/curvas_reader_pairs.csv`.

`pair_dispersion.csv` carries, per task and reader class, `within_image_sd`: the standard deviation of the pair
scores within an image, averaged over the images that hold at least two pairs, with `n_images_used` counting
them. A two-reader source has one pair per image and therefore no within-image spread; those rows carry an empty
`within_image_sd` and report `sd_across_images` instead, the spread of the single pair score over images. It
describes how much pair scores vary within an image; it is not the confidence interval of the agreement value. To
judge whether a difference between two models is resolved, use the number of scans needed, n ≈ (1.96 s/m)², where m
is the margin and s the standard deviation of the scan-by-scan difference between the two models (article, Appendix
S1); `per_rater_scores/cases_needed.csv` is a worked example.

**CURVAS image orientation.** 89 of the 90 CURVAS image files carry no orientation codes, and their arrays are
stored in left-posterior-superior order. Software that falls back to a default orientation sees the body mirrored
from front to back. Every CURVAS model and starting-outline result in this release was computed after giving these
images that orientation.

## What is here, and what is deliberately not

**No images and no masks.** This release contains derived statistics only. Every row is keyed by the original
dataset's identifier; obtain the underlying data from the original source under its own licence.

**Task-level rows of all {n_total} sources are released; per-image and per-reader rows of {n_released}.** The
per-image and per-reader rows of {n_withheld} sources are withheld because their licences are unstated, forbid
derivatives or permit group-level derivatives only (see `licences.tsv`). They are withheld on licensing grounds,
not quality:

{withheld_list}

If you maintain one of these sources and its terms permit publishing derived agreement statistics, tell us and the
per-image rows ship in the next version.

**Two levels of release, and what governs each.** The code is MIT licensed. The data rows are of two kinds.
*Task-level* rows carry one number per task, measure and reader group: these are summary facts about published
data, reported in the article itself, and they are released for every source. *Per-image* rows carry one value
per image and reader pair, stand closest to the annotations themselves, and are released only for the sources
whose licence explicitly permits redistribution of derived statistics; `licences.tsv` records which, with the
licence read and where it was read. We assert no licence of our own over the task-level rows beyond attribution
(CC BY 4.0), and we do not relicense anything that came from a source: where a source's licence restricts use, it
continues to govern what you may do with anything derived from it. In particular the per-image rows of RIGA
(CC BY-NC 4.0) and MultiOrg (CC BY-NC-SA 4.0) may not be used commercially, and MultiOrg's share-alike term
requires that anything you build from its rows is released under the same terms. If a use depends on the licence,
check `licences.tsv` and the original source rather than this summary.

## Files

| file | contents |
|---|---|
| `task_floor.csv` | the layer: reader agreement per (dataset, task, pair class, measure), with mean, median, IQR and CI, for every source |
| `consensus_comparator.csv` | pairwise reader agreement and the consensus comparator on the same images (three or more readers) |
| `pair_dispersion.csv` | how far reader pairs on one image sit from each other, per task |
| `per_image.csv` | per-image agreement, for sources whose licence permits it |
| `per_rater.csv` | per-reader means, same sources; description only, never a ranking of readers |
| `licences.tsv` | what each source permits, where that was read, and what ships |
| `reporting_items.md` | the eleven items to report when releasing a multireader segmentation dataset (the article's Table 2) |
| `candidate_log.csv` | the datasets identified during selection, the conditions each met or failed, and the outcome (the article's Table S1) |
| `published_values/` | each published agreement value beside its recomputation under the dataset's own and the common definition, the reproduction counts, the C3RO rounding table and `verify_counts.py`, which recounts them |
| `per_rater_scores/` | the per-reader model analysis: each model's score on each case against each individual reference reader, the registry with pinned revisions, and a verifier. See its own README |

## Per-reader model scores

`per_rater_scores/` holds what three off-the-shelf models score on each case against **each individual
reference reader**, with the model registry and an independent verifier. A model's score depends on which
reader drew the reference, so read a model against the agreement reached by **that same reader**, not against
the panel-averaged value in `task_floor.csv`. Scores only: no predicted masks and no images. That folder's
README explains the tables and how the intervals were drawn.

## Reading the numbers responsibly

* **Filter on `floor_reportable`.** It excludes rows that describe how a reference was built rather than
  independent agreement (repeat outlines by one reader, readers who revised one another's outlines, outlines
  produced by an algorithm) and any task in which no reader ever marked the structure.
* **Sources with an algorithm's starting outline** (CURVAS, MultiOrg, PDACVI, and the flood-fill outlines of
  IMA++): readers corrected or grew an automatic or seeded pre-segmentation.
* **Box sources are a separate stream.** `annotation_type == 'box'` reports box IoU and box F1; never pool it with
  pixel-mask agreement. The consensus comparator is scored on the foreground in Dice and NSD for every source, so
  for box and instance sources it is not on the same measure as the reader agreement.
* **Sparse classes inflate agreement.** Where a class is rare, most reader pairs are both empty and score 1.0.
  Every value carries `frac_informative` (the share of pairs in which both masks are non-empty); for a rare class,
  quote the value beside the agreement with both-empty pairs excluded, which the paper's supplemental material
  reports.
* **A two-reader source has no between-reader spread**; its interval comes from the image bootstrap alone.
* **QUBIQ 2021 distances are in voxels**, because its masks are released without voxel spacing.
* **RIGA's reader labels** are assumed to denote the same ophthalmologist on every image; its pairwise agreement
  does not depend on this, its per-reader rows do.

## Cite

The paper, plus **every source you actually use**: cite each source's own publication, as that source asks.
`licences.tsv` records the licence of each source and where it was read, which is where to check whether a
source attaches conditions to reuse; it does not restate the citation each source prefers.

Version {version}. Adding sources without changing any measure, aggregation or spread definition gives v1.x; a
change to any of those definitions gives v2.
"""


def main() -> None:
    floors = pd.read_csv(LAYER / "task_floor.csv")
    per_image = pd.read_csv(LAYER / "per_image.csv")
    per_rater = pd.read_csv(LAYER / "per_rater.csv")

    datasets = sorted(floors.dataset.unique())
    licences = build_licences(datasets)
    ships = set(licences[licences.per_image_rows_released == "yes"].dataset)
    withheld = sorted(set(datasets) - ships)

    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    RELEASE.mkdir(parents=True)

    floors.to_csv(RELEASE / "task_floor.csv", index=False)
    print(f"  task_floor.csv: {len(floors)} rows, all {len(datasets)} sources")
    comparator = consensus_table(floors)
    comparator.to_csv(RELEASE / "consensus_comparator.csv", index=False)
    print(f"  consensus_comparator.csv: {len(comparator)} rows")
    # The scale on which a difference between two models is read. Task-level, so it ships for
    # every source under the same reading as task_floor.csv.
    dispersion = pd.read_csv(RESULTS / "pair_dispersion.csv")
    dispersion.to_csv(RELEASE / "pair_dispersion.csv", index=False)
    print(f"  pair_dispersion.csv: {len(dispersion)} rows")
    for name, frame in (("per_image", per_image), ("per_rater", per_rater)):
        kept = frame[frame.dataset.isin(ships)]
        kept.to_csv(RELEASE / f"{name}.csv", index=False)
        print(f"  {name}.csv: {len(kept)} of {len(frame)} rows ({len(frame) - len(kept)} withheld)")

    build_per_rater(ships)
    build_published_values()
    pd.read_csv(RESULTS / "manuscript" / "candidate_log.csv").to_csv(RELEASE / "candidate_log.csv", index=False)
    print("  candidate_log.csv")

    licences.to_csv(RELEASE / "licences.tsv", sep="\t", index=False)
    (RELEASE / "reporting_items.md").write_text(REPORTING_ITEMS, encoding="utf-8")

    withheld_list = "\n".join(f"- **{row.dataset}**: {readme_reason(row.withheld_reason)}"
                              for _, row in licences[licences.per_image_rows_released == "no"].iterrows())
    (RELEASE / "README.md").write_text(
        README.format(version=VERSION, n_released=len(ships), n_total=len(datasets), n_withheld=len(withheld),
                      withheld_list=withheld_list),
        encoding="utf-8",
    )

    print(f"\nper-image rows released for {len(ships)}/{len(datasets)} sources: {sorted(ships)}")
    print(f"per-image rows withheld for {len(withheld)}: {withheld}")
    print(f"\nwrote {RELEASE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
