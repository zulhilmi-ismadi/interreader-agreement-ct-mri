# The GTU layer, v1.1

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

**Task-level rows of all 18 sources are released; per-image and per-reader rows of 11.** The
per-image and per-reader rows of 7 sources are withheld because their licences are unstated, forbid
derivatives or permit group-level derivatives only (see `licences.tsv`). They are withheld on licensing grounds,
not quality:

- **cytocrowd**: licence not stated; per-image rows not shared until it is confirmed
- **gleason2019**: licence not stated; per-image rows not shared until it is confirmed
- **ima**: the licence's NoDerivatives term does not permit derived per-image rows to be redistributed
- **mesmer**: licence not stated; per-image rows not shared until it is confirmed
- **msseg2016**: data use agreement permits group-level results only; per-image and per-reader rows not shared, task-level rows released
- **qubiq21**: licence not stated; per-image rows not shared until it is confirmed
- **stare**: licence not stated; per-image rows not shared until it is confirmed

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

Version 1.1. Adding sources without changing any measure, aggregation or spread definition gives v1.x; a
change to any of those definitions gives v2.
