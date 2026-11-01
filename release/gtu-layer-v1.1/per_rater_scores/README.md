# Per-reader model scores, v1.1

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

47 CSVs. In the column and file names, `rater` means reader and `floor` means the reader agreement
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

Every interval in these tables comes from 2,000 draws at seed 20260917, resampling cases and paired
across models, so two models are always compared on the same resample.

## The models

`models.json` carries, for each model, its repository, the **pinned revision** that produced the scores,
the inference recipe, the classes used per structure, and the date it ran. It also lists one model that
was considered and not used, with the reason, so that the full set of candidate models is visible.

One of the three is prompt-driven and so is defined as much by its prompt rule as by its weights. The
rule used:

> A square box covering 18% of the frame, centred on the brightest region of the green channel - the usual optic-nerve-head cue in a fundus photograph. Applied identically to every image; no rater mask is read.

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
