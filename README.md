# Interreader agreement and scoring in public CT and MRI segmentation datasets

Code and derived results for the article "Judging Segmentation AI against Radiologists: Interreader Agreement
and Scoring in Public CT and MRI Datasets" (release v1.1).

The code measures how closely the readers of public multireader segmentation datasets agree, from each
reader's released outlines and with one set of definitions for every dataset; checks how each reference was
built; recomputes published agreement values from the released files; and, on the CURVAS multiorgan CT
challenge, compares leaderboard gaps and public models with the change produced by using a different
radiologist as the reference.

No images or outlines are redistributed. The derived tables are in `release/gtu-layer-v1.1/`; start there if
you only need the agreement values.

## Contents

| Path | What it holds |
|---|---|
| `release/gtu-layer-v1.1/` | The derived results: agreement per structure, per image and per reader, the consensus comparator, the published values beside their recomputation, the per-reader model scores, the licence of each source, and the reporting items. See its README |
| `code/gtu/` | Data model, mask loading, agreement metrics (Dice, IoU, normalized surface Dice, HD95, instance F1, panoptic quality) and the leave-one-reader-out consensus |
| `code/loaders/` | One loader per dataset: reads the released files into the common schema and checks their structure |
| `code/*.py` | The analyses (listed below) |
| `code/manuscript/` | Descriptive values and secondary analyses that read `results/` |
| `code/tests/` | Hand-computable checks of every metric, the consensus and the mask loading |

The tables keep their original column names: `rater` means reader, and `floor` means the agreement between
readers for a structure.

## Installation

Python 3.11 or later.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt           # the agreement analyses
.venv/bin/pip install -r requirements-models.txt    # only for running the public segmentation models
python -m pytest code/tests
```

The model experiment (`per_rater_curvas.py`, `per_rater_infer.py`) needs PyTorch and MONAI and is best run in
its own environment with a GPU. Model weights are downloaded at the revisions pinned in
`release/gtu-layer-v1.1/per_rater_scores/models.json`.

## Data

Download each dataset from its source and place it under `data/gtu-raw/<name>/` (LIDC-IDRI crops under
`data/LIDC-IDRI-Kohl/`). Each loader's docstring gives the expected layout and the checks it runs. The source of
record and licence of every dataset are in `release/gtu-layer-v1.1/licences.tsv`; MSSEG-2016 is available
from OFSEP under its data access agreement.

| Name | Dataset | Imaging |
|---|---|---|
| `c3ro` | C3RO crowdsourced radiotherapy segmentations | CT |
| `curvas` | CURVAS multirater multiorgan abdominal CT | CT |
| `pdacvi` | CURVAS-PDACVI pancreatic tumour | CT |
| `liverhccseg` | LiverHccSeg | MRI |
| `msseg2016` | MSSEG-2016 multiple sclerosis lesions | MRI |
| `qubiq21` | QUBIQ 2021 (training and validation splits) | CT, MRI |
| `lidc` | LIDC-IDRI 2D crops (Kohl et al, 2018); independence check only | CT |
| `arvaniti`, `chasedb1`, `cryonuseg`, `cytocrowd`, `gleason2019`, `ima`, `mesmer`, `multiorg`, `nucls`, `riga`, `stare` | Comparison set from histology, microscopy, cytology, fundus photography and dermoscopy | — |

## Running the analyses

Run from the repository root. Each script states its inputs and outputs in its docstring.

```
# 1. Agreement for every dataset, then one merged table (results/layer/)
python code/compute_layer.py --sources c3ro,curvas,pdacvi,liverhccseg,msseg2016,qubiq21
python code/compute_layer.py --sources lidc,arvaniti,chasedb1,cryonuseg,cytocrowd,gleason2019,ima,mesmer,multiorg,nucls,riga,stare
python code/merge_layer.py

# 2. Published values (needs data/derived/ from tidy_liverhccseg.py)
python code/tidy_liverhccseg.py
python code/validate_published.py
python code/validate_msseg.py
python code/validate_mesmer_fig3d.py

# 3. Leaderboard and model experiment (CURVAS)
python code/leaderboard_curvas.py
python code/leaderboard_null.py
python code/per_rater_curvas.py --model <name> --task <organ>   # see --list; GPU environment
python code/curvas_seed_inference.py                           # regenerates the TotalSegmentator starting outline
python code/curvas_seed_screen.py

# 4. Everything that reads results/ only (any order after steps 1–3)
python code/<script>.py
```

## Where each result comes from

| Article | Scripts |
|---|---|
| Datasets identified and selected (Appendix S1, Table S1) | `census.py`, `make_tables.py`, `candidate_log.py` |
| Reader independence (Results; Appendix S1) | `provenance_checks.py`, `nucls_boundary_check.py`, `ima_tool_contrast.py`, `sensitivity_s5.py` (each reader left out), `qubiq_audit.py`, `curvas_seed_inference.py`, `curvas_seed_screen.py` |
| Interreader agreement (Table 3, Figure 2) | `compute_layer.py`, `merge_layer.py`, `two_way_bootstrap.py`, `both_drew_share.py`, `consensus_comparator_summary.py`, `kendall_tau.py`, `cross_dataset_differences.py`, `manuscript/supplementary_analyses.py` |
| Published values (Table 4, Figure 3) | `tidy_liverhccseg.py`, `validate_published.py`, `validate_msseg.py`, `validate_mesmer_fig3d.py`, `reproduction_counts.py`, `report_reproduction_tolerance.py`, `verify_counts.py` |
| Leaderboard gaps (Figure 4) | `leaderboard_curvas.py`, `leaderboard_null.py`, `report_leaderboard_e2.py`, `curvas_reference_shift.py`, `curvas_model_vs_agreement.py` |
| Reference radiologist and scans (Figure 5) | `per_rater_curvas.py`, `curvas_liver_decomposition.py`, `curvas_fixed_reference_control.py`, `cases_needed.py`, `cross_dataset_differences.py`, `verify_curvas.py` |
| Comparison set (Appendix S2) | the agreement scripts above, `per_rater_infer.py`, `per_rater_models.py`, `riga_image_subsampling.py`, `leaderboard_ctc.py` |
| Secondary and sensitivity analyses (Appendix S3) | `sensitivity.py`, `sensitivity_s5.py`, `sensitivity_s7_s8.py`, `report_s1_s3_e4.py`, `pair_dispersion.py`, `metric_reorder.py`, `layer_totals.py`, `manuscript/descriptive_counts.py`, `manuscript/descriptive_checks.py`, `manuscript/table5_per_rater.py` |
| The released tables | `make_release.py` |

The figures are drawn from the result tables these scripts write; the plotting code is not included.

## Licence

Code: MIT (see `LICENSE`). The derived tables in `release/` follow the licence of each source dataset, as listed
in `licences.tsv`; rows are withheld where a source's terms do not allow them to be shared.

## Citation

See `CITATION.cff`.
