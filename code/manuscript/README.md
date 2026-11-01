# Descriptive values and secondary analyses

These scripts read `results/` (after the agreement layer and the analyses in `code/`) and write to
`results/manuscript/`. Python and library versions are pinned in `requirements.txt`.

```
python code/manuscript/supplementary_analyses.py   # empty-pair views, interval units, C3RO expert and non-expert groups
python code/manuscript/descriptive_counts.py       # distinct pairs, reproduction counts, modality breakdown
python code/manuscript/descriptive_checks.py       # comparator on pixel floors, identical masks, gap counts
python code/manuscript/table5_per_rater.py         # per-reader model scores beside reader agreement
```

The `values_*.md` files in `results/manuscript/` list each value with the file it comes from.
