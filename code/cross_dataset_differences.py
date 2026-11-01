"""Cross-dataset differences in interreader Dice, and the interval for the liver variance ratio.

    python code/cross_dataset_differences.py

1. Differences in interreader Dice between datasets for the same organ: CURVAS liver minus LiverHccSeg liver,
   and CURVAS pancreas minus QUBIQ 2021 pancreas (task 01). The two datasets of each comparison are
   independent, so each is resampled separately (10 000 resamples, seed 20261028): CURVAS and LiverHccSeg by
   scan, QUBIQ 2021 by its paired-volume cluster. The interval is the 95% percentile interval of the
   difference of means.
2. Approximate 95% interval for the liver variance ratio of `curvas_liver_decomposition.py`: the
   between-radiologist variance (mean within-resample sample variance) is treated as a sample variance with
   2 degrees of freedom (three radiologists), and the chi-square interval for the true variance is divided
   into the across-scan variance. It reflects the sampling of three radiologists only.
3. An alternative definition: the sample variance of the three full-sample margins, and the ratio it gives.

Reads results/layer/per_image.csv and results/per_rater/curvas_liver_margin_decomposition.csv.
Writes results/manuscript/cross_dataset_differences.csv.
"""
from __future__ import annotations

import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "manuscript" / "cross_dataset_differences.csv"
SEED, B = 20261028, 10000
rows = []

d = pd.read_csv(ROOT / "results" / "layer" / "per_image.csv")
rng = np.random.default_rng(SEED)


def boot(x: pd.DataFrame, clustered: bool) -> np.ndarray:
    if not clustered:
        v = x["dice"].to_numpy()
        return v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    groups = [g["dice"].to_numpy() for _, g in x.groupby("cluster_id")]
    return np.array([np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
                     for _ in range(B)])


for organ, (da, ta), (db, tb), cl in [
        ("liver", ("curvas", "liver"), ("liverhccseg", "liver"), False),
        ("pancreas", ("curvas", "pancreas"), ("qubiq21", "pancreas_task01"), True)]:
    xa = d[(d.dataset == da) & (d.task == ta)]
    xb = d[(d.dataset == db) & (d.task == tb)]
    diff = boot(xa, False) - boot(xb, cl)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    rows.append({"quantity": f"cross_dataset_difference_{organ}", "comparison": f"{da} {ta} minus {db} {tb}",
                 "value": xa.dice.mean() - xb.dice.mean(), "ci_low": lo, "ci_high": hi,
                 "n_a": len(xa), "n_b": len(xb), "clusters_b": xb.cluster_id.nunique() if cl else len(xb),
                 "resamples": B, "seed": SEED})

dec = pd.read_csv(ROOT / "results" / "per_rater" / "curvas_liver_margin_decomposition.csv")
v_scans = float(dec["variance_across_scan_resamples_mean"].iloc[0])
v_refs = float(dec["variance_across_radiologists_sample"].iloc[0])
sig_lo, sig_hi = 2 * v_refs / chi2.ppf(0.975, 2), 2 * v_refs / chi2.ppf(0.025, 2)
rows.append({"quantity": "variance_ratio_chi2_interval", "comparison": "across-scan / across-radiologist variance",
             "value": v_scans / v_refs, "ci_low": v_scans / sig_hi, "ci_high": v_scans / sig_lo,
             "n_a": 3, "n_b": None, "clusters_b": None, "resamples": None, "seed": None})
margins = dec["margin_point"].tolist()
v_point = st.variance(margins)
rows.append({"quantity": "variance_of_full_sample_margins", "comparison": "sample variance of the three margins",
             "value": v_point, "ci_low": None, "ci_high": None, "n_a": 3, "n_b": None, "clusters_b": None,
             "resamples": None, "seed": None})
rows.append({"quantity": "variance_ratio_full_sample_margins", "comparison": "across-scan / variance of three margins",
             "value": v_scans / v_point, "ci_low": None, "ci_high": None, "n_a": 3, "n_b": None, "clusters_b": None,
             "resamples": None, "seed": None})
OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(pd.DataFrame(rows).to_string())
