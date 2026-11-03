"""Independent verification of the CURVAS per-rater outputs.

    python code/verify_curvas.py

Recomputes every emitted table from the raw per-pair CSVs with plain pandas/numpy, without
importing per_rater_models, and compares: the model ranks, margins, rank stability, fixed-reference
control and scans needed; each model against reader agreement; the leaderboard's top team against
agreement and against each radiologist scored the challenge's way; the reference-radiologist shift
from the released reader pairs; and the liver margin decomposition (both variance divisors) with the
scans needed for both targets. Prints PASS/FAIL per claim and exits non-zero on any failure.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent  # the release root
OUT = ROOT / "per_rater_scores"
EXPECTED = {"pancreas": 3, "liver": 2, "kidney": 1}  # admissible models per structure
CASES, RATERS = 90, 3  # all 90 public CURVAS cases, three radiologists each
fails, checks = [], 0


def check(label, ok, detail=""):
    global checks
    checks += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fails.append(label)


def close(a, b, tol=5e-5):
    return bool(np.all(np.abs(np.asarray(a, float) - np.asarray(b, float)) <= tol))


# ---------------------------------------------------------------- floors and dispersion, read raw
# the released layer, filtered to the Dice floor of the independent-rater class. `mean` here is the
# same number the internal table calls `primary`: checked identical for all three CURVAS organs.
floors = pd.read_csv(ROOT / "task_floor.csv")
floors = (floors[(floors.dataset == "curvas") & (floors.metric == "dice")
                 & (floors.pair_class == "P1")]
          .rename(columns={"mean": "primary"}).set_index("task"))
disp = pd.read_csv(ROOT / "pair_dispersion.csv")
disp = disp[(disp.dataset == "curvas") & (disp.metric == "dice") & disp.within_image_sd.notna()]
# per-image reader agreement (mean over the three reader pairs on each scan), for the model-versus-agreement checks
per_image = pd.read_csv(ROOT / "per_image.csv")
per_image = per_image[(per_image.dataset == "curvas") & (per_image.pair_class == "P1")]
DRAWS, SEED = 2000, 20260917

for task, n_models in EXPECTED.items():
    print(f"\n=== {task} ({n_models} model{'s' if n_models > 1 else ''}) ===")
    raw = pd.read_csv(OUT / f"curvas_model_vs_rater_{task}.csv")

    # ---- shape of the raw table
    check(f"{task}: row count == {CASES * RATERS * n_models}",
          len(raw) == CASES * RATERS * n_models, f"{len(raw)} rows")
    check(f"{task}: {CASES} distinct cases", raw.image_id.nunique() == CASES,
          f"{raw.image_id.nunique()}")
    check(f"{task}: {RATERS} distinct raters", raw.reference_rater.nunique() == RATERS,
          f"{raw.reference_rater.nunique()}")
    check(f"{task}: {n_models} distinct models", raw.model.nunique() == n_models,
          f"{sorted(raw.model.unique())}")
    check(f"{task}: every (case, model, rater) appears exactly once",
          not raw.duplicated(["image_id", "model", "reference_rater"]).any())
    check(f"{task}: no missing dice values", raw.dice.notna().all())
    check(f"{task}: dice within [0, 1]", bool(raw.dice.between(0, 1).all()))
    check(f"{task}: task column is '{task}' throughout", (raw.task == task).all())
    check(f"{task}: dataset column is 'curvas' throughout", (raw.dataset == "curvas").all())

    # ---- ranks recomputed from raw
    mine = raw.groupby(["reference_rater", "model"]).dice.mean().reset_index()
    mine["rank"] = mine.groupby("reference_rater").dice.rank(ascending=False, method="min").astype(int)
    emitted = pd.read_csv(OUT / f"curvas_model_ranks_{task}.csv")
    merged = emitted.merge(mine, on=["reference_rater", "model"], suffixes=("_e", "_m"))
    check(f"{task}: ranks table covers every (rater, model)", len(merged) == len(emitted) == len(mine),
          f"emitted {len(emitted)}, mine {len(mine)}")
    check(f"{task}: mean_score matches recomputation", close(merged.mean_score, merged.dice),
          f"max delta {np.abs(merged.mean_score - merged.dice).max():.2e}")
    check(f"{task}: rank matches recomputation", (merged["rank_e"] == merged["rank_m"]).all())

    # ---- per-image reader agreement reproduces the published value for the organ
    floor = float(floors.loc[task, "primary"])
    agree = per_image[per_image.task == task].set_index("image_id").dice
    check(f"{task}: per-image reader agreement covers the {CASES} cases", agree.index.nunique() == CASES)
    check(f"{task}: mean per-image reader agreement == published value {floor:.4f}", close(agree.mean(), floor))

    if n_models < 2:
        for suffix in ("rank_stability", "rank_stability_bootstrap", "model_margins",
                       "model_ranks_bootstrap"):
            p = OUT / f"curvas_{suffix}_{task}.csv"
            check(f"{task}: no {suffix} file emitted (one model, no ordering invented)",
                  not p.exists())
        continue

    # ---- dispersion, read straight from pair_dispersion.csv
    rows = disp[disp.task == task]
    pref = rows[rows.pair_class == "P1"]
    spread = float((pref if not pref.empty else rows).iloc[0].within_image_sd)

    # ---- margins recomputed from raw
    emargin = pd.read_csv(OUT / f"curvas_model_margins_{task}.csv")
    ok_margin = ok_disp = ok_flag = ok_best = True
    for _, row in emargin.iterrows():
        block = mine[mine.reference_rater == row.reference_rater].sort_values("dice", ascending=False)
        m = float(block.iloc[0].dice - block.iloc[1].dice)
        ok_margin &= close(row.margin, m)
        ok_best &= (row.best_model == block.iloc[0].model) and (row.runner_up == block.iloc[1].model)
        ok_disp &= close(row.rater_pair_dispersion, spread)
        ok_flag &= bool(row.margin_exceeds_dispersion) == bool(m > spread)
    check(f"{task}: margin matches best-minus-runner-up", ok_margin)
    check(f"{task}: best_model / runner_up match the ordering", ok_best)
    check(f"{task}: rater_pair_dispersion == pair_dispersion.csv ({spread:.4f})", ok_disp)
    check(f"{task}: margin_exceeds_dispersion is the stated comparison", ok_flag)

    # ---- point stability recomputed
    estab = pd.read_csv(OUT / f"curvas_rank_stability_{task}.csv")
    wide = mine.pivot(index="model", columns="reference_rater", values="dice").dropna()
    ok_tau = ok_win = True
    for _, row in estab.iterrows():
        a, b = wide[row.rater_a], wide[row.rater_b]
        ok_tau &= close(row.kendall_tau, kendalltau(a, b).statistic)
        ok_win &= bool(row.same_winner) == bool(a.idxmax() == b.idxmax())
    check(f"{task}: point Kendall tau matches", ok_tau)
    check(f"{task}: same_winner matches", ok_win)

    # ---- bootstrap reproduced with the declared seed and draws
    draws, seed = 2000, 20260917
    piv = raw.pivot_table(index="image_id", columns=["model", "reference_rater"], values="dice")
    values, cols = piv.to_numpy(float), list(piv.columns)
    present, filled = ~np.isnan(values), np.nan_to_num(values)
    rng = np.random.default_rng(seed)
    resampled = np.empty((draws, values.shape[1]))
    for d in range(draws):
        pick = rng.integers(0, values.shape[0], values.shape[0])
        totals, counts = filled[pick].sum(axis=0), present[pick].sum(axis=0)
        resampled[d] = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    pos = {c: i for i, c in enumerate(cols)}
    models = sorted(raw.model.unique())
    eboot = pd.read_csv(OUT / f"curvas_rank_stability_bootstrap_{task}.csv")
    ok_rate = ok_btau = True
    for _, row in eboot.iterrows():
        A = resampled[:, [pos[(m, row.rater_a)] for m in models]]
        B = resampled[:, [pos[(m, row.rater_b)] for m in models]]
        rate = float((A.argmax(axis=1) == B.argmax(axis=1)).mean())
        taus = np.array([kendalltau(A[d], B[d]).statistic for d in range(draws)])
        ok_rate &= close(row.same_winner_rate, rate, 5e-4)
        ok_btau &= close(row.kendall_tau, float(np.nanmean(taus)), 5e-3)
    check(f"{task}: bootstrap same_winner_rate reproduces at seed {seed}", ok_rate)
    check(f"{task}: bootstrap mean tau reproduces", ok_btau)

    # ---- fixed-reference control: same resamples, reference radiologist held fixed
    control = pd.read_csv(OUT / "curvas_fixed_reference_control.csv")
    control = control[control.task == task].set_index("reference_radiologist")
    ok_ctrl = len(control) == RATERS
    for rater in sorted(raw.reference_rater.unique()):
        col = [pos[(m, rater)] for m in models]
        full = int(np.nanmean(values[:, col], axis=0).argmax())
        changes = int((resampled[:, col].argmax(axis=1) != full).sum())
        ok_ctrl &= (control.loc[rater, "full_sample_winner"] == models[full]
                    and int(control.loc[rater, "winner_changes"]) == changes)
    check(f"{task}: fixed-reference control (winner changes per radiologist) reproduces", ok_ctrl,
          ", ".join(str(int(v)) for v in control.winner_changes))

    # ---- scans needed to tell the two models apart, n = (1.96 s / m)^2 (two-model structures)
    if n_models == 2:
        needed = pd.read_csv(OUT / "cases_needed.csv").set_index("reference_radiologist")
        ok_n = len(needed) == RATERS
        for rater, block in raw.groupby("reference_rater"):
            wide2 = block.pivot_table(index="image_id", columns="model", values="dice").dropna()
            better, other = wide2.mean().sort_values(ascending=False).index[:2]
            diff = wide2[better] - wide2[other]
            m, s = float(diff.mean()), float(diff.std(ddof=1))
            n = (1.96 * s / m) ** 2
            ok_n &= (close(needed.loc[rater, "margin_m"], m, 1e-9) and close(needed.loc[rater, "per_case_sd_s"], s, 1e-9)
                     and int(needed.loc[rater, "n_needed_rounded"]) == int(round(n)))
        check(f"{task}: cases_needed.csv (m, s and n per radiologist) reproduces", ok_n,
              ", ".join(str(int(v)) for v in needed.n_needed_rounded))

# ---------------------------------------------------------------- models against reader agreement
print("\n=== models against reader agreement (curvas_model_vs_agreement_summary.csv, 90 scans, Dice) ===")
summary = pd.read_csv(OUT / "curvas_model_vs_agreement_summary.csv")
summary = summary[(summary.case_set == "all_90") & (summary.metric == "dice")].set_index("organ")
first = pd.read_csv(OUT / "curvas_model_vs_rater_liver.csv")
cases = sorted(first.image_id.unique())
picks = np.random.default_rng(SEED).integers(0, len(cases), (DRAWS, len(cases)))


def interval(v):
    d = v[picks].mean(axis=1)
    return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


best_vectors, agree_vectors = {}, {}
for task in EXPECTED:
    raw = pd.read_csv(OUT / f"curvas_model_vs_rater_{task}.csv")
    agree = per_image[per_image.task == task].set_index("image_id").dice.reindex(cases).to_numpy(float)
    per_case = raw.groupby(["model", "image_id"]).dice.mean()  # each model, references averaged, per scan
    by_model = {m: per_case.loc[m].reindex(cases).to_numpy(float) for m in sorted(raw.model.unique())}
    best = max(by_model, key=lambda m: by_model[m].mean())
    best_vectors[task], agree_vectors[task] = by_model[best], agree
    row = summary.loc[task]
    d, lo, hi = interval(by_model[best] - agree)
    check(f"{task}: best model is {best}", row.best_model == best)
    check(f"{task}: reader agreement and best-model score reproduce",
          close(row.reader_agreement, agree.mean(), 6e-5) and close(row.best_model_score, by_model[best].mean(), 6e-5),
          f"{row.reader_agreement:.4f}, {row.best_model_score:.4f}")
    check(f"{task}: difference and paired 95% CI reproduce at seed {SEED}",
          close([row.difference, row.difference_ci_low, row.difference_ci_high], [d, lo, hi], 6e-5),
          f"{row.difference:+.4f} ({row.difference_ci_low:+.4f}, {row.difference_ci_high:+.4f})")
    check(f"{task}: the best model falls short of reader agreement (CI below zero)", hi < 0)

row = summary.loc["organ_average"]
d, lo, hi = interval(np.mean([best_vectors[t] for t in EXPECTED], axis=0)
                     - np.mean([agree_vectors[t] for t in EXPECTED], axis=0))
check("organ average: difference and paired 95% CI reproduce",
      close([row.difference, row.difference_ci_low, row.difference_ci_high], [d, lo, hi], 6e-5),
      f"{row.difference:+.4f} ({row.difference_ci_low:+.4f}, {row.difference_ci_high:+.4f})")

# ---------------------------------------------------------------- the leaderboard's top team against agreement
print("\n=== leaderboard top team against reader agreement (curvas_leaderboard_vs_agreement.csv, 90 scans) ===")
board = pd.read_csv(OUT / "curvas_leaderboard_vs_agreement.csv")
board = board[board.case_set == "all_90"]
top = float(board.top_team_dsc.iloc[0])
agree_avg = np.mean([agree_vectors[t] for t in EXPECTED], axis=0)
a, a_lo, a_hi = interval(agree_avg)
row = board[board.quantity == "pairwise reader agreement"].iloc[0]
check("pairwise reader agreement, organ-averaged, and its 95% CI reproduce",
      close([row.value, row.ci_low, row.ci_high], [a, a_lo, a_hi], 6e-5), f"{row.value:.4f} ({row.ci_low:.4f}, {row.ci_high:.4f})")
check("top team minus agreement and its 95% CI reproduce",
      close([row.top_team_minus_value, row.top_team_minus_value_ci_low, row.top_team_minus_value_ci_high],
            [top - a, top - a_hi, top - a_lo], 6e-5),
      f"{row.top_team_minus_value:+.4f} ({row.top_team_minus_value_ci_low:+.4f}, {row.top_team_minus_value_ci_high:+.4f})")
region = pd.read_csv(OUT / "curvas_model_consensus_region.csv")
region = region[region.kind == "radiologist"]
rad_vectors = {}
for scorer in sorted(region.scorer.unique()):
    per_organ = [region[(region.scorer == scorer) & (region.organ == o)].set_index("image_id")
                 .dice_consensus_region.reindex(cases).to_numpy(float) for o in EXPECTED]
    rad_vectors[scorer] = np.mean(per_organ, axis=0)
rad_vectors["mean_of_three"] = np.mean(
    [region[region.organ == o].groupby("image_id").dice_consensus_region.mean().reindex(cases).to_numpy(float)
     for o in EXPECTED], axis=0)
ok_rad = True
for scorer, v in rad_vectors.items():
    r = board[(board.quantity.str.startswith("radiologist")) & (board.scorer == scorer)].iloc[0]
    m, lo, hi = interval(v)
    ok_rad &= close([r.value, r.ci_low, r.ci_high, r.top_team_minus_value], [m, lo, hi, top - m], 6e-5)
check("each radiologist scored the challenge's way, and their mean, reproduce with intervals", ok_rad,
      ", ".join(f"{v.mean():.4f}" for v in rad_vectors.values()))

# ---------------------------------------------------------------- reference-radiologist shift, from the reader pairs
print("\n=== reference-radiologist shift (curvas_reader_pairs.csv -> curvas_reference_shift.csv) ===")
reader_pairs = pd.read_csv(OUT / "curvas_reader_pairs.csv")
shift_table = pd.read_csv(OUT / "curvas_reference_shift.csv").set_index(["basis", "subset"])
ok_rows = True
for subset, keep in (("all 90 scans", None), ("65 testing-split scans", "testing_set")):
    rp = reader_pairs if keep is None else reader_pairs[reader_pairs.split == keep]
    for basis in ("organ-averaged", "liver", "kidney", "pancreas"):
        df = (rp.groupby(["image_id", "rater_a", "rater_b"]).dice.mean().reset_index() if basis == "organ-averaged"
              else rp[rp.task == basis])
        both = pd.concat([df, df.rename(columns={"rater_a": "rater_b", "rater_b": "rater_a"})])
        wide = both.pivot_table(index="image_id", columns=["rater_a", "rater_b"], values="dice")
        means = wide.mean()
        raters = sorted(set(df.rater_a) | set(df.rater_b))
        shifts, spreads = [], []
        for p in raters:
            q, r = [x for x in raters if x != p]
            shifts.append(abs(means[(p, q)] - means[(p, r)]))
            spreads.append((wide[(p, q)] - wide[(p, r)]).abs())
        spread = float(pd.concat(spreads, axis=1).mean(axis=1).mean())
        e = shift_table.loc[(basis, subset)]
        ok = close([e.shift_r1, e.shift_r2, e.shift_r3, e.shift_mean, e.shift_max, e.per_scan_spread],
                   [*shifts, np.mean(shifts), max(shifts), spread], 1e-9) and int(e.n_scans) == len(wide)
        ok_rows &= ok
        if basis == "organ-averaged" and subset == "all 90 scans":
            check("organ-averaged shifts of the three radiologists, their mean and maximum reproduce (90 scans)", ok,
                  f"{', '.join(f'{s:.4f}' for s in shifts)}; mean {np.mean(shifts):.4f}, max {max(shifts):.4f}")
            rng = np.random.default_rng(SEED)
            resampled = []
            for _ in range(DRAWS):
                w = wide.iloc[rng.integers(0, len(wide), len(wide))].mean()
                resampled.append(np.mean([abs(w[(p, [x for x in raters if x != p][0])]
                                              - w[(p, [x for x in raters if x != p][1])]) for p in raters]))
            lo, hi = np.percentile(resampled, [2.5, 97.5])
            check(f"2000-scan interval of the mean shift reproduces at seed {SEED}",
                  close([e.shift_mean_ci_low, e.shift_mean_ci_high], [lo, hi], 1e-9), f"({lo:.4f}, {hi:.4f})")
check("shift, maximum and per-scan spread reproduce for every organ and for the testing split", ok_rows)

# ---------------------------------------------------------------- liver margin: scans against reference radiologist
print("\n=== liver margin decomposition (curvas_liver_margin_decomposition.csv) ===")
liver = pd.read_csv(OUT / "curvas_model_vs_rater_liver.csv")
dec = pd.read_csv(OUT / "curvas_liver_margin_decomposition.csv")
first_m, second_m = dec["first"].iloc[0], dec["second"].iloc[0]
piv = liver.pivot_table(index="image_id", columns=["model", "reference_rater"], values="dice")
refs = sorted(liver.reference_rater.unique())
per_scan = np.stack([(piv[(first_m, r)] - piv[(second_m, r)]).to_numpy(float) for r in refs], axis=1)
rng = np.random.default_rng(SEED)
margins = np.array([per_scan[rng.integers(0, len(per_scan), len(per_scan))].mean(axis=0) for _ in range(DRAWS)])
v_scans = margins.var(axis=0, ddof=1).mean()
v_sample, v_pop = margins.var(axis=1, ddof=1).mean(), margins.var(axis=1, ddof=0).mean()
check("paired liver margins per radiologist reproduce", close(dec.margin_point, per_scan.mean(axis=0), 1e-9),
      ", ".join(f"{m:.4f}" for m in per_scan.mean(axis=0)))
check("variance across scan resamples reproduces", close(dec.variance_across_scan_resamples_mean.iloc[0], v_scans, 1e-12),
      f"{v_scans:.3e}")
check("variance ratio with the sample divisor reproduces and is 44.9 (+/- 0.5)",
      close(dec.variance_ratio_sample.iloc[0], v_scans / v_sample, 1e-6) and abs(v_scans / v_sample - 44.9) <= 0.5,
      f"{v_scans / v_sample:.1f} ({v_sample:.3e})")
check("variance ratio with the population divisor reproduces and is 67.3 (+/- 0.5)",
      close(dec.variance_ratio_population.iloc[0], v_scans / v_pop, 1e-6) and abs(v_scans / v_pop - 67.3) <= 0.5,
      f"{v_scans / v_pop:.1f} ({v_pop:.3e})")

# ---------------------------------------------------------------- scans needed, both targets
needed = pd.read_csv(OUT / "cases_needed.csv").set_index("reference_radiologist")
z80 = 1.959963984540054 + 0.8416212335729143
ok_both = True
for i, r in enumerate(refs):
    m, s = per_scan[:, i].mean(), per_scan[:, i].std(ddof=1)
    ok_both &= close([needed.loc[r, "n_needed"], needed.loc[r, "n_ci_80_power"]], [(1.96 * s / m) ** 2, (z80 * s / m) ** 2], 1e-6)
check("scans needed for both targets (margin = 1.96 SE; 95% CI excluding zero with 80% power) reproduce", ok_both,
      "; ".join(f"{needed.loc[r, 'n_needed']:.1f} and {needed.loc[r, 'n_ci_80_power']:.1f}" for r in refs))

print(f"\n{checks} checks, {len(fails)} failed")
for f in fails:
    print(f"  FAILED: {f}")
sys.exit(1 if fails else 0)
