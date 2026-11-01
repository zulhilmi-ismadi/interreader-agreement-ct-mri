"""Reporting table for the CURVAS leaderboard pairing: the share of adjacent gaps inside the spread, its
interval, and the density-aware scale.

    python code/report_leaderboard_e2.py

The headline quantity is the share of adjacent leaderboard gaps smaller than the
reference-substitution spread. On its own it cannot be read as evidence about annotator disagreement:
with seven teams over a narrow range most adjacent gaps are small whatever
the annotators did. The manuscript therefore reports it beside the share that team spacing alone
produces and beside two scale comparisons that need no null (spread in median-gap units, spread as a
share of the score range).

Nothing is recomputed here. Every value is transcribed from the files written by
`leaderboard_curvas.py`, `leaderboard_null.py` and `sensitivity_s7_s8.py`, so that the manuscript
quotes one table whose provenance is explicit, and the table is regenerated whenever those files are.
The one derived figure is the median of the six adjacent gaps, which is checked against the
spread-over-median-gap ratio already stored in `wp_l_null_summary.csv`.

Two facts about the cases are stated rather than computed: the spread uses all 90 public CURVAS scans,
and the leaderboard was scored on 60 test cases (Riera-Marin et al. 2025, read from the docstring of
`leaderboard_curvas.py`, which records the source). The spread is therefore not restricted to the
scored cases.

Writes results/manuscript/e2_curvas_reporting.csv.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "manuscript" / "e2_curvas_reporting.csv"

PRE = "reported"
POST = "spacing control"
CONTEXT = "context"


def leaderboard_case_count() -> int:
    """The number of test cases the published leaderboard scored, as recorded with its source."""
    text = (ROOT / "code" / "leaderboard_curvas.py").read_text(encoding="utf-8")
    found = re.search(r"over (\d+) test cases", text)
    if found is None:
        raise SystemExit("leaderboard case count not found in code/leaderboard_curvas.py")
    return int(found.group(1))


def main() -> None:
    summary = pd.read_csv(RESULTS / "wp_l_curvas_summary.csv").iloc[0]
    null = pd.read_csv(RESULTS / "wp_l_null_summary.csv").set_index("leaderboard").loc["CURVAS"]
    gaps = pd.read_csv(RESULTS / "wp_l_curvas_gaps.csv").sort_values("rank_above")
    board = pd.read_csv(ROOT / "data" / "derived" / "curvas_riera2025_leaderboard.csv")
    s7 = pd.read_csv(RESULTS / "s7_leaderboards.csv")
    s7 = s7[(s7.leaderboard == "CURVAS")].set_index("statistic")

    band = float(summary.rss_band)
    n_gaps = int(summary.n_adjacent_gaps)
    n_pairs = int(summary.n_method_pairs)
    median_gap = float(np.median(gaps.gap))
    score_range = (board.dsc_percent.max() - board.dsc_percent.min()) / 100.0

    ratio_stored = float(null.median_band_over_median_gap)
    if not np.isclose(band / median_gap, ratio_stored, rtol=1e-6):
        raise SystemExit(f"median gap {median_gap} does not reproduce the stored ratio {ratio_stored}")
    if not np.isclose(band / score_range, float(null.median_band_over_range), rtol=1e-6):
        raise SystemExit("score range does not reproduce the stored band-over-range ratio")
    if int(gaps.inside_band.sum()) != round(float(summary.e2_adjacent_inside) * n_gaps):
        raise SystemExit("gap table and summary disagree on the number of gaps inside the spread")

    rows: list[dict] = []

    def add(quantity, value, status, source, ci_low=np.nan, ci_high=np.nan, count="", note=""):
        rows.append(dict(quantity=quantity, value=value, ci_low=ci_low, ci_high=ci_high, count=count,
                         status=status, source=source, note=note))

    for _, g in gaps.iterrows():
        add(f"adjacent_gap_{int(g.rank_above)}", float(g.gap), CONTEXT, "results/wp_l_curvas_gaps.csv",
            note=f"{g.team_above} {g.dsc_above:.4f} minus {g.team_below} {g.dsc_below:.4f}; "
                 f"{'inside' if g.inside_band else 'outside'} the spread; teams ordered by DSC")
    add("median_adjacent_gap", median_gap, CONTEXT, "results/wp_l_curvas_gaps.csv",
        note=f"median of the {n_gaps} adjacent gaps")
    add("score_range", score_range, CONTEXT, "data/derived/curvas_riera2025_leaderboard.csv",
        note="highest minus lowest team DSC")
    add("reference_substitution_spread", band, PRE, "results/wp_l_curvas_summary.csv",
        ci_low=float(summary.band_ci_low), ci_high=float(summary.band_ci_high),
        note="organ-averaged; 95% CI by resampling scans (B = 1000)")
    add("e2_adjacent_gaps_inside_spread", float(summary.e2_adjacent_inside), PRE,
        "results/wp_l_curvas_summary.csv",
        ci_low=float(summary.e2_ci_low), ci_high=float(summary.e2_ci_high),
        count=f"{int(gaps.inside_band.sum())}/{n_gaps}",
        note="95% CI resamples the spread over scans; published team scores are fixed")
    add("e2_density_prediction", float(null.e2_density_null), POST, "results/wp_l_null_summary.csv",
        note="expected share if the same number of teams were spread uniformly over the observed range "
             "(10,000 draws)")
    add("e2_observed_minus_density", float(null.e2_excess_over_null), POST, "results/wp_l_null_summary.csv")
    add("spread_in_median_gaps", ratio_stored, POST, "results/wp_l_null_summary.csv",
        note="spread divided by the median adjacent gap")
    add("spread_share_of_score_range", float(null.median_band_over_range), POST,
        "results/wp_l_null_summary.csv")
    add("teams_within_spread_of_a_typical_team", float(null.median_indistinguishable), POST,
        "results/wp_l_null_summary.csv")
    add("e6_all_pairs_inside_spread", float(summary.e6_all_pairs_inside), PRE,
        "results/wp_l_curvas_summary.csv",
        ci_low=float(summary.e6_ci_low), ci_high=float(summary.e6_ci_high),
        count=f"{round(float(summary.e6_all_pairs_inside) * n_pairs)}/{n_pairs}")
    if "sd" in s7.index:
        sd_row = s7.loc["sd"]
        add("e2_sd_variant", float(sd_row.e2), "sensitivity", "results/s7_leaderboards.csv",
            count=f"{round(float(sd_row.e2) * n_gaps)}/{n_gaps}",
            note=f"spread taken as the SD of directed scores ({float(sd_row.mean_band):.4f}) instead of "
                 f"the range")
    add("organ_averaged_reader_agreement", float(summary.floor_organ_averaged), PRE,
        "results/wp_l_curvas_summary.csv",
        note="pairwise radiologist Dice averaged over liver, kidney and pancreas per scan, then over scans")
    add("organ_averaging", np.nan, CONTEXT, "results/wp_l_curvas_summary.csv",
        note="the spread is computed per scan on the Dice averaged over the three organs, matching the "
             "leaderboard's single organ-averaged DSC per team")
    add("n_cases_leaderboard_scored", leaderboard_case_count(), CONTEXT,
        "code/leaderboard_curvas.py (Riera-Marin et al. 2025)",
        note="test cases on which the published team DSC was computed")
    add("n_cases_spread", int(summary.n_annotated_cases), CONTEXT, "results/wp_l_curvas_summary.csv",
        note="public scans with three radiologists' masks used for the spread; they include the scored "
             "test cases, so the spread is not restricted to them")
    add("n_teams", int(summary.n_teams), CONTEXT, "results/wp_l_curvas_summary.csv")

    table = pd.DataFrame(rows, dtype=object)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)

    e2 = table.set_index("quantity")
    print(f"CURVAS leaderboard: {int(summary.n_teams)} teams, spread {band:.4f} "
          f"[{summary.band_ci_low:.4f}, {summary.band_ci_high:.4f}]")
    print(f"  adjacent gaps: {', '.join(f'{g:.4f}' for g in gaps.gap)}; median {median_gap:.5f}")
    print(f"  adjacent gaps inside {e2.loc['e2_adjacent_gaps_inside_spread', 'count']} = {summary.e2_adjacent_inside:.4f} "
          f"[{summary.e2_ci_low:.4f}, {summary.e2_ci_high:.4f}]; density prediction "
          f"{null.e2_density_null:.4f}")
    print(f"  spread = {ratio_stored:.2f} median gaps = {null.median_band_over_range:.1%} of the range")
    print(f"  all pairs inside {e2.loc['e6_all_pairs_inside_spread', 'count']} = {summary.e6_all_pairs_inside:.4f}")
    print(f"  cases: leaderboard {leaderboard_case_count()}, spread {int(summary.n_annotated_cases)}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
