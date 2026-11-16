"""Build the per-item paper-vs-reproduction table (Fig. 7 and Fig. 8) from artifacts.

Reads ``results/fig7_comparison.json`` (``utils/compare_with_paper.py``),
``results/paper_values_fig7_fig8.json``, the Phase 3 sweep summaries
(Fig. 8 gaps) and ``results/phase5_comparison.json`` (the claim-support
column states its verdicts), and writes ``<out>.csv`` and ``<out>.md`` with the columns
Paper item | Claim tested | Paper value | Reproduced value | Delta | Metric
agreement | Trend agreement | Claim support | Seeds | Comment.

Agreement categories (bits/site, paper metric NLL - log2(N)/n; declared with
the table, from the plot-reading precision in the paper-values file and the
h = 1 test-split spread over 20 split seeds, SD <= 0.021, LOG.md):

- ``match``: |delta| <= 0.01;
- ``within test-split spread``: test split of a randomly drawn quarter,
  0.01 < |delta| <= 2 SD of that data set's h = 1 test value over 20 split
  seeds (SD 0.0035 Promoters, 0.018 SPECT pooled, 0.021 biofam; LOG.md);
  the SPECT standard split is fixed, so it has no split spread;
- ``biofam offset (R12)``: biofam, -0.08 <= delta <= -0.02 (the documented
  model-independent data offset of about -0.05);
- ``differs``: anything else.

Trend agreement compares the sign of HMM - BE at the same (data set, split,
h) between paper and reproduction (|difference| <= 0.002 counts as a tie).
Fig. 8 rows compare the training gap in bits per sequence: ``match`` when
|delta| <= 0.06 (twice the +-0.03 reading precision), else ``differs``; the
trend column states whether the 5-sigma decision (gap above threshold)
agrees.

Usage::

    python3 utils/build_results_table.py --phase3 <E5_SWEEP_DIR> <POOLED_SWEEP_DIR> \
        --out results/paper_item_table
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
TIE_BITS = 0.002
DATASET_LABEL = {
    ("biofam", "e5_fig7"): "biofam",
    ("promoter_gene_sequences", "e5_fig7"): "Promoters",
    ("spect_heart", "e5_fig7"): "SPECT standard split (as stated)",
    ("spect_heart", "e5_fig7_spect_pooled"): "SPECT pooled split (paper-inferred)",
}
COMPARISON_ARM = {
    "biofam": "biofam",
    "Promoters": "promoter_gene_sequences",
    "SPECT standard split (as stated)": "spect_heart",
    "SPECT pooled split (paper-inferred)": "spect_heart_pooled",
}
TEST_SPLIT_SD = {
    "biofam": 0.021,
    "Promoters": 0.0035,
    "SPECT pooled split (paper-inferred)": 0.018,
}


def _h_list(values: list[int]) -> str:
    return ", ".join(str(v) for v in values) if values else "none"


def _verdict(comparison: dict, claim: str, arm: str) -> dict:
    return next(e for e in comparison["verdicts"][claim] if e["arm"] == arm)


def c3_support(phase5: dict, label: str) -> str:
    """Return the Phase 5 C3 verdicts (arm A train, arm B test) for one arm."""
    arm = COMPARISON_ARM[label]
    train = _verdict(phase5, "C3_learns_better_armA_train", arm)
    test = _verdict(phase5, "C3_generalizes_better_armB_test", arm)
    return (
        f"Phase 5 C3: train (arm A) {train['verdict']}, h = {_h_list(train['advantage_holds_at_h'])}; "
        f"test (arm B) {test['verdict']}, h = {_h_list(test['advantage_holds_at_h'])}"
    )


def c4_support(phase5: dict, label: str) -> str:
    """Return the Phase 5 C4 decision (5 sigma, and after parameter matching) for one arm."""
    entry = _verdict(phase5, "C4", COMPARISON_ARM[label])
    return (
        f"Phase 5 C4: 5 sigma at h = {_h_list(entry['five_sigma_at_h'])}; "
        f"survives parameter matching (B3) at h = {_h_list(entry['survives_parameter_matching_at_h'])}"
    )


def _sweep_family(sweep_id: str) -> str:
    return "e5_fig7_spect_pooled" if "spect_pooled" in sweep_id else "e5_fig7"


def metric_agreement(dataset: str, label: str, split: str, delta: float) -> str:
    if abs(delta) <= 0.01:
        return "match"
    if (
        split == "test"
        and label in TEST_SPLIT_SD
        and abs(delta) <= 2 * TEST_SPLIT_SD[label]
    ):
        return "within test-split spread"
    if dataset == "biofam" and -0.08 <= delta <= -0.02:
        return "biofam offset (R12)"
    return "differs"


def _sign(value: float) -> int:
    return 0 if abs(value) <= TIE_BITS else (1 if value > 0 else -1)


def fig7_rows(comparison: dict, phase5: dict) -> list[dict]:
    cells = {}
    for row in comparison["rows"]:
        if row.get("paper_plot_read") is None:
            continue
        label = DATASET_LABEL[(row["dataset"], _sweep_family(row["sweep_id"]))]
        cells[(label, row["split"], row["bond_dim"], row["model"])] = row
    out = []
    for (label, split, h, model), row in sorted(cells.items()):
        other = "hmm_baum_welch" if model == "basis_enhanced" else "basis_enhanced"
        partner = cells[(label, split, h, other)]
        hmm, be = (row, partner) if model == "hmm_baum_welch" else (partner, row)
        paper_trend = _sign(hmm["paper_plot_read"] - be["paper_plot_read"])
        ours_trend = _sign(
            hmm["reproduced_best_train_trial"] - be["reproduced_best_train_trial"]
        )
        delta = row["difference_best_train_minus_paper"]
        name = (
            "Basis-enhanced 2-gram"
            if model == "basis_enhanced"
            else "HMM (Baum-Welch, converged)"
        )
        comment = []
        if label.startswith("SPECT standard"):
            comment.append("paper values match the pooled split, not this one (R6)")
        if model == "hmm_baum_welch" and split == "train" and delta < -0.01:
            comment.append(
                "our HMM is converged; the paper's matches ~50-100 EM iterations"
            )
        out.append({
            "Paper item": f"Fig. 7 {label}, {split}, h = {h}, {name}",
            "Claim tested": "C3 (best of 10)",
            "Paper value": round(row["paper_plot_read"], 3),
            "Reproduced value": round(row["reproduced_best_train_trial"], 4),
            "Delta": round(delta, 4),
            "Metric agreement": metric_agreement(row["dataset"], label, split, delta),
            "Trend agreement": "same ranking" if paper_trend == ours_trend
            else f"different (paper {['BE worse', 'tie', 'BE better'][paper_trend + 1]}, ours {['BE worse', 'tie', 'BE better'][ours_trend + 1]})",
            "Claim support": c3_support(phase5, label),
            "Seeds": row["trials"],
            "Comment": "; ".join(comment),
        })  # fmt: skip
    return out


def fig8_rows(paper: dict, phase5: dict, phase3_dirs: list[Path]) -> list[dict]:
    gaps = paper["fig8_gap_bits_per_sequence"]
    thresholds = paper["fig8_threshold_bits_per_sequence"]
    out = []
    for sweep_dir in phase3_dirs:
        summary = json.loads((sweep_dir / "summary.json").read_text())
        family = _sweep_family(summary["sweep_id"])
        for entry in summary["fig8"]:
            dataset, h = entry["dataset"], entry["bond_dim"]
            label = DATASET_LABEL[(dataset, family)]
            if dataset not in gaps or label.startswith("SPECT standard"):
                continue
            if str(h) not in gaps[dataset] or h not in (4, 8, 12):
                continue
            ours_gap = entry["gap_bits_per_sequence_nll_bits_per_site"]
            ours_threshold = entry["lrt_5sigma_threshold_paper_df_bits_per_sequence"]
            paper_gap = gaps[dataset][str(h)]
            paper_threshold = thresholds[dataset][str(h)]
            same_decision = (ours_gap > ours_threshold) == (paper_gap > paper_threshold)
            out.append({
                "Paper item": f"Fig. 8 {label}, h = {h}, training gap (bits/sequence)",
                "Claim tested": "C4 (5 sigma)",
                "Paper value": paper_gap,
                "Reproduced value": round(ours_gap, 3),
                "Delta": round(ours_gap - paper_gap, 3),
                "Metric agreement": "match" if abs(ours_gap - paper_gap) <= 0.06 else "differs",
                "Trend agreement": f"{'same' if same_decision else 'different'} 5-sigma decision: paper {'yes' if paper_gap > paper_threshold else 'no'}, ours {'yes' if ours_gap > ours_threshold else 'no'} (threshold paper {paper_threshold}, ours {ours_threshold:.2f})",
                "Claim support": c4_support(phase5, label),
                "Seeds": 10,
                "Comment": "gap = n x (best-of-10 HMM train - best-of-10 BE train); df = h^2 - 1 (R7)",
            })  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase3", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    comparison = json.loads((RESULTS / "fig7_comparison.json").read_text())
    paper = json.loads((RESULTS / "paper_values_fig7_fig8.json").read_text())
    phase5 = json.loads((RESULTS / "phase5_comparison.json").read_text())
    rows = fig7_rows(comparison, phase5) + fig8_rows(
        paper, phase5, [p.resolve() for p in args.phase3]
    )
    columns = list(rows[0].keys())
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["Metric agreement"]] = counts.get(row["Metric agreement"], 0) + 1
    print(len(rows), "rows", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
