"""Build the Fig. 7 / Fig. 8 / Fig. 13 summary of an E5 sweep from structured artifacts.

Reads ``sweep_status.json`` and every COMPLETED run's ``metrics.json`` (never
logs) and writes ``summary.json`` and ``summary.csv`` next to the status file.

Per (dataset, model, h) cell over the 10 trials (seeds):

- ``best_train``: the trial with the lowest final train NLL, ties within
  1e-6 bits going to the lowest seed (the paper plots "the minimum achieved
  loss over ten trials", p14); its train and test values are reported, so the
  test value is not selected on test;
- ``min_test``: the lowest final test value over trials (the reading in which
  the test panel is itself a minimum over trials; selection on test, reported
  only to reproduce the figure, LOG.md R5);
- mean, standard deviation and SEM over trials (Fig. 13, p23).

Metrics are per-site NLL in bits, the KL to the split's empirical
distribution, and the paper's plotted estimate NLL - log2(N)/n (R1, calibrated
in Phase 3; lib/metrics.py). Fig. 8 (p14): gap = n x (best HMM train value -
best basis-enhanced train value) in bits per sequence, and the 5-sigma Wilks
threshold with two df rules: the free-parameter difference planned before the
sweep, and df = h^2 - 1, which reproduces the paper's solid curves (R7).

Phase 5 sweeps (``sweep_status.json`` with ``planned_runs``) are summarised
per (dataset, config variant, model label, h, hyperparameters) at each run's
recorded checkpoint (``metrics.json`` key ``checkpoint``: the final epoch, or
the best-validation epoch in arm B), with the validation split added and the
best-of-10 also selected on validation (``best_validation_*``, same tie
rule). Model label ``basis_enhanced_real`` is the basis-enhanced kind with
field "real" (control B4).

Usage::

    python3 utils/summarize_fig7.py <SWEEP_DIR>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from lib.metrics import (  # noqa: E402
    likelihood_ratio_threshold_bits,
    paper_inferred_lrt_degrees_of_freedom,
    paper_kl_estimate_bits_per_site,
)

METRICS = (
    "nll_bits_per_site",
    "kl_to_empirical_bits_per_site",
    "paper_kl_estimate_bits_per_site",
)
SPLITS = ("train", "test")
# Pre-declared in the E5 sweep plan (LOG.md): ties within 1e-6 bits go to the lowest seed.
TIE_TOLERANCE_BITS = 1e-6
# Phase 5 amendment (LOG.md): a basis-enhanced run whose unitarity error at the
# reported checkpoint exceeds this is INVALID (its likelihood is not normalised).
UNITARITY_TOLERANCE = 1e-6


def select_best_train_trial(trials: list[dict]) -> dict:
    """Apply the pre-declared best-of-10 rule of the E5 sweep plan (LOG.md).

    The trial with the lowest final train NLL; trials within
    ``TIE_TOLERANCE_BITS`` of that minimum are tied and the lowest seed wins.
    """
    lowest = min(t["metrics"]["final"]["train"]["nll_bits_per_site"] for t in trials)
    tied = [
        t
        for t in trials
        if t["metrics"]["final"]["train"]["nll_bits_per_site"] - lowest
        <= TIE_TOLERANCE_BITS
    ]
    return min(tied, key=lambda t: t["seed"])


def _stats(values: list[float]) -> dict:
    count = len(values)
    mean = sum(values) / count
    std = (
        math.sqrt(sum((v - mean) ** 2 for v in values) / (count - 1))
        if count > 1
        else 0.0
    )
    return {
        "mean": mean,
        "std": std,
        "sem": std / math.sqrt(count) if count else float("nan"),
    }


def select_best_validation_trial(trials: list[dict]) -> dict:
    """Best-of-10 on validation NLL at the checkpoint, same tie rule as on train."""
    lowest = min(t["evaluation"]["validation"]["nll_bits_per_site"] for t in trials)
    tied = [
        t
        for t in trials
        if t["evaluation"]["validation"]["nll_bits_per_site"] - lowest
        <= TIE_TOLERANCE_BITS
    ]
    return min(tied, key=lambda t: t["seed"])


def _write(sweep_dir: Path, summary: dict, rows: list[dict]) -> None:
    (sweep_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    columns = sorted({key for row in rows for key in row})
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            [
                {k: json.dumps(v) if isinstance(v, dict) else v for k, v in r.items()}
                for r in rows
            ]
        )


def _cell_key(candidate: dict) -> tuple:
    return (
        candidate["dataset"],
        candidate["config_variant"],
        candidate["model"],
        candidate["bond_dim"],
        json.dumps(candidate["hparams"], sort_keys=True),
    )


def summarize_planned(sweep_dir: Path, status: dict) -> dict:
    """Summarise a spec sweep (Phase 5) cell by cell at the recorded checkpoint."""
    expected: dict[tuple, set] = {}
    for planned in status["planned_runs"]:
        expected.setdefault(_cell_key(planned), set()).add(planned["seed"])
    trials_by_cell: dict[tuple, list[dict]] = {}
    invalid: dict[tuple, list[dict]] = {}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        planned = run["sweep_candidate"]
        label = metrics["model_kind"] + (
            "_real"
            if metrics["model_kind"] == "basis_enhanced" and metrics["field"] == "real"
            else ""
        )
        if (label, metrics["dataset"], metrics["bond_dim"]) != (
            planned["model"],
            planned["dataset"],
            planned["bond_dim"],
        ):
            raise ValueError(f"{run['metrics_path']} does not match its planned cell")
        evaluation = metrics["checkpoint"]["evaluation"]
        drift = metrics["history"][metrics["checkpoint"]["index"]].get(
            "unitarity_error", 0.0
        )
        if drift > UNITARITY_TOLERANCE:
            invalid.setdefault(_cell_key(planned), []).append(
                {"seed": run["seed"], "unitarity_error_at_checkpoint": drift}
            )
            continue
        for final in evaluation.values():
            final["paper_kl_estimate_bits_per_site"] = paper_kl_estimate_bits_per_site(
                final["nll_bits_per_site"],
                final["num_sequences"],
                metrics["sequence_length"],
            )
            if not all(math.isfinite(final[m]) for m in METRICS):
                raise ValueError(f"Non-finite metric in {run['metrics_path']}")
        trials_by_cell.setdefault(_cell_key(planned), []).append(
            {
                "seed": run["seed"],
                "metrics": metrics,
                "evaluation": evaluation,
                "step": metrics["checkpoint"]["step"],
            }
        )

    rows = []
    for key, seeds_expected in expected.items():
        dataset, variant, model, bond_dim, hparams = key
        trials = trials_by_cell.get(key, [])
        seeds = sorted(t["seed"] for t in trials)
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds in cell {key}")
        row = {
            "dataset": dataset,
            "config_variant": variant,
            "arm": dataset + ("_pooled" if "_pooled" in variant else ""),
            "model": model,
            "bond_dim": bond_dim,
            "hparams": json.loads(hparams),
            "successful_runs": len(trials),
            "expected_runs": len(seeds_expected),
            "status": "COMPLETE" if set(seeds) == seeds_expected else "PARTIAL",
            "invalid_runs": invalid.get(key, []),
        }
        if trials:
            first = trials[0]["metrics"]
            splits = [
                s
                for s in ("train", "validation", "test")
                if s in trials[0]["evaluation"]
            ]
            row["sequence_length"] = first["sequence_length"]
            row["free_parameter_count"] = first["free_parameter_count"]
            row["checkpoint_rule"] = first["checkpoint"]["rule"]
            row["mean_checkpoint_step"] = sum(t["step"] for t in trials) / len(trials)
            row["wall_seconds_total"] = sum(
                t["metrics"]["wall_seconds"] for t in trials
            )
            row["num_train"] = trials[0]["evaluation"]["train"]["num_sequences"]
            best_train = min(
                [
                    t
                    for t in trials
                    if t["evaluation"]["train"]["nll_bits_per_site"]
                    - min(x["evaluation"]["train"]["nll_bits_per_site"] for x in trials)
                    <= TIE_TOLERANCE_BITS
                ],
                key=lambda t: t["seed"],
            )
            row["best_train_seed"] = best_train["seed"]
            best_validation = (
                select_best_validation_trial(trials) if "validation" in splits else None
            )
            if best_validation:
                row["best_validation_seed"] = best_validation["seed"]
            for split in splits:
                for metric in METRICS:
                    values = [t["evaluation"][split][metric] for t in trials]
                    row[f"best_train_{split}_{metric}"] = best_train["evaluation"][
                        split
                    ][metric]
                    if best_validation:
                        row[f"best_validation_{split}_{metric}"] = best_validation[
                            "evaluation"
                        ][split][metric]
                    row[f"min_{split}_{metric}"] = min(values)
                    for name, value in _stats(values).items():
                        row[f"{name}_{split}_{metric}"] = value
        rows.append(row)
    rows.sort(
        key=lambda r: (
            r["arm"],
            r["config_variant"],
            r["model"],
            json.dumps(r["hparams"], sort_keys=True),
            r["bond_dim"],
        )
    )
    complete = all(row["status"] == "COMPLETE" for row in rows)
    summary = {
        "sweep_id": status["sweep_id"],
        "coverage": "COMPLETE SWEEP" if complete else "PARTIAL SWEEP",
        "cells": rows,
        "fig8": [],
        "generated_by": "utils/summarize_fig7.py",
    }
    _write(sweep_dir, summary, rows)
    return summary


def summarize(sweep_dir: Path) -> dict:
    """Return the summary mapping and write summary.json / summary.csv."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    # Spec-driven sweeps (tuning, finals) list their candidates as ``blocks``;
    # the plain Fig. 7 sweeps keep the Fig. 7 / Fig. 8 format below, even
    # though the coordinator now also records ``planned_runs`` for them.
    if "blocks" in status["plan"]:
        return summarize_planned(sweep_dir, status)
    plan = status["plan"]
    cells: dict[tuple, list[dict]] = {}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        for split in SPLITS:
            final = metrics["final"][split]
            final["paper_kl_estimate_bits_per_site"] = paper_kl_estimate_bits_per_site(
                final["nll_bits_per_site"],
                final["num_sequences"],
                metrics["sequence_length"],
            )
        values = [
            metrics["final"][split][metric] for split in SPLITS for metric in METRICS
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite metric in {run['metrics_path']}")
        key = (metrics["dataset"], metrics["model_kind"], metrics["bond_dim"])
        cells.setdefault(key, []).append({"seed": run["seed"], "metrics": metrics})

    rows = []
    expected_seeds = set(plan["seeds"])
    for dataset in plan["datasets"]:
        for kind in plan["model_kinds"]:
            for bond_dim in plan["bond_dims"]:
                trials = cells.get((dataset, kind, bond_dim), [])
                seeds = sorted(trial["seed"] for trial in trials)
                if len(seeds) != len(set(seeds)):
                    raise ValueError(
                        f"Duplicate seeds in cell {(dataset, kind, bond_dim)}"
                    )
                row = {
                    "dataset": dataset,
                    "model": kind,
                    "bond_dim": bond_dim,
                    "successful_runs": len(trials),
                    "expected_runs": len(expected_seeds),
                    "status": "COMPLETE" if set(seeds) == expected_seeds else "PARTIAL",
                }
                if trials:
                    first = trials[0]["metrics"]
                    row["sequence_length"] = first["sequence_length"]
                    row["free_parameter_count"] = first["free_parameter_count"]
                    best = select_best_train_trial(trials)
                    row["best_train_seed"] = best["seed"]
                    for split in SPLITS:
                        for metric in METRICS:
                            values = [
                                t["metrics"]["final"][split][metric] for t in trials
                            ]
                            row[f"best_train_{split}_{metric}"] = best["metrics"][
                                "final"
                            ][split][metric]
                            row[f"min_{split}_{metric}"] = min(values)
                            for name, value in _stats(values).items():
                                row[f"{name}_{split}_{metric}"] = value
                    row["num_train"] = first["final"]["train"]["num_sequences"]
                rows.append(row)

    fig8 = []
    for dataset in plan["datasets"]:
        for bond_dim in plan["bond_dims"]:
            hmm = next(
                (
                    r
                    for r in rows
                    if (r["dataset"], r["model"], r["bond_dim"])
                    == (dataset, "hmm_baum_welch", bond_dim)
                ),
                None,
            )
            enhanced = next(
                (
                    r
                    for r in rows
                    if (r["dataset"], r["model"], r["bond_dim"])
                    == (dataset, "basis_enhanced", bond_dim)
                ),
                None,
            )
            if (
                not hmm
                or not enhanced
                or not hmm["successful_runs"]
                or not enhanced["successful_runs"]
            ):
                continue
            length = enhanced["sequence_length"]
            extra = enhanced["free_parameter_count"] - hmm["free_parameter_count"]
            entry = {
                "dataset": dataset,
                "bond_dim": bond_dim,
                "extra_parameters": extra,
            }
            for metric in METRICS:
                gap = length * (
                    hmm[f"min_train_{metric}"] - enhanced[f"min_train_{metric}"]
                )
                entry[f"gap_bits_per_sequence_{metric}"] = gap
            entry["lrt_5sigma_threshold_bits_per_sequence"] = (
                likelihood_ratio_threshold_bits(enhanced["num_train"], extra)
                if extra > 0
                else None
            )
            paper_df = paper_inferred_lrt_degrees_of_freedom(bond_dim)
            entry["paper_inferred_df"] = paper_df
            entry["lrt_5sigma_threshold_paper_df_bits_per_sequence"] = (
                likelihood_ratio_threshold_bits(enhanced["num_train"], paper_df)
                if paper_df > 0
                else None
            )
            fig8.append(entry)

    complete = all(row["status"] == "COMPLETE" for row in rows)
    summary = {
        "sweep_id": status["sweep_id"],
        "coverage": "COMPLETE SWEEP" if complete else "PARTIAL SWEEP",
        "cells": rows,
        "fig8": fig8,
        "generated_by": "utils/summarize_fig7.py",
    }
    _write(sweep_dir, summary, rows)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args(argv)
    summary = summarize(args.sweep_dir.resolve())
    print(
        f"{summary['coverage']}: {len(summary['cells'])} cells -> {args.sweep_dir / 'summary.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
