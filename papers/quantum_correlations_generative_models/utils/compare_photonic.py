"""Compare the Phase 4 MerLin adaptation with the Phase 3 models, per (arm, h).

Applies the rules pre-declared in ``LOG.md`` (Phase 4 Adaptation Plan) to the
paper metric NLL - log2(N)/n (``paper_kl_estimate_bits_per_site``):

- ``matches_basis_enhanced``: |best-of-10 train (MerLin) - best-of-10 train
  (basis-enhanced)| <= 0.005 bits/site;
- ``beats_converged_hmm_<split>``: mean (MerLin) < mean (HMM) by more than
  2 x sqrt(SEM_MerLin^2 + SEM_HMM^2), on train and on test.

Best-of-10 is the trial with the lowest final train NLL (ties within 1e-6
bits to the lowest seed, ``utils/summarize_fig7.py``). An arm is a dataset
plus the sweep's ``config_variant`` (``spect_heart_pooled`` for the pooled
SPECT split). Herald success (log2 probability that all n heralds succeed,
``metrics.json`` key ``photonic.herald_success_log2_per_sequence``) and
wall-clock are read from every MerLin run.

Usage::

    python3 utils/compare_photonic.py --merlin <SWEEP_DIR> [...] \
        --phase3 <SWEEP_DIR> [...] --out results/photonic_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

METRIC = "paper_kl_estimate_bits_per_site"
MATCH_TOLERANCE_BITS = 0.005
SEM_MULTIPLIER = 2.0
MERLIN_KIND = "photonic_merlin"
PHASE3_KINDS = ("basis_enhanced", "hmm_baum_welch")


def _arm(dataset: str, config_variant: str | None) -> str:
    return dataset + ("_pooled" if config_variant == "_pooled" else "")


def load_cells(sweep_dirs: list[Path]) -> dict[tuple[str, str, int], dict]:
    """Return summary cells keyed by (arm, model, h), with the source sweep id."""
    cells = {}
    for sweep_dir in sweep_dirs:
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        summary = json.loads((sweep_dir / "summary.json").read_text())
        if summary["coverage"] != "COMPLETE SWEEP":
            raise ValueError(f"{sweep_dir} is not a COMPLETE SWEEP")
        variant = status["plan"].get("config_variant")
        for cell in summary["cells"]:
            key = (_arm(cell["dataset"], variant), cell["model"], cell["bond_dim"])
            if key in cells:
                raise ValueError(f"Cell {key} appears in more than one sweep")
            cells[key] = {**cell, "sweep_id": summary["sweep_id"]}
    return cells


def load_merlin_run_records(sweep_dirs: list[Path]) -> dict[tuple[str, int], list]:
    """Return per-run photonic records keyed by (arm, h)."""
    records: dict[tuple[str, int], list] = {}
    for sweep_dir in sweep_dirs:
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        variant = status["plan"].get("config_variant")
        for run in status["runs"]:
            if run["status"] != "COMPLETED":
                continue
            metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
            key = (_arm(metrics["dataset"], variant), metrics["bond_dim"])
            records.setdefault(key, []).append(
                {
                    "seed": metrics["seed"],
                    "herald_log2": metrics["photonic"][
                        "herald_success_log2_per_sequence"
                    ],
                    "wall_seconds": metrics["wall_seconds"],
                    "photonic": metrics["photonic"],
                }
            )
    return records


def compare(merlin_dirs: list[Path], phase3_dirs: list[Path]) -> dict:
    """Return the comparison rows and the per-arm counts."""
    merlin = load_cells(merlin_dirs)
    phase3 = load_cells(phase3_dirs)
    runs = load_merlin_run_records(merlin_dirs)
    rows = []
    for (arm, kind, bond_dim), cell in sorted(merlin.items()):
        if kind != MERLIN_KIND:
            continue
        enhanced = phase3[(arm, "basis_enhanced", bond_dim)]
        hmm = phase3[(arm, "hmm_baum_welch", bond_dim)]
        records = runs[(arm, bond_dim)]
        best_record = next(r for r in records if r["seed"] == cell["best_train_seed"])
        row = {
            "arm": arm,
            "bond_dim": bond_dim,
            "merlin_sweep": cell["sweep_id"],
            "phase3_sweep": enhanced["sweep_id"],
            "merlin_trials": cell["successful_runs"],
            "merlin_free_parameters": cell["free_parameter_count"],
            "basis_enhanced_free_parameters": enhanced["free_parameter_count"],
            "hmm_free_parameters": hmm["free_parameter_count"],
            "merlin_best_train_seed": cell["best_train_seed"],
        }
        for split in ("train", "test"):
            for name, source in (
                ("merlin", cell),
                ("basis_enhanced", enhanced),
                ("hmm", hmm),
            ):
                row[f"{name}_best_train_{split}"] = source[
                    f"best_train_{split}_{METRIC}"
                ]
                row[f"{name}_mean_{split}"] = source[f"mean_{split}_{METRIC}"]
                row[f"{name}_sem_{split}"] = source[f"sem_{split}_{METRIC}"]
            margin = SEM_MULTIPLIER * math.hypot(
                row[f"merlin_sem_{split}"], row[f"hmm_sem_{split}"]
            )
            row[f"merlin_minus_hmm_mean_{split}"] = (
                row[f"merlin_mean_{split}"] - row[f"hmm_mean_{split}"]
            )
            row[f"beats_hmm_margin_{split}"] = margin
            row[f"beats_converged_hmm_{split}"] = (
                row[f"merlin_minus_hmm_mean_{split}"] < -margin
            )
        row["merlin_minus_basis_enhanced_best_train"] = (
            row["merlin_best_train_train"] - row["basis_enhanced_best_train_train"]
        )
        row["matches_basis_enhanced"] = (
            abs(row["merlin_minus_basis_enhanced_best_train"]) <= MATCH_TOLERANCE_BITS
        )
        heralds = [r["herald_log2"] for r in records]
        row["herald_success_log2_best_trial"] = best_record["herald_log2"]
        row["herald_success_log2_min"] = min(heralds)
        row["herald_success_log2_max"] = max(heralds)
        row["herald_success_log2_mean"] = sum(heralds) / len(heralds)
        row["wall_seconds_total"] = sum(r["wall_seconds"] for r in records)
        row["wall_seconds_max"] = max(r["wall_seconds"] for r in records)
        row["seeds"] = sorted(r["seed"] for r in records)
        row["num_modes"] = best_record["photonic"]["num_modes"]
        row["photon_number"] = best_record["photonic"]["photon_number"]
        row["preserves_role"] = (
            row["matches_basis_enhanced"] or row["beats_converged_hmm_train"]
        )
        rows.append(row)

    arms = {}
    for row in rows:
        arm = arms.setdefault(
            row["arm"],
            {
                "cells": 0,
                "matches_basis_enhanced": [],
                "beats_converged_hmm_train": [],
                "beats_converged_hmm_test": [],
            },
        )
        arm["cells"] += 1
        for flag in (
            "matches_basis_enhanced",
            "beats_converged_hmm_train",
            "beats_converged_hmm_test",
        ):
            if row[flag]:
                arm[flag].append(row["bond_dim"])
    return {
        "metric": METRIC,
        "rules": {
            "matches_basis_enhanced": f"|best-of-10 train MerLin - best-of-10 train BE| <= {MATCH_TOLERANCE_BITS}",
            "beats_converged_hmm": f"mean MerLin < mean HMM - {SEM_MULTIPLIER} x sqrt(SEM_MerLin^2 + SEM_HMM^2)",
            "source": "LOG.md, Phase 4 Adaptation Plan (declared before the runs)",
        },
        "label": "MerLin adaptation (simulated), not the paper's model",
        "arms": arms,
        "rows": rows,
        "generated_by": "utils/compare_photonic.py",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merlin", type=Path, nargs="+", required=True)
    parser.add_argument("--phase3", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(
        [p.resolve() for p in args.merlin], [p.resolve() for p in args.phase3]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    rows = result["rows"]
    columns = list(rows[0].keys())
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result["arms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
