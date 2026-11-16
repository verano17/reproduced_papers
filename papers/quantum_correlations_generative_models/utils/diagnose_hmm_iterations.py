"""Diagnose the HMM baseline: best-of-10 Fig. 7 values under Baum-Welch caps.

Phase 3 gap diagnosis (LOG.md). Every HMM run stores the train and test
evaluation after every EM iteration in ``metrics.json``, so the value the
baseline would have reached with a smaller iteration cap K is read from the
history (a run that stopped earlier keeps its final value). No run is
re-executed. For each (dataset, h) and cap K the script reports the best-of-10
trial (lowest train NLL at K) in the paper's metric NLL - log2(N)/n, and the
cap that best matches the paper's plotted HMM points (root-mean-square error
over the h values read from Fig. 7). The matching cap is a diagnostic of how
far the paper's baseline is from Baum-Welch convergence, not a tuned setting.

Usage::

    python3 utils/diagnose_hmm_iterations.py <SWEEP_DIR> [<SWEEP_DIR> ...] \
        --paper-values results/paper_values_fig7_fig8.json --out results/hmm_iteration_caps.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from lib.metrics import paper_kl_estimate_bits_per_site  # noqa: E402

CAPS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
TIE_TOLERANCE_BITS = 1e-6


def _paper_metric_history(metrics: dict, split: str) -> list[float]:
    length = metrics["sequence_length"]
    values = []
    for entry in metrics["history"]:
        evaluation = entry["evaluation"][split]
        values.append(
            paper_kl_estimate_bits_per_site(
                evaluation["nll_bits_per_site"], evaluation["num_sequences"], length
            )
        )
    return values


def diagnose(sweep_dirs: list[Path], paper_values: dict) -> dict:
    """Return per-(sweep, dataset, h) cap tables and the best-matching caps."""
    result = {"caps": list(CAPS), "sweeps": []}
    for sweep_dir in sweep_dirs:
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        histories: dict[tuple, list] = {}
        for run in status["runs"]:
            if run["status"] != "COMPLETED":
                continue
            if run["candidate"]["model"] != "hmm_baum_welch":
                continue
            metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
            key = (metrics["dataset"], metrics["bond_dim"])
            histories.setdefault(key, []).append(
                {
                    "seed": run["seed"],
                    "train": _paper_metric_history(metrics, "train"),
                    "test": _paper_metric_history(metrics, "test"),
                }
            )
        cells = []
        for (dataset, bond_dim), trials in sorted(histories.items()):
            row = {"dataset": dataset, "bond_dim": bond_dim, "trials": len(trials)}
            for cap in CAPS:
                at_cap = [
                    (t["train"][min(cap, len(t["train"]) - 1)], t["seed"], t)
                    for t in trials
                ]
                lowest = min(value for value, _, _ in at_cap)
                # Same rule as utils/summarize_fig7.py: ties within 1e-6 bits
                # of the lowest train value go to the lowest seed.
                best = min(
                    (
                        entry
                        for entry in at_cap
                        if entry[0] - lowest <= TIE_TOLERANCE_BITS
                    ),
                    key=lambda entry: entry[1],
                )[2]
                index = min(cap, len(best["train"]) - 1)
                row[f"train_at_{cap}"] = best["train"][index]
                row[f"test_at_{cap}"] = best["test"][index]
            cells.append(row)
        matches = []
        for dataset in sorted({row["dataset"] for row in cells}):
            paper = paper_values["fig7_best_of_ten"][dataset]["train"]["hmm_baum_welch"]
            points = [
                (row, paper[str(row["bond_dim"])])
                for row in cells
                if row["dataset"] == dataset
                and str(row["bond_dim"]) in paper
                and row["bond_dim"] > 1
            ]
            errors = {
                cap: math.sqrt(
                    sum((row[f"train_at_{cap}"] - value) ** 2 for row, value in points)
                    / len(points)
                )
                for cap in CAPS
            }
            best_cap = min(errors, key=errors.get)
            matches.append(
                {
                    "dataset": dataset,
                    "rms_error_by_cap": errors,
                    "best_matching_cap": best_cap,
                    "rms_error_at_best_cap": errors[best_cap],
                    "rms_error_at_final": errors[CAPS[-1]],
                }
            )
        result["sweeps"].append(
            {"sweep_id": status["sweep_id"], "cells": cells, "paper_match": matches}
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    parser.add_argument("--paper-values", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    paper_values = json.loads(args.paper_values.read_text())
    result = diagnose([path.resolve() for path in args.sweep_dirs], paper_values)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for sweep in result["sweeps"]:
        for match in sweep["paper_match"]:
            print(
                f"{sweep['sweep_id']} {match['dataset']}: best-matching cap "
                f"{match['best_matching_cap']} (RMS {match['rms_error_at_best_cap']:.4f}); "
                f"RMS at cap {CAPS[-1]}: {match['rms_error_at_final']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
