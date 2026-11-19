"""Export every sweep run's per-kernel metrics to one durable CSV.

Reads ``outdir/sweeps/*/sweep_status.json`` and each COMPLETED run's
``metrics.json`` and ``run_status.json``; writes one row per
(sweep, data seed, Sobol seed, query kernel, method). The notebook reads this
file so that individual seeds can be shown without the disposable
``outdir/``.

Usage (from the paper folder):
    python utils/export_run_table.py --out results/all_runs_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SWEEPS_DIR = Path(__file__).resolve().parents[1] / "outdir" / "sweeps"
FIELDS = (
    "alignment_centered",
    "alignment_uncentered",
    "rmse",
    "avg_nlpd",
    "coverage_2sigma",
    "noise_variance",
    "train_log_marginal",
)


def rows() -> list[dict]:
    """Collect one row per completed run, kernel and method."""
    table = []
    for status_path in sorted(SWEEPS_DIR.glob("*/sweep_status.json")):
        status = json.loads(status_path.read_text())
        if status["status"] != "COMPLETED":
            raise ValueError(f"{status_path.parent.name} is {status['status']}.")
        for run in status["runs"]:
            metrics = json.loads((status_path.parent / run["metrics_path"]).read_text())
            run_status = json.loads(
                (status_path.parent / run["run_status"]).read_text()
            )
            feature_error = metrics.get("feature_error") or {}
            photonic = metrics.get("photonic") or {}
            for kernel, methods in metrics["kernels"].items():
                for method, values in methods.items():
                    row = {
                        "sweep": status["sweep_id"],
                        "config": run["candidate"]["config"],
                        "data_seed": run["candidate"]["data_seed"],
                        "sobol_seed": run["candidate"]["sobol_seed"],
                        "run_id": run["run_id"],
                        "code_commit": run_status["code"]["commit"][:7],
                        "kernel": kernel,
                        "method": method,
                        "feature_sup_norm": feature_error.get("sup_norm"),
                        "postselection_mean": photonic.get(
                            "postselection_probability_mean"
                        ),
                    }
                    row.update({field: values.get(field) for field in FIELDS})
                    table.append(row)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    table = rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(
        f"Wrote {len(table)} rows from {len({r['sweep'] for r in table})} sweeps to {args.out}"
    )


if __name__ == "__main__":
    main()
