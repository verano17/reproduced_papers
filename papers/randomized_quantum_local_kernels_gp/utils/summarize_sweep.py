"""Build the result table of a repetition sweep from its structured artifacts.

Reads ``sweep_status.json`` and every listed run's ``metrics.json``, checks
that the observed (data seed, Sobol seed) set equals the planned set and that
every included value is finite, and writes ``summary.json`` and
``summary.md`` next to the sweep status. Values are aggregated as
mean ± sample standard deviation (ddof = 1) over repetitions, at full
precision. These sweeps estimate seed variability; they do not select a
candidate.

Usage:
    python utils/summarize_sweep.py <sweep_dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

METHODS = ("learned_mixture", "uniform_mixture", "best_single_kernel")
METRICS = (
    "alignment_centered",
    "alignment_uncentered",
    "rmse",
    "avg_nlpd",
    "coverage_2sigma",
    "noise_variance",
    "effective_num_kernels",
)
FEATURE_METRICS = ("sup_norm", "sup_norm_one_local", "sup_norm_two_local", "mean_abs")


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else float("nan"),
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(array.size),
    }


def summarize(sweep_dir: Path) -> dict:
    """Aggregate the completed runs of a sweep.

    Parameters
    ----------
    sweep_dir : pathlib.Path
        Directory holding ``sweep_status.json`` and the run folders.

    Returns
    -------
    dict
        Coverage report and per-kernel, per-method statistics.

    Raises
    ------
    ValueError
        If a completed run has a missing or non-finite metric, or a
        repetition appears twice.
    """
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    planned = {tuple(rep["seeds"]) for rep in status["plan"]["repetitions"]}
    observed: dict[tuple, dict] = {}
    missing_or_failed = []
    for run in status["runs"]:
        key = (run["candidate"]["data_seed"], run["candidate"]["sobol_seed"])
        if run["status"] != "COMPLETED":
            missing_or_failed.append({"seeds": key, "status": run["status"]})
            continue
        if key in observed:
            raise ValueError(f"Duplicate completed repetition {key}.")
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        observed[key] = metrics
    for key in planned - set(observed):
        if not any(tuple(item["seeds"]) == key for item in missing_or_failed):
            missing_or_failed.append({"seeds": key, "status": "MISSING"})

    first = next(iter(observed.values())) if observed else {"kernels": {}}
    kernels = sorted(first["kernels"])
    table: dict = {}
    for kernel in kernels:
        table[kernel] = {}
        # Mixture sweeps report learned/uniform/best-single; parametric GP
        # baselines report a single "parametric_gp" method.
        methods = [
            m for m in METHODS + ("parametric_gp",) if m in first["kernels"][kernel]
        ]
        for method in methods:
            table[kernel][method] = {}
            metrics_here = [m for m in METRICS if m in first["kernels"][kernel][method]]
            for metric in metrics_here:
                values = [
                    observed[key]["kernels"][kernel][method][metric]
                    for key in sorted(observed)
                ]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError(f"Non-finite {metric} for {kernel}/{method}.")
                table[kernel][method][metric] = _stats(values)
    feature_error = (
        {
            metric: _stats(
                [observed[key]["feature_error"][metric] for key in sorted(observed)]
            )
            for metric in FEATURE_METRICS
        }
        if observed and first.get("feature_error") is not None
        else {}
    )
    complete = set(observed) == planned and not missing_or_failed

    def primary(kernel_metrics: dict) -> dict:
        if "learned_mixture" in kernel_metrics:
            return kernel_metrics["learned_mixture"]
        return kernel_metrics["parametric_gp"]

    per_run = {
        f"data{key[0]}_sobol{key[1]}": {
            kernel: primary(observed[key]["kernels"][kernel])["alignment_centered"]
            for kernel in kernels
        }
        for key in sorted(observed)
    }
    return {
        "sweep_id": status["sweep_id"],
        "candidate": status["plan"]["candidate"],
        "coverage": "COMPLETE SWEEP" if complete else "PARTIAL SWEEP",
        "planned_runs": len(planned),
        "completed_runs": len(observed),
        "missing_or_failed": [
            {"seeds": list(i["seeds"]), "status": i["status"]}
            for i in missing_or_failed
        ],
        "aggregation": "mean and sample std (ddof=1) over (data_seed, sobol_seed) repetitions",
        "feature_error": feature_error,
        "kernels": table,
        "per_run_alignment_centered": per_run,
    }


def to_markdown(summary: dict) -> str:
    """Render the summary as a Markdown table (values at 4 decimals for display)."""
    lines = [
        f"# Sweep {summary['sweep_id']}: {summary['coverage']}",
        "",
        f"Candidate: `{json.dumps(summary['candidate'])}`  ",
        f"Completed {summary['completed_runs']} / {summary['planned_runs']} planned runs. "
        f"{summary['aggregation']}.",
        "",
        "| Kernel | Method | Centered alignment | Uncentered alignment | RMSE | Avg NLPD | 2σ coverage |",
        "|---|---|---|---|---|---|---|",
    ]
    for kernel, methods in summary["kernels"].items():
        for method, metrics in methods.items():
            cells = [
                f"{metrics[m]['mean']:.4f} ± {metrics[m]['std']:.4f}"
                if m in metrics
                else "n/a"
                for m in (
                    "alignment_centered",
                    "alignment_uncentered",
                    "rmse",
                    "avg_nlpd",
                    "coverage_2sigma",
                )
            ]
            lines.append(f"| {kernel} | {method} | " + " | ".join(cells) + " |")
    if summary["feature_error"]:
        fe = summary["feature_error"]["sup_norm"]
        lines += [
            "",
            f"Feature sup-norm error: {fe['mean']:.4f} ± {fe['std']:.4f} (max {fe['max']:.4f}).",
        ]
    if summary["missing_or_failed"]:
        lines += ["", f"Missing or failed: {summary['missing_or_failed']}"]
    return "\n".join(lines) + "\n"


def write_summary(sweep_dir: Path) -> dict:
    """Write ``summary.json`` and ``summary.md`` into ``sweep_dir`` and return the summary."""
    summary = summarize(sweep_dir)
    (sweep_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (sweep_dir / "summary.md").write_text(to_markdown(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    summary = write_summary(args.sweep_dir.resolve())
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
