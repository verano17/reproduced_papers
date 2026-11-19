"""Phase 5 comparison table and pairwise decisions from structured sweep artifacts.

Implements the frozen rule of the "Phase 5 Comparison Plan" in LOG.md:
system A is better than B on a metric only if the paired 95% bootstrap CI of
A − B (paired by data seed and Sobol seed) excludes 0 AND the mean difference
exceeds the practical tolerance (alignment 0.01, RMSE 5% relative, NLPD 0.05).
Otherwise the pair is a tie. Directions: alignment ↑, RMSE ↓, NLPD ↓.

Every sweep must be COMPLETED with exactly the 15 planned seed pairs and
finite metrics; otherwise the script raises.

Usage (from the paper folder):
    python utils/compare_baselines.py --out results/phase5_comparison
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SWEEPS_DIR = PROJECT_DIR / "outdir" / "sweeps"
KERNELS = ("rbf", "matern32", "nonstat_ls")
METRICS = ("alignment_centered", "rmse", "avg_nlpd", "coverage_2sigma")
PAPER_TABLE1 = {"rbf": 0.9791, "matern32": 0.9920, "nonstat_ls": 0.9538}
EXPECTED_PAIRS = {(d, s) for d in (12345, 1, 2, 3, 4) for s in (12345, 1, 2)}

# (sweep id, display name, role, provenance, comparison group)
SYSTEMS = (
    (
        "S5_table1_original_sharednoise",
        "Qubit dictionary (authors' protocol)",
        "paper method",
        "local reproduction",
        "authors",
    ),
    (
        "M1_table1_merlin",
        "Photonic dictionary (authors' protocol)",
        "MerLin adaptation",
        "local, simulated",
        "authors",
    ),
    (
        "C1_table1_classical_rff",
        "B3 RFF dictionary (authors' protocol)",
        "fair classical baseline",
        "local",
        "authors",
    ),
    (
        "C3_table1_classical_rbfdict",
        "B4 RBF dictionary (authors' protocol)",
        "classical baseline",
        "local",
        "authors",
    ),
    (
        "S6_table1_lbfgs",
        "Qubit dictionary (converged MKL)",
        "paper method, converged optimizer",
        "local reproduction",
        "converged",
    ),
    (
        "M2_table1_merlin_lbfgs",
        "Photonic dictionary (converged MKL)",
        "MerLin adaptation",
        "local, simulated",
        "converged",
    ),
    (
        "C2_table1_classical_rff_lbfgs",
        "B3 RFF dictionary (converged MKL)",
        "fair classical baseline",
        "local",
        "converged",
    ),
    (
        "C4_table1_classical_rbfdict_lbfgs",
        "B4 RBF dictionary (converged MKL)",
        "classical baseline",
        "local",
        "converged",
    ),
    (
        "C5_table1_classical_gp_true",
        "B1 true-family GP (ML-II)",
        "oracle reference",
        "local",
        "reference",
    ),
    (
        "C6_table1_classical_gp_rbf",
        "B2 RBF GP (ML-II)",
        "standard practice",
        "local",
        "reference",
    ),
    (
        "S7_table1_exact_features_lbfgs",
        "Qubit dictionary, exact features (converged MKL)",
        "paper method, shot-noise-free diagnostic",
        "local reproduction",
        "converged",
    ),
    (
        "M3_table1_merlin_mixed",
        "Photonic dictionary, corrected encoding (authors' protocol)",
        "MerLin adaptation",
        "local, simulated",
        "authors",
    ),
    (
        "M4_table1_merlin_mixed_lbfgs",
        "Photonic dictionary, corrected encoding (converged MKL)",
        "MerLin adaptation",
        "local, simulated",
        "converged",
    ),
)
# Pairwise questions: (A, B). Quantum/photonic vs matched classical within an
# optimizer group, and every dictionary vs the parametric references.
PAIRS = (
    ("S5_table1_original_sharednoise", "C1_table1_classical_rff"),
    ("S5_table1_original_sharednoise", "C3_table1_classical_rbfdict"),
    ("M1_table1_merlin", "C1_table1_classical_rff"),
    ("S6_table1_lbfgs", "C2_table1_classical_rff_lbfgs"),
    ("S6_table1_lbfgs", "C4_table1_classical_rbfdict_lbfgs"),
    ("M2_table1_merlin_lbfgs", "C2_table1_classical_rff_lbfgs"),
    ("S6_table1_lbfgs", "C6_table1_classical_gp_rbf"),
    ("S6_table1_lbfgs", "C5_table1_classical_gp_true"),
    ("C2_table1_classical_rff_lbfgs", "C5_table1_classical_gp_true"),
    # Added by the 2026-11-19T16:00:08Z amendment (independent review 2).
    ("S7_table1_exact_features_lbfgs", "C2_table1_classical_rff_lbfgs"),
    ("S7_table1_exact_features_lbfgs", "S6_table1_lbfgs"),
    ("M4_table1_merlin_mixed_lbfgs", "C2_table1_classical_rff_lbfgs"),
    ("M4_table1_merlin_mixed_lbfgs", "S6_table1_lbfgs"),
)
TOLERANCE = {
    "alignment_centered": ("absolute", 0.01),
    "rmse": ("relative", 0.05),
    "avg_nlpd": ("absolute", 0.05),
}
HIGHER_IS_BETTER = {"alignment_centered": True, "rmse": False, "avg_nlpd": False}


def _timestamp(text: str) -> dt.datetime:
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def load_sweep(sweep_id: str) -> dict:
    """Per-seed-pair primary metrics plus wall-clock of a validated, complete sweep."""
    sweep_dir = SWEEPS_DIR / sweep_id
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep_id} is {status['status']}, not COMPLETED.")
    runs, seconds = {}, []
    for run in status["runs"]:
        key = (run["candidate"]["data_seed"], run["candidate"]["sobol_seed"])
        if key in runs:
            raise ValueError(f"{sweep_id}: duplicate seed pair {key}.")
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        run_status = json.loads((sweep_dir / run["run_status"]).read_text())
        if run_status["status"] != "COMPLETED":
            raise ValueError(f"{sweep_id}: run {run['run_id']} is not COMPLETED.")
        seconds.append(
            (
                _timestamp(run_status["completed_at"])
                - _timestamp(run_status["started_at"])
            ).total_seconds()
        )
        primary = {}
        for kernel in KERNELS:
            entry = metrics["kernels"][kernel]
            method = (
                entry["learned_mixture"]
                if "learned_mixture" in entry
                else entry["parametric_gp"]
            )
            values = {metric: float(method[metric]) for metric in METRICS}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError(f"{sweep_id}: non-finite metric in {run['run_id']}.")
            primary[kernel] = values
        runs[key] = primary
    if set(runs) != EXPECTED_PAIRS:
        raise ValueError(
            f"{sweep_id}: observed seed pairs differ from the 15 planned pairs."
        )
    return {
        "runs": runs,
        "seconds": seconds,
        "started": status["started_at"],
        "completed": status["completed_at"],
    }


def _bootstrap_ci(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    means = rng.choice(differences, size=(10_000, differences.size), replace=True).mean(
        axis=1
    )
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def decide(a: np.ndarray, b: np.ndarray, metric: str) -> dict:
    """Apply the frozen decision rule to paired values of systems A and B."""
    differences = a - b
    low, high = _bootstrap_ci(differences)
    mean = float(differences.mean())
    kind, tolerance = TOLERANCE[metric]
    threshold = (
        tolerance * float(np.mean(np.abs(b))) if kind == "relative" else tolerance
    )
    significant = low > 0 or high < 0
    if significant and abs(mean) > threshold:
        a_better = (mean > 0) == HIGHER_IS_BETTER[metric]
        verdict = "A better" if a_better else "B better"
    else:
        verdict = "tie"
    return {
        "mean_difference": mean,
        "ci95": [low, high],
        "threshold": threshold,
        "verdict": verdict,
    }


def _paired_values(
    sweep: dict, keys: list, kernel: str, metric: str, aggregate: str
) -> np.ndarray:
    values = {k: sweep["runs"][k][kernel][metric] for k in keys}
    if aggregate == "run":
        return np.array([values[k] for k in keys])
    # Data-seed aggregation: average over Sobol seeds, giving one value per
    # data seed (n = 5). Deterministic baselines (B4) and ML-II GPs barely
    # vary across Sobol seeds, so this is the honest replication unit.
    data_seeds = sorted({k[0] for k in keys})
    return np.array(
        [np.mean([values[k] for k in keys if k[0] == d]) for d in data_seeds]
    )


def build(out: Path, aggregate: str = "run") -> dict:
    """Assemble the table, pairwise decisions and timing; write JSON and Markdown.

    ``aggregate`` is "run" (15 paired runs) or "data_seed" (5 data-seed means).
    """
    sweeps = {sweep_id: load_sweep(sweep_id) for sweep_id, *_ in SYSTEMS}
    keys = sorted(EXPECTED_PAIRS)
    table = []
    for sweep_id, name, role, provenance, group in SYSTEMS:
        runs = sweeps[sweep_id]["runs"]
        row = {
            "sweep": sweep_id,
            "system": name,
            "role": role,
            "provenance": provenance,
            "group": group,
            "runs": f"{len(runs)}/15",
            "median_run_seconds": float(np.median(sweeps[sweep_id]["seconds"])),
            "metrics": {},
        }
        for kernel in KERNELS:
            row["metrics"][kernel] = {}
            for metric in METRICS:
                values = np.array([runs[k][kernel][metric] for k in keys])
                row["metrics"][kernel][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                }
        table.append(row)
    pairs = []
    for a_id, b_id in PAIRS:
        for kernel in KERNELS:
            for metric in TOLERANCE:
                a = _paired_values(sweeps[a_id], keys, kernel, metric, aggregate)
                b = _paired_values(sweeps[b_id], keys, kernel, metric, aggregate)
                pairs.append(
                    {
                        "A": a_id,
                        "B": b_id,
                        "kernel": kernel,
                        "metric": metric,
                        **decide(a, b, metric),
                    }
                )
    result = {
        "paper_table1": PAPER_TABLE1,
        "table": table,
        "pairs": pairs,
        "aggregate": aggregate,
        "rule": "A better/B better only if paired 95% bootstrap CI excludes 0 and |mean diff| > tolerance "
        "(alignment 0.01, RMSE 5% relative, NLPD 0.05); otherwise tie",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    out.with_suffix(".md").write_text(to_markdown(result))
    return result


def _cell(metrics: dict, kernel: str, metric: str) -> str:
    return (
        f"{metrics[kernel][metric]['mean']:.4f} ± {metrics[kernel][metric]['std']:.4f}"
    )


def to_markdown(result: dict) -> str:
    """Render the comparison (display rounding only; decisions use full precision)."""
    lines = [
        "# Phase 5 comparison (generated by utils/compare_baselines.py)",
        "",
        "Scope labels: all local values are synthetic-data, 15 runs (5 data seeds x 3 Sobol seeds); "
        "MerLin rows are a simulated adaptation; the paper row is paper-reported (single run).",
        "",
        "| System | Role | Provenance | Runs | rbf align | matern32 align | nonstat align | rbf RMSE | matern32 RMSE | nonstat RMSE | rbf NLPD | matern32 NLPD | nonstat NLPD | Median s/run |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    paper = result["paper_table1"]
    lines.append(
        f"| Paper Table I (qubit, single run) | paper method | paper-reported | 1 | {paper['rbf']:.4f} | {paper['matern32']:.4f} | {paper['nonstat_ls']:.4f} | — | — | — | — | — | — | — |"
    )
    for row in result["table"]:
        m = row["metrics"]

        lines.append(
            f"| {row['system']} | {row['role']} | {row['provenance']} | {row['runs']} | "
            + " | ".join(_cell(m, k, "alignment_centered") for k in KERNELS)
            + " | "
            + " | ".join(_cell(m, k, "rmse") for k in KERNELS)
            + " | "
            + " | ".join(_cell(m, k, "avg_nlpd") for k in KERNELS)
            + f" | {row['median_run_seconds']:.0f} |"
        )
    lines += [
        "",
        f"Decision rule: {result['rule']}. Pairing unit: {result['aggregate']}.",
        "",
        "| A | B | Kernel | Metric | Mean A−B [95% CI] | Tolerance | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in result["pairs"]:
        lines.append(
            f"| {p['A']} | {p['B']} | {p['kernel']} | {p['metric']} | {p['mean_difference']:+.4f} "
            f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | {p['threshold']:.4f} | {p['verdict']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--aggregate",
        choices=["run", "data_seed"],
        default="run",
        help="Pair 15 runs (frozen plan) or 5 data-seed means (robustness check)",
    )
    args = parser.parse_args()
    build(args.out, args.aggregate)
    print(args.out.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
