"""Paired statistics across repetition sweeps and against Table I.

Reads the completed runs of each sweep (``metrics.json`` via
``sweep_status.json``), pairs them by (data_seed, sobol_seed), and reports:

- per sweep and query kernel: mean ± std of the learned-mixture centered
  alignment, the paper value, its z-score and the fraction of seeds at or
  above the paper value;
- within each sweep: learned mixture minus uniform mixture and minus best
  single kernel (paired differences);
- between sweeps: paired differences of the learned-mixture metric against a
  reference sweep.

Paired differences are summarized by their mean, a 95% percentile bootstrap
interval (10 000 resamples, seed 0) and the two-sided Wilcoxon signed-rank
p-value. Results are written as JSON and Markdown.

Usage:
    python utils/compare_sweeps.py --sweeps outdir/sweeps/S1_table1_original ... \
        --reference S1_table1_original --out results/table1_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# Table I of the paper (test centered alignment, as computed by the authors'
# code; the notebook prints 0.97914153, 0.99196248, 0.9537627).
PAPER_TABLE1 = {"rbf": 0.9791, "matern32": 0.9920, "nonstat_ls": 0.9538}
METRICS = (
    "alignment_centered",
    "alignment_uncentered",
    "rmse",
    "avg_nlpd",
    "coverage_2sigma",
)


def load_sweep(sweep_dir: Path) -> dict[tuple, dict]:
    """Map (data_seed, sobol_seed) to metrics.json for every completed run."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep_dir.name} is {status['status']}, not COMPLETED.")
    runs = {}
    for run in status["runs"]:
        key = (run["candidate"]["data_seed"], run["candidate"]["sobol_seed"])
        runs[key] = json.loads((sweep_dir / run["metrics_path"]).read_text())
    return runs


def paired_summary(differences: np.ndarray) -> dict:
    """Mean, bootstrap 95% CI and Wilcoxon p-value of paired differences."""
    rng = np.random.default_rng(0)
    resamples = rng.choice(
        differences, size=(10_000, differences.size), replace=True
    ).mean(axis=1)
    p_value = float(wilcoxon(differences).pvalue) if np.any(differences != 0) else 1.0
    return {
        "mean": float(differences.mean()),
        "ci95": [
            float(np.percentile(resamples, 2.5)),
            float(np.percentile(resamples, 97.5)),
        ],
        "wilcoxon_p": p_value,
        "n": int(differences.size),
        "num_positive": int(np.sum(differences > 0)),
    }


def compare(sweeps: dict[str, dict], reference: str) -> dict:
    """Compute all paired comparisons; see module docstring."""
    keys = sorted(set.intersection(*(set(runs) for runs in sweeps.values())))
    if any(len(runs) != len(keys) for runs in sweeps.values()):
        raise ValueError("Sweeps do not cover the same (data_seed, sobol_seed) pairs.")
    kernels = sorted(next(iter(sweeps.values()))[keys[0]]["kernels"])
    result: dict = {
        "pairs": [list(k) for k in keys],
        "per_sweep": {},
        "between_sweeps": {},
    }

    def values(name, kernel, method, metric):
        return np.array(
            [sweeps[name][k]["kernels"][kernel][method][metric] for k in keys]
        )

    for name in sweeps:
        entry = {}
        for kernel in kernels:
            learned = values(name, kernel, "learned_mixture", "alignment_centered")
            paper = PAPER_TABLE1[kernel]
            entry[kernel] = {
                "learned_alignment_centered": {
                    "mean": float(learned.mean()),
                    "std": float(learned.std(ddof=1)),
                    "paper": paper,
                    "z_paper": float((paper - learned.mean()) / learned.std(ddof=1)),
                    "fraction_at_or_above_paper": float(np.mean(learned >= paper)),
                },
                "learned_minus_uniform": {
                    metric: paired_summary(
                        values(name, kernel, "learned_mixture", metric)
                        - values(name, kernel, "uniform_mixture", metric)
                    )
                    for metric in METRICS
                },
                "learned_minus_best_single": {
                    metric: paired_summary(
                        values(name, kernel, "learned_mixture", metric)
                        - values(name, kernel, "best_single_kernel", metric)
                    )
                    for metric in METRICS
                },
            }
        result["per_sweep"][name] = entry
    for name in sweeps:
        if name == reference:
            continue
        result["between_sweeps"][f"{name}_minus_{reference}"] = {
            kernel: {
                metric: paired_summary(
                    values(name, kernel, "learned_mixture", metric)
                    - values(reference, kernel, "learned_mixture", metric)
                )
                for metric in METRICS
            }
            for kernel in kernels
        }
    return result


def to_markdown(result: dict) -> str:
    """Render the main comparison tables."""
    lines = ["# Table I comparison (test centered alignment, learned mixture)", ""]
    lines += [
        "| Sweep | Kernel | Mean ± std (n) | Paper | z | Seeds ≥ paper |",
        "|---|---|---|---|---|---|",
    ]
    for name, entry in result["per_sweep"].items():
        for kernel, stats in entry.items():
            s = stats["learned_alignment_centered"]
            n = stats["learned_minus_uniform"]["alignment_centered"]["n"]
            lines.append(
                f"| {name} | {kernel} | {s['mean']:.4f} ± {s['std']:.4f} ({n}) | {s['paper']:.4f} "
                f"| {s['z_paper']:+.2f} | {s['fraction_at_or_above_paper']:.2f} |"
            )
    lines += ["", "## Learned minus uniform mixture (paired)", ""]
    lines += [
        "| Sweep | Kernel | Metric | Mean diff [95% CI] | Wilcoxon p | #>0 / n |",
        "|---|---|---|---|---|---|",
    ]
    for name, entry in result["per_sweep"].items():
        for kernel, stats in entry.items():
            for metric in ("alignment_centered", "rmse", "avg_nlpd"):
                d = stats["learned_minus_uniform"][metric]
                lines.append(
                    f"| {name} | {kernel} | {metric} | {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
                    f"| {d['wilcoxon_p']:.3g} | {d['num_positive']}/{d['n']} |"
                )
    lines += ["", "## Between sweeps (learned mixture, paired)", ""]
    lines += [
        "| Comparison | Kernel | Metric | Mean diff [95% CI] | Wilcoxon p |",
        "|---|---|---|---|---|",
    ]
    for name, entry in result["between_sweeps"].items():
        for kernel, metrics in entry.items():
            for metric in ("alignment_centered", "rmse", "avg_nlpd"):
                d = metrics[metric]
                lines.append(
                    f"| {name} | {kernel} | {metric} | {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
                    f"| {d['wilcoxon_p']:.3g} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweeps", nargs="+", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="Output path without extension"
    )
    args = parser.parse_args()
    sweeps = {path.name: load_sweep(path.resolve()) for path in args.sweeps}
    result = compare(sweeps, args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    markdown = to_markdown(result)
    args.out.with_suffix(".md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
