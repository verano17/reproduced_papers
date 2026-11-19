"""Phase 5 figure: test centered alignment vs test RMSE for every system.

Reads the structured comparison produced by ``utils/compare_baselines.py``
(``<input>.json``); one panel per query kernel, mean ± std over 15 runs,
marker shape by role. The paper's Table I alignment is drawn as a vertical
line (paper-reported, single run, no RMSE reported). All local values are
synthetic-data; MerLin points are a simulated adaptation.

Usage:
    python utils/plot_phase5_comparison.py --input results/phase5_comparison.json \
        --output results/phase5_alignment_vs_rmse.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KERNELS = ("rbf", "matern32", "nonstat_ls")
MARKERS = {"authors": "o", "converged": "s", "reference": "^"}
COLORS = {
    "paper method": "tab:blue",
    "paper method, converged optimizer": "tab:blue",
    "paper method, shot-noise-free diagnostic": "tab:cyan",
    "MerLin adaptation": "tab:purple",
    "fair classical baseline": "tab:orange",
    "classical baseline": "tab:green",
    "oracle reference": "black",
    "standard practice": "tab:gray",
}


def plot(comparison: dict, output: Path) -> None:
    """Draw alignment vs RMSE per kernel."""
    figure, axes = plt.subplots(1, len(KERNELS), figsize=(15, 4.6))
    for axis, kernel in zip(axes, KERNELS):
        for row in comparison["table"]:
            align = row["metrics"][kernel]["alignment_centered"]
            rmse = row["metrics"][kernel]["rmse"]
            axis.errorbar(
                align["mean"],
                rmse["mean"],
                xerr=align["std"],
                yerr=rmse["std"],
                fmt=MARKERS[row["group"]],
                color=COLORS[row["role"]],
                capsize=3,
                label=row["system"],
            )
        axis.axvline(
            comparison["paper_table1"][kernel],
            color="red",
            linestyle="--",
            linewidth=1,
            label="paper Table I alignment (single run)",
        )
        axis.set_title(kernel)
        axis.set_xlabel("test centered alignment (higher is better)")
        axis.set_ylabel("test RMSE (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=7)
    figure.suptitle(
        "Phase 5: alignment vs predictive error (synthetic-data, 15 runs per system; "
        "circles = authors' MKL protocol, squares = converged MKL, triangles = parametric GPs; "
        "MerLin = simulated adaptation)",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.14, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.input.read_text()), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
