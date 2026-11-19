"""Plot Table I: test centered alignment per query kernel across sweeps.

Each sweep is drawn as per-seed points plus mean ± std for the learned
mixture, with the uniform-mixture mean as a hollow marker and the paper's
Table I value as a horizontal tick. Reads only ``sweep_status.json`` and the
runs' ``metrics.json``; writes the PNG into every sweep folder and into
``results/``.

Usage:
    python utils/plot_table1_alignment.py --sweeps outdir/sweeps/S1_... outdir/sweeps/S2_... \
        --label "partial, synthetic-data: 15 seeds" --results-dir results
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_sweeps import PAPER_TABLE1, load_sweep  # noqa: E402

KERNELS = ("rbf", "matern32", "nonstat_ls")


def plot(sweep_dirs: list[Path], label: str, output: Path) -> None:
    """Draw the alignment comparison figure to ``output``."""
    sweeps = {path.name: load_sweep(path) for path in sweep_dirs}
    figure, axes = plt.subplots(
        1, len(KERNELS), figsize=(4.2 * len(KERNELS), 4.0), sharey=True
    )
    offsets = np.linspace(-0.25, 0.25, len(sweeps))
    for axis, kernel in zip(axes, KERNELS):
        for offset, (name, runs) in zip(offsets, sweeps.items()):
            learned = np.array(
                [
                    m["kernels"][kernel]["learned_mixture"]["alignment_centered"]
                    for m in runs.values()
                ]
            )
            uniform = np.array(
                [
                    m["kernels"][kernel]["uniform_mixture"]["alignment_centered"]
                    for m in runs.values()
                ]
            )
            x = np.full(learned.size, offset) + np.random.default_rng(0).uniform(
                -0.03, 0.03, learned.size
            )
            points = axis.scatter(x, learned, s=10, alpha=0.5)
            axis.errorbar(
                offset,
                learned.mean(),
                yerr=learned.std(ddof=1),
                fmt="o",
                color=points.get_facecolor()[0],
                capsize=4,
                label=f"{name} learned",
            )
            axis.scatter(
                offset + 0.06,
                uniform.mean(),
                marker="D",
                facecolors="none",
                edgecolors=points.get_facecolor()[0],
                label=f"{name} uniform (mean)",
            )
        axis.axhline(
            PAPER_TABLE1[kernel],
            color="black",
            linestyle="--",
            linewidth=1,
            label="paper Table I",
        )
        axis.set_title(kernel)
        axis.set_xticks([])
        axis.set_xlabel("sweep")
    axes[0].set_ylabel("test centered kernel alignment")
    axes[-1].legend(fontsize=7, loc="lower right")
    figure.suptitle(f"Table I reproduction ({label})", fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweeps", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--label", required=True, help="Result-scope label shown on the figure"
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--name", default="table1_alignment.png")
    args = parser.parse_args()
    sweep_dirs = [path.resolve() for path in args.sweeps]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    target = args.results_dir / args.name
    plot(sweep_dirs, args.label, target)
    for sweep_dir in sweep_dirs:
        shutil.copyfile(target, sweep_dir / args.name)
    print(f"Wrote {target} and copies in {len(sweep_dirs)} sweep folders")


if __name__ == "__main__":
    main()
