"""Re-plot Fig. 4 (test predictions) and Fig. 5 (RBF Gram heatmaps) for one run.

Reads ``predictions.npz`` from the given run folder, adds a result-scope
label to the titles, and saves the figures into the run folder and into
``results/`` with a caller-chosen prefix.

Usage:
    python utils/plot_run_figures.py --run-dir outdir/run_... \
        --label "preliminary, single seed" --results-dir results --prefix table1_original
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
# Repository root (for runtime_lib) and papers/ (for the paper package).
sys.path.insert(0, str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR.parent))

from randomized_quantum_local_kernels_gp.lib.plotting import (  # noqa: E402
    plot_gram_heatmaps,
    plot_test_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    predictions = np.load(run_dir / "predictions.npz")
    names = tuple(f"{name} ({args.label})" for name in predictions["kernel_names"])
    args.results_dir.mkdir(parents=True, exist_ok=True)

    fig4 = run_dir / f"{args.prefix}_fig4_test_predictions.png"
    plot_test_predictions(
        predictions["test_inputs"],
        predictions["test_targets"],
        predictions["means"],
        predictions["predictive_std"],
        names,
        fig4,
    )
    outputs = [fig4]
    for index, name in enumerate(predictions["kernel_names"]):
        heatmap = run_dir / f"{args.prefix}_fig5_gram_heatmap_{name}.png"
        plot_gram_heatmaps(
            predictions["true_test_grams"][index],
            predictions["learned_test_grams"][index],
            f"{name} ({args.label})",
            heatmap,
        )
        outputs.append(heatmap)
    for path in outputs:
        shutil.copyfile(path, args.results_dir / path.name)
    print(
        "Wrote "
        + ", ".join(path.name for path in outputs)
        + f" to {run_dir} and {args.results_dir}"
    )


if __name__ == "__main__":
    main()
