"""Re-plot Fig. 3 (the synthetic GP datasets) for one run.

Regenerates the run's dataset from its ``config_snapshot.json`` with
``lib.data.make_dataset`` (deterministic in the data seed), checks it against
the dataset SHA-256 recorded in ``run_status.json``, and saves the figure to
the run folder and to ``results/``.

Usage:
    python utils/plot_fig3_datasets.py --run-dir outdir/run_... \
        --label "synthetic-data, code settings" --results-dir results --prefix table1_original_paperseed
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR.parent))

from randomized_quantum_local_kernels_gp.lib.data import make_dataset  # noqa: E402
from randomized_quantum_local_kernels_gp.lib.plotting import plot_datasets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "config_snapshot.json").read_text())
    recorded = json.loads((run_dir / "run_status.json").read_text())["dataset"][
        "sha256"
    ]
    dataset = make_dataset(cfg)
    if dataset.fingerprint() != recorded:
        raise ValueError(
            f"Regenerated dataset {dataset.fingerprint()} differs from the run's {recorded}."
        )
    names = tuple(f"{name} ({args.label})" for name in dataset.kernel_names)
    output = run_dir / f"{args.prefix}_fig3_datasets.png"
    plot_datasets(dataset.inputs, dataset.labels, names, output)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output, args.results_dir / output.name)
    print(
        f"Wrote {output.name} to {run_dir} and {args.results_dir} (dataset sha256 {recorded[:12]})"
    )


if __name__ == "__main__":
    main()
