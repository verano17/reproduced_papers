"""Compare a sweep's Fig. 7 cells with the values read from the paper.

Reads ``summary.json`` of each passed sweep (``utils/summarize_fig7.py``) and
``results/paper_values_fig7_fig8.json``, and writes one row per (sweep,
dataset, split, model, h) where the paper value was read: reproduced
best-of-10 (best-on-train trial and min-over-trials), mean +- SEM over the 10
trials, the paper value, and the difference, all in the paper's metric
NLL - log2(N)/n (LOG.md R1). A per-(sweep, dataset, model, split) RMS
difference over h >= 2 is written alongside.

Usage::

    python3 utils/compare_with_paper.py <SWEEP_DIR> [<SWEEP_DIR> ...] \
        --paper-values results/paper_values_fig7_fig8.json --out results/fig7_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

METRIC = "paper_kl_estimate_bits_per_site"


def compare(
    sweep_dirs: list[Path], paper_values: dict
) -> tuple[list[dict], list[dict]]:
    """Return (rows, rms) comparison tables."""
    rows = []
    for sweep_dir in sweep_dirs:
        summary = json.loads((sweep_dir / "summary.json").read_text())
        for cell in summary["cells"]:
            paper_dataset = paper_values["fig7_best_of_ten"].get(cell["dataset"])
            if not paper_dataset or not cell["successful_runs"]:
                continue
            for split in ("train", "test"):
                paper_value = paper_dataset[split][cell["model"]].get(
                    str(cell["bond_dim"])
                )
                if paper_value is None:
                    continue
                reproduced = cell[f"best_train_{split}_{METRIC}"]
                rows.append(
                    {
                        "sweep_id": summary["sweep_id"],
                        "dataset": cell["dataset"],
                        "split": split,
                        "model": cell["model"],
                        "bond_dim": cell["bond_dim"],
                        "trials": cell["successful_runs"],
                        "reproduced_best_train_trial": reproduced,
                        "reproduced_min_over_trials": cell[f"min_{split}_{METRIC}"],
                        "reproduced_mean": cell[f"mean_{split}_{METRIC}"],
                        "reproduced_sem": cell[f"sem_{split}_{METRIC}"],
                        "paper_plot_read": paper_value,
                        "difference_best_train_minus_paper": reproduced - paper_value,
                    }
                )
    rms = []
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        if row["bond_dim"] < 2:
            continue
        key = (row["sweep_id"], row["dataset"], row["model"], row["split"])
        groups.setdefault(key, []).append(row["difference_best_train_minus_paper"])
    for (sweep_id, dataset, model, split), differences in groups.items():
        rms.append(
            {
                "sweep_id": sweep_id,
                "dataset": dataset,
                "model": model,
                "split": split,
                "points": len(differences),
                "rms_difference": math.sqrt(
                    sum(d * d for d in differences) / len(differences)
                ),
                "mean_difference": sum(differences) / len(differences),
            }
        )
    return rows, rms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    parser.add_argument("--paper-values", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output path stem")
    args = parser.parse_args(argv)
    rows, rms = compare(
        [path.resolve() for path in args.sweep_dirs],
        json.loads(args.paper_values.read_text()),
    )
    args.out.with_suffix(".json").write_text(
        json.dumps({"rows": rows, "rms": rms, "metric": METRIC}, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for entry in rms:
        print(
            f"{entry['sweep_id']:38s} {entry['dataset']:24s} {entry['model']:15s} "
            f"{entry['split']:5s} RMS {entry['rms_difference']:.4f} "
            f"mean {entry['mean_difference']:+.4f} (n={entry['points']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
