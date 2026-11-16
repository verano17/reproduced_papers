"""Split-seed robustness of the Promoters held-out result (post-Phase-5 check).

The fair comparison (Phase 5) uses one random test quarter (split seed 0).
This script repeats the validation-tuned protocol (arm B) on further split
seeds, with the hyperparameters selected on split seed 0 (not retuned per
split), for BE, B1, B2 and B3 at the planned bond dimensions, 10 seeds per
cell. Each run goes through ``runner.train_and_evaluate`` and writes the
usual run artifacts under ``<out>/runs/``; the per-run metrics are collected
into ``<out>/robustness_runs.csv`` and the per-split decision into
``<out>/robustness_summary.json``.

Decision per (split seed, h), on test NLL at the best-validation
checkpoint: BE beats each of B1, B2, B3 if mean(control) - mean(BE) >
2 sqrt(SEM_BE^2 + SEM_control^2) (the Phase 5 separation rule).

Usage::

    python3 utils/robustness_split_seeds.py --selection results/phase5_tuning_selection.json \
        --split-seeds 1 2 3 4 5 --bond-dims 8 12 --workers 6 --out outdir/robustness_split_seeds_<stamp>
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
ARM = "promoter_gene_sequences"
ALPHABET = 4
CONFIG = {
    "BE": "e5_fig7_promoters_val_original",
    "B1": "e5_fig7_promoters_val_classical",
    "B2": "e5_fig7_promoters_val_gradient",
}
SEEDS = list(range(10))


def _paths() -> None:
    for path in (REPO_ROOT, PROJECT_DIR.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def selected_values(selection_path: Path) -> dict:
    """Return the arm-B values selected on split seed 0 for this arm."""
    selection = json.loads(selection_path.read_text())
    chosen = {
        (s["model"], s["order"]): s["selected"]
        for s in selection["selections"]
        if s["arm"] == ARM
    }
    return {
        "BE": {"training__lr": chosen[("basis_enhanced", None)]},
        "B1": {"training__pseudo_count": chosen[("hmm_baum_welch", None)]},
        "B2": {"training__lr": chosen[("hmm_gradient", None)]},
    }


def planned_runs(split_seeds, bond_dims, values) -> list[dict]:
    _paths()
    from quantum_correlations_generative_models.lib.basis_enhanced import (
        free_parameter_count,
        parameter_matched_hmm_bond_dim,
    )

    runs = []
    for split_seed in split_seeds:
        for h in bond_dims:
            matched = parameter_matched_hmm_bond_dim(
                free_parameter_count(h, ALPHABET), ALPHABET
            )
            for label, model_bond_dim, config_label in (
                ("BE", h, "BE"),
                ("B1", h, "B1"),
                ("B2", h, "B2"),
                ("B3", matched, "B1"),
            ):
                for seed in SEEDS:
                    runs.append(
                        {
                            "split_seed": split_seed,
                            "h": h,
                            "label": label,
                            "bond_dim": model_bond_dim,
                            "config": CONFIG[config_label],
                            "overrides": values[config_label],
                            "seed": seed,
                        }
                    )
    return runs


def run_one(job: dict, out_dir: str) -> dict:
    _paths()
    import torch

    torch.set_num_threads(1)
    from quantum_correlations_generative_models.lib import runner

    from runtime_lib.config import deep_update, load_config
    from runtime_lib.dtypes import resolve_config_dtypes

    cfg = deep_update(
        load_config(PROJECT_DIR / "configs" / "defaults.json"),
        load_config(PROJECT_DIR / "configs" / f"{job['config']}.json"),
    )
    cfg["seed"] = job["seed"]
    cfg["model"]["bond_dim"] = job["bond_dim"]
    cfg["dataset"]["split_seed"] = job["split_seed"]
    for dotted, value in job["overrides"].items():
        section, key = dotted.split("__")
        cfg[section][key] = value
    cfg = resolve_config_dtypes(cfg)
    run_dir = (
        Path(out_dir)
        / "runs"
        / f"split{job['split_seed']}_{job['label']}_h{job['h']:02d}_bd{job['bond_dim']:02d}_s{job['seed']}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    with contextlib.redirect_stderr(io.StringIO()):
        runner.train_and_evaluate(cfg, run_dir)
    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    evaluation = metrics["checkpoint"]["evaluation"]
    return {
        **{k: job[k] for k in ("split_seed", "h", "label", "bond_dim", "seed")},
        "status": status["status"],
        "checkpoint_step": metrics["checkpoint"]["step"],
        "validation_nll": evaluation["validation"]["nll_bits_per_site"],
        "test_nll": evaluation["test"]["nll_bits_per_site"],
        "unitarity_error_at_checkpoint": metrics["history"][
            metrics["checkpoint"]["index"]
        ].get("unitarity_error"),
        "run_dir": str(run_dir.relative_to(out_dir)),
    }


def summarise(rows: list[dict]) -> dict:
    def stats(values):
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        return mean, sd / math.sqrt(len(values))

    valid = [
        r
        for r in rows
        if r["status"] == "COMPLETED"
        and (
            r["unitarity_error_at_checkpoint"] is None
            or r["unitarity_error_at_checkpoint"] <= 1e-6
        )
    ]
    decisions = []
    for split_seed in sorted({r["split_seed"] for r in rows}):
        for h in sorted({r["h"] for r in rows}):
            cell = {
                label: [
                    r["test_nll"]
                    for r in valid
                    if (r["split_seed"], r["h"], r["label"]) == (split_seed, h, label)
                ]
                for label in ("BE", "B1", "B2", "B3")
            }
            entry = {
                "split_seed": split_seed,
                "h": h,
                "runs": {k: len(v) for k, v in cell.items()},
            }
            be_mean, be_sem = stats(cell["BE"])
            entry["BE_mean_test"] = be_mean
            beats = []
            for label in ("B1", "B2", "B3"):
                mean, sem = stats(cell[label])
                advantage = mean - be_mean
                margin = 2 * math.hypot(be_sem, sem)
                entry[f"{label}_mean_test"] = mean
                entry[f"BE_advantage_over_{label}"] = advantage
                entry[f"margin_{label}"] = margin
                beats.append(advantage > margin)
            entry["BE_beats_B1_B2_B3"] = all(beats)
            decisions.append(entry)
    return {"decisions": decisions, "excluded_runs": len(rows) - len(valid)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--bond-dims", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    values = selected_values(args.selection)
    jobs = planned_runs(args.split_seeds, args.bond_dims, values)
    (out_dir / "plan.json").write_text(
        json.dumps({"values": values, "runs": jobs}, indent=2) + "\n"
    )
    print(f"{len(jobs)} runs -> {out_dir}", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_one, job, str(out_dir)) for job in jobs]
        for done, future in enumerate(futures, start=1):
            rows.append(future.result())
            if done % 20 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} runs finished", flush=True)
    with (out_dir / "robustness_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows)
    (out_dir / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    for entry in summary["decisions"]:
        print(
            f"split {entry['split_seed']} h={entry['h']:2d}: BE beats B1, B2, B3 = {entry['BE_beats_B1_B2_B3']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
