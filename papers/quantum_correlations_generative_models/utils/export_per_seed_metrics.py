"""Export one row per planned run (per seed) of the evidence sweeps to a CSV.

The notebook shows individual seeds and must not depend on the disposable
``outdir/``. This script reads each sweep's ``sweep_status.json`` and the
``metrics.json`` of every run, and writes ``results/per_seed_metrics.csv``.

Columns: sweep_id, dataset, arm (dataset, ``_pooled`` for the pooled SPECT
split), protocol (``A`` paper protocol, ``B`` validation-tuned), model,
bond_dim, hparams (JSON), seed, status (``valid`` / ``invalid`` /
``failed``), checkpoint_rule, checkpoint_step, nll_bits_per_site on
train / validation / test at the reported checkpoint, the paper metric
NLL - log2(N)/n on train / test, unitarity error at the checkpoint (BE / B4),
free_parameter_count, wall_seconds.

A run is ``invalid`` when its unitarity error at the checkpoint exceeds
1e-6 (LOG.md, Phase 5 amendment), ``failed`` when no attempt completed.

Usage::

    python3 utils/export_per_seed_metrics.py <SWEEP_DIR> [...] --out results/per_seed_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PAPERS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPERS_DIR))
from quantum_correlations_generative_models.lib.metrics import (  # noqa: E402
    paper_kl_estimate_bits_per_site,
)

UNITARITY_TOLERANCE = 1e-6
COLUMNS = [
    "sweep_id", "dataset", "arm", "protocol", "model", "bond_dim", "hparams", "seed",
    "status", "checkpoint_rule", "checkpoint_step",
    "train_nll_bits_per_site", "validation_nll_bits_per_site", "test_nll_bits_per_site",
    "train_paper_metric", "test_paper_metric", "unitarity_error_at_checkpoint",
    "free_parameter_count", "wall_seconds",
]  # fmt: skip


def _label(metrics: dict) -> str:
    kind = metrics["model_kind"]
    return kind + (
        "_real" if kind == "basis_enhanced" and metrics["field"] == "real" else ""
    )


def _variant(status: dict, run: dict) -> str:
    if "sweep_candidate" in run:
        return run["sweep_candidate"]["config_variant"]
    return status["plan"].get("config_variant") or ""


def export(sweep_dirs: list[Path]) -> list[dict]:
    rows = []
    for sweep_dir in sweep_dirs:
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        attempts: dict[tuple, list] = {}
        for run in status["runs"]:
            candidate = run.get("sweep_candidate", run["candidate"])
            key = (json.dumps(candidate, sort_keys=True), run["seed"])
            attempts.setdefault(key, []).append(run)
        for (candidate_json, seed), runs in sorted(attempts.items()):
            candidate = json.loads(candidate_json)
            variant = _variant(status, runs[0])
            row = {
                "sweep_id": status["sweep_id"],
                "dataset": candidate["dataset"],
                "arm": candidate["dataset"] + ("_pooled" if "_pooled" in variant else ""),
                "protocol": "B" if variant.endswith("_val") else "A",
                "model": candidate["model"] if "sweep_candidate" in runs[0]
                else candidate["model"] + ("_real" if candidate.get("field") == "real"
                                           and candidate["model"] == "basis_enhanced" else ""),
                "bond_dim": candidate["bond_dim"],
                "hparams": json.dumps(candidate.get("hparams", {}), sort_keys=True),
                "seed": seed,
            }  # fmt: skip
            completed = [r for r in runs if r["status"] == "COMPLETED"]
            if not completed:
                rows.append({**row, "status": "failed"})
                continue
            run = completed[-1]
            metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
            row["model"] = _label(metrics)
            checkpoint = metrics.get("checkpoint") or {
                "rule": "final",
                "step": metrics["history"][-1].get(
                    "epoch", metrics["history"][-1].get("iteration")
                ),
                "index": len(metrics["history"]) - 1,
                "evaluation": metrics["final"],
            }
            evaluation = checkpoint["evaluation"]
            drift = metrics["history"][checkpoint["index"]].get("unitarity_error")
            length = metrics["sequence_length"]
            row.update(
                {
                    "status": "invalid"
                    if drift is not None and drift > UNITARITY_TOLERANCE
                    else "valid",
                    "checkpoint_rule": checkpoint["rule"],
                    "checkpoint_step": checkpoint["step"],
                    "unitarity_error_at_checkpoint": drift,
                    "free_parameter_count": metrics["free_parameter_count"],
                    "wall_seconds": round(metrics["wall_seconds"], 3),
                }
            )
            for split in ("train", "validation", "test"):
                if split in evaluation:
                    row[f"{split}_nll_bits_per_site"] = evaluation[split][
                        "nll_bits_per_site"
                    ]
            for split in ("train", "test"):
                row[f"{split}_paper_metric"] = paper_kl_estimate_bits_per_site(
                    evaluation[split]["nll_bits_per_site"],
                    evaluation[split]["num_sequences"],
                    length,
                )
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = export([d.resolve() for d in args.sweep_dirs])
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[tuple, int] = {}
    for row in rows:
        key = (row["sweep_id"], row["status"])
        counts[key] = counts.get(key, 0) + 1
    for key, value in sorted(counts.items()):
        print(*key, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
