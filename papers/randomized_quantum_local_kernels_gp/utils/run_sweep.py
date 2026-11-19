"""Coordinator for repetition sweeps over data and Sobol seeds.

Runs one base config for every (data_seed, sobol_seed) pair, sequentially,
through the shared runtime. Each run gets an overlay config that carries the
sweep ID, candidate, seeds and repetition index, so its ``run_status.json``
matches the sweep entry. The coordinator alone writes ``sweep.log`` and
``sweep_status.json`` under ``outdir/sweeps/<sweep_id>/``; each run keeps its
own ``run.log`` in its run folder there.

Usage (from the paper folder):
    python utils/run_sweep.py --sweep-id S1_table1_original \
        --config configs/table1_original.json \
        --data-seeds 12345 1 2 3 4 --sobol-seeds 12345 1 2
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_DIR.parent))

from randomized_quantum_local_kernels_gp.lib.experiment_logging import (  # noqa: E402
    configure_logging,
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)
from randomized_quantum_local_kernels_gp.utils.summarize_sweep import write_summary  # noqa: E402
from runtime_lib.config import deep_update, load_config  # noqa: E402

logger = logging.getLogger("sweep")


def _run_entry(
    sweep_dir: Path,
    run_dir: Path | None,
    run_id: str,
    candidate: dict,
    seed: dict,
    repetition: int,
    status: str,
) -> dict:
    relative = run_dir.relative_to(sweep_dir).as_posix() if run_dir else None
    return {
        "run_id": run_id,
        "candidate": candidate,
        "seed": seed,
        "repetition": repetition,
        "status": status,
        "run_dir": relative,
        "run_log": f"{relative}/run.log" if relative else None,
        "run_status": f"{relative}/run_status.json" if relative else None,
        "config_path": f"{relative}/config_snapshot.json" if relative else None,
        "metrics_path": f"{relative}/metrics.json" if relative else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-seeds", required=True, type=int, nargs="+")
    parser.add_argument("--sobol-seeds", required=True, type=int, nargs="+")
    parser.add_argument(
        "--outdir", type=Path, default=PROJECT_DIR / "outdir" / "sweeps"
    )
    args = parser.parse_args()

    sweep_dir = (args.outdir / args.sweep_id).resolve()
    if sweep_dir.exists():
        raise FileExistsError(f"Sweep directory already exists: {sweep_dir}")
    overlay_dir = sweep_dir / "run_configs"
    overlay_dir.mkdir(parents=True)
    configure_logging(sweep_dir / "sweep.log")

    base_config_path = (
        (PROJECT_DIR / args.config).resolve()
        if not args.config.is_absolute()
        else args.config
    )
    resolved = deep_update(
        load_config(PROJECT_DIR / "configs" / "defaults.json"),
        load_config(base_config_path),
    )
    candidate_name = base_config_path.stem
    repetitions = [
        {"index": index, "seeds": [data_seed, sobol_seed]}
        for index, (data_seed, sobol_seed) in enumerate(
            (d, s) for d in args.data_seeds for s in args.sobol_seeds
        )
    ]
    status = {
        "schema_version": 1,
        "sweep_id": args.sweep_id,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "expected_runs": len(repetitions),
        "runs": [],
        "summary_path": None,
        "selected_candidates": [],
        "error": None,
        "plan": {
            "candidate": {
                "config": candidate_name,
                "feature_mode": resolved["features"]["mode"],
            },
            "repetitions": repetitions,
            "selection": "none: repetition sweep for seed variability",
            "metric": "alignment_centered (test), plus RMSE, NLPD, coverage",
        },
    }
    write_json(sweep_dir / "sweep_status.json", status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | config=%s",
        args.sweep_id,
        len(repetitions),
        args.config,
    )

    try:
        for repetition in repetitions:
            data_seed, sobol_seed = repetition["seeds"]
            candidate = {
                "config": candidate_name,
                "data_seed": data_seed,
                "sobol_seed": sobol_seed,
            }
            run_key = f"data{data_seed}_sobol{sobol_seed}"
            overlay = {
                "description": f"Sweep {args.sweep_id} overlay of {args.config} ({run_key}).",
                "experiment": {
                    "sweep_id": args.sweep_id,
                    "candidate": candidate,
                    "repetition": repetition["index"],
                },
                "data": {"seed": data_seed},
                "circuit": {"sobol_seed": sobol_seed},
            }
            merged = deep_update(json.loads(json.dumps(resolved)), overlay)
            overlay_path = overlay_dir / f"{run_key}.json"
            overlay_path.write_text(json.dumps(merged, indent=2) + "\n")
            seed = {
                "global": merged.get("seed"),
                "data": data_seed,
                "sobol": sobol_seed,
                "features": merged["features"].get("seed"),
            }
            existing = set(sweep_dir.glob("run_*"))
            pending = _run_entry(
                sweep_dir,
                None,
                run_key,
                candidate,
                seed,
                repetition["index"],
                "RUNNING",
            )
            status["runs"].append(pending)
            write_json(sweep_dir / "sweep_status.json", status)
            logger.info(
                "CANDIDATE_STARTED | run_id=%s | candidate=%s", run_key, candidate
            )
            logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))

            command = [
                sys.executable,
                str(REPO_ROOT / "implementation.py"),
                "--paper",
                PROJECT_DIR.name,
                "--config",
                str(overlay_path),
                "--outdir",
                str(sweep_dir),
            ]
            completed = subprocess.run(
                command, cwd=REPO_ROOT, capture_output=True, text=True
            )
            new_dirs = sorted(set(sweep_dir.glob("run_*")) - existing)
            run_dir = new_dirs[-1] if new_dirs else None
            run_status = None
            if run_dir and (run_dir / "run_status.json").exists():
                run_status = json.loads((run_dir / "run_status.json").read_text())
            succeeded = (
                completed.returncode == 0
                and run_status is not None
                and run_status["status"] == "COMPLETED"
                and (run_dir / "metrics.json").exists()
            )
            # The run's own ID is its folder name (written by the runner), so
            # the sweep entry uses it once the folder exists.
            entry = _run_entry(
                sweep_dir,
                run_dir,
                run_dir.name if run_dir else run_key,
                candidate,
                seed,
                repetition["index"],
                "COMPLETED" if succeeded else "FAILED",
            )
            status["runs"][-1] = entry
            write_json(sweep_dir / "sweep_status.json", status)
            if succeeded:
                logger.info(
                    "CANDIDATE_COMPLETED | run_id=%s | metrics=%s",
                    run_key,
                    entry["metrics_path"],
                )
            else:
                tail = (completed.stderr or "")[-2000:]
                logger.error(
                    "CANDIDATE_FAILED | run_id=%s | returncode=%d | stderr_tail=%s",
                    run_key,
                    completed.returncode,
                    tail,
                )
            logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))

        summary = write_summary(sweep_dir)
        status["summary_path"] = "summary.json"
        all_completed = all(run["status"] == "COMPLETED" for run in status["runs"])
        status["status"] = (
            "COMPLETED"
            if all_completed and summary["coverage"] == "COMPLETE SWEEP"
            else "PARTIAL"
        )
        status["completed_at"] = utc_now()
        write_json(sweep_dir / "sweep_status.json", status)
        if status["status"] == "COMPLETED":
            logger.info(
                "SWEEP_COMPLETED | summary=%s | selected=none (repetition sweep)",
                "summary.json",
            )
        else:
            logger.info(
                "SWEEP_PARTIAL | summary=%s | missing_or_failed=%s",
                "summary.json",
                summary["missing_or_failed"],
            )
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(sweep_dir / "sweep_status.json", status)
        logger.exception("SWEEP_FAILED | sweep_id=%s", args.sweep_id)
        raise


if __name__ == "__main__":
    main()
