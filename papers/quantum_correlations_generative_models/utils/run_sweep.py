"""Coordinator for the E5 (Fig. 7) sweep: datasets x models x bond dimensions x seeds.

Each planned run is one ``implementation.py`` call in its own process with
its own ``--outdir`` (so parallel runs never share a run directory or a log
file) and one thread (``OMP_NUM_THREADS=1``) to avoid oversubscription. The
coordinator alone writes ``sweep.log`` and ``sweep_status.json``. A planned
run gets at most two attempts; a failed attempt stays in ``runs`` as FAILED.
Progress for running runs is the number of completed epochs / EM iterations
counted in that run's own ``run.log`` (progress only; metrics come from
``metrics.json``). At the end ``utils/summarize_fig7.py`` builds the summary
from structured artifacts.

Phase 5 sweeps are described by a JSON spec (``--spec``): a list of blocks,
each with ``datasets``, ``config_variant``, ``model_kinds``, ``bond_dims`` (a
list, or a mapping dataset -> list), ``seeds`` and an optional ``grid`` of
CLI hyperparameters (e.g. ``{"lr": [0.001, 0.01]}``); every grid point is one
candidate. ``kgram_counting`` has no bond dimension (runs at ``bond_dim`` 0)
and is deterministic.

Usage (from the repository root)::

    python3 papers/quantum_correlations_generative_models/utils/run_sweep.py \
        --sweep-name e5_fig7 --workers 6
    python3 papers/quantum_correlations_generative_models/utils/run_sweep.py \
        --sweep-name phase5_tuning --spec papers/quantum_correlations_generative_models/configs/sweeps/phase5_tuning.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
PAPER = PROJECT_DIR.name
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from lib.experiment_logging import (  # noqa: E402
    configure_logging,
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)
from utils.summarize_fig7 import summarize  # noqa: E402

logger = logging.getLogger("sweep")
DATASET_CONFIG = {
    "biofam": "biofam",
    "promoter_gene_sequences": "promoters",
    "spect_heart": "spect",
}
# Sweep model label -> config suffix. ``basis_enhanced_real`` is the
# basis_enhanced kind with field "real" (Phase 5 control B4).
KIND_SUFFIX = {
    "basis_enhanced": "original",
    "hmm_baum_welch": "classical",
    "photonic_merlin": "merlin",
    "basis_enhanced_real": "real",
    "hmm_gradient": "gradient",
    "kgram_counting": "kgram",
}
NO_BOND_DIM_KINDS = {"kgram_counting"}
# The paper's two models; the Phase 4 MerLin adaptation is opt-in (--model-kinds).
DEFAULT_MODEL_KINDS = ["basis_enhanced", "hmm_baum_welch"]
COMPLEX_KINDS = {"basis_enhanced", "photonic_merlin"}
PROGRESS_PATTERN = re.compile(
    r"TRAIN_(EPOCH|ITERATION)_COMPLETED \| (?:epoch|iteration)=(\d+)/(\d+)"
)
# Relative cost used only to launch the longest runs first.
COST = {
    "biofam": 1500 * 16 * 75,
    "promoter_gene_sequences": 80 * 57 * 150,
    "spect_heart": 80 * 23 * 150,
}


def code_fingerprint() -> str:
    """Hash the paper's code and configs so every run of the sweep uses the same state."""
    digest = hashlib.sha256()
    for path in sorted(
        [
            *PROJECT_DIR.glob("lib/*.py"),
            *PROJECT_DIR.glob("configs/*.json"),
            PROJECT_DIR / "cli.json",
        ]
    ):
        digest.update(path.relative_to(PROJECT_DIR).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def blocks_from_args(args) -> list[dict]:
    """Return the sweep blocks: the ``--spec`` file, or one block from the CLI."""
    if args.spec:
        blocks = json.loads(Path(args.spec).read_text())["blocks"]
        for block in blocks:
            unknown = set(block["model_kinds"]) - set(KIND_SUFFIX)
            if unknown:
                raise ValueError(f"Unknown model kinds in spec: {sorted(unknown)}")
        return blocks
    return [
        {
            "datasets": args.datasets,
            "config_variant": args.config_variant,
            "model_kinds": args.model_kinds,
            "bond_dims": args.bond_dims,
            "seeds": args.seeds,
            "grid": {},
        }
    ]


def _grid_points(grid: dict) -> list[dict]:
    keys = sorted(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[k] for k in keys))]  # fmt: skip


def planned_runs(blocks: list[dict]) -> list[dict]:
    runs = []
    for block in blocks:
        for dataset in block["datasets"]:
            bond_dims = block["bond_dims"]
            if isinstance(bond_dims, dict):
                bond_dims = bond_dims[dataset]
            for kind in block["model_kinds"]:
                kind_bond_dims = [0] if kind in NO_BOND_DIM_KINDS else bond_dims
                for hparams in _grid_points(block.get("grid", {})):
                    for bond_dim in kind_bond_dims:
                        for seed in block["seeds"]:
                            runs.append(
                                {
                                    "dataset": dataset,
                                    "kind": kind,
                                    "bond_dim": bond_dim,
                                    "seed": seed,
                                    "config_variant": block["config_variant"],
                                    "hparams": hparams,
                                }
                            )
    keys = [_plan_key(run) for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("The sweep plan contains duplicate runs")
    runs.sort(
        key=lambda r: (
            r["kind"] != "basis_enhanced",
            -COST[r["dataset"]] * r["bond_dim"] ** 2,
            r["seed"],
        )
    )
    return runs


def _plan_key(plan: dict) -> tuple:
    return (
        plan["dataset"],
        plan["config_variant"],
        plan["kind"],
        plan["bond_dim"],
        plan["seed"],
        json.dumps(plan["hparams"], sort_keys=True),
    )


def _hparam_tag(hparams: dict) -> str:
    return "".join(f"_{key}{value:g}" for key, value in sorted(hparams.items()))


def sweep_candidate(plan: dict) -> dict:
    """Return the planned identity of a run (its summary cell plus seed)."""
    return {
        "dataset": plan["dataset"],
        "config_variant": plan["config_variant"],
        "model": plan["kind"],
        "bond_dim": plan["bond_dim"],
        "hparams": plan["hparams"],
    }


def run_one(plan: dict, sweep_dir: Path, sweep_id: str, attempt: int) -> dict:
    tag = (
        f"{plan['dataset']}{plan['config_variant']}_{plan['kind']}_h{plan['bond_dim']:02d}"
        f"{_hparam_tag(plan['hparams'])}_s{plan['seed']}_a{attempt}"
    )
    outdir = sweep_dir / "runs" / tag
    config = (
        f"configs/e5_fig7_{DATASET_CONFIG[plan['dataset']]}{plan['config_variant']}"
        f"_{KIND_SUFFIX[plan['kind']]}.json"
    )
    command = [
        sys.executable, "implementation.py", "--paper", PAPER, "--config", config,
        "--bond-dim", str(plan["bond_dim"]), "--seed", str(plan["seed"]),
        "--outdir", str(outdir), "--sweep-id", sweep_id, "--log-level", "info",
    ]  # fmt: skip
    for key, value in sorted(plan["hparams"].items()):
        command += [f"--{key.replace('_', '-')}", repr(value)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    plan["_outdir"] = outdir
    completed = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )
    return {
        "plan": plan,
        "attempt": attempt,
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-2000:],
    }


def find_run_dir(outdir: Path) -> Path | None:
    candidates = sorted(outdir.glob("run_*"))
    return candidates[-1] if candidates else None


def entry_for(
    plan: dict, attempt: int, sweep_dir: Path, status: str, command: str
) -> dict:
    run_dir = find_run_dir(plan["_outdir"])
    rel = (
        (lambda name: os.path.relpath(run_dir / name, sweep_dir))
        if run_dir
        else (lambda name: None)
    )
    candidate = {
        "experiment": "sequence_model",
        "dataset": plan["dataset"],
        "model": plan["kind"],
        "field": "complex" if plan["kind"] in COMPLEX_KINDS else "real",
        "bond_dim": plan["bond_dim"],
    }
    return {
        "sweep_candidate": sweep_candidate(plan),
        "run_id": run_dir.name if run_dir else None,
        "candidate": candidate,
        "seed": plan["seed"],
        "repetition": plan["seed"],
        "attempt": attempt,
        "status": status,
        "command": command,
        "run_dir": os.path.relpath(run_dir, sweep_dir) if run_dir else None,
        "run_log": rel("run.log"),
        "run_status": rel("run_status.json"),
        "config_path": rel("config_snapshot.json"),
        "metrics_path": rel("metrics.json"),
    }


def read_progress(plan: dict) -> dict | None:
    run_dir = find_run_dir(plan["_outdir"]) if "_outdir" in plan else None
    if run_dir is None or not (run_dir / "run.log").exists():
        return None
    matches = PROGRESS_PATTERN.findall(
        (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    )
    if not matches:
        return None
    unit, current, total = matches[-1]
    return {
        "current": int(current),
        "total": int(total),
        "unit": "epochs" if unit == "EPOCH" else "em_iterations",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-name", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIG))
    parser.add_argument(
        "--model-kinds",
        nargs="+",
        default=DEFAULT_MODEL_KINDS,
        choices=list(KIND_SUFFIX),
    )
    parser.add_argument("--bond-dims", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--config-variant",
        default="",
        help="Infix of the config pair, e.g. '_pooled' -> configs/e5_fig7_spect_pooled_*.json",
    )
    parser.add_argument(
        "--spec",
        default=None,
        help="JSON sweep spec with blocks (Phase 5); replaces the dataset/model/bond-dim/seed options",
    )
    args = parser.parse_args(argv)
    blocks = blocks_from_args(args)

    sweep_id = f"{args.sweep_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    sweep_dir = PROJECT_DIR / "outdir" / "sweeps" / sweep_id
    sweep_dir.mkdir(parents=True)
    configure_logging(sweep_dir / "sweep.log")
    plans = planned_runs(blocks)
    fingerprint = code_fingerprint()
    status = {
        "schema_version": 1,
        "sweep_id": sweep_id,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": {**git_state(REPO_ROOT), "paper_code_sha256": fingerprint},
        "expected_runs": len(plans),
        "plan": (
            {
                "spec": args.spec,
                "blocks": blocks,
                "workers": args.workers,
                "max_attempts": args.max_attempts,
            }
            if args.spec
            else {
                "datasets": args.datasets,
                "model_kinds": args.model_kinds,
                "bond_dims": args.bond_dims,
                "seeds": args.seeds,
                "workers": args.workers,
                "max_attempts": args.max_attempts,
                "config_variant": args.config_variant,
            }
        ),
        "planned_runs": [
            {**sweep_candidate(plan), "seed": plan["seed"]} for plan in plans
        ],
        "runs": [],
        "summary_path": None,
        "selected_candidates": [],
        "error": None,
    }
    write_json(sweep_dir / "sweep_status.json", status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | workers=%d | code=%s",
        sweep_id,
        len(plans),
        args.workers,
        fingerprint[:12],
    )
    running: dict = {}

    def refresh() -> None:
        for plan, _attempt, index in running.values():
            progress = read_progress(plan)
            if progress:
                status["runs"][index]["progress"] = progress
        write_json(sweep_dir / "sweep_status.json", status)
        logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))

    try:
        queue = [(plan, 1) for plan in plans]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while queue or running:
                while queue and len(running) < args.workers:
                    plan, attempt = queue.pop(0)
                    future = pool.submit(run_one, plan, sweep_dir, sweep_id, attempt)
                    status["runs"].append(
                        {
                            "sweep_candidate": sweep_candidate(plan),
                            "run_id": None,
                            "candidate": {
                                "dataset": plan["dataset"],
                                "model": plan["kind"],
                                "bond_dim": plan["bond_dim"],
                            },
                            "seed": plan["seed"],
                            "repetition": plan["seed"],
                            "attempt": attempt,
                            "status": "RUNNING",
                            "run_dir": None,
                            "run_log": None,
                            "run_status": None,
                            "config_path": None,
                            "metrics_path": None,
                        }
                    )
                    running[future] = (plan, attempt, len(status["runs"]) - 1)
                    logger.info(
                        "CANDIDATE_STARTED | dataset=%s%s | model=%s | h=%d | hparams=%s | seed=%d | attempt=%d",
                        plan["dataset"], plan["config_variant"], plan["kind"], plan["bond_dim"],
                        json.dumps(plan["hparams"], sort_keys=True), plan["seed"], attempt,
                    )  # fmt: skip
                    write_json(sweep_dir / "sweep_status.json", status)
                done, _ = wait(list(running), timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    plan, attempt, index = running.pop(future)
                    result = future.result()
                    run_dir = find_run_dir(plan["_outdir"])
                    run_status = (
                        json.loads((run_dir / "run_status.json").read_text())
                        if run_dir and (run_dir / "run_status.json").exists()
                        else {}
                    )
                    ok = (
                        result["returncode"] == 0
                        and run_status.get("status") == "COMPLETED"
                        and (run_dir / "metrics.json").exists()
                    )
                    entry = entry_for(
                        plan,
                        attempt,
                        sweep_dir,
                        "COMPLETED" if ok else "FAILED",
                        result["command"],
                    )
                    if run_status.get("candidate"):
                        entry["candidate"] = run_status["candidate"]
                    status["runs"][index] = entry
                    if ok:
                        logger.info(
                            "CANDIDATE_COMPLETED | run_id=%s | metrics=%s",
                            entry["run_id"],
                            entry["metrics_path"],
                        )
                    else:
                        cause = (run_status.get("error") or {}).get(
                            "message"
                        ) or result["stderr_tail"][-300:]
                        logger.error(
                            "CANDIDATE_FAILED | run_id=%s | attempt=%d | returncode=%d | cause=%s",
                            entry["run_id"],
                            attempt,
                            result["returncode"],
                            cause,
                        )
                        if attempt < args.max_attempts:
                            queue.insert(0, (plan, attempt + 1))
                refresh()
        if code_fingerprint() != fingerprint:
            raise RuntimeError(
                "Paper code or configs changed during the sweep; runs are not comparable"
            )
        summary = summarize(sweep_dir)
        status["summary_path"] = "summary.json"
        planned = {_plan_key(p) for p in plans}
        observed = {
            _plan_key(
                {
                    **r["sweep_candidate"],
                    "kind": r["sweep_candidate"]["model"],
                    "seed": r["seed"],
                }
            )
            for r in status["runs"]
            if r["status"] == "COMPLETED"
        }
        complete = observed == planned and summary["coverage"] == "COMPLETE SWEEP"
        has_failed_attempt = any(r["status"] == "FAILED" for r in status["runs"])
        status["status"] = (
            "COMPLETED" if complete and not has_failed_attempt else "PARTIAL"
        )
        status["coverage"] = summary["coverage"]
        status["completed_at"] = utc_now()
        write_json(sweep_dir / "sweep_status.json", status)
        event = (
            "SWEEP_COMPLETED" if status["status"] == "COMPLETED" else "SWEEP_PARTIAL"
        )
        logger.info(
            "%s | summary=summary.json | coverage=%s | completed_runs=%d/%d",
            event,
            summary["coverage"],
            len(observed),
            len(planned),
        )
        return 0 if status["status"] == "COMPLETED" else 1
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(sweep_dir / "sweep_status.json", status)
        logger.exception("SWEEP_FAILED | sweep_id=%s", sweep_id)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
