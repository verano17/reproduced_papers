from __future__ import annotations

import json

import numpy as np
import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config


def _tiny_config() -> dict:
    cfg = load_runtime_ready_config()
    cfg["data"]["num_points"] = 24
    cfg["circuit"]["num_draws"] = 3
    cfg["features"]["num_snapshots"] = 64
    cfg["mkl"]["max_iterations"] = 5
    cfg["mkl"]["approx_rank"] = 4
    return cfg


def _write_snapshot(cfg: dict, run_dir) -> None:
    serializable = {key: value for key, value in cfg.items() if key != "dtype"}
    serializable["dtype"] = "float64"
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(serializable), encoding="utf-8"
    )


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_parses_paper_flags():
    parser, _ = build_project_cli_parser()
    args = parser.parse_args(
        [
            "--task",
            "gp_mkl",
            "--feature-mode",
            "pauli_shadow",
            "--num-draws",
            "16",
            "--gradient",
            "exact",
        ]
    )
    assert args.task == "gp_mkl"
    assert args.feature_mode == "pauli_shadow"
    assert args.num_draws == 16
    assert args.gradient == "exact"


@pytest.mark.parametrize(
    "path", sorted((PROJECT_DIR / "configs").glob("*.json")), ids=lambda p: p.name
)
def test_configs_are_json_with_description(path):
    cfg = json.loads(path.read_text())
    assert cfg.get("description")
    assert "<<" not in path.read_text()
    from randomized_quantum_local_kernels_gp.lib.runner import SUPPORTED_TASKS

    assert cfg["experiment"]["task"] in SUPPORTED_TASKS


@pytest.mark.parametrize("mode", ["exact", "code_gaussian", "pauli_shadow"])
def test_train_and_evaluate_writes_contract_artifacts(tmp_path, mode):
    from randomized_quantum_local_kernels_gp.lib import runner

    cfg = _tiny_config()
    cfg["features"]["mode"] = mode
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    _write_snapshot(cfg, run_dir)

    runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    assert status["metrics_path"] == "metrics.json"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics["kernels"]) == {"rbf", "matern32", "nonstat_ls"}
    learned = metrics["kernels"]["rbf"]["learned_mixture"]
    assert -1.0 <= learned["alignment_centered"] <= 1.0
    assert np.isfinite(learned["avg_nlpd"])
    assert "uniform_mixture" in metrics["kernels"]["rbf"]
    if mode == "exact":
        assert metrics["feature_error"]["sup_norm"] == 0.0
    else:
        assert metrics["feature_error"]["sup_norm"] > 0.0
    log = (run_dir / "run.log").read_text()
    for event in (
        "RUN_STARTED",
        "DATASET_READY",
        "TRAINING_STARTED",
        "EVALUATION_COMPLETED",
        "METRICS_WRITTEN",
        "RUN_COMPLETED",
    ):
        assert event in log
    assert (run_dir / "predictions.npz").exists()
    assert (run_dir / "fig4_test_predictions.png").exists()


def test_unknown_task_fails(tmp_path):
    from randomized_quantum_local_kernels_gp.lib import runner

    cfg = _tiny_config()
    cfg["experiment"]["task"] = "not_a_task"
    with pytest.raises(ValueError, match="experiment.task"):
        runner.train_and_evaluate(cfg, tmp_path)


def test_non_cpu_device_fails(tmp_path):
    from randomized_quantum_local_kernels_gp.lib import runner

    cfg = _tiny_config()
    cfg["device"] = "cuda:0"
    with pytest.raises(ValueError, match="NumPy/CPU"):
        runner.train_and_evaluate(cfg, tmp_path)
