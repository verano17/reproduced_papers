from __future__ import annotations

import json
from copy import deepcopy

import pytest
from common import load_runtime_ready_config
from quantum_correlations_generative_models.lib import runner


def _run(tmp_path, overrides):
    cfg = deepcopy(load_runtime_ready_config())
    cfg.setdefault("logging", {"level": "info"})
    cfg.update(overrides)
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)
    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return run_dir, status, metrics


def test_defaults_train_basis_enhanced(tmp_path):
    run_dir, status, metrics = _run(
        tmp_path, {"training": {**load_runtime_ready_config()["training"], "epochs": 1}}
    )
    assert status["status"] == "COMPLETED"
    assert metrics["final"]["train"]["nll_bits_per_site"] > 0
    assert (run_dir / "model_state.pt").exists()


def test_baum_welch_run(tmp_path):
    base = load_runtime_ready_config()
    _, status, metrics = _run(
        tmp_path,
        {
            "model": {"kind": "hmm_baum_welch", "bond_dim": 2, "field": "real"},
            "training": {**base["training"], "em_iterations": 3, "em_tolerance": 0.0},
        },
    )
    assert status["status"] == "COMPLETED"
    train_curve = [
        entry["evaluation"]["train"]["nll_bits_per_site"]
        for entry in metrics["history"]
    ]
    assert all(
        later <= earlier + 1e-12 for earlier, later in zip(train_curve, train_curve[1:])
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "experiment": "theorem1_toy",
            "theorem1": {
                "num_pairs": [3, 5],
                "statevector_check_pairs": 5,
                "chosen_pairs": [0, 2, 4],
            },
        },
        {
            "experiment": "robust_l1",
            "robust_l1": {"walk_steps": [1, 2], "num_pairs": [7], "kgram_orders": [1]},
        },
        {"experiment": "mermin_peres", "mermin_peres": {"count_qubits": [1, 2]}},
    ],
    ids=["theorem1_toy", "robust_l1", "mermin_peres"],
)
def test_exact_experiments_complete(tmp_path, overrides):
    _, status, metrics = _run(tmp_path, overrides)
    assert status["status"] == "COMPLETED"
    assert metrics["experiment"] == overrides["experiment"]


def test_unknown_experiment_fails_with_status(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, {"experiment": "nope"})
    status = json.loads((tmp_path / "run_test" / "run_status.json").read_text())
    assert status["status"] == "FAILED" and status["error"]["type"] == "ValueError"
