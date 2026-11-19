"""Checks of the Phase 5 classical baselines (lib/classical_baselines.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from common import load_runtime_ready_config
from randomized_quantum_local_kernels_gp.lib.classical_baselines import (
    fit_parametric_gp,
    parametric_gram,
    rbf_dictionary_grams,
    rff_dictionary_features,
)
from randomized_quantum_local_kernels_gp.lib.gp import factorize
from randomized_quantum_local_kernels_gp.lib.kernels import (
    cosine_wrapper_gram,
    query_kernel_matrix,
)

INPUTS = np.linspace(-1, 1, 40)[:, None]


def test_rff_features_shape_norm_and_seeding():
    features, lengthscales = rff_dictionary_features(
        INPUTS, 8, 24, sobol_seed=3, lengthscale_range=(0.02, 2.0)
    )
    assert features.shape == (8, 40, 24)
    assert np.allclose(np.linalg.norm(features, axis=-1), 1.0)
    assert np.all((lengthscales >= 0.02) & (lengthscales <= 2.0))
    again, _ = rff_dictionary_features(
        INPUTS, 8, 24, sobol_seed=3, lengthscale_range=(0.02, 2.0)
    )
    other, _ = rff_dictionary_features(
        INPUTS, 8, 24, sobol_seed=4, lengthscale_range=(0.02, 2.0)
    )
    assert np.array_equal(features, again) and not np.array_equal(features, other)
    with pytest.raises(ValueError, match="even"):
        rff_dictionary_features(
            INPUTS, 2, 5, sobol_seed=0, lengthscale_range=(0.1, 1.0)
        )


def test_wrapped_rff_approximates_shifted_rbf():
    # With many features the cosine-wrapped RFF kernel tends to ½(1 + RBF_ℓ).
    features, _ = rff_dictionary_features(
        INPUTS, 1, 4096, sobol_seed=0, lengthscale_range=(0.3, 0.3)
    )
    wrapped = cosine_wrapper_gram(features[0], features[0])
    target = 0.5 * (
        1.0 + query_kernel_matrix(INPUTS, INPUTS, "rbf", {"lengthscale": 0.3})
    )
    assert np.max(np.abs(wrapped - target)) < 0.03


def test_rbf_dictionary_lengthscales_span_range():
    grams, lengthscales = rbf_dictionary_grams(INPUTS, 16, (0.02, 2.0))
    assert grams.shape == (16, 40, 40)
    assert lengthscales[0] == pytest.approx(0.02) and lengthscales[-1] == pytest.approx(
        2.0
    )
    assert np.allclose(np.diagonal(grams, axis1=1, axis2=2), 1.0)


def test_parametric_fit_recovers_rbf_and_beats_true_params():
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-1, 1, size=(150, 1))
    gram = query_kernel_matrix(inputs, inputs, "rbf", {"lengthscale": 0.3})
    targets = rng.multivariate_normal(np.zeros(150), gram + 0.002 * np.eye(150))
    fit = fit_parametric_gp("rbf", inputs, targets, restarts=4, seed=1, label="toy")
    assert 0.2 < fit.params["lengthscale"] < 0.45
    at_truth = factorize(gram, 0.002, targets).log_marginal
    assert fit.log_marginal >= at_truth - 1e-6
    assert len(fit.restarts) == 4
    rebuilt = parametric_gram("rbf", fit.params, inputs, inputs)
    assert np.allclose(np.diag(rebuilt), fit.params["amplitude"])


def test_parametric_nonstationary_fit_orders_lengthscales():
    rng = np.random.default_rng(2)
    inputs = rng.uniform(-1, 1, size=(80, 1))
    params = {"ell_min": 0.1, "ell_max": 0.8, "c": 2.0, "sigmoid_sign": 1}
    gram = query_kernel_matrix(inputs, inputs, "nonstat_ls", params)
    targets = rng.multivariate_normal(np.zeros(80), gram + 0.002 * np.eye(80))
    fit = fit_parametric_gp(
        "nonstat_ls", inputs, targets, restarts=3, seed=0, label="toy"
    )
    assert fit.params["ell_max"] > fit.params["ell_min"] > 0
    assert fit.params["sigmoid_sign"] == 1


def _run(tmp_path, cfg):
    from randomized_quantum_local_kernels_gp.lib import runner

    run_dir = tmp_path / "run_baseline"
    run_dir.mkdir()
    snapshot = {k: v for k, v in cfg.items() if k != "dtype"} | {"dtype": "float64"}
    (run_dir / "config_snapshot.json").write_text(json.dumps(snapshot))
    runner.train_and_evaluate(cfg, run_dir)
    assert (
        json.loads((run_dir / "run_status.json").read_text())["status"] == "COMPLETED"
    )
    return json.loads((run_dir / "metrics.json").read_text())


@pytest.mark.parametrize("backend", ["classical_rff", "rbf_dictionary"])
def test_runner_dictionary_baselines(tmp_path, backend):
    cfg = load_runtime_ready_config()
    cfg["data"]["num_points"] = 24
    cfg["circuit"]["num_draws"] = 4
    cfg["mkl"]["max_iterations"] = 5
    cfg["mkl"]["approx_rank"] = 4
    cfg["features"].update(backend=backend, mode="exact")
    metrics = _run(tmp_path, cfg)
    assert metrics["feature_backend"] == backend
    assert metrics["num_base_kernels"] == 4
    assert (
        metrics["classical"]["lengthscale_range"]
        == cfg["classical"]["lengthscale_range"]
    )
    if backend == "classical_rff":
        assert (
            metrics["num_features"] == 24
            and metrics["feature_error"]["sup_norm"] == 0.0
        )
    else:
        assert metrics["num_features"] is None and metrics["feature_error"] is None
    assert "uniform_mixture" in metrics["kernels"]["rbf"]


@pytest.mark.parametrize("family", ["true", "rbf"])
def test_runner_parametric_baselines(tmp_path, family):
    cfg = load_runtime_ready_config()
    cfg["data"]["num_points"] = 30
    cfg["experiment"]["task"] = "gp_parametric"
    cfg["parametric"] = {"family": family, "restarts": 2}
    metrics = _run(tmp_path, cfg)
    entry = metrics["kernels"]["matern32"]["parametric_gp"]
    assert entry["family"] == ("matern32" if family == "true" else "rbf")
    assert np.isfinite(entry["alignment_centered"]) and len(entry["restarts"]) == 2
