"""GP likelihood, gradient, kernel-metric and MKL optimizer checks."""

from __future__ import annotations

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401  (puts the repository on sys.path)
from randomized_quantum_local_kernels_gp.lib.gp import factorize, predict
from randomized_quantum_local_kernels_gp.lib.kernels import (
    cosine_wrapper_gram,
    kernel_alignment,
    query_kernel_matrix,
)
from randomized_quantum_local_kernels_gp.lib.mkl import (
    MKLSettings,
    _exact_gradient,
    _LowRankHutchinson,
    fit_mixture,
    fit_mixtures,
    mixture_gram,
    softmax,
)
from scipy.stats import multivariate_normal


def _rbf_dictionary(inputs: np.ndarray, lengthscales) -> np.ndarray:
    return np.stack(
        [
            query_kernel_matrix(inputs, inputs, "rbf", {"lengthscale": ell})
            for ell in lengthscales
        ]
    )


def test_log_marginal_matches_scipy():
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-1, 1, size=(30, 1))
    gram = query_kernel_matrix(inputs, inputs, "matern32", {"lengthscale": 0.3})
    targets = rng.normal(size=30)
    state = factorize(gram, 0.01, targets)
    reference = multivariate_normal(np.zeros(30), gram + 0.01 * np.eye(30)).logpdf(
        targets
    )
    assert state.log_marginal == pytest.approx(reference, rel=1e-10)


def test_exact_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    inputs = rng.uniform(-1, 1, size=(25, 1))
    grams = _rbf_dictionary(inputs, [0.05, 0.2, 0.6])
    targets = rng.normal(size=25)
    weights = np.array([0.2, 0.5, 0.3])
    gradient = _exact_gradient(
        grams, factorize(mixture_gram(weights, grams), 0.05, targets)
    )
    step = 1e-6
    for r in range(3):
        shift = np.zeros(3)
        shift[r] = step
        up = factorize(mixture_gram(weights + shift, grams), 0.05, targets).log_marginal
        down = factorize(
            mixture_gram(weights - shift, grams), 0.05, targets
        ).log_marginal
        assert gradient[r] == pytest.approx((up - down) / (2 * step), rel=1e-5)


def test_lowrank_hutchinson_approaches_exact_gradient_at_full_rank():
    rng = np.random.default_rng(2)
    inputs = rng.uniform(-1, 1, size=(20, 1))
    grams = _rbf_dictionary(inputs, [0.1, 0.4])
    targets = rng.normal(size=20)
    state = factorize(mixture_gram(np.array([0.5, 0.5]), grams), 0.1, targets)
    approx = _LowRankHutchinson(grams, rank=20, probes=20000, seed=0).gradient(state)
    assert np.allclose(approx, _exact_gradient(grams, state), rtol=0.05, atol=0.05)


def test_cosine_wrapper_range_and_alignment_properties():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(15, 6))
    gram = cosine_wrapper_gram(features, features)
    assert np.all(gram >= -1e-12) and np.all(gram <= 1 + 1e-12)
    assert np.allclose(np.diag(gram), 1.0)
    other = query_kernel_matrix(
        rng.uniform(-1, 1, (15, 1)),
        rng.uniform(-1, 1, (15, 1)),
        "rbf",
        {"lengthscale": 0.3},
    )
    other = 0.5 * (other + other.T)
    # Centered alignment is invariant to positive affine rescaling, as used for Table I.
    assert kernel_alignment(3.0 * gram + 2.0, other, centered=True) == pytest.approx(
        kernel_alignment(gram, other, centered=True)
    )
    assert kernel_alignment(gram, gram, centered=False) == pytest.approx(1.0)


def test_prediction_interpolates_noise_free_training_points():
    inputs = np.linspace(-1, 1, 12)[:, None]
    gram = query_kernel_matrix(inputs, inputs, "rbf", {"lengthscale": 0.4})
    targets = np.sin(3 * inputs[:, 0])
    state = factorize(gram, 1e-8, targets)
    mean, variance = predict(state, gram, np.diag(gram))
    assert np.allclose(mean, targets, atol=1e-4)
    assert np.all(variance < 1e-5)


@pytest.mark.parametrize("gradient", ["exact", "lowrank_hutchinson"])
def test_mkl_selects_the_generating_lengthscale(gradient):
    rng = np.random.default_rng(4)
    inputs = rng.uniform(-1, 1, size=(80, 1))
    lengthscales = [0.03, 0.1, 0.3, 1.0]
    grams = _rbf_dictionary(inputs, lengthscales)
    targets = rng.multivariate_normal(np.zeros(80), grams[2] + 0.01 * np.eye(80))
    settings = MKLSettings(
        gradient=gradient,
        max_iterations=200,
        tolerance=1e-8,
        coordinate_rounds=3,
        initial_noise_variance=1.0,
        fit_noise_variance=True,
        approx_rank=40,
        hutchinson_probes=64,
        approx_seed=0,
    )
    result = fit_mixture(grams, targets, settings, label="toy")
    assert np.isclose(result.weights.sum(), 1.0)
    assert int(np.argmax(result.weights)) == 2
    uniform = factorize(
        mixture_gram(np.full(4, 0.25), grams), result.noise_variance, targets
    )
    assert result.log_marginal > uniform.log_marginal
    accepted = [
        h["log_marginal"]
        for h in result.history
        if "iteration" in h and h["round"] == 1
    ]
    assert all(b >= a - 1e-9 for a, b in zip(accepted, accepted[1:]))


def _toy_problem(seed: int, generating: list[int], size: int = 60):
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(-1, 1, size=(size, 1))
    grams = _rbf_dictionary(inputs, [0.03, 0.1, 0.3, 1.0])
    targets = [
        rng.multivariate_normal(np.zeros(size), grams[g] + noise * np.eye(size))
        for g, noise in zip(generating, (0.01, 0.0005))
    ]
    return grams, targets


def _protocol_settings(**overrides) -> MKLSettings:
    base = {
        "gradient": "exact",
        "max_iterations": 50,
        "tolerance": 1e-6,
        "coordinate_rounds": 3,
        "initial_noise_variance": 1.0,
        "fit_noise_variance": True,
    }
    base.update(overrides)
    return MKLSettings(**base)


def test_fit_mixtures_per_kernel_equals_fit_mixture():
    grams, targets = _toy_problem(5, [2, 1])
    settings = _protocol_settings()
    joint = fit_mixtures(grams, targets, settings, ["a", "b"])
    for result, y in zip(joint, targets):
        single = fit_mixture(grams, y, settings, "x")
        assert np.allclose(result.weights, single.weights)
        assert result.noise_variance == single.noise_variance


def test_shared_noise_coupling_reuses_last_gp_noise():
    grams, targets = _toy_problem(6, [2, 1])
    shared = fit_mixtures(
        grams, targets, _protocol_settings(noise_coupling="authors_shared"), ["a", "b"]
    )
    # Round-1 grid result of the last GP is the σ² every GP ascends with in round 2.
    last_round1_noise = next(
        h["noise_variance"]
        for h in shared[1].history
        if h.get("noise_search") and h["round"] == 1
    )
    round2_ascent = [
        h for h in shared[0].history if h.get("iteration") and h["round"] == 2
    ]
    assert round2_ascent, "GP 0 should ascend in round 2 for this toy problem"
    assert all(h["noise_variance"] == last_round1_noise for h in round2_ascent)
    # A single GP has nobody to share with: identical to per-kernel fitting.
    alone = fit_mixtures(
        grams, targets[:1], _protocol_settings(noise_coupling="authors_shared"), ["a"]
    )
    per_kernel = fit_mixture(grams, targets[0], _protocol_settings(), "a")
    assert np.allclose(alone[0].weights, per_kernel.weights)


def test_lbfgs_reaches_at_least_protocol_and_uniform_likelihood():
    grams, targets = _toy_problem(7, [2, 1])
    for y in targets:
        protocol = fit_mixture(grams, y, _protocol_settings(), "p")
        uniform = fit_mixture(grams, y, _protocol_settings(max_iterations=0), "u")
        converged = fit_mixture(
            grams,
            y,
            _protocol_settings(optimizer="lbfgs", max_iterations=500, tolerance=1e-8),
            "l",
        )
        assert converged.log_marginal >= protocol.log_marginal - 1e-6
        assert converged.log_marginal >= uniform.log_marginal - 1e-6
        # Local optimality: small random logit / log σ² perturbations do not improve L.
        rng = np.random.default_rng(0)
        for _ in range(5):
            logits = converged.logits + 1e-3 * rng.normal(size=converged.logits.size)
            noise = converged.noise_variance * np.exp(1e-3 * rng.normal())
            perturbed = factorize(
                mixture_gram(softmax(logits), grams), noise, y
            ).log_marginal
            assert perturbed <= converged.log_marginal + 1e-6
