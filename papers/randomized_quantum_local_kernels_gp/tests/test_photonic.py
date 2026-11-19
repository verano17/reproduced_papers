"""Checks of the MerLin photonic adaptation (lib/photonic_features.py)."""

from __future__ import annotations

import json

import numpy as np
import perceval as pcvl
import pytest
from common import load_runtime_ready_config
from randomized_quantum_local_kernels_gp.lib.photonic_features import (
    PhotonicArchitecture,
    build_circuit,
    build_layer,
    encoding_phases,
    local_parity_features,
    outcome_probabilities,
    photonic_features,
    postselection_probability,
    sample_frequencies,
    sample_photonic_draws,
)

ARCH = PhotonicArchitecture(num_modes=6, num_photons=3, num_layers=2, input_dim=1)


def _perceval_distribution(arch, params, phases_row):
    """Independent SLOS simulation with the data phases substituted numerically."""
    circuit = build_circuit(arch, params)
    for parameter in circuit.get_parameters():
        index = int(parameter.name[2:])
        parameter.set_value(float(phases_row[index]))
    backend = pcvl.BackendFactory.get_backend("SLOS")
    backend.set_circuit(circuit)
    backend.set_input_state(pcvl.BasicState(arch.input_state))
    return {tuple(state): float(p) for state, p in backend.prob_distribution().items()}


def test_architecture_validation_and_input_state():
    assert PhotonicArchitecture(12, 4, 4, 1).input_state == [
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
    ]
    assert ARCH.num_features == 11
    assert ARCH.mixing_pairs == ((0, 1), (2, 3), (4, 5), (1, 2), (3, 4))
    with pytest.raises(ValueError, match="num_photons"):
        PhotonicArchitecture(6, 1, 2, 1)


def test_sobol_draws_are_seeded_and_in_range():
    first = sample_photonic_draws(ARCH, 4, seed=3)
    second = sample_photonic_draws(ARCH, 4, seed=3)
    other = sample_photonic_draws(ARCH, 4, seed=4)
    assert np.array_equal(first[2].theta, second[2].theta)
    assert not np.array_equal(first[2].theta, other[2].theta)
    for params in first:
        assert np.all(np.abs(params.delta) < 1) and np.all(np.abs(params.xi) <= np.pi)
        for block in (params.alpha, params.theta, params.omega):
            assert np.all(np.abs(block) <= np.pi)
        assert params.theta.shape == (2, 5)


@pytest.mark.parametrize("detector", ["pnr", "threshold"])
def test_merlin_matches_independent_perceval_simulation(detector):
    params = sample_photonic_draws(ARCH, 2, seed=1)[1]
    inputs = np.array([[-0.7], [0.2], [0.9]])
    phases = encoding_phases(ARCH, params, inputs)
    probabilities, patterns = outcome_probabilities(
        build_layer(ARCH, params, detector), phases
    )
    for row in range(inputs.shape[0]):
        reference = _perceval_distribution(ARCH, params, phases[row])
        if detector == "threshold":
            collision_free = {k: v for k, v in reference.items() if max(k) <= 1}
            total = sum(collision_free.values())
            reference = {k: v / total for k, v in collision_free.items()}
        ours = {tuple(p): float(v) for p, v in zip(patterns, probabilities[row])}
        # Perceval omits zero-probability outcomes; MerLin lists the whole space.
        assert set(reference) <= set(ours)
        assert max(abs(ours[k] - reference.get(k, 0.0)) for k in ours) < 1e-10


def test_postselection_probability_and_threshold_space():
    params = sample_photonic_draws(ARCH, 1, seed=2)[0]
    phases = encoding_phases(ARCH, params, np.array([[0.4], [-0.5]]))
    rate = postselection_probability(ARCH, params, phases)
    assert np.all((rate > 0) & (rate <= 1))
    probabilities, patterns = outcome_probabilities(
        build_layer(ARCH, params, "threshold"), phases
    )
    # The selected UNBUNCHED space has only collision-free outcomes (C(6, 3) = 20).
    assert patterns.shape == (20, 6) and patterns.max() == 1
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_parity_features_bounded_and_sampling_unbiased():
    params = sample_photonic_draws(ARCH, 1, seed=5)[0]
    inputs = np.array([[0.3]])
    phases = encoding_phases(ARCH, params, inputs)
    probabilities, patterns = outcome_probabilities(
        build_layer(ARCH, params, "threshold"), phases
    )
    exact = local_parity_features(probabilities, patterns, "threshold")
    assert exact.shape == (1, 11) and np.all(np.abs(exact) <= 1 + 1e-12)
    rng = np.random.default_rng(0)
    repeats = np.stack(
        [
            local_parity_features(
                sample_frequencies(probabilities, 500, rng), patterns, "threshold"
            )
            for _ in range(400)
        ]
    )
    # Each parity estimate has variance <= 1/500; 5-sigma tolerance on the mean of 400.
    assert np.max(np.abs(repeats.mean(axis=0) - exact)) < 5 * np.sqrt(1 / 500 / 400)
    same = photonic_features(
        ARCH, params, inputs, "threshold", 200, np.random.default_rng(9)
    )
    again = photonic_features(
        ARCH, params, inputs, "threshold", 200, np.random.default_rng(9)
    )
    assert np.array_equal(same["used"], again["used"])


@pytest.mark.parametrize("detector", ["threshold", "pnr"])
def test_runner_end_to_end_photonic(tmp_path, detector):
    from randomized_quantum_local_kernels_gp.lib import runner

    cfg = load_runtime_ready_config()
    cfg["data"]["num_points"] = 24
    cfg["circuit"]["num_draws"] = 3
    cfg["mkl"]["max_iterations"] = 5
    cfg["mkl"]["approx_rank"] = 4
    cfg["features"].update(
        backend="merlin_photonic", mode="photon_sampled", num_snapshots=64
    )
    cfg["photonic"] = {
        "num_modes": 6,
        "num_photons": 2,
        "num_layers": 2,
        "detector": detector,
        "initial_mixing": "balanced_brickwork",
    }
    run_dir = tmp_path / "run_photonic"
    run_dir.mkdir()
    snapshot = {k: v for k, v in cfg.items() if k != "dtype"} | {"dtype": "float64"}
    (run_dir / "config_snapshot.json").write_text(json.dumps(snapshot))

    runner.train_and_evaluate(cfg, run_dir)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["feature_backend"] == "merlin_photonic"
    assert metrics["num_features"] == 11
    hardware = metrics["photonic"]
    assert hardware["num_modes"] == 6 and hardware["num_photons"] == 2
    assert hardware["computation_space"] == (
        "UNBUNCHED" if detector == "threshold" else "FOCK"
    )
    assert hardware["shots_per_input_and_draw"] == 64
    assert ("postselection_probability_mean" in hardware) == (detector == "threshold")
    assert (
        json.loads((run_dir / "run_status.json").read_text())["status"] == "COMPLETED"
    )


def test_photonic_backend_rejects_qubit_feature_modes(tmp_path):
    from randomized_quantum_local_kernels_gp.lib import runner

    cfg = load_runtime_ready_config()
    cfg["features"].update(backend="merlin_photonic", mode="pauli_shadow")
    with pytest.raises(ValueError, match="features.mode"):
        runner.train_and_evaluate(cfg, tmp_path)


@pytest.mark.parametrize("mixing", ["none", "balanced_brickwork"])
def test_first_layer_encoding_effect_depends_on_initial_mixing(mixing):
    """Phases on a Fock input are a global phase; an initial brickwork makes them act."""
    arch = PhotonicArchitecture(
        num_modes=6, num_photons=3, num_layers=2, input_dim=1, initial_mixing=mixing
    )
    params = sample_photonic_draws(arch, 1, seed=4)[0]
    layer = build_layer(arch, params, "pnr")
    phases = encoding_phases(arch, params, np.array([[0.5]]))
    perturbed = phases.copy()
    perturbed[:, : arch.num_modes] += np.random.default_rng(0).uniform(
        -1, 1, arch.num_modes
    )
    base, _ = outcome_probabilities(layer, phases)
    moved, _ = outcome_probabilities(layer, perturbed)
    change = np.max(np.abs(base - moved))
    if mixing == "none":
        assert change < 1e-12
    else:
        assert change > 1e-3


def test_initial_mixing_is_validated():
    with pytest.raises(ValueError, match="initial_mixing"):
        PhotonicArchitecture(
            num_modes=6,
            num_photons=2,
            num_layers=1,
            input_dim=1,
            initial_mixing="other",
        )
