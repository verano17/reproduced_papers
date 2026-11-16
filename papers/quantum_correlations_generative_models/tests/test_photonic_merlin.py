"""Phase 4 MerLin adaptation: heralded two-photon photonic quantum HMM."""

from __future__ import annotations

import itertools
import json
import warnings

import merlin as ml
import pytest
import torch
from common import PROJECT_DIR, load_runtime_ready_config
from quantum_correlations_generative_models.lib import runner
from quantum_correlations_generative_models.lib.photonic_hmm_merlin import (
    COMPUTATION_SPACE,
    NUM_PHOTONS,
    PhotonicQuantumHMM,
)

from runtime_lib.config import deep_update

warnings.filterwarnings("ignore", category=UserWarning)


def _all_sequences(alphabet: int, length: int) -> torch.Tensor:
    return torch.tensor(list(itertools.product(range(alphabet), repeat=length)))


def _single_photon_unitary(model: PhotonicQuantumHMM) -> torch.Tensor:
    """U[out, in] of the same mesh, read from a one-photon layer with shared phases."""
    builder = ml.CircuitBuilder(n_modes=model.num_modes)
    builder.add_entangling_layer()
    single = ml.QuantumLayer(
        builder=builder,
        n_photons=1,
        measurement_strategy=ml.MeasurementStrategy.amplitudes(
            computation_space=ml.ComputationSpace.FOCK
        ),
        dtype=torch.float64,
    )
    single.load_state_dict(model.layer.state_dict())
    index = {tuple(key): i for i, key in enumerate(single.output_keys)}
    inputs = torch.zeros(model.num_modes, len(index), dtype=torch.complex128)
    for mode in range(model.num_modes):
        inputs[mode, index[model._occupation(mode)]] = 1.0
    with torch.no_grad():
        out = single(inputs)
    columns = [index[model._occupation(mode)] for mode in range(model.num_modes)]
    return out[:, columns].T


def test_layer_uses_fock_space_and_two_photons():
    model = PhotonicQuantumHMM(3, 2, seed=0)
    assert COMPUTATION_SPACE == ml.ComputationSpace.FOCK
    assert NUM_PHOTONS == 2
    assert all(sum(key) == 2 for key in model.layer.output_keys)
    # FOCK keeps bunched outcomes (two photons in one mode).
    assert any(max(key) == 2 for key in model.layer.output_keys)
    assert model.num_modes == 5 and model.probe_mode == 3


def test_kraus_operators_are_two_photon_permanents():
    model = PhotonicQuantumHMM(4, 3, seed=1)
    unitary = _single_photon_unitary(model)
    h, probe = 4, model.probe_mode
    with torch.no_grad():
        kraus = model.kraus_operators()
    for symbol in range(3):
        data = h + symbol
        expected = (
            unitary[:h, :h] * unitary[data, probe]
            + unitary[:h, probe][:, None] * unitary[data, :h][None, :]
        )
        torch.testing.assert_close(kraus[symbol], expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("bond_dim,alphabet,length", [(1, 2, 4), (3, 2, 5), (2, 3, 3)])
def test_distribution_is_normalised(bond_dim, alphabet, length):
    model = PhotonicQuantumHMM(bond_dim, alphabet, seed=2)
    with torch.no_grad():
        total = model.log_prob(_all_sequences(alphabet, length)).exp().sum()
    assert float(total) == pytest.approx(1.0, abs=1e-10)


def test_herald_success_equals_enumerated_mass():
    model = PhotonicQuantumHMM(3, 2, seed=3)
    sequences = _all_sequences(2, 4)
    with torch.no_grad():
        kraus = model.kraus_operators()
        psi = model.initial_state().expand(len(sequences), -1)
        for site in range(4):
            psi = torch.einsum("zba,za->zb", kraus[sequences[:, site]], psi)
        enumerated = psi.abs().pow(2).sum()
        success = model.log_herald_success(kraus, 4).exp()
    assert float(success) == pytest.approx(float(enumerated), rel=1e-10)
    assert 0.0 < float(success) < 1.0


def test_gradients_reach_mesh_phases_and_initial_memory():
    model = PhotonicQuantumHMM(3, 2, seed=4)
    loss = -model.log_prob(torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]])).mean()
    loss.backward()
    named = dict(model.named_parameters())
    assert any(name.startswith("layer.") for name in named)
    for name, parameter in named.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert any(parameter.grad.abs().sum() > 0 for parameter in model.layer.parameters())


def test_seeding_is_deterministic():
    sequences = torch.tensor([[0, 1, 1, 0]])
    first = PhotonicQuantumHMM(3, 2, seed=5)
    second = PhotonicQuantumHMM(3, 2, seed=5)
    other = PhotonicQuantumHMM(3, 2, seed=6)
    with torch.no_grad():
        assert torch.equal(first.log_prob(sequences), second.log_prob(sequences))
        assert not torch.equal(first.log_prob(sequences), other.log_prob(sequences))


def test_runner_trains_merlin_model_from_config(tmp_path):
    overlay = json.loads((PROJECT_DIR / "configs" / "smoke_merlin.json").read_text())
    cfg = deep_update(load_runtime_ready_config(), overlay)
    cfg.setdefault("logging", {"level": "info"})
    cfg["model"]["bond_dim"] = 3
    runner.train_and_evaluate(cfg, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    assert metrics["model_kind"] == "photonic_merlin"
    assert metrics["bond_dim"] == 3
    assert metrics["photonic"]["photon_number"] == 2
    assert metrics["photonic"]["num_modes"] == 3 + 2
    assert "FOCK" in metrics["photonic"]["computation_space"].upper()
    assert len(metrics["history"]) == cfg["training"]["epochs"] + 1
    assert metrics["photonic"]["herald_success_log2_per_sequence"] < 0


@pytest.mark.parametrize("bond_dim", [3, 6])
def test_post_emission_memory_spans_at_most_two_dimensions(bond_dim):
    """K_m psi = U[h+m, h] W psi + (U[h+m, :h] . psi) c lies in span{W psi, c}.

    W = U[:h, :h] and c = U[:h, h] are shared by every symbol, so whatever h
    is, the memories reachable from one state after one emission span at most
    two dimensions (the structural limit discussed in LOG.md, Phase 4 results).
    """
    model = PhotonicQuantumHMM(bond_dim, 8, seed=7)
    with torch.no_grad():
        kraus = model.kraus_operators()
        psi = model.initial_state()
        successors = torch.einsum("mba,a->bm", kraus, psi)
    singular_values = torch.linalg.svdvals(successors)
    assert int((singular_values > 1e-10 * singular_values[0]).sum()) == 2
