"""Circuit, Sobol-layout and feature-estimator checks against Qiskit and theory."""

from __future__ import annotations

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401  (puts the repository on sys.path)
from randomized_quantum_local_kernels_gp.lib.circuits import (
    RingArchitecture,
    map_sobol_point,
    sample_parameter_draws,
    simulate_states,
)
from randomized_quantum_local_kernels_gp.lib.features import (
    code_gaussian_features,
    exact_features,
    observable_supports,
    pauli_shadow_features,
)

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import ParameterVector  # noqa: E402
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402

ARCH = RingArchitecture(num_qubits=4, num_layers=4, input_dim=1)


def _qiskit_circuit(arch: RingArchitecture) -> QuantumCircuit:
    """Ring circuit of Eq. 8-9 / Fig. 2 written directly in Qiskit."""
    qc = QuantumCircuit(arch.num_qubits)
    x = ParameterVector("x", arch.input_dim)
    qc.h(range(arch.num_qubits))
    for layer in range(arch.num_layers):
        xi = ParameterVector(f"xi_{layer}", arch.input_dim)
        delta = ParameterVector(f"delta_{layer}", arch.num_qubits)
        alpha = ParameterVector(f"alpha_{layer}", arch.num_qubits)
        beta = ParameterVector(f"beta_{layer}", len(arch.ring_edges))
        omega = ParameterVector(f"omega_{layer}", arch.num_qubits)
        gamma = sum(xi[j] * x[j] for j in range(arch.input_dim))
        for q in range(arch.num_qubits):
            qc.rz(delta[q] * gamma, q)
        for q in range(arch.num_qubits):
            qc.rx(alpha[q], q)
        for e, (i, j) in enumerate(arch.ring_edges):
            qc.rzz(beta[e], i, j)
        for q in range(arch.num_qubits):
            qc.ry(omega[q], q)
    return qc


def _assignment(qc: QuantumCircuit, params, x_value: float) -> dict:
    values = {}
    for p in qc.parameters:
        name, index = p.name.split("[")
        index = int(index[:-1])
        if name == "x":
            values[p] = x_value
            continue
        group, layer = name.split("_")
        values[p] = float(getattr(params, group)[int(layer), index])
    return values


def test_statevectors_match_qiskit():
    params = sample_parameter_draws(ARCH, num_draws=4, seed=7)[3]
    inputs = np.array([[-0.8], [0.1], [0.65]])
    ours = simulate_states(ARCH, params, inputs)
    qc = _qiskit_circuit(ARCH)
    for row, x_value in enumerate(inputs[:, 0]):
        reference = Statevector(
            qc.assign_parameters(_assignment(qc, params, x_value))
        ).data
        assert np.allclose(ours[row], reference, atol=1e-10)


def test_sobol_layout_follows_qiskit_parameter_order():
    """Coordinates map to ξ blocks first, then non-data parameters in Qiskit's sorted order."""
    rng = np.random.default_rng(0)
    point = rng.uniform(size=ARCH.sobol_dimension)
    params = map_sobol_point(ARCH, point)
    qc = _qiskit_circuit(ARCH)
    others = [p.name for p in qc.parameters if not p.name.startswith(("x[", "xi_"))]
    cursor = ARCH.num_layers * (ARCH.input_dim + 1)
    for offset, name in enumerate(others):
        group_layer, index = name.split("[")
        group, layer = group_layer.split("_")
        u = point[cursor + offset]
        expected = 2 * u - 1 if group == "delta" else (2 * u - 1) * np.pi
        assert getattr(params, group)[int(layer), int(index[:-1])] == pytest.approx(
            expected
        )
    for layer in range(ARCH.num_layers):
        sign = np.sign(2 * point[2 * layer] - 1)
        assert params.xi[layer, 0] == pytest.approx(sign * np.pi * point[2 * layer + 1])


def test_exact_features_match_qiskit_expectations():
    params = sample_parameter_draws(ARCH, num_draws=2, seed=3)[1]
    states = simulate_states(ARCH, params, np.array([[0.3]]))
    ours = exact_features(states, ARCH.num_qubits)[0]
    reference = []
    for support in observable_supports(ARCH.num_qubits):
        label = ["I"] * ARCH.num_qubits
        for qubit, pauli in support:
            label[ARCH.num_qubits - 1 - qubit] = (
                pauli  # Qiskit labels are little-endian
            )
        reference.append(
            Statevector(states[0]).expectation_value(SparsePauliOp("".join(label))).real
        )
    assert ours.shape == (24,)
    assert np.allclose(ours, reference, atol=1e-10)


def test_pauli_shadow_is_unbiased_and_noisier_than_code_model():
    params = sample_parameter_draws(ARCH, num_draws=1, seed=11)[0]
    states = simulate_states(ARCH, params, np.array([[0.4], [-0.2]]))
    exact = exact_features(states, ARCH.num_qubits)
    rng = np.random.default_rng(5)
    repeats = 300
    shadow = np.stack(
        [
            pauli_shadow_features(states, ARCH.num_qubits, 400, 1, rng)
            for _ in range(repeats)
        ]
    )
    code = np.stack(
        [code_gaussian_features(exact, 400, 1, rng) for _ in range(repeats)]
    )
    # Mean of single-group estimates is unbiased; 5-sigma tolerance from the
    # worst-case shadow variance 9 / 400 per estimate.
    tolerance = 5.0 * np.sqrt(9.0 / 400 / repeats)
    assert np.max(np.abs(shadow.mean(axis=0) - exact)) < tolerance
    num_one_local = 3 * ARCH.num_qubits
    shadow_var = shadow.var(axis=0)
    code_var = code.var(axis=0)
    # Pauli shadows: Var = (3^k - μ²)/N versus (1 - μ²)/N for direct measurement.
    assert np.all(shadow_var[:, :num_one_local] > 1.5 * code_var[:, :num_one_local])
    assert np.all(shadow_var[:, num_one_local:] > 4.0 * code_var[:, num_one_local:])


def test_runner_input_scale_reaches_encoding(tmp_path):
    import json

    from common import load_runtime_ready_config
    from randomized_quantum_local_kernels_gp.lib import runner

    def run(scale, name):
        cfg = load_runtime_ready_config()
        cfg["data"]["num_points"] = 20
        cfg["circuit"].update(num_draws=2, input_scale=scale)
        cfg["features"]["mode"] = "exact"
        cfg["mkl"]["max_iterations"] = 2
        cfg["evaluation"]["save_features"] = True
        run_dir = tmp_path / name
        run_dir.mkdir()
        snapshot = {k: v for k, v in cfg.items() if k != "dtype"} | {"dtype": "float64"}
        (run_dir / "config_snapshot.json").write_text(json.dumps(snapshot))
        runner.train_and_evaluate(cfg, run_dir)
        return np.load(run_dir / "features.npz")["exact"]

    base, scaled = run(1.0, "a"), run(2.5, "b")
    assert base.shape == scaled.shape and not np.allclose(base, scaled)
