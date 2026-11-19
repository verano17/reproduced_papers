"""Shallow ring circuits U(x, θ_r) and their Sobol parameter draws.

Implements Sec. IV-A and IV-C of Aaraba et al. (QCNC 2026) as a NumPy
statevector simulator for small qubit counts (the paper uses n = 4).

Circuit (Eq. 3, 8, 9 and Fig. 2), acting on |+>^n:

    U(x, θ) = Π_ℓ W_ℓ G_ℓ(x),
    G_ℓ(x)  = ⊗_q R_Z(δ_{ℓ,q} · ξ_ℓᵀ x),
    W_ℓ     = (Π_q R_Y(ω_{ℓ,q})) (Π_{(i,j) ∈ E_ring} R_ZZ(β_{ℓ,ij})) (Π_q R_X(α_{ℓ,q})).

Gate conventions follow Qiskit, which the authors' code uses:
R_P(t) = exp(-i t P / 2) and R_ZZ(t) = exp(-i t Z⊗Z / 2). Eq. 9 of the paper
writes exp(-i β Z Z) instead (a factor 2 in β); the code convention is the one
behind Table I. Qubit q is bit q of the basis-state index (Qiskit little-endian
ordering), so a statevector index k has qubit q in state (k >> q) & 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

# Qiskit sorts circuit parameters by name; the authors' Sobol layout consumes
# Sobol coordinates in that order. Reproducing the order makes our θ_r equal
# to theirs for the same scrambling seed.
_ANGLE_GROUPS = ("alpha", "beta", "delta", "omega")


@dataclass(frozen=True)
class RingArchitecture:
    """Ring circuit shape.

    Parameters
    ----------
    num_qubits : int
        Number of qubits n (ring of n qubits).
    num_layers : int
        Number of data re-uploading layers L.
    input_dim : int
        Input dimension ζ.
    """

    num_qubits: int
    num_layers: int
    input_dim: int

    def __post_init__(self) -> None:
        if self.num_qubits < 3:
            raise ValueError(f"A ring needs at least 3 qubits, got {self.num_qubits}.")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}.")
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}.")

    @property
    def ring_edges(self) -> tuple[tuple[int, int], ...]:
        """Edges (0,1), (1,2), ..., (n-2, n-1), (n-1, 0) in application order."""
        n = self.num_qubits
        return tuple((q, (q + 1) % n) for q in range(n))

    @property
    def sobol_dimension(self) -> int:
        """Sobol dimension: L·(ζ + 1) for ξ plus L·(3n + |E|) angles/offsets."""
        n, L = self.num_qubits, self.num_layers
        return L * (self.input_dim + 1) + L * (3 * n + len(self.ring_edges))


@dataclass(frozen=True)
class CircuitParameters:
    """One parameter draw θ_r (Eq. 12).

    Attributes
    ----------
    xi : np.ndarray
        Projection vectors, shape (L, ζ).
    delta : np.ndarray
        Generator offsets δ_{ℓ,q}, shape (L, n).
    alpha : np.ndarray
        R_X mixer angles, shape (L, n).
    beta : np.ndarray
        R_ZZ entangler angles per ring edge, shape (L, |E|).
    omega : np.ndarray
        R_Y mixer angles, shape (L, n).
    """

    xi: np.ndarray
    delta: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    omega: np.ndarray


def _qiskit_layer_order(num_layers: int) -> list[int]:
    # Qiskit compares ParameterVector names as strings ("alpha_10" < "alpha_2").
    return sorted(range(num_layers), key=str)


def map_sobol_point(arch: RingArchitecture, point: np.ndarray) -> CircuitParameters:
    """Map one Sobol point u ∈ [0, 1)^d to circuit parameters (Sec. IV-C).

    ξ_ℓ uses ζ coordinates for a direction in [-1, 1]^ζ (normalized to unit
    norm) and one coordinate for a radius π·u^{1/ζ} ∈ (0, π), as in the
    authors' code (the paper only states "radius < π"). δ maps to (-1, 1) and
    α, β, ω map to (-π, π) by affine rescaling.

    Parameters
    ----------
    arch : RingArchitecture
        Circuit shape.
    point : np.ndarray
        Sobol point of shape (arch.sobol_dimension,).

    Returns
    -------
    CircuitParameters
        The parameter draw.

    Raises
    ------
    ValueError
        If the point has the wrong dimension.
    """
    point = np.asarray(point, dtype=float)
    if point.shape != (arch.sobol_dimension,):
        raise ValueError(
            f"Sobol point has shape {point.shape}, expected ({arch.sobol_dimension},)."
        )
    n, L, zeta = arch.num_qubits, arch.num_layers, arch.input_dim
    num_edges = len(arch.ring_edges)
    cursor = 0

    xi = np.empty((L, zeta))
    for layer in _qiskit_layer_order(L):
        direction = 2.0 * point[cursor : cursor + zeta] - 1.0
        cursor += zeta
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            direction = np.ones(zeta)
            norm = np.linalg.norm(direction)
        radius = np.pi * point[cursor] ** (1.0 / zeta)
        cursor += 1
        xi[layer] = radius * direction / norm

    sizes = {"alpha": n, "beta": num_edges, "delta": n, "omega": n}
    groups: dict[str, np.ndarray] = {}
    for name in _ANGLE_GROUPS:
        values = np.empty((L, sizes[name]))
        for layer in _qiskit_layer_order(L):
            values[layer] = point[cursor : cursor + sizes[name]]
            cursor += sizes[name]
        if name == "delta":
            groups[name] = 2.0 * values - 1.0
        else:
            groups[name] = (2.0 * values - 1.0) * np.pi
    if cursor != arch.sobol_dimension:
        raise RuntimeError(
            f"Consumed {cursor} Sobol coordinates, expected {arch.sobol_dimension}."
        )
    return CircuitParameters(xi=xi, **groups)


def sample_parameter_draws(
    arch: RingArchitecture, num_draws: int, seed: int
) -> list[CircuitParameters]:
    """Draw θ_1..θ_R from a scrambled Sobol sequence (Sec. IV-C).

    Like the authors' code, 2^ceil(log2 R) points are drawn with
    ``random_base2`` and the first R are kept, so the sequence keeps its
    prefix property.

    Parameters
    ----------
    arch : RingArchitecture
        Circuit shape.
    num_draws : int
        Number of draws R.
    seed : int
        Scrambling seed of ``scipy.stats.qmc.Sobol``.

    Returns
    -------
    list of CircuitParameters
        R parameter draws.
    """
    if num_draws < 1:
        raise ValueError(f"num_draws must be >= 1, got {num_draws}.")
    engine = qmc.Sobol(d=arch.sobol_dimension, scramble=True, seed=seed)
    exponent = int(np.ceil(np.log2(num_draws)))
    points = engine.random_base2(m=exponent)[:num_draws]
    return [map_sobol_point(arch, point) for point in points]


def _apply_single_qubit(
    states: np.ndarray, gates: np.ndarray, qubit: int
) -> np.ndarray:
    """Apply per-sample 2x2 gates to one qubit.

    states : (B, 2**n) complex; gates : (B, 2, 2) or (2, 2).
    """
    batch, dim = states.shape
    high = dim >> (qubit + 1)
    low = 1 << qubit
    view = states.reshape(batch, high, 2, low)
    if gates.ndim == 2:
        out = np.einsum("ab,hjbl->hjal", gates, view)
    else:
        out = np.einsum("hab,hjbl->hjal", gates, view)
    return out.reshape(batch, dim)


def _rz_gates(angles: np.ndarray) -> np.ndarray:
    gates = np.zeros(angles.shape + (2, 2), dtype=complex)
    gates[..., 0, 0] = np.exp(-0.5j * angles)
    gates[..., 1, 1] = np.exp(0.5j * angles)
    return gates


def _rx_gate(angle: float) -> np.ndarray:
    c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry_gate(angle: float) -> np.ndarray:
    c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _zz_phases(num_qubits: int, i: int, j: int, angle: float) -> np.ndarray:
    indices = np.arange(1 << num_qubits)
    parity = ((indices >> i) & 1) ^ ((indices >> j) & 1)
    # Z_i Z_j eigenvalue is +1 for even parity and -1 for odd parity.
    eigen = 1.0 - 2.0 * parity
    return np.exp(-0.5j * angle * eigen)


def simulate_states(
    arch: RingArchitecture, params: CircuitParameters, inputs: np.ndarray
) -> np.ndarray:
    """Statevectors |ψ_r(x)⟩ = U(x, θ_r)|+⟩^n for a batch of inputs.

    Parameters
    ----------
    arch : RingArchitecture
        Circuit shape.
    params : CircuitParameters
        One parameter draw θ_r.
    inputs : np.ndarray
        Inputs of shape (M, ζ).

    Returns
    -------
    np.ndarray
        Complex statevectors of shape (M, 2**n), qubit q = bit q of the index.
    """
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] != arch.input_dim:
        raise ValueError(
            f"inputs must have shape (M, {arch.input_dim}), got {inputs.shape}."
        )
    n = arch.num_qubits
    dim = 1 << n
    batch = inputs.shape[0]
    states = np.full((batch, dim), 1.0 / np.sqrt(dim), dtype=complex)
    for layer in range(arch.num_layers):
        projected = inputs @ params.xi[layer]  # γ_ℓ(x), shape (M,)
        for q in range(n):
            states = _apply_single_qubit(
                states, _rz_gates(params.delta[layer, q] * projected), q
            )
        for q in range(n):
            states = _apply_single_qubit(states, _rx_gate(params.alpha[layer, q]), q)
        for edge_index, (i, j) in enumerate(arch.ring_edges):
            states = (
                states * _zz_phases(n, i, j, params.beta[layer, edge_index])[None, :]
            )
        for q in range(n):
            states = _apply_single_qubit(states, _ry_gate(params.omega[layer, q]), q)
    return states
