"""Local Pauli feature maps φ_r(x) and their finite-shot estimators.

Observables (Sec. V, footnote 3): X_q, Y_q, Z_q on every qubit and
X_qX_{q+1}, Y_qY_{q+1}, Z_qZ_{q+1} on every ring edge, so m = 6n.
Feature order: for each qubit q, (X_q, Y_q, Z_q); then for each ring edge,
sorted by qubit pair, (X X, Y Y, Z Z). The cosine wrapper (Eq. 21) is
invariant to this order; it only matters for which seeded noise draw lands on
which feature.

Three feature modes are provided:

- ``exact``: Tr[O_j ρ_r(x)] from the statevector.
- ``code_gaussian``: the noise model of the authors' code. Each feature gets
  K median-of-means groups whose means are Gaussian with variance
  (1 - μ²)/batch, i.e. the variance of measuring O_j directly. This is what
  produced Table I, but it is not a classical-shadow simulation.
- ``pauli_shadow``: a true single-qubit-Pauli classical-shadow simulation
  (Sec. IV-B): each snapshot picks a uniformly random basis in {X, Y, Z} per
  qubit, measures all qubits, and the inverse channel gives the estimator
  3^k Π_q (±1) for a k-local Pauli whose bases all match (0 otherwise). All
  observables share the same snapshots, aggregated by median of means.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)

PAULI_LABELS = ("X", "Y", "Z")
FEATURE_MODES = ("exact", "code_gaussian", "pauli_shadow")

_PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
# Rotations mapping the eigenbasis of each Pauli to the computational basis,
# so that outcome bit b corresponds to eigenvalue (-1)^b.
_BASIS_ROTATION = {
    "X": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0),
    "Y": np.array([[1, -1j], [1, 1j]], dtype=complex) / np.sqrt(2.0),
    "Z": np.eye(2, dtype=complex),
}


def observable_supports(num_qubits: int) -> list[tuple[tuple[int, str], ...]]:
    """List the m = 6n local observables as tuples of (qubit, Pauli label).

    Parameters
    ----------
    num_qubits : int
        Ring size n.

    Returns
    -------
    list of tuple
        Observable supports in feature order.
    """
    observables: list[tuple[tuple[int, str], ...]] = []
    for q in range(num_qubits):
        for label in PAULI_LABELS:
            observables.append(((q, label),))
    # Ring edges sorted by (min, max) qubit, e.g. (0,1), (0,3), (1,2), (2,3)
    # for n = 4. This is the authors' feature order; it does not change any
    # kernel (Eq. 21 is order invariant) but makes the seeded noise draws of
    # ``code_gaussian`` land on the same features as in their run.
    edges = sorted(
        {tuple(sorted((q, (q + 1) % num_qubits))) for q in range(num_qubits)}
    )
    for first, second in edges:
        for label in PAULI_LABELS:
            observables.append(((first, label), (second, label)))
    return observables


def _apply_local_operator(
    states: np.ndarray, operator: np.ndarray, qubit: int
) -> np.ndarray:
    batch, dim = states.shape
    high = dim >> (qubit + 1)
    low = 1 << qubit
    view = states.reshape(batch, high, 2, low)
    return np.einsum("ab,hjbl->hjal", operator, view).reshape(batch, dim)


def exact_features(states: np.ndarray, num_qubits: int) -> np.ndarray:
    """Exact expectation values ⟨ψ|O_j|ψ⟩ (Eq. 4-5).

    Parameters
    ----------
    states : np.ndarray
        Statevectors of shape (M, 2**n).
    num_qubits : int
        Ring size n.

    Returns
    -------
    np.ndarray
        Real features of shape (M, 6n).
    """
    features = []
    for support in observable_supports(num_qubits):
        transformed = states
        for qubit, label in support:
            transformed = _apply_local_operator(transformed, _PAULI[label], qubit)
        features.append(np.real(np.sum(np.conj(states) * transformed, axis=1)))
    return np.stack(features, axis=1)


def median_of_means_groups(num_snapshots: int, num_groups: int | None) -> int:
    """Number of median-of-means groups K.

    When ``num_groups`` is None, use the authors' heuristic
    K = max(1, min(16, floor(sqrt(N)))).
    """
    if num_snapshots < 1:
        raise ValueError(f"num_snapshots must be >= 1, got {num_snapshots}.")
    if num_groups is None:
        num_groups = max(1, min(16, int(np.sqrt(num_snapshots))))
    return min(num_groups, num_snapshots)


def code_gaussian_features(
    exact: np.ndarray,
    num_snapshots: int,
    num_groups: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Authors' noise model: median of K Gaussian batch means per feature.

    Parameters
    ----------
    exact : np.ndarray
        Exact features μ of any shape.
    num_snapshots : int
        Shots N per (draw, input).
    num_groups : int or None
        Median-of-means groups K (None = authors' heuristic).
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    np.ndarray
        Noisy features with the same shape as ``exact``.
    """
    groups = median_of_means_groups(num_snapshots, num_groups)
    batch_size = max(num_snapshots // groups, 1)
    mu = np.clip(exact, -1.0, 1.0)
    std = np.sqrt(np.maximum(1.0 - mu**2, 0.0) / batch_size)
    group_means = mu[..., None] + std[..., None] * rng.standard_normal(
        mu.shape + (groups,)
    )
    return np.median(group_means, axis=-1)


def _basis_outcome_estimator_table(
    num_qubits: int,
) -> tuple[list[tuple[str, ...]], np.ndarray]:
    """Shadow estimator value for every (basis setting, outcome, observable).

    Returns the list of 3^n basis settings and a table of shape
    (3^n, 2^n, m) with entries 3^k Π_q (-1)^{b_q} when all k Pauli labels of
    the observable match the measured bases, else 0.
    """
    settings = list(itertools.product(PAULI_LABELS, repeat=num_qubits))
    supports = observable_supports(num_qubits)
    dim = 1 << num_qubits
    outcomes = np.arange(dim)
    table = np.zeros((len(settings), dim, len(supports)))
    for s_index, setting in enumerate(settings):
        for j, support in enumerate(supports):
            if all(setting[qubit] == label for qubit, label in support):
                sign = np.ones(dim)
                for qubit, _ in support:
                    sign = sign * (1.0 - 2.0 * ((outcomes >> qubit) & 1))
                table[s_index, :, j] = (3.0 ** len(support)) * sign
    return settings, table


def _basis_probabilities(
    states: np.ndarray, settings: list[tuple[str, ...]]
) -> np.ndarray:
    """Born probabilities after rotating every qubit to its measured basis.

    Returns an array of shape (M, 3^n, 2^n).
    """
    probabilities = np.empty((states.shape[0], len(settings), states.shape[1]))
    for s_index, setting in enumerate(settings):
        rotated = states
        for qubit, label in enumerate(setting):
            rotated = _apply_local_operator(rotated, _BASIS_ROTATION[label], qubit)
        probabilities[:, s_index, :] = np.abs(rotated) ** 2
    totals = probabilities.sum(axis=-1, keepdims=True)
    if not np.allclose(totals, 1.0, atol=1e-10):
        raise ValueError(
            "Statevectors are not normalized; Born probabilities do not sum to 1."
        )
    # Remove rounding drift only, so multinomial sampling accepts the vectors.
    probabilities /= totals
    return probabilities


def pauli_shadow_features(
    states: np.ndarray,
    num_qubits: int,
    num_snapshots: int,
    num_groups: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Classical-shadow estimates of all local features (Sec. IV-B).

    The N snapshots of each state are split into K groups. Within a group,
    basis settings are drawn uniformly (multinomial over 3^n settings), then
    outcomes are drawn from the Born distribution of each setting
    (multinomial). Group means of the shadow estimator are aggregated by
    their median. Sampling counts instead of individual shots is exact in
    distribution, because the estimator is a sum over snapshots.

    Parameters
    ----------
    states : np.ndarray
        Statevectors of shape (M, 2**n).
    num_qubits : int
        Ring size n.
    num_snapshots : int
        Snapshots N per state.
    num_groups : int or None
        Median-of-means groups K (None = authors' heuristic).
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    np.ndarray
        Estimated features of shape (M, 6n).
    """
    groups = median_of_means_groups(num_snapshots, num_groups)
    if num_snapshots % groups != 0:
        raise ValueError(
            f"num_snapshots={num_snapshots} is not divisible by {groups} median-of-means groups."
        )
    snapshots_per_group = num_snapshots // groups
    settings, table = _basis_outcome_estimator_table(num_qubits)
    probabilities = _basis_probabilities(states, settings)
    num_states, num_settings, dim = probabilities.shape
    setting_counts = rng.multinomial(
        snapshots_per_group,
        np.full(num_settings, 1.0 / num_settings),
        size=(num_states, groups),
    )
    outcome_counts = rng.multinomial(setting_counts, probabilities[:, None, :, :])
    flat_counts = outcome_counts.reshape(num_states, groups, num_settings * dim)
    group_means = (
        flat_counts @ table.reshape(num_settings * dim, -1) / snapshots_per_group
    )
    return np.median(group_means, axis=1)


def estimate_features(
    states: np.ndarray,
    num_qubits: int,
    mode: str,
    num_snapshots: int | None,
    num_groups: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Feature vector φ̂_r(x) for one draw under the chosen feature mode.

    Parameters
    ----------
    states : np.ndarray
        Statevectors of shape (M, 2**n).
    num_qubits : int
        Ring size n.
    mode : str
        One of ``FEATURE_MODES``.
    num_snapshots : int or None
        Snapshots N; required unless mode is ``exact``.
    num_groups : int or None
        Median-of-means groups K (None = authors' heuristic).
    rng : np.random.Generator
        Random generator used by the noisy modes.

    Returns
    -------
    np.ndarray
        Features of shape (M, 6n).
    """
    if mode not in FEATURE_MODES:
        raise ValueError(
            f"Unknown feature mode {mode!r}; expected one of {FEATURE_MODES}."
        )
    if mode == "exact":
        return exact_features(states, num_qubits)
    if num_snapshots is None:
        raise ValueError(f"Feature mode {mode!r} requires features.num_snapshots.")
    if mode == "code_gaussian":
        return code_gaussian_features(
            exact_features(states, num_qubits), num_snapshots, num_groups, rng
        )
    return pauli_shadow_features(states, num_qubits, num_snapshots, num_groups, rng)
