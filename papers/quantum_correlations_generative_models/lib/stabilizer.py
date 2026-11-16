"""Exact checks for Theorem 2 ingredients and the App. E robustness lemmas.

Theorem 2 (Sec. IV C, p8-11): the Mermin-Peres square of Table I (p10) with
the example of Eq. 11 (p9) and the stabilizer-state count of Lemma 1 (Eq. 17).
App. E (p21-22): the S3 random walk of Eqs. E3-E5 and the GHZ Bell-test bound
of Lemma 6 (Eq. E9-E10).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

IDENTITY = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2.0)
PHASE_S = np.diag([1, 1j]).astype(np.complex128)


def two_qubit(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return first (x) second (qubit 1 is the left factor)."""
    return np.kron(first, second)


def mermin_peres_checks() -> dict:
    """Verify Table I (p10), Eq. 11 (p9), Eq. 12 and Eq. 14 (p10).

    A = Z1, a = Z2, B = X2, b = X1; |psi1> = |00>, |psi2> = |++>,
    |psi3> = (|00> + |01> + |10> - |11>) / 2.
    """
    A, a = two_qubit(PAULI_Z, IDENTITY), two_qubit(IDENTITY, PAULI_Z)
    B, b = two_qubit(IDENTITY, PAULI_X), two_qubit(PAULI_X, IDENTITY)
    psi1 = np.array([1, 0, 0, 0], dtype=np.complex128)
    psi2 = np.full(4, 0.5, dtype=np.complex128)
    psi3 = np.array([1, 1, 1, -1], dtype=np.complex128) / 2.0

    def commutator(x, y):
        return float(np.abs(x @ y - y @ x).max())

    def anticommutator(x, y):
        return float(np.abs(x @ y + y @ x).max())

    rows = {
        "psi1": (psi1, [A, a, A @ a]),
        "psi2": (psi2, [B, b, B @ b]),
        "psi3": (psi3, [A @ B, a @ b, A @ B @ a @ b]),
    }
    eigen_errors = {
        name: max(float(np.abs(op @ state - state).max()) for op in ops)
        for name, (state, ops) in rows.items()
    }
    column_products = [
        A @ B @ (A @ B),
        a @ b @ (a @ b),
        (A @ a) @ (B @ b) @ (A @ B @ a @ b),
    ]
    column_signs = [
        float(np.real(np.trace(product)) / 4) for product in column_products
    ]
    y1y2 = two_qubit(PAULI_Y, PAULI_Y)
    projector = (np.eye(4) + y1y2) / 2
    projected1 = projector @ psi1
    projected2 = projector @ psi2
    expected1 = np.array([1, 0, 0, -1]) / 2
    expected2 = np.array([0, 1, 1, 0]) / 2
    return {
        "commutators_zero_max": max(
            commutator(A, a), commutator(B, b), commutator(A, B), commutator(a, b)
        ),
        "anticommutators_zero_max": max(anticommutator(A, b), anticommutator(a, B)),
        "row_eigenvalue_plus_one_errors": eigen_errors,
        "column_product_signs": column_signs,
        "ABab_equals_Y1Y2_error": float(np.abs(A @ B @ a @ b - y1y2).max()),
        "eq12_prob_plus_one_on_psi3": float(np.real(psi3.conj() @ projector @ psi3)),
        "eq14_projected_psi1_error": float(np.abs(projected1 - expected1).max()),
        "eq14_projected_psi2_error": float(np.abs(projected2 - expected2).max()),
        "eq14_overlap_abs": float(abs(projected1.conj() @ projected2)),
    }


def _canonical_key(vector: np.ndarray) -> tuple:
    pivot = vector[np.argmax(np.abs(vector) > 1e-9)]
    normalised = vector * (abs(pivot) / pivot)
    return tuple(np.round(normalised.real, 9)) + tuple(np.round(normalised.imag, 9))


def enumerate_stabilizer_states(num_qubits: int) -> int:
    """Count n-qubit stabilizer states by Clifford orbit search from |0...0>."""
    dim = 2**num_qubits
    generators = []
    for qubit in range(num_qubits):
        for gate in (HADAMARD, PHASE_S):
            factors = [IDENTITY] * num_qubits
            factors[qubit] = gate
            operator = factors[0]
            for factor in factors[1:]:
                operator = np.kron(operator, factor)
            generators.append(operator)
    for control, target in itertools.permutations(range(num_qubits), 2):
        operator = np.zeros((dim, dim), dtype=np.complex128)
        for basis in range(dim):
            bits = [(basis >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]
            if bits[control]:
                bits[target] ^= 1
            image = sum(bit << (num_qubits - 1 - q) for q, bit in enumerate(bits))
            operator[image, basis] = 1
        generators.append(operator)
    start = np.zeros(dim, dtype=np.complex128)
    start[0] = 1
    seen = {_canonical_key(start)}
    frontier = [start]
    while frontier:
        next_frontier = []
        for state in frontier:
            for operator in generators:
                image = operator @ state
                key = _canonical_key(image)
                if key not in seen:
                    seen.add(key)
                    next_frontier.append(image)
        frontier = next_frontier
    return len(seen)


def stabilizer_count_formula(num_qubits: int) -> int:
    """Exact count 2^n prod_{k=0}^{n-1} (2^{n-k} + 1) (Aaronson-Gottesman, paper Ref. [76])."""
    return 2**num_qubits * math.prod(
        2 ** (num_qubits - k) + 1 for k in range(num_qubits)
    )


# S3 acting on the Pauli axes (X, Y, Z) up to signs; elements as permutation tuples.
S3_ELEMENTS = list(itertools.permutations(range(3)))


def clifford_axis_permutation(unitary: np.ndarray) -> tuple[int, int, int]:
    """Return the permutation of (X, Y, Z) induced by U P U^dagger, ignoring signs."""
    paulis = [PAULI_X, PAULI_Y, PAULI_Z]
    images = []
    for pauli in paulis:
        conjugated = unitary @ pauli @ unitary.conj().T
        overlaps = [abs(np.trace(other.conj().T @ conjugated)) / 2 for other in paulis]
        images.append(int(np.argmax(overlaps)))
    return tuple(images)


def s3_walk_transfer_matrix() -> np.ndarray:
    """Transfer matrix of the App. E walk (Eq. E4): one step applies H S^a H S^b, a, b uniform."""
    matrix = np.zeros((6, 6))
    for start_index, start in enumerate(S3_ELEMENTS):
        for a, b in itertools.product(range(2), repeat=2):
            step = (
                HADAMARD
                @ np.linalg.matrix_power(PHASE_S, a)
                @ HADAMARD
                @ np.linalg.matrix_power(PHASE_S, b)
            )
            step_perm = clifford_axis_permutation(step)
            composed = tuple(step_perm[start[axis]] for axis in range(3))
            matrix[start_index, S3_ELEMENTS.index(composed)] += 0.25
    return matrix


PRINTED_E4 = (
    np.array(
        [
            [1, 1, 1, 0, 1, 0],
            [1, 1, 1, 0, 1, 0],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 1],
            [0, 1, 0, 1, 1, 1],
        ]
    )
    / 4.0
)  # Eq. E4 as read from the 300-dpi crop of p21


def matches_printed_e4(matrix: np.ndarray) -> bool:
    """Return True if ``matrix`` equals the printed E4 under some relabelling of S3."""
    for order in itertools.permutations(range(6)):
        if np.allclose(matrix[np.ix_(order, order)], PRINTED_E4):
            return True
    return False


def identity_or_s_probability(steps: int) -> float:
    """Probability that k walk steps from I end at I or S up to Paulis (Eq. E5 left side)."""
    matrix = s3_walk_transfer_matrix()
    start = np.zeros(6)
    start[S3_ELEMENTS.index((0, 1, 2))] = 1
    final = start @ np.linalg.matrix_power(matrix, steps)
    s_perm = clifford_axis_permutation(PHASE_S)
    return float(final[S3_ELEMENTS.index((0, 1, 2))] + final[S3_ELEMENTS.index(s_perm)])


def ghz_lhv_max_success() -> float:
    """Max over deterministic local strategies of sum_a q(a) 1[t(a) in sol_a], q(a) = 1/8 (Eq. E10).

    sol_a: the GHZ constraint of Eq. 5 for even a1 + a2 + a3; no constraint for
    odd sums. Local strategies t_i = t_i(a_i) cover all local hidden variable
    models by convexity.
    """
    best = 0.0
    strategies = list(itertools.product(range(2), repeat=2))  # (t(0), t(1)) per party
    for f1, f2, f3 in itertools.product(strategies, repeat=3):
        wins = 0
        for a in itertools.product(range(2), repeat=3):
            if sum(a) % 2:
                wins += 1
                continue
            t = (f1[a[0]], f2[a[1]], f3[a[2]])
            wins += int((1j) ** sum(a) * (-1) ** sum(t) == 1)
        best = max(best, wins / 8)
    return best
