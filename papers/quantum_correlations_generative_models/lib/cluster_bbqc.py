"""Theorem 1 toy model: the cluster-state basis-enhanced 2-gram model (Fig. 2a, p4).

Circuit (read from the 300-dpi crop of Fig. 2a): ``num_pairs`` pairs of
qubits, all prepared in |+>. In pair i the first qubit (basis qubit, outcome
b_i) controls an S gate on the second (physical) qubit; CZ gates join
consecutive physical qubits into a 1D cluster state; each physical qubit gets
H and every qubit is measured in the computational basis (outcome s_i on the
physical qubit). Hence b_i = 0 measures the cluster qubit in X and b_i = 1 in
S^dagger X S = Y (caption, p4).

Each pair is one variable x_i = 2 b_i + s_i in {0, 1, 2, 3}. Because CZ and
controlled-S are diagonal, the amplitude is

    Psi(b, s) = 2^{-n/2} sum_z (-1)^{sum_i z_i z_{i+1}}
                prod_i (1/2) (-1)^{s_i z_i} i^{b_i z_i},

with z the physical computational-basis values (one 1/sqrt2 from the basis
qubit's |+>, one from the final H). ``exact_distribution`` contracts this as a
transfer matrix over z and returns all 4^n probabilities.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

PAIR_VALUES = 4  # x = 2 b + s


def decode(x: int) -> tuple[int, int]:
    """Return (b, s) of a pair value x = 2 b + s."""
    return x // 2, x % 2


def exact_distribution(num_pairs: int) -> np.ndarray:
    """Return q over all pairs as an array of shape ``(4,) * num_pairs``."""
    site_factor = np.zeros((PAIR_VALUES, 2), dtype=np.complex128)  # f[x, z]
    for x in range(PAIR_VALUES):
        b, s = decode(x)
        for z in range(2):
            site_factor[x, z] = 0.5 * (-1) ** (s * z) * (1j) ** (b * z) / math.sqrt(2.0)
    cz_sign = np.array([[1.0, 1.0], [1.0, -1.0]])  # (-1)^{z z'}
    amplitude = site_factor.copy()  # shape (4^i, 2)
    for _ in range(1, num_pairs):
        propagated = amplitude @ cz_sign  # sum over previous z with (-1)^{z z'}
        amplitude = (propagated[:, None, :] * site_factor[None, :, :]).reshape(-1, 2)
    probabilities = np.abs(amplitude.sum(axis=1)) ** 2
    return probabilities.reshape((PAIR_VALUES,) * num_pairs)


def statevector_distribution(num_pairs: int) -> np.ndarray:
    """Simulate the literal Fig. 2a gate sequence on 2 * num_pairs qubits.

    Qubit 2i is the basis qubit of pair i, qubit 2i+1 its physical qubit.
    Used only to cross-check ``exact_distribution`` (10 qubits for 5 pairs).
    """
    num_qubits = 2 * num_pairs
    state = np.full(2**num_qubits, 2 ** (-num_qubits / 2), dtype=np.complex128)
    tensor = state.reshape((2,) * num_qubits)

    def apply_diagonal(qubits: tuple[int, ...], phase_fn) -> None:
        nonlocal tensor
        for values in itertools.product(range(2), repeat=len(qubits)):
            phase = phase_fn(*values)
            if phase == 1:
                continue
            index = [slice(None)] * num_qubits
            for qubit, value in zip(qubits, values):
                index[qubit] = value
            tensor[tuple(index)] *= phase

    for pair in range(num_pairs):  # controlled-S: phase i when both are 1
        apply_diagonal((2 * pair, 2 * pair + 1), lambda c, t: 1j if (c and t) else 1)
    for pair in range(num_pairs - 1):  # CZ between consecutive physical qubits
        apply_diagonal(
            (2 * pair + 1, 2 * pair + 3), lambda a, b: -1 if (a and b) else 1
        )
    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2.0)
    for pair in range(num_pairs):
        tensor = np.moveaxis(
            np.tensordot(hadamard, tensor, axes=([1], [2 * pair + 1])), 0, 2 * pair + 1
        )
    probabilities = np.abs(tensor) ** 2  # axes: b_0, s_0, b_1, s_1, ...
    return probabilities.reshape((PAIR_VALUES,) * num_pairs)


def ghz_conditional(distribution: np.ndarray, chosen_pairs: list[int]) -> np.ndarray:
    """Return q(b, s of the chosen pairs | every other pair = (0, 0)), normalised.

    Shape ``(2, 2, 2, 2, 2, 2)`` indexed as (b1, s1, b2, s2, b3, s3), the
    conditioning of Eq. 4 (p7): ancillary pairs measured in X with outcome 0.
    """
    num_pairs = distribution.ndim
    index: list = [0] * num_pairs
    for pair in chosen_pairs:
        index[pair] = slice(None)
    block = distribution[tuple(index)].reshape((2, 2) * len(chosen_pairs))
    return block / block.sum()


def eq4_reference(real_part: bool) -> np.ndarray:
    """Return the Eq. 4 expression 1 + i^{b1+b2+b3} (-1)^{s1+s2+s3}, normalised.

    The printed expression is complex when b1 + b2 + b3 is odd. With
    ``real_part`` the real part is used, which is the probability rule for an
    X/Y measurement of a GHZ state; without it the modulus of the printed
    complex number is used, for comparison only.
    """
    values = np.zeros((2,) * 6)
    for b1, s1, b2, s2, b3, s3 in itertools.product(range(2), repeat=6):
        term = 1 + (1j) ** (b1 + b2 + b3) * (-1) ** (s1 + s2 + s3)
        values[b1, s1, b2, s2, b3, s3] = term.real if real_part else abs(term)
    return values / values.sum()


def eq5_violations(conditional: np.ndarray, tolerance: float = 1e-12) -> int:
    """Count (b, s) with b1 xor b2 xor b3 = 0 where support and Eq. 5 disagree."""
    violations = 0
    for b1, s1, b2, s2, b3, s3 in itertools.product(range(2), repeat=6):
        if (b1 + b2 + b3) % 2:
            continue
        satisfies = (1j) ** (b1 + b2 + b3) * (-1) ** (s1 + s2 + s3) == 1
        nonzero = conditional[b1, s1, b2, s2, b3, s3] > tolerance
        violations += int(satisfies != nonzero)
    return violations


def local_support_matches(conditional: np.ndarray, tolerance: float = 1e-12) -> dict:
    """Exhaustively test the factorised (local hidden variable) form of Eq. 6.

    Eq. 6 (p8): the k-gram conditional factorises as f1(s1, b1) f2(s2, b2)
    f3(s3, b3). Its support is a product of three non-empty subsets of
    {(b, s)}. Enumerate all 15^3 such products and count those equal to the
    support of the quantum conditional.
    """
    target = conditional.reshape(4, 4, 4) > tolerance
    subsets = [mask for mask in itertools.product([False, True], repeat=4) if any(mask)]
    exact_matches = 0
    smallest_difference = None
    for first, second, third in itertools.product(subsets, repeat=3):
        product_support = np.einsum("i,j,k->ijk", first, second, third).astype(bool)
        difference = int((product_support != target).sum())
        exact_matches += int(difference == 0)
        smallest_difference = (
            difference
            if smallest_difference is None
            else min(smallest_difference, difference)
        )
    return {
        "num_product_supports": len(subsets) ** 3,
        "exact_matches": exact_matches,
        "min_symmetric_difference": smallest_difference,
        "target_support_size": int(target.sum()),
    }


def ml_kgram(distribution: np.ndarray, order: int) -> np.ndarray:
    """Return the maximum-likelihood k-gram fit of ``distribution``.

    A k-gram model conditions each variable on its k - 1 predecessors (Fig.
    1a, p3: a 4-gram uses p(x_l | x_{l-1}, x_{l-2}, x_{l-3})). The fit that
    minimises KL(q || p) uses q's own window marginals:
    p(x) = prod_i q(x_i | x_{max(0, i-k+1)} .. x_{i-1}).
    """
    num_vars = distribution.ndim
    log_model = np.zeros_like(distribution)
    with np.errstate(divide="ignore", invalid="ignore"):
        for site in range(num_vars):
            window = list(range(max(0, site - order + 1), site + 1))
            others = tuple(axis for axis in range(num_vars) if axis not in window)
            joint = distribution.sum(axis=others, keepdims=True)
            parents = joint.sum(axis=site, keepdims=True)
            log_model = log_model + np.log(joint) - np.log(parents)
    model = np.exp(log_model)
    model[~np.isfinite(model)] = 0.0
    return model


def compare_distributions(
    model: np.ndarray, target: np.ndarray, tolerance: float = 1e-12
) -> dict:
    """Return support, KL and l1 comparisons of ``model`` (p) against ``target`` (q)."""
    model_support = model > tolerance
    target_support = target > tolerance
    with np.errstate(divide="ignore", invalid="ignore"):
        forward_terms = np.where(
            target_support, target * (np.log2(target) - np.log2(model)), 0.0
        )
        reverse_terms = np.where(
            model_support, model * (np.log2(model) - np.log2(target)), 0.0
        )
    return {
        "support_equal": bool(np.array_equal(model_support, target_support)),
        "model_mass_outside_target_support": float(model[~target_support].sum()),
        "target_mass_outside_model_support": float(target[~model_support].sum()),
        "kl_target_to_model_bits": float(forward_terms.sum()),
        "kl_model_to_target_bits": float(reverse_terms.sum()),
        "l1_distance": float(np.abs(model - target).sum()),
    }


# Single-qubit stabilizer states used by the App. F hidden process (p23).
STABILIZER_LABELS = ["|0>", "|1>", "|+>", "|->", "|+i>", "|-i>"]
_STABILIZER_VECTORS = (
    np.array([[1, 0], [0, 1], [1, 1], [1, -1], [1, 1j], [1, -1j]], dtype=np.complex128)
    / np.array([1, 1, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)])[:, None]
)


def _stabilizer_index(vector: np.ndarray) -> int:
    vector = vector / np.linalg.norm(vector)
    overlaps = np.abs(_STABILIZER_VECTORS.conj() @ vector)
    index = int(np.argmax(overlaps))
    if not math.isclose(overlaps[index], 1.0, abs_tol=1e-9):
        raise ValueError("Vector is not a single-qubit stabilizer state")
    return index


def six_state_hidden_process() -> tuple[np.ndarray, np.ndarray, int]:
    """Build the 6-state hidden process of App. F (p23).

    The hidden state is the logical single-qubit state psi on the next
    unmeasured physical qubit. Measuring a non-final pair with outcome (b, s)
    occurs with probability 1/4 and maps psi to H diag(1, (-1)^s i^b) psi
    (teleportation along the cluster); the final pair is measured directly
    with probability (1/2) |<s| H S^b |psi>|^2.

    Returns
    -------
    tuple
        ``transfer[x, a, c]`` (probability of emitting x and moving from state
        a to c on a non-final pair), ``final_emission[a, x]`` and the index of
        the initial state |+>.
    """
    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2.0)
    transfer = np.zeros((PAIR_VALUES, 6, 6))
    final_emission = np.zeros((6, PAIR_VALUES))
    for x in range(PAIR_VALUES):
        b, s = decode(x)
        phases = np.diag([1, (-1) ** s * (1j) ** b])
        for state in range(6):
            next_state = _stabilizer_index(
                hadamard @ phases @ _STABILIZER_VECTORS[state]
            )
            transfer[x, state, next_state] = 0.25
            basis_phase = np.diag([1, (1j) ** b])  # <s| H S^b |psi>
            amplitude = (hadamard @ basis_phase @ _STABILIZER_VECTORS[state])[s]
            final_emission[state, x] = 0.5 * abs(amplitude) ** 2
    return transfer, final_emission, STABILIZER_LABELS.index("|+>")


def hidden_process_distribution(num_pairs: int) -> np.ndarray:
    """Return the distribution generated by the App. F 6-state hidden process."""
    transfer, final_emission, initial = six_state_hidden_process()
    messages = np.zeros((1, 6))
    messages[0, initial] = 1.0
    for _ in range(num_pairs - 1):
        messages = np.einsum("za,xac->zxc", messages, transfer).reshape(-1, 6)
    probabilities = messages @ final_emission
    return probabilities.reshape((PAIR_VALUES,) * num_pairs)
