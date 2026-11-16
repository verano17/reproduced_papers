"""E10: the Fig. 10b collective-gate construction of control-R = Rz(sum_i w_i x_i) (p17).

Read from a 300-dpi crop of p17: the right-hand side is a fan-out (the target
qubit controls X on every x_i), Rz(w_i / 2) on each x_i, the same fan-out,
then Rz(-w_i / 2) on each x_i. With Rz(t) = exp(-i t Z / 2), the circuit acts
on the target, for every computational-basis control x, as
Rz(W / 2 - theta(x)) up to a phase that depends on x only, where
theta(x) = sum_i w_i x_i and W = sum_i w_i. The caption's control-R =
Rz(theta) therefore holds up to the Rz sign convention, a fixed
control-independent Rz(W / 2) on the target, and a diagonal phase on the
controls (irrelevant for computational-basis controls). The depth does not
depend on k, which is the point of App. B2.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest


def rz(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def kron_all(matrices) -> np.ndarray:
    result = np.ones((1, 1), dtype=complex)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def fan_out(num_controls: int) -> np.ndarray:
    """Target (last qubit) controls X on every other qubit."""
    dimension = 2 ** (num_controls + 1)
    matrix = np.zeros((dimension, dimension), dtype=complex)
    for bits in itertools.product((0, 1), repeat=num_controls + 1):
        target = bits[-1]
        image = tuple(bit ^ target for bit in bits[:-1]) + (target,)
        matrix[_index(image), _index(bits)] = 1.0
    return matrix


def _index(bits) -> int:
    return int("".join(map(str, bits)), 2)


def fig10b_circuit(weights: np.ndarray) -> np.ndarray:
    identity = np.eye(2)
    first = kron_all([rz(w / 2) for w in weights] + [identity])
    second = kron_all([rz(-w / 2) for w in weights] + [identity])
    fan = fan_out(len(weights))
    return second @ fan @ first @ fan


@pytest.mark.parametrize("num_controls", [1, 2, 3, 4])
def test_fig10b_is_controlled_rz_up_to_fixed_rotation_and_control_phases(num_controls):
    weights = np.random.default_rng(num_controls).normal(size=num_controls)
    circuit = fig10b_circuit(weights)
    total = weights.sum()
    for controls in itertools.product((0, 1), repeat=num_controls):
        rows = [_index(controls + (target,)) for target in (0, 1)]
        columns = circuit[:, rows]
        block = columns[rows, :]
        assert np.linalg.norm(columns) == pytest.approx(np.linalg.norm(block))
        expected = rz(total / 2 - float(np.dot(weights, controls)))
        phase = np.vdot(expected.ravel(), block.ravel())
        phase /= abs(phase)
        np.testing.assert_allclose(block, phase * expected, atol=1e-12)
