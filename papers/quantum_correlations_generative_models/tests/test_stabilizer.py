from __future__ import annotations

import pytest
from common import PROJECT_DIR  # noqa: F401
from quantum_correlations_generative_models.lib import stabilizer as st


def test_mermin_peres_square_relations():
    checks = st.mermin_peres_checks()
    assert (
        checks["commutators_zero_max"] == 0 and checks["anticommutators_zero_max"] == 0
    )
    assert checks["column_product_signs"] == [1.0, 1.0, -1.0]
    assert max(checks["row_eigenvalue_plus_one_errors"].values()) < 1e-14
    assert checks["eq12_prob_plus_one_on_psi3"] == pytest.approx(1.0)
    assert checks["eq14_overlap_abs"] < 1e-14


@pytest.mark.parametrize("num_qubits,count", [(1, 6), (2, 60)])
def test_stabilizer_counts(num_qubits, count):
    assert (
        st.enumerate_stabilizer_states(num_qubits)
        == st.stabilizer_count_formula(num_qubits)
        == count
    )


def test_s3_walk_matches_eq_e4_and_e5():
    assert st.matches_printed_e4(st.s3_walk_transfer_matrix())
    for steps in range(1, 6):
        assert st.identity_or_s_probability(steps) == pytest.approx(
            1 / 3 + (2 / 3) * 4.0**-steps
        )


def test_local_strategies_win_at_most_seven_eighths():
    assert st.ghz_lhv_max_success() == pytest.approx(7 / 8)
