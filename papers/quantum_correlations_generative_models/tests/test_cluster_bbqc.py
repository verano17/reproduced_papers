from __future__ import annotations

import numpy as np
from common import PROJECT_DIR  # noqa: F401
from quantum_correlations_generative_models.lib import cluster_bbqc as cb


def test_transfer_matrix_matches_literal_circuit():
    for pairs in (2, 3, 5):
        assert (
            np.abs(
                cb.exact_distribution(pairs) - cb.statevector_distribution(pairs)
            ).max()
            < 1e-14
        )


def test_eq4_holds_as_real_part_and_eq5_support():
    conditional = cb.ghz_conditional(cb.exact_distribution(5), [0, 2, 4])
    assert np.abs(conditional - cb.eq4_reference(real_part=True)).max() < 1e-14
    assert cb.eq5_violations(conditional) == 0


def test_no_local_product_support_reproduces_the_ghz_support():
    conditional = cb.ghz_conditional(cb.exact_distribution(5), [0, 2, 4])
    assert cb.local_support_matches(conditional)["exact_matches"] == 0


def test_six_state_hidden_process_is_exact():
    for pairs in (2, 4, 6):
        assert (
            np.abs(
                cb.hidden_process_distribution(pairs) - cb.exact_distribution(pairs)
            ).max()
            < 1e-14
        )


def test_full_order_kgram_is_exact():
    target = cb.exact_distribution(4)
    assert (
        cb.compare_distributions(cb.ml_kgram(target, 4), target)["l1_distance"] < 1e-12
    )
