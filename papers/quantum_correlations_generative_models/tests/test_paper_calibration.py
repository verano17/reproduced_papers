"""Phase 3 calibrations of the paper's plotted metric (R1) and Fig. 8 test (R7)."""

from __future__ import annotations

import math

import pytest
from common import PROJECT_DIR  # noqa: F401  (puts the repo root on sys.path)
from quantum_correlations_generative_models.lib.metrics import (
    likelihood_ratio_threshold_bits,
    paper_inferred_lrt_degrees_of_freedom,
    paper_kl_estimate_bits_per_site,
)


def test_paper_kl_estimate_subtracts_uniform_entropy_per_site():
    assert paper_kl_estimate_bits_per_site(2.0, 80, 57) == pytest.approx(
        2.0 - math.log2(80) / 57
    )


def test_paper_kl_estimate_is_negative_when_the_model_beats_log2_n():
    assert paper_kl_estimate_bits_per_site(0.5, 1500, 16) < 0


# Solid curves of Fig. 8 (p14), read from a 300-dpi crop at h = 2, 4, ..., 12.
FIG8_THRESHOLDS = {
    80: [0.30, 0.54, 0.85, 1.23, 1.69, 2.22],  # Promoter Gene Sequences
    200: [0.12, 0.21, 0.33, 0.48, 0.67, 0.87],  # SPECT Heart, pooled split
}


@pytest.mark.parametrize("num_train,read_values", FIG8_THRESHOLDS.items())
def test_inferred_df_reproduces_fig8_thresholds(num_train, read_values):
    for bond_dim, read in zip(range(2, 13, 2), read_values):
        df = paper_inferred_lrt_degrees_of_freedom(bond_dim)
        assert likelihood_ratio_threshold_bits(num_train, df) == pytest.approx(
            read, abs=0.03
        )
