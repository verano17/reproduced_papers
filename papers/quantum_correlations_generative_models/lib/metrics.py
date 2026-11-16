"""Metrics for the sequence-model experiments (Sec. V, p12-14).

The paper plots a "stochastic estimate of the KL divergence (normalized by the
sequence length)" in bits (Fig. 7, p14) built from the negative
log-likelihood of Eq. B7 (p19). The offset that turns the NLL into the plotted
value is not stated (LOG.md, R1), so every run reports:

- ``nll_bits_per_site``: -(1 / (N n)) sum_x log2 p(x), the unambiguous metric;
- ``empirical_entropy_bits_per_site``: entropy of the split's empirical
  distribution divided by n;
- ``kl_to_empirical_bits_per_site``: their difference, the KL divergence from
  the split's empirical distribution to the model, one candidate reading of
  the plotted quantity (calibrated in Phase 3).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy import stats


def empirical_entropy_bits_per_site(sequences: np.ndarray) -> float:
    """Return H(empirical distribution of rows) / n in bits."""
    counts = np.array(
        list(Counter(map(tuple, sequences.tolist())).values()), dtype=np.float64
    )
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum() / sequences.shape[1])


def sequence_metrics(log_probs_nats: np.ndarray, sequences: np.ndarray) -> dict:
    """Return per-site NLL, empirical entropy and KL-to-empirical in bits.

    Parameters
    ----------
    log_probs_nats : numpy.ndarray
        Natural-log model probabilities of each row, shape ``(N,)``.
    sequences : numpy.ndarray
        The evaluated rows, shape ``(N, n)``.
    """
    length = sequences.shape[1]
    nll = float(-log_probs_nats.mean() / (length * math.log(2.0)))
    entropy = empirical_entropy_bits_per_site(sequences)
    return {
        "num_sequences": int(sequences.shape[0]),
        "nll_bits_per_site": nll,
        "empirical_entropy_bits_per_site": entropy,
        "kl_to_empirical_bits_per_site": nll - entropy,
    }


def likelihood_ratio_threshold_bits(
    num_train: int, extra_parameters: int, sigma: float = 5.0
) -> float:
    """Return the training-NLL gap (bits per sequence) needed to reject the null.

    Wilks' theorem with ``extra_parameters`` degrees of freedom: the null (the
    larger model reduces to the smaller one) is rejected at a one-sided
    Gaussian ``sigma`` level when 2 N ln(2) * gap_bits exceeds the chi-square
    quantile. This is the planned reading of Fig. 8 (LOG.md, R7), not a
    procedure stated in the paper.
    """
    tail = stats.norm.sf(sigma)
    statistic = stats.chi2.isf(tail, extra_parameters)
    return float(statistic / (2.0 * num_train * math.log(2.0)))


def paper_kl_estimate_bits_per_site(
    nll_bits_per_site: float, num_sequences: int, sequence_length: int
) -> float:
    """Return the paper's plotted "stochastic estimate of the KL divergence".

    Phase 3 calibration of R1 (LOG.md): the plotted value is
    ``NLL - log2(N) / n``, the KL divergence from a uniform distribution over
    the N evaluated rows (treated as distinct) to the model, per site. It
    reproduces the paper's h = 1 points within plot precision (Promoters
    train and test, biofam test, and SPECT under the pooled split), and it is
    the only candidate that can be negative when rows repeat, as the biofam
    training curve of Fig. 7a is.

    Parameters
    ----------
    nll_bits_per_site : float
        Mean negative log-likelihood per site in bits.
    num_sequences : int
        Number of evaluated rows N.
    sequence_length : int
        Sequence length n.
    """
    return nll_bits_per_site - math.log2(num_sequences) / sequence_length


def paper_inferred_lrt_degrees_of_freedom(bond_dim: int) -> int:
    """Return the degrees of freedom that reproduce the Fig. 8 thresholds.

    Inferred in Phase 3 (LOG.md, R7): with a one-sided 5 sigma Wilks test,
    ``df = h**2 - 1`` reproduces both solid curves of Fig. 8 (p14) within 0.02
    bits for h = 2..12, with N_train = 80 (Promoters) and N_train = 200
    (SPECT). It is independent of M, unlike the free-parameter difference
    ``M h (h - 1)`` of the two models as parameterised here.
    """
    return bond_dim * bond_dim - 1
