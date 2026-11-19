"""Test-split evaluation of a fitted kernel mixture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gp import factorize, predict, predictive_metrics
from .kernels import kernel_alignment
from .mkl import mixture_gram


@dataclass
class MixtureEvaluation:
    """Test predictions and metrics of one mixture on one query-kernel GP.

    Attributes
    ----------
    metrics : dict
        Alignments (centered, uncentered), predictive metrics and train log
        marginal likelihood.
    mean : np.ndarray
        Posterior mean at test inputs.
    predictive_std : np.ndarray
        Standard deviation of the noisy test targets.
    test_gram : np.ndarray
        Learned (noise-free) test Gram matrix K_ts.
    """

    metrics: dict
    mean: np.ndarray
    predictive_std: np.ndarray
    test_gram: np.ndarray


def evaluate_mixture(
    grams: np.ndarray,
    weights: np.ndarray,
    noise_variance: float,
    train_index: np.ndarray,
    test_index: np.ndarray,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    true_test_gram: np.ndarray,
) -> MixtureEvaluation:
    """Evaluate a mixture K = Σ_r w_r K^(r) on the test split.

    Parameters
    ----------
    grams : np.ndarray
        Base Gram matrices on all M inputs, shape (R, M, M).
    weights : np.ndarray
        Mixture weights, shape (R,).
    noise_variance : float
        Fitted σ².
    train_index, test_index : np.ndarray
        Split indices into the M inputs.
    train_targets, test_targets : np.ndarray
        Noisy targets of the two splits.
    true_test_gram : np.ndarray
        Query-kernel Gram matrix K*_ts on the test inputs.

    Returns
    -------
    MixtureEvaluation
        Metrics and predictions.
    """
    full_gram = mixture_gram(weights, grams)
    train_gram = full_gram[np.ix_(train_index, train_index)]
    cross_gram = full_gram[np.ix_(test_index, train_index)]
    test_gram = full_gram[np.ix_(test_index, test_index)]
    state = factorize(train_gram, noise_variance, train_targets)
    mean, latent_variance = predict(state, cross_gram, np.diag(test_gram))
    metrics = predictive_metrics(test_targets, mean, latent_variance, noise_variance)
    metrics.update(
        {
            "alignment_centered": kernel_alignment(
                test_gram, true_test_gram, centered=True
            ),
            "alignment_uncentered": kernel_alignment(
                test_gram, true_test_gram, centered=False
            ),
            "train_log_marginal": state.log_marginal,
            "noise_variance": float(noise_variance),
            "effective_num_kernels": float(1.0 / np.sum(np.asarray(weights) ** 2)),
            "max_weight": float(np.max(weights)),
        }
    )
    return MixtureEvaluation(
        metrics=metrics,
        mean=mean,
        predictive_std=np.sqrt(latent_variance + noise_variance),
        test_gram=test_gram,
    )
