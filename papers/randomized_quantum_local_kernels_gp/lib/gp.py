"""Exact Gaussian-process regression with a fixed kernel (Sec. IV-D, Eq. 14, 22-24)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class CholeskyState:
    """Factorization of Σ_tr = K_tr + σ² I and the weight vector α = Σ_tr⁻¹ y.

    Attributes
    ----------
    lower : np.ndarray
        Lower Cholesky factor of Σ_tr.
    alpha : np.ndarray
        Σ_tr⁻¹ y_tr.
    log_marginal : float
        Exact log marginal likelihood of Eq. 14, including the
        -(M_tr/2) log 2π constant.
    """

    lower: np.ndarray
    alpha: np.ndarray
    log_marginal: float


def factorize(
    train_gram: np.ndarray, noise_variance: float, targets: np.ndarray
) -> CholeskyState:
    """Cholesky-factorize Σ_tr and evaluate the log marginal likelihood (Eq. 14).

    Parameters
    ----------
    train_gram : np.ndarray
        Noise-free training Gram matrix K_tr, shape (M_tr, M_tr).
    noise_variance : float
        Observation noise σ² > 0.
    targets : np.ndarray
        Training targets y_tr, shape (M_tr,).

    Returns
    -------
    CholeskyState
        Factor, α and log marginal likelihood.
    """
    if noise_variance <= 0.0:
        raise ValueError(f"noise_variance must be positive, got {noise_variance}.")
    size = train_gram.shape[0]
    lower = np.linalg.cholesky(train_gram + noise_variance * np.eye(size))
    alpha = cho_solve((lower, True), targets)
    log_marginal = (
        -0.5 * float(targets @ alpha)
        - float(np.sum(np.log(np.diag(lower))))
        - 0.5 * size * LOG_2PI
    )
    return CholeskyState(lower=lower, alpha=alpha, log_marginal=log_marginal)


def predict(
    state: CholeskyState,
    cross_gram: np.ndarray,
    test_gram_diagonal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and latent variance at test inputs (Eq. 23).

    Parameters
    ----------
    state : CholeskyState
        Factorization from ``factorize``.
    cross_gram : np.ndarray
        K_ts,tr, shape (M_ts, M_tr).
    test_gram_diagonal : np.ndarray
        diag(K_ts), shape (M_ts,).

    Returns
    -------
    mean : np.ndarray
        Posterior mean, shape (M_ts,).
    latent_variance : np.ndarray
        Posterior variance of the latent function (without σ²), shape (M_ts,).
    """
    mean = cross_gram @ state.alpha
    projected = solve_triangular(state.lower, cross_gram.T, lower=True)
    latent_variance = test_gram_diagonal - np.sum(projected**2, axis=0)
    return mean, np.maximum(latent_variance, 0.0)


def predictive_metrics(
    targets: np.ndarray,
    mean: np.ndarray,
    latent_variance: np.ndarray,
    noise_variance: float,
) -> dict[str, float]:
    """Test metrics for noisy targets under N(mean, latent_variance + σ²).

    The authors' notebook reports the same quantities (RMSE, MAE, average
    NLPD, standardized-residual mean and variance, ±1σ/±2σ coverage).

    Returns
    -------
    dict
        ``rmse``, ``mae``, ``avg_nlpd``, ``z_mean``, ``z_var``,
        ``coverage_1sigma``, ``coverage_2sigma``.
    """
    variance = latent_variance + noise_variance
    residual = targets - mean
    z_scores = residual / np.sqrt(variance)
    log_density = -0.5 * (residual**2 / variance + np.log(variance) + LOG_2PI)
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "avg_nlpd": float(-np.mean(log_density)),
        "z_mean": float(np.mean(z_scores)),
        "z_var": float(np.var(z_scores)),
        "coverage_1sigma": float(np.mean(np.abs(z_scores) <= 1.0)),
        "coverage_2sigma": float(np.mean(np.abs(z_scores) <= 2.0)),
    }
