"""Classical fair baselines (Phase 5); not part of the paper.

B3, matched random-Fourier-feature dictionary: R base kernels, each from
m = 2K features φ_r(x) = [cos ω_rj x, sin ω_rj x]_j / √K with a per-kernel
lengthscale ℓ_r (log-uniform on a fixed range) and Gaussian frequencies
ω_rj = Φ⁻¹(u_j)/ℓ_r drawn from a scrambled Sobol point. Wrapped by the same
cosine kernel (Eq. 21) and fitted by the same MKL as the quantum dictionary.

B4, multi-lengthscale RBF dictionary: R exact RBF kernels with lengthscales
log-spaced on a fixed range, fitted by the same MKL.

B1/B2, parametric GPs: a kernel family (the query kernel's own family for B1,
RBF for B2) times an amplitude, with all hyperparameters and σ² fitted by
ML-II (L-BFGS-B on the exact log marginal likelihood, log-space parameters,
several log-uniform restarts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc

from .gp import factorize
from .kernels import query_kernel_matrix

logger = logging.getLogger(__name__)

PARAMETRIC_FAMILIES = ("rbf", "matern32", "nonstat_ls")


def _log_uniform(unit: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.exp(np.log(low) + unit * (np.log(high) - np.log(low)))


def rff_dictionary_features(
    inputs: np.ndarray,
    num_kernels: int,
    num_features: int,
    sobol_seed: int,
    lengthscale_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """B3 features for every base kernel.

    Parameters
    ----------
    inputs : np.ndarray
        Inputs of shape (M, 1).
    num_kernels : int
        Dictionary size R.
    num_features : int
        Features per base kernel m (even; K = m/2 frequencies).
    sobol_seed : int
        Scrambling seed of the Sobol design (one point per base kernel).
    lengthscale_range : tuple of float
        (low, high) of the log-uniform per-kernel lengthscale.

    Returns
    -------
    features : np.ndarray
        Shape (R, M, m).
    lengthscales : np.ndarray
        Shape (R,).
    """
    if num_features % 2 != 0:
        raise ValueError(
            f"num_features must be even (cos/sin pairs), got {num_features}."
        )
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] != 1:
        raise ValueError(f"inputs must have shape (M, 1), got {inputs.shape}.")
    frequencies_per_kernel = num_features // 2
    engine = qmc.Sobol(d=1 + frequencies_per_kernel, scramble=True, seed=sobol_seed)
    points = engine.random_base2(m=int(np.ceil(np.log2(num_kernels))))[:num_kernels]
    lengthscales = _log_uniform(points[:, 0], *lengthscale_range)
    # Keep the inverse normal finite at the (measure-zero) Sobol boundary.
    standard = norm.ppf(np.clip(points[:, 1:], 1e-12, 1.0 - 1e-12))
    frequencies = standard / lengthscales[:, None]  # (R, K)
    phases = frequencies[:, None, :] * inputs[None, :, :1]  # (R, M, K)
    features = np.concatenate([np.cos(phases), np.sin(phases)], axis=-1)
    return features / np.sqrt(frequencies_per_kernel), lengthscales


def rbf_dictionary_grams(
    inputs: np.ndarray, num_kernels: int, lengthscale_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """B4 Gram matrices (R, M, M) for lengthscales log-spaced over ``lengthscale_range``."""
    lengthscales = np.exp(
        np.linspace(
            np.log(lengthscale_range[0]), np.log(lengthscale_range[1]), num_kernels
        )
    )
    grams = np.stack(
        [
            query_kernel_matrix(inputs, inputs, "rbf", {"lengthscale": ell})
            for ell in lengthscales
        ]
    )
    return grams, lengthscales


@dataclass
class ParametricFit:
    """ML-II fit of one parametric GP.

    Attributes
    ----------
    family : str
        Kernel family.
    params : dict
        Fitted kernel hyperparameters (family parameters and ``amplitude``).
    noise_variance : float
        Fitted σ².
    log_marginal : float
        Train log marginal likelihood at the best restart.
    restarts : list of dict
        Final log marginal likelihood and success flag per restart.
    """

    family: str
    params: dict
    noise_variance: float
    log_marginal: float
    restarts: list


def _unpack(family: str, theta: np.ndarray) -> tuple[dict, float, float]:
    values = np.exp(theta)
    if family in ("rbf", "matern32"):
        return {"lengthscale": values[0]}, values[1], values[2]
    if family == "nonstat_ls":
        params = {
            "ell_min": values[0],
            "ell_max": values[0] + values[1],
            "c": values[2],
            "sigmoid_sign": 1,
        }
        return params, values[3], values[4]
    raise ValueError(
        f"Unknown family {family!r}; expected one of {PARAMETRIC_FAMILIES}."
    )


def parametric_gram(
    family: str, params: dict, inputs_a: np.ndarray, inputs_b: np.ndarray
) -> np.ndarray:
    """Amplitude × family kernel; ``params`` holds the family parameters and ``amplitude``."""
    family_params = {k: v for k, v in params.items() if k != "amplitude"}
    return params["amplitude"] * query_kernel_matrix(
        inputs_a, inputs_b, family, family_params
    )


def _restart_start(family: str, rng: np.random.Generator) -> np.ndarray:
    def draw(low, high):
        return float(np.log(_log_uniform(rng.uniform(), low, high)))

    amplitude, noise = draw(0.1, 10.0), draw(1e-4, 1e-1)
    if family in ("rbf", "matern32"):
        return np.array([draw(0.02, 2.0), amplitude, noise])
    return np.array(
        [draw(0.02, 2.0), draw(0.02, 2.0), draw(0.1, 10.0), amplitude, noise]
    )


def fit_parametric_gp(
    family: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    restarts: int,
    seed: int,
    label: str,
) -> ParametricFit:
    """ML-II fit of amplitude × family + σ² by L-BFGS-B from log-uniform restarts.

    Parameters
    ----------
    family : str
        One of ``PARAMETRIC_FAMILIES``.
    inputs, targets : np.ndarray
        Training data, shapes (M, 1) and (M,).
    restarts : int
        Number of random restarts.
    seed : int
        Seed of the restart initializations.
    label : str
        Name used in log records.

    Returns
    -------
    ParametricFit
        Best restart by train log marginal likelihood.
    """
    if family not in PARAMETRIC_FAMILIES:
        raise ValueError(
            f"Unknown family {family!r}; expected one of {PARAMETRIC_FAMILIES}."
        )
    rng = np.random.default_rng(seed)

    def negative_log_marginal(theta: np.ndarray) -> float:
        params, amplitude, noise = _unpack(family, theta)
        gram = amplitude * query_kernel_matrix(inputs, inputs, family, params)
        try:
            return -factorize(gram, noise, targets).log_marginal
        except np.linalg.LinAlgError:
            return 1e10

    best, history = None, []
    for restart in range(1, restarts + 1):
        solution = minimize(
            negative_log_marginal,
            _restart_start(family, rng),
            method="L-BFGS-B",
            bounds=[(-12.0, 6.0)] * (3 if family in ("rbf", "matern32") else 5),
        )
        history.append(
            {
                "restart": restart,
                "log_marginal": -float(solution.fun),
                "success": bool(solution.success),
            }
        )
        logger.info(
            "TRAIN_RESTART_COMPLETED | kernel=%s | family=%s | restart=%d/%d | log_marginal=%.6f"
            " | success=%s | iterations=%d",
            label,
            family,
            restart,
            restarts,
            -float(solution.fun),
            solution.success,
            solution.nit,
        )
        if best is None or solution.fun < best.fun:
            best = solution
    params, amplitude, noise = _unpack(family, best.x)
    params = {k: float(v) if k != "sigmoid_sign" else v for k, v in params.items()}
    params["amplitude"] = float(amplitude)
    return ParametricFit(
        family=family,
        params=params,
        noise_variance=float(noise),
        log_marginal=-float(best.fun),
        restarts=history,
    )
