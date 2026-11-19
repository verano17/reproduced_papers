"""Query kernels, the cosine wrapper and kernel-alignment metrics.

Query kernels (Sec. V-A, Eq. 16-19) for scalar inputs on [-1, 1]:

- ``rbf``: exp(-(x - x')² / (2ℓ²)).
- ``matern32``: (1 + √3|x - x'|/ℓ) exp(-√3|x - x'|/ℓ).
- ``nonstat_ls``: Paciorek-Schervish non-stationary SE,
  sqrt(2ℓ(x)ℓ(x') / (ℓ(x)² + ℓ(x')²)) exp(-(x - x')² / (ℓ(x)² + ℓ(x')²)),
  with ℓ(x) = ℓ_min + (ℓ_max - ℓ_min) · sigmoid(s·c·x).

The sigmoid sign s is configurable because the paper and the code disagree:
Eq. 19 uses 1/(1 + e^{cx}) (s = -1, ℓ decreasing in x) while the authors'
code uses 1/(1 + e^{-cx}) (s = +1, ℓ increasing in x).
"""

from __future__ import annotations

import numpy as np

QUERY_KERNEL_TYPES = ("rbf", "matern32", "nonstat_ls")


def _require(params: dict, key: str, kernel_type: str):
    if key not in params:
        raise KeyError(f"Query kernel {kernel_type!r} requires parameter {key!r}.")
    return params[key]


def nonstationary_lengthscale(inputs: np.ndarray, params: dict) -> np.ndarray:
    """Input-dependent lengthscale ℓ(x) of Eq. 19 for scalar inputs.

    Parameters
    ----------
    inputs : np.ndarray
        Inputs of shape (M,) or (M, 1).
    params : dict
        Keys ``ell_min``, ``ell_max``, ``c`` and ``sigmoid_sign`` (+1 or -1).

    Returns
    -------
    np.ndarray
        Lengthscales of shape (M,).
    """
    ell_min = float(_require(params, "ell_min", "nonstat_ls"))
    ell_max = float(_require(params, "ell_max", "nonstat_ls"))
    slope = float(_require(params, "c", "nonstat_ls"))
    sign = float(_require(params, "sigmoid_sign", "nonstat_ls"))
    if sign not in (1.0, -1.0):
        raise ValueError(f"sigmoid_sign must be +1 or -1, got {sign}.")
    t = np.asarray(inputs, dtype=float).reshape(-1)
    return ell_min + (ell_max - ell_min) / (1.0 + np.exp(-sign * slope * t))


def query_kernel_matrix(
    inputs_a: np.ndarray, inputs_b: np.ndarray, kernel_type: str, params: dict
) -> np.ndarray:
    """Cross Gram matrix of a query kernel for scalar inputs.

    Parameters
    ----------
    inputs_a, inputs_b : np.ndarray
        Inputs of shape (M_a, 1) and (M_b, 1).
    kernel_type : str
        One of ``QUERY_KERNEL_TYPES``.
    params : dict
        Kernel hyperparameters (``lengthscale`` for stationary kernels; see
        ``nonstationary_lengthscale`` for ``nonstat_ls``).

    Returns
    -------
    np.ndarray
        Gram matrix of shape (M_a, M_b).
    """
    a = np.asarray(inputs_a, dtype=float)
    b = np.asarray(inputs_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 1 or b.shape[1] != 1:
        raise ValueError(
            "Query kernels are implemented for scalar inputs of shape (M, 1)."
        )
    diff = a[:, 0][:, None] - b[:, 0][None, :]
    if kernel_type == "rbf":
        ell = float(_require(params, "lengthscale", kernel_type))
        return np.exp(-0.5 * diff**2 / ell**2)
    if kernel_type == "matern32":
        ell = float(_require(params, "lengthscale", kernel_type))
        scaled = np.sqrt(3.0) * np.abs(diff) / ell
        return (1.0 + scaled) * np.exp(-scaled)
    if kernel_type == "nonstat_ls":
        ell_a = nonstationary_lengthscale(a, params)[:, None]
        ell_b = nonstationary_lengthscale(b, params)[None, :]
        denom = ell_a**2 + ell_b**2
        return np.sqrt(2.0 * ell_a * ell_b / denom) * np.exp(-(diff**2) / denom)
    raise ValueError(
        f"Unknown query kernel {kernel_type!r}; expected one of {QUERY_KERNEL_TYPES}."
    )


def cosine_wrapper_gram(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """Normalized cosine wrapper of Eq. 21, κ = ½(1 + cos∠(φ(x), φ(x'))) ∈ [0, 1].

    Parameters
    ----------
    features_a, features_b : np.ndarray
        Feature matrices of shape (..., M_a, m) and (..., M_b, m).

    Returns
    -------
    np.ndarray
        Gram matrices of shape (..., M_a, M_b).
    """
    eps = 1e-12
    unit_a = features_a / (np.linalg.norm(features_a, axis=-1, keepdims=True) + eps)
    unit_b = features_b / (np.linalg.norm(features_b, axis=-1, keepdims=True) + eps)
    return 0.5 * (1.0 + unit_a @ np.swapaxes(unit_b, -1, -2))


def center_gram(gram: np.ndarray) -> np.ndarray:
    """Double-center a square Gram matrix, H K H with H = I - 11ᵀ/n."""
    row_means = gram.mean(axis=0, keepdims=True)
    col_means = gram.mean(axis=1, keepdims=True)
    return gram - row_means - col_means + gram.mean()


def kernel_alignment(gram_a: np.ndarray, gram_b: np.ndarray, centered: bool) -> float:
    """Frobenius cosine between two Gram matrices.

    ``centered=False`` is Eq. 25 of the paper. ``centered=True`` is the
    centered alignment that the authors' code computes for Table I.

    Parameters
    ----------
    gram_a, gram_b : np.ndarray
        Square Gram matrices of equal shape.
    centered : bool
        Whether to double-center both matrices first.

    Returns
    -------
    float
        Alignment value.
    """
    if gram_a.shape != gram_b.shape:
        raise ValueError(f"Shape mismatch: {gram_a.shape} vs {gram_b.shape}.")
    if centered:
        gram_a, gram_b = center_gram(gram_a), center_gram(gram_b)
    numerator = float(np.sum(gram_a * gram_b))
    denominator = float(np.linalg.norm(gram_a) * np.linalg.norm(gram_b))
    if denominator == 0.0:
        raise ValueError("Alignment is undefined for a zero (centered) Gram matrix.")
    return numerator / denominator
