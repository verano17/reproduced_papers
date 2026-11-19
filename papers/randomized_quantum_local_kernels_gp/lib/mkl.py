"""GP multiple-kernel learning with softmax mixture weights (Sec. IV-D, Eq. 13-15).

The objective is the exact GP log marginal likelihood of the mixture kernel
K_w = Σ_r w_r K^(r) with w = softmax(a). Its gradient with respect to the
linear weights is Eq. 15 with ∂Σ/∂w_r = K^(r):

    ∂L/∂w_r = ½ (αᵀ K^(r) α - Tr[Σ⁻¹ K^(r)]),

and the chain rule through the softmax gives ∂L/∂a_j = w_j (∂L/∂w_j - ⟨∂L/∂w, w⟩).

Two optimizers are available (``MKLSettings.optimizer``).

``"authors_protocol"`` reproduces the authors' code (the paper only says
"(stochastic) gradient ascent"):

- logits start at zero (uniform weights) and σ² at ``initial_noise_variance``;
- ``coordinate_rounds`` alternations of (A) logit ascent and (B) a σ² grid
  search;
- (A) gradient ascent in logit space with backtracking line search (step 1,
  factor 0.5, minimum 1e-6) up to ``max_iterations`` iterations, stopping
  when the relative improvement is below ``tolerance``. With the exact
  gradient the Armijo condition (c = 1e-4) is used; with the approximate
  gradient only non-decrease of the exact objective is required;
- (B) σ² is chosen from 9 log-spaced values over two decades on either side
  of the current value (current value included), maximizing the exact
  objective.

With ``noise_coupling = "authors_shared"`` the GPs of all query kernels are
fitted round by round, as in the authors' code, and step (A) of every GP uses
a single shared σ² that step (B) overwrites with each GP's selected value in
turn. From round 2 on, every GP's logits are therefore ascended at the σ² of
the last GP. Step (B) and prediction use each GP's own σ². This reproduces a
quirk of the released code (``_noise_plus_jitter`` in ``noisy_mk_learner.py``)
that shaped Table I. ``noise_coupling = "per_kernel"`` keeps each GP's own σ²
throughout.

``"lbfgs"`` is a converged reference optimizer, not in the paper: L-BFGS-B
jointly over the logits and log σ² with the exact gradient
(∂L/∂log σ² = σ² · ½(αᵀα - Tr[Σ⁻¹])), started from uniform weights and
``initial_noise_variance``.

``gradient = "lowrank_hutchinson"`` reproduces the authors' approximate
gradient: each K^(r) is replaced by its rank-g eigen-truncation Φ_r Φ_rᵀ for
the quadratic term, and Tr[Σ⁻¹ K^(r)] is estimated with S Rademacher probes
Z as (1/S) Σ_s (Σ⁻¹ z_s)ᵀ (Φ_r Φ_rᵀ z_s). The objective itself stays exact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize

from .gp import LOG_2PI, CholeskyState, factorize

logger = logging.getLogger(__name__)

GRADIENT_METHODS = ("exact", "lowrank_hutchinson")
OPTIMIZERS = ("authors_protocol", "lbfgs")
NOISE_COUPLINGS = ("per_kernel", "authors_shared")
# Bounds on σ² for L-BFGS-B, wide enough not to bind for these data
# (true σ² = 0.002, label scale about 0.2-1.2).
_LBFGS_NOISE_BOUNDS = (1e-8, 10.0)


@dataclass(frozen=True)
class MKLSettings:
    """Optimizer settings (see module docstring).

    Attributes
    ----------
    gradient : str
        "exact" or "lowrank_hutchinson".
    max_iterations : int
        Logit-ascent iterations per coordinate round (authors' protocol) or
        L-BFGS-B iterations (lbfgs).
    tolerance : float
        Relative-improvement stopping threshold (authors' protocol) or
        projected-gradient tolerance (lbfgs).
    coordinate_rounds : int
        Number of (logits, σ²) alternations (authors' protocol only).
    initial_noise_variance : float
        Starting σ².
    fit_noise_variance : bool
        Whether σ² is fitted.
    approx_rank : int or None
        Truncation rank g (required for "lowrank_hutchinson").
    hutchinson_probes : int or None
        Number of probes S (required for "lowrank_hutchinson").
    approx_seed : int or None
        Seed of the Rademacher probes (required for "lowrank_hutchinson").
    optimizer : str
        "authors_protocol" or "lbfgs". Default value is "authors_protocol".
    noise_coupling : str
        "per_kernel" or "authors_shared" (authors' protocol only). Default
        value is "per_kernel".
    """

    gradient: str
    max_iterations: int
    tolerance: float
    coordinate_rounds: int
    initial_noise_variance: float
    fit_noise_variance: bool
    approx_rank: int | None = None
    hutchinson_probes: int | None = None
    approx_seed: int | None = None
    optimizer: str = "authors_protocol"
    noise_coupling: str = "per_kernel"

    def __post_init__(self) -> None:
        if self.gradient not in GRADIENT_METHODS:
            raise ValueError(
                f"Unknown gradient {self.gradient!r}; expected one of {GRADIENT_METHODS}."
            )
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer {self.optimizer!r}; expected one of {OPTIMIZERS}."
            )
        if self.noise_coupling not in NOISE_COUPLINGS:
            raise ValueError(
                f"Unknown noise_coupling {self.noise_coupling!r}; expected one of {NOISE_COUPLINGS}."
            )
        if self.gradient == "lowrank_hutchinson" and None in (
            self.approx_rank,
            self.hutchinson_probes,
            self.approx_seed,
        ):
            raise ValueError(
                "lowrank_hutchinson needs approx_rank, hutchinson_probes and approx_seed."
            )
        if self.optimizer == "lbfgs":
            if self.gradient != "exact":
                raise ValueError("The lbfgs optimizer uses the exact gradient only.")
            if self.noise_coupling != "per_kernel":
                raise ValueError(
                    "noise_coupling applies to the authors' protocol only."
                )


@dataclass
class MKLResult:
    """Fitted mixture for one GP.

    Attributes
    ----------
    weights : np.ndarray
        Mixture weights w on the simplex, shape (R,).
    logits : np.ndarray
        Softmax logits a, shape (R,).
    noise_variance : float
        Selected σ².
    log_marginal : float
        Exact train log marginal likelihood at the solution (with this GP's
        own σ²).
    iterations : int
        Total accepted logit-ascent or L-BFGS-B iterations.
    history : list of dict
        One entry per accepted iteration and per σ² search.
    seconds : float
        Wall-clock fitting time, set by the caller. Default value is 0.0.
    """

    weights: np.ndarray
    logits: np.ndarray
    noise_variance: float
    log_marginal: float
    iterations: int
    history: list
    seconds: float = field(default=0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = np.exp(logits - np.max(logits))
    return shifted / shifted.sum()


def mixture_gram(weights: np.ndarray, grams: np.ndarray) -> np.ndarray:
    """K_w = Σ_r w_r K^(r) for grams of shape (R, M, M')."""
    return np.tensordot(weights, grams, axes=(0, 0))


class _LowRankHutchinson:
    """Precomputed rank-g factors and probe products for the approximate gradient."""

    def __init__(self, grams: np.ndarray, rank: int, probes: int, seed: int) -> None:
        num_kernels, size, _ = grams.shape
        rank = max(1, min(rank, size))
        self.probes = np.random.default_rng(seed).choice(
            [-1.0, 1.0], size=(size, probes)
        )
        self.factors = np.empty((num_kernels, size, rank))
        self.factor_probe_products = np.empty((num_kernels, size, probes))
        for r in range(num_kernels):
            eigenvalues, eigenvectors = np.linalg.eigh(grams[r])
            order = np.argsort(eigenvalues)[::-1][:rank]
            factor = eigenvectors[:, order] * np.sqrt(
                np.clip(eigenvalues[order], 0.0, None)
            )
            self.factors[r] = factor
            self.factor_probe_products[r] = factor @ (factor.T @ self.probes)

    def gradient(self, state: CholeskyState) -> np.ndarray:
        quadratic = np.sum(
            np.einsum("rij,i->rj", self.factors, state.alpha) ** 2, axis=1
        )
        solved_probes = cho_solve((state.lower, True), self.probes)
        trace = (
            np.einsum("is,ris->r", solved_probes, self.factor_probe_products)
            / self.probes.shape[1]
        )
        return 0.5 * (quadratic - trace)


def _exact_gradient(grams: np.ndarray, state: CholeskyState) -> np.ndarray:
    inverse = cho_solve((state.lower, True), np.eye(state.lower.shape[0]))
    quadratic = np.einsum("i,rij,j->r", state.alpha, grams, state.alpha)
    trace = np.einsum("ij,rij->r", inverse, grams)
    return 0.5 * (quadratic - trace)


class _GPObjective:
    """Log marginal likelihood and weight gradient of one GP's mixture."""

    def __init__(self, grams: np.ndarray, targets: np.ndarray, settings: MKLSettings):
        self.grams = grams
        self.targets = targets
        self.approximation = None
        if settings.gradient == "lowrank_hutchinson":
            self.approximation = _LowRankHutchinson(
                grams,
                settings.approx_rank,
                settings.hutchinson_probes,
                settings.approx_seed,
            )

    def evaluate(
        self, weights: np.ndarray, noise: float
    ) -> tuple[CholeskyState, np.ndarray]:
        state = factorize(mixture_gram(weights, self.grams), noise, self.targets)
        if self.approximation is None:
            return state, _exact_gradient(self.grams, state)
        return state, self.approximation.gradient(state)


@dataclass
class _ProtocolState:
    """Mutable optimizer state of one GP under the authors' protocol."""

    logits: np.ndarray
    noise: float
    iterations: int = 0
    history: list = field(default_factory=list)


def _ascend_logits(
    objective: _GPObjective,
    track: _ProtocolState,
    ascent_noise: float,
    settings: MKLSettings,
    label: str,
    round_index: int,
) -> None:
    """Step (A): backtracking gradient ascent on the logits at ``ascent_noise``."""
    logits = track.logits
    weights = softmax(logits)
    state, gradient_w = objective.evaluate(weights, ascent_noise)
    constant = 0.5 * objective.targets.shape[0] * LOG_2PI
    for iteration in range(1, settings.max_iterations + 1):
        previous = state.log_marginal
        gradient_a = weights * (gradient_w - float(gradient_w @ weights))
        squared_norm = float(gradient_a @ gradient_a)
        if squared_norm < 1e-16:
            logger.info(
                "MKL_STOP | kernel=%s | round=%d | reason=small_gradient",
                label,
                round_index,
            )
            break
        step = 1.0
        accepted = False
        while step >= 1e-6:
            trial_logits = logits + step * gradient_a
            trial_weights = softmax(trial_logits)
            trial_state, trial_gradient_w = objective.evaluate(
                trial_weights, ascent_noise
            )
            if objective.approximation is None:
                accept = (
                    trial_state.log_marginal >= previous + 1e-4 * step * squared_norm
                )
            else:
                accept = trial_state.log_marginal >= previous - 1e-10
            if accept:
                logits, weights = trial_logits, trial_weights
                state, gradient_w = trial_state, trial_gradient_w
                accepted = True
                break
            step *= 0.5
        if not accepted:
            logger.info(
                "MKL_STOP | kernel=%s | round=%d | reason=no_improvement",
                label,
                round_index,
            )
            break
        track.iterations += 1
        # The authors' stopping rule divides by |L| computed without the
        # -(M/2) log 2π constant; keep that convention so the tolerance stops
        # at the same point.
        relative = abs(state.log_marginal - previous) / (
            abs(previous + constant) + 1e-12
        )
        track.history.append(
            {
                "round": round_index,
                "iteration": iteration,
                "log_marginal": state.log_marginal,
                "step": step,
                "noise_variance": ascent_noise,
            }
        )
        logger.info(
            "TRAIN_ITERATION_COMPLETED | kernel=%s | round=%d/%d | iteration=%d/%d"
            " | log_marginal=%.6f | rel_improvement=%.3e | step=%.3g | noise_variance=%.3e",
            label,
            round_index,
            settings.coordinate_rounds,
            iteration,
            settings.max_iterations,
            state.log_marginal,
            relative,
            step,
            ascent_noise,
        )
        if relative < settings.tolerance:
            break
    track.logits = logits


def _search_noise(
    objective: _GPObjective,
    track: _ProtocolState,
    settings: MKLSettings,
    label: str,
    round_index: int,
) -> None:
    """Step (B): 9-point log grid over two decades around the GP's own σ²."""
    weights = softmax(track.logits)
    exponent = np.log10(track.noise)
    grid = np.unique(
        np.concatenate(([track.noise], np.logspace(exponent - 2.0, exponent + 2.0, 9)))
    )
    gram = mixture_gram(weights, objective.grams)
    best_noise, best_value = track.noise, -np.inf
    for candidate in grid:
        value = factorize(gram, float(candidate), objective.targets).log_marginal
        if value > best_value:
            best_noise, best_value = float(candidate), value
    track.noise = best_noise
    track.history.append(
        {
            "round": round_index,
            "noise_search": True,
            "noise_variance": best_noise,
            "log_marginal": best_value,
        }
    )
    logger.info(
        "TRAIN_NOISE_SEARCH_COMPLETED | kernel=%s | round=%d/%d | noise_variance=%.3e"
        " | log_marginal=%.6f",
        label,
        round_index,
        settings.coordinate_rounds,
        best_noise,
        best_value,
    )


def _result_from_track(objective: _GPObjective, track: _ProtocolState) -> MKLResult:
    weights = softmax(track.logits)
    state = factorize(
        mixture_gram(weights, objective.grams), track.noise, objective.targets
    )
    return MKLResult(
        weights=weights,
        logits=track.logits,
        noise_variance=track.noise,
        log_marginal=state.log_marginal,
        iterations=track.iterations,
        history=track.history,
    )


def _fit_lbfgs(objective: _GPObjective, settings: MKLSettings, label: str) -> MKLResult:
    num_kernels = objective.grams.shape[0]
    size = objective.targets.shape[0]
    history: list[dict] = []

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, float]:
        if settings.fit_noise_variance:
            return parameters[:num_kernels], float(np.exp(parameters[-1]))
        return parameters, float(settings.initial_noise_variance)

    def negative_objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits, noise = unpack(parameters)
        weights = softmax(logits)
        state, gradient_w = objective.evaluate(weights, noise)
        gradient_a = weights * (gradient_w - float(gradient_w @ weights))
        if not settings.fit_noise_variance:
            return -state.log_marginal, -gradient_a
        inverse_trace = float(np.trace(cho_solve((state.lower, True), np.eye(size))))
        gradient_log_noise = (
            noise * 0.5 * (float(state.alpha @ state.alpha) - inverse_trace)
        )
        return -state.log_marginal, -np.concatenate((gradient_a, [gradient_log_noise]))

    def record(intermediate_result) -> None:
        iteration = len(history) + 1
        _, noise = unpack(intermediate_result.x)
        history.append(
            {
                "iteration": iteration,
                "log_marginal": -float(intermediate_result.fun),
                "noise_variance": noise,
            }
        )
        logger.info(
            "TRAIN_ITERATION_COMPLETED | kernel=%s | optimizer=lbfgs | iteration=%d/%d"
            " | log_marginal=%.6f | noise_variance=%.3e",
            label,
            iteration,
            settings.max_iterations,
            -float(intermediate_result.fun),
            noise,
        )

    start = np.zeros(num_kernels)
    bounds = [(None, None)] * num_kernels
    if settings.fit_noise_variance:
        start = np.concatenate((start, [np.log(settings.initial_noise_variance)]))
        bounds.append(tuple(np.log(_LBFGS_NOISE_BOUNDS)))
    solution = minimize(
        negative_objective,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        callback=record,
        options={
            "maxiter": settings.max_iterations,
            "gtol": settings.tolerance,
            "ftol": 1e-12,
        },
    )
    logger.info(
        "MKL_STOP | kernel=%s | optimizer=lbfgs | success=%s | message=%s | iterations=%d",
        label,
        solution.success,
        solution.message,
        solution.nit,
    )
    logits, noise = unpack(solution.x)
    weights = softmax(logits)
    state = factorize(mixture_gram(weights, objective.grams), noise, objective.targets)
    return MKLResult(
        weights=weights,
        logits=logits,
        noise_variance=noise,
        log_marginal=state.log_marginal,
        iterations=int(solution.nit),
        history=history,
    )


def fit_mixture(
    grams: np.ndarray,
    targets: np.ndarray,
    settings: MKLSettings,
    label: str,
) -> MKLResult:
    """Maximize one GP's log marginal likelihood over softmax weights and σ².

    Parameters
    ----------
    grams : np.ndarray
        Training Gram matrices K^(r), shape (R, M_tr, M_tr).
    targets : np.ndarray
        Training targets, shape (M_tr,).
    settings : MKLSettings
        Optimizer settings; ``noise_coupling`` must be "per_kernel" (use
        ``fit_mixtures`` for the authors' shared-σ² coupling).
    label : str
        Name used in log records (e.g. the query kernel).

    Returns
    -------
    MKLResult
        Fitted weights, σ² and optimization history.
    """
    if settings.noise_coupling != "per_kernel":
        raise ValueError("fit_mixture fits one GP; use fit_mixtures for shared σ².")
    objective = _GPObjective(grams, targets, settings)
    if settings.optimizer == "lbfgs":
        return _fit_lbfgs(objective, settings, label)
    track = _ProtocolState(
        logits=np.zeros(grams.shape[0]), noise=float(settings.initial_noise_variance)
    )
    for round_index in range(1, settings.coordinate_rounds + 1):
        _ascend_logits(objective, track, track.noise, settings, label, round_index)
        if settings.fit_noise_variance:
            _search_noise(objective, track, settings, label, round_index)
    return _result_from_track(objective, track)


def fit_mixtures(
    grams: np.ndarray,
    targets_by_kernel: list[np.ndarray],
    settings: MKLSettings,
    labels: list[str],
) -> list[MKLResult]:
    """Fit the mixtures of all query-kernel GPs that share one dictionary.

    With ``noise_coupling = "per_kernel"`` this is ``fit_mixture`` per GP.
    With ``"authors_shared"`` the GPs are fitted round by round and step (A)
    uses a shared σ² that each GP's step (B) overwrites (see module
    docstring).

    Parameters
    ----------
    grams : np.ndarray
        Training Gram matrices, shape (R, M_tr, M_tr).
    targets_by_kernel : list of np.ndarray
        Training targets of each GP.
    settings : MKLSettings
        Optimizer settings.
    labels : list of str
        One name per GP, for log records.

    Returns
    -------
    list of MKLResult
        One result per GP, in input order.
    """
    if settings.noise_coupling == "per_kernel":
        per_kernel = MKLSettings(
            **{**settings.__dict__, "noise_coupling": "per_kernel"}
        )
        return [
            fit_mixture(grams, targets, per_kernel, label)
            for targets, label in zip(targets_by_kernel, labels)
        ]
    if settings.optimizer != "authors_protocol":
        raise ValueError("authors_shared coupling requires the authors' protocol.")
    objectives = [
        _GPObjective(grams, targets, settings) for targets in targets_by_kernel
    ]
    tracks = [
        _ProtocolState(
            logits=np.zeros(grams.shape[0]),
            noise=float(settings.initial_noise_variance),
        )
        for _ in targets_by_kernel
    ]
    shared_noise = float(settings.initial_noise_variance)
    for round_index in range(1, settings.coordinate_rounds + 1):
        for objective, track, label in zip(objectives, tracks, labels):
            _ascend_logits(objective, track, shared_noise, settings, label, round_index)
        if settings.fit_noise_variance:
            for objective, track, label in zip(objectives, tracks, labels):
                _search_noise(objective, track, settings, label, round_index)
                shared_noise = track.noise
    return [
        _result_from_track(objective, track)
        for objective, track in zip(objectives, tracks)
    ]
