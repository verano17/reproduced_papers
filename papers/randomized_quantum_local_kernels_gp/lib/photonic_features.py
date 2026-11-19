"""MerLin photonic adaptation of the randomized local feature generator.

This is an ADAPTATION, not part of the paper. It replaces the paper's n-qubit
ring circuits (Sec. IV-A) with linear-optical circuits that keep the same
scientific role: a random dictionary of shallow, geometrically local, data
re-uploading circuits whose few-body local observables are the features.

Circuit for one draw r on m modes with n photons (L layers):

    for each layer ℓ:
        PS(δ_ℓq · ξ_ℓᵀ x) on every mode q        (data re-uploading, as RZ(δγ))
        PS(α_ℓq) on every mode q                  (fixed random phases)
        BS(θ) on pairs (0,1), (2,3), ...          (nearest-neighbour mixing,
        BS(θ) on pairs (1,2), (3,4), ...           brickwork on an open chain)
        PS(ω_ℓq) on every mode q

All non-data parameters are fixed per draw (no training), sampled from a
scrambled Sobol sequence: δ → (−1, 1); α, θ, ω → (−π, π); ξ_ℓ = sign · π·u
for scalar inputs, as in the qubit sampler.

Measurement: click patterns (threshold detectors, MerLin
``ComputationSpace.UNBUNCHED`` = Fock distribution postselected on n
distinct clicks) or photon numbers (``"pnr"``, ``ComputationSpace.FOCK``).
Features are local parity expectations, the photonic analogue of Z and ZZ:

    ⟨Π_q⟩ and ⟨Π_q Π_(q+1)⟩,   Π_q = (−1)^(click_q)  or  (−1)^(n_q) for PNR,

giving m + (m − 1) features. All are estimated from the same detection
events, so no randomized-basis (shadow) protocol is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import merlin as ml
import numpy as np
import perceval as pcvl
import torch
from scipy.stats import qmc

DETECTORS = ("threshold", "pnr")
# "none": the first data layer acts directly on the Fock input state, where
# phases are only a global phase, so layer 0's encoding has no effect
# (Phase 4 version, found in independent review 2). "balanced_brickwork":
# a fixed 50:50 beam-splitter brickwork (even then odd pairs) before the first
# data layer spreads every photon over neighbouring modes, the photonic
# analogue of preparing |+>^n, so all L data layers act.
INITIAL_MIXING = ("none", "balanced_brickwork")
_SPACE = {"threshold": ml.ComputationSpace.UNBUNCHED, "pnr": ml.ComputationSpace.FOCK}


@dataclass(frozen=True)
class PhotonicArchitecture:
    """Shape of the photonic local circuits.

    Parameters
    ----------
    num_modes : int
        Number of optical modes m (open chain).
    num_photons : int
        Number of photons n (at least 2; a single photon is a linear-optics
        baseline, not a quantum adaptation).
    num_layers : int
        Number of re-uploading layers L.
    input_dim : int
        Input dimension ζ (1 for this paper's tasks).
    """

    num_modes: int
    num_photons: int
    num_layers: int
    input_dim: int
    initial_mixing: str = "none"

    def __post_init__(self) -> None:
        if self.initial_mixing not in INITIAL_MIXING:
            raise ValueError(
                f"initial_mixing must be one of {INITIAL_MIXING}, got {self.initial_mixing!r}."
            )
        if self.num_modes < 3:
            raise ValueError(f"num_modes must be >= 3, got {self.num_modes}.")
        if not 2 <= self.num_photons <= self.num_modes:
            raise ValueError(
                f"num_photons must be in [2, num_modes], got {self.num_photons}."
            )
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}.")
        if self.input_dim != 1:
            raise ValueError(
                "The photonic adaptation implements scalar inputs (input_dim = 1)."
            )

    @property
    def mixing_pairs(self) -> tuple[tuple[int, int], ...]:
        """Beam-splitter pairs in application order: even pairs, then odd pairs."""
        even = tuple((q, q + 1) for q in range(0, self.num_modes - 1, 2))
        odd = tuple((q, q + 1) for q in range(1, self.num_modes - 1, 2))
        return even + odd

    @property
    def input_state(self) -> list[int]:
        """Photons spread evenly along the chain, e.g. modes 0, 3, 6, 9 for m=12, n=4.

        Spreading the photons puts every mode inside some photon's light cone
        after a few brickwork layers (cookbook principle 1).
        """
        occupied = {
            int(round(i * self.num_modes / self.num_photons))
            for i in range(self.num_photons)
        }
        if len(occupied) != self.num_photons:
            raise ValueError("Could not place photons on distinct modes.")
        return [1 if q in occupied else 0 for q in range(self.num_modes)]

    @property
    def num_features(self) -> int:
        return 2 * self.num_modes - 1

    @property
    def sobol_dimension(self) -> int:
        """Per layer: ξ sign and radius (2), δ (m), α (m), θ (m − 1), ω (m)."""
        return self.num_layers * (2 + 4 * self.num_modes - 1)


@dataclass(frozen=True)
class PhotonicParameters:
    """One fixed parameter draw.

    Attributes
    ----------
    xi : np.ndarray
        Projection scalars ξ_ℓ, shape (L,).
    delta, alpha, omega : np.ndarray
        Per-mode values, shape (L, m).
    theta : np.ndarray
        Beam-splitter angles in ``mixing_pairs`` order, shape (L, m − 1).
    """

    xi: np.ndarray
    delta: np.ndarray
    alpha: np.ndarray
    theta: np.ndarray
    omega: np.ndarray


def map_photonic_sobol_point(
    arch: PhotonicArchitecture, point: np.ndarray
) -> PhotonicParameters:
    """Map one Sobol point u ∈ [0, 1)^d to photonic parameters.

    Layout: for each layer, [ξ sign, ξ radius, δ (m), α (m), θ (m − 1), ω (m)].
    """
    point = np.asarray(point, dtype=float)
    if point.shape != (arch.sobol_dimension,):
        raise ValueError(
            f"Sobol point has shape {point.shape}, expected ({arch.sobol_dimension},)."
        )
    m, L = arch.num_modes, arch.num_layers
    per_layer = point.reshape(L, 2 + 4 * m - 1)
    sign = np.where(2.0 * per_layer[:, 0] - 1.0 >= 0.0, 1.0, -1.0)
    xi = sign * np.pi * per_layer[:, 1]
    cursor = 2
    blocks = {}
    for name, size in (("delta", m), ("alpha", m), ("theta", m - 1), ("omega", m)):
        values = per_layer[:, cursor : cursor + size]
        cursor += size
        blocks[name] = (
            2.0 * values - 1.0 if name == "delta" else (2.0 * values - 1.0) * np.pi
        )
    return PhotonicParameters(xi=xi, **blocks)


def sample_photonic_draws(
    arch: PhotonicArchitecture, num_draws: int, seed: int
) -> list[PhotonicParameters]:
    """R parameter draws from a scrambled Sobol sequence (prefix of 2^ceil(log2 R) points)."""
    if num_draws < 1:
        raise ValueError(f"num_draws must be >= 1, got {num_draws}.")
    engine = qmc.Sobol(d=arch.sobol_dimension, scramble=True, seed=seed)
    points = engine.random_base2(m=int(np.ceil(np.log2(num_draws))))[:num_draws]
    return [map_photonic_sobol_point(arch, point) for point in points]


def _input_name(index: int) -> str:
    # Zero padding keeps MerLin's name-sorted input order equal to column order.
    return f"in{index:04d}"


def build_circuit(
    arch: PhotonicArchitecture, params: PhotonicParameters
) -> pcvl.Circuit:
    """Perceval circuit for one draw; the data phases are named input parameters."""
    m = arch.num_modes
    circuit = pcvl.Circuit(m)
    if arch.initial_mixing == "balanced_brickwork":
        # Perceval BS(θ) has reflectivity cos²(θ/2); θ = π/2 is 50:50.
        for first, _ in arch.mixing_pairs:
            circuit.add(first, pcvl.BS(theta=float(np.pi / 2)))
    for layer in range(arch.num_layers):
        for q in range(m):
            circuit.add(q, pcvl.PS(pcvl.P(_input_name(layer * m + q))))
        for q in range(m):
            circuit.add(q, pcvl.PS(float(params.alpha[layer, q])))
        for pair_index, (first, _) in enumerate(arch.mixing_pairs):
            circuit.add(first, pcvl.BS(theta=float(params.theta[layer, pair_index])))
        for q in range(m):
            circuit.add(q, pcvl.PS(float(params.omega[layer, q])))
    return circuit


def encoding_phases(
    arch: PhotonicArchitecture, params: PhotonicParameters, inputs: np.ndarray
) -> np.ndarray:
    """Data phases δ_ℓq · ξ_ℓ x for all inputs, shape (M, L·m), columns in parameter order."""
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] != 1:
        raise ValueError(f"inputs must have shape (M, 1), got {inputs.shape}.")
    projected = inputs[:, :1] * params.xi[None, :]  # γ_ℓ(x), shape (M, L)
    return (projected[:, :, None] * params.delta[None, :, :]).reshape(
        inputs.shape[0], -1
    )


def build_layer(
    arch: PhotonicArchitecture, params: PhotonicParameters, detector: str
) -> ml.QuantumLayer:
    """MerLin layer with no trainable parameters for one draw."""
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector {detector!r}; expected one of {DETECTORS}.")
    return ml.QuantumLayer(
        circuit=build_circuit(arch, params),
        input_state=arch.input_state,
        n_photons=arch.num_photons,
        input_parameters=["in"],
        trainable_parameters=[],
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=_SPACE[detector]
        ),
        dtype=torch.float64,
    )


def outcome_probabilities(
    layer: ml.QuantumLayer, phases: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic outcome probabilities (M, K) and occupation patterns (K, m)."""
    with torch.no_grad():
        probabilities = layer(torch.as_tensor(phases, dtype=torch.float64)).numpy()
    patterns = np.asarray(layer.output_keys, dtype=int)
    return probabilities, patterns


def parity_signs(patterns: np.ndarray, detector: str) -> np.ndarray:
    """Π_q per outcome: (−1)^click for threshold, (−1)^n for PNR; shape (K, m)."""
    if detector == "threshold":
        return 1.0 - 2.0 * (patterns > 0)
    return 1.0 - 2.0 * (patterns % 2)


def local_parity_features(
    probabilities: np.ndarray, patterns: np.ndarray, detector: str
) -> np.ndarray:
    """⟨Π_q⟩ for every mode, then ⟨Π_q Π_(q+1)⟩ for neighbours; shape (M, 2m − 1).

    ``probabilities`` may be exact probabilities or empirical frequencies.
    """
    signs = parity_signs(patterns, detector)
    single = probabilities @ signs
    neighbours = probabilities @ (signs[:, :-1] * signs[:, 1:])
    return np.hstack([single, neighbours])


def sample_frequencies(
    probabilities: np.ndarray, num_shots: int, rng: np.random.Generator
) -> np.ndarray:
    """Empirical outcome frequencies from ``num_shots`` (postselected) detection events per row."""
    if num_shots < 1:
        raise ValueError(f"num_shots must be >= 1, got {num_shots}.")
    totals = probabilities.sum(axis=1, keepdims=True)
    if not np.allclose(totals, 1.0, atol=1e-9):
        raise ValueError("Outcome probabilities do not sum to 1.")
    counts = rng.multinomial(num_shots, probabilities / totals)
    return counts / num_shots


def postselection_probability(
    arch: PhotonicArchitecture, params: PhotonicParameters, phases: np.ndarray
) -> np.ndarray:
    """Probability that all n photons exit in distinct modes (threshold n-click events), shape (M,)."""
    fock_layer = build_layer(arch, params, "pnr")
    probabilities, patterns = outcome_probabilities(fock_layer, phases)
    collision_free = patterns.max(axis=1) <= 1
    return probabilities[:, collision_free].sum(axis=1)


def photonic_features(
    arch: PhotonicArchitecture,
    params: PhotonicParameters,
    inputs: np.ndarray,
    detector: str,
    num_shots: int | None,
    rng: np.random.Generator | None,
) -> dict[str, np.ndarray]:
    """Exact and (optionally) shot-sampled local features for one draw.

    Returns
    -------
    dict
        ``exact`` (M, 2m − 1), ``used`` (sampled when ``num_shots`` is set,
        else exact) and, for threshold detection, ``postselection`` (M,).
    """
    phases = encoding_phases(arch, params, inputs)
    layer = build_layer(arch, params, detector)
    probabilities, patterns = outcome_probabilities(layer, phases)
    exact = local_parity_features(probabilities, patterns, detector)
    used = exact
    if num_shots is not None:
        if rng is None:
            raise ValueError("A random generator is required for shot sampling.")
        used = local_parity_features(
            sample_frequencies(probabilities, num_shots, rng), patterns, detector
        )
    result = {"exact": exact, "used": used}
    if detector == "threshold":
        result["postselection"] = postselection_probability(arch, params, phases)
    return result
