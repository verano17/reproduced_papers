"""MerLin adaptation (Phase 4): heralded two-photon photonic quantum HMM.

This is an **adaptation**, not the paper's model. It keeps the scientific
role of the basis-enhanced 2-gram model of Fig. 6a (p12), a coherent quantum
memory of dimension h whose hidden paths interfere with complex amplitudes
and that emits one symbol per site by measuring an ancilla, in a form that
linear optics can implement (LOG.md, Photonic Translation Assessment, M3).

Optical layout (modes 0..h-1 hidden, modes h..h+M-1 data):

- memory: one photon spread over the h hidden modes (bond dimension h);
- at every site a probe photon enters mode ``h`` (the first data mode) and
  one trainable universal MZI mesh, shared by all sites (translation
  invariance), acts on all h + M modes;
- photon-number-resolving detectors on the data modes herald a symbol when
  they count exactly one photon; the symbol m is its data mode, and the
  hidden photon stays coherent for the next site.

For the hidden basis input ``|a>`` the heralded amplitude to ``|b>`` with
symbol m is the 2 x 2 permanent ``U[b, a] U[h+m, h] + U[h+m, a] U[b, h]``;
it is read from MerLin's two-photon amplitudes, giving Kraus operators
``K[m, b, a]``. The model distribution is the globally post-selected one
(all n heralds succeed):

    q(x) = ||K_{x_n} ... K_{x_1} psi_0||^2 / Z,
    Z    = sum_x ||K_{x_n} ... K_{x_1} psi_0||^2   (herald success probability),

with ``Z`` computed exactly from the channel ``rho -> sum_m K_m rho K_m^dag``.
"""

from __future__ import annotations

import math

import merlin as ml
import torch

COMPUTATION_SPACE = ml.ComputationSpace.FOCK
NUM_PHOTONS = 2


class PhotonicQuantumHMM(torch.nn.Module):
    """Heralded two-photon photonic quantum HMM simulated with MerLin.

    Parameters
    ----------
    bond_dim : int
        Number of hidden modes h (dimension of the one-photon memory).
    alphabet_size : int
        Number of data modes M (one per symbol).
    seed : int
        Seed of the mesh and initial-memory initialisation.
    """

    def __init__(self, bond_dim: int, alphabet_size: int, seed: int) -> None:
        super().__init__()
        self.bond_dim = int(bond_dim)
        self.alphabet_size = int(alphabet_size)
        h, M = self.bond_dim, self.alphabet_size
        self.num_modes = h + M
        self.probe_mode = h
        torch.manual_seed(seed)
        builder = ml.CircuitBuilder(n_modes=self.num_modes)
        builder.add_entangling_layer(seed=seed)
        self.layer = ml.QuantumLayer(
            builder=builder,
            n_photons=NUM_PHOTONS,
            measurement_strategy=ml.MeasurementStrategy.amplitudes(
                computation_space=COMPUTATION_SPACE
            ),
            dtype=torch.float64,
        )
        generator = torch.Generator().manual_seed(seed)
        self.initial_real = torch.nn.Parameter(
            torch.randn(h, generator=generator, dtype=torch.float64)
        )
        self.initial_imag = torch.nn.Parameter(
            torch.randn(h, generator=generator, dtype=torch.float64)
        )
        index = {tuple(key): i for i, key in enumerate(self.layer.output_keys)}
        input_states = torch.zeros(h, len(index), dtype=torch.complex128)
        for hidden in range(h):
            input_states[hidden, index[self._occupation(hidden, self.probe_mode)]] = 1.0
        self.register_buffer("input_states", input_states)
        # heralded_index[m, b]: output state with the memory photon in hidden
        # mode b and the probe-heralded photon in data mode h + m.
        heralded_index = torch.tensor(
            [
                [index[self._occupation(hidden, h + symbol)] for hidden in range(h)]
                for symbol in range(M)
            ]
        )
        self.register_buffer("heralded_index", heralded_index)

    def _occupation(self, *modes: int) -> tuple[int, ...]:
        occupation = [0] * self.num_modes
        for mode in modes:
            occupation[mode] += 1
        return tuple(occupation)

    def initial_state(self) -> torch.Tensor:
        """Return the normalised initial memory amplitudes, shape ``(h,)``."""
        amplitudes = torch.complex(self.initial_real, self.initial_imag)
        return amplitudes / torch.linalg.vector_norm(amplitudes)

    def kraus_operators(self) -> torch.Tensor:
        """Return ``K[m, b, a]``, the heralded maps from hidden mode a to b for symbol m."""
        amplitudes = self.layer(self.input_states)  # (a, Fock output states)
        return amplitudes[:, self.heralded_index].permute(1, 2, 0)

    def log_herald_success(self, kraus: torch.Tensor, length: int) -> torch.Tensor:
        """Return ``ln Z``, the log probability that all ``length`` heralds succeed."""
        psi = self.initial_state()
        rho = torch.outer(psi, psi.conj())
        log_success = torch.zeros((), dtype=torch.float64, device=psi.device)
        for _ in range(length):
            rho = torch.einsum("mba,ac,mdc->bd", kraus, rho, kraus.conj())
            trace = torch.diagonal(rho).sum().real
            log_success = log_success + torch.log(trace)
            rho = rho / trace
        return log_success

    def log_prob(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return natural-log probabilities ``ln q(x)`` of a batch of sequences.

        Parameters
        ----------
        sequences : torch.Tensor
            Integer tensor of shape ``(batch, n)`` with values in ``0..M-1``.
        """
        batch, length = sequences.shape
        kraus = self.kraus_operators()
        psi = self.initial_state().expand(batch, -1)
        log_joint = torch.zeros(batch, dtype=torch.float64, device=sequences.device)
        for site in range(length):
            psi = torch.einsum("zba,za->zb", kraus[sequences[:, site]], psi)
            norm_squared = psi.abs().pow(2).sum(dim=1)
            log_joint = log_joint + torch.log(norm_squared)
            psi = psi / torch.sqrt(norm_squared)[:, None]
        return log_joint - self.log_herald_success(kraus, length)

    def trainable_parameter_count(self) -> int:
        """Return the number of real trainable parameters (mesh phases + memory)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def herald_success_bits(model: PhotonicQuantumHMM, length: int) -> float:
    """Return log2 of the probability that a length-n sequence is fully heralded."""
    with torch.no_grad():
        return float(
            model.log_herald_success(model.kraus_operators(), length)
        ) / math.log(2.0)
