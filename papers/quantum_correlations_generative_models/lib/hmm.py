"""Translation-invariant classical hidden Markov model and Baum-Welch (p13).

The HMM has h hidden states, an initial distribution ``pi``, a transition
matrix ``A[j, a] = p(a | j)`` and an emission matrix ``E[j, m] = p(m | j)``,
shared by all sites (general form of Fig. 1b, p3). It is trained with the
Baum-Welch EM algorithm (paper Ref. [78]) on the full training set per
iteration. The paper does not state the initialisation, iteration cap,
stopping rule or smoothing (LOG.md, R3): rows are drawn from a flat Dirichlet
and the cap, tolerance and pseudo-count come from the config.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _dirichlet_rows(
    rows: int, columns: int, generator: torch.Generator
) -> torch.Tensor:
    samples = -torch.log(
        torch.rand(rows, columns, generator=generator, dtype=torch.float64)
    )
    return samples / samples.sum(dim=1, keepdim=True)


@dataclass
class HiddenMarkovModel:
    """Parameters of a translation-invariant HMM.

    Attributes
    ----------
    initial : torch.Tensor
        Shape ``(h,)``.
    transition : torch.Tensor
        Row-stochastic, shape ``(h, h)``.
    emission : torch.Tensor
        Row-stochastic, shape ``(h, M)``.
    """

    initial: torch.Tensor
    transition: torch.Tensor
    emission: torch.Tensor

    @classmethod
    def random(
        cls, bond_dim: int, alphabet_size: int, generator: torch.Generator
    ) -> HiddenMarkovModel:
        """Draw parameters from flat Dirichlet distributions."""
        return cls(
            initial=_dirichlet_rows(1, bond_dim, generator)[0],
            transition=_dirichlet_rows(bond_dim, bond_dim, generator),
            emission=_dirichlet_rows(bond_dim, alphabet_size, generator),
        )

    def to(self, device: str | torch.device) -> HiddenMarkovModel:
        return HiddenMarkovModel(
            self.initial.to(device),
            self.transition.to(device),
            self.emission.to(device),
        )

    def _forward(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scaled forward messages ``(batch, n, h)`` and scales ``(batch, n)``."""
        batch, length = sequences.shape
        forward = torch.empty(
            batch,
            length,
            self.initial.numel(),
            dtype=torch.float64,
            device=sequences.device,
        )
        scales = torch.empty(
            batch, length, dtype=torch.float64, device=sequences.device
        )
        message = self.initial[None, :] * self.emission[:, sequences[:, 0]].T
        for site in range(length):
            if site > 0:
                message = (forward[:, site - 1] @ self.transition) * self.emission[
                    :, sequences[:, site]
                ].T
            scales[:, site] = message.sum(dim=1)
            forward[:, site] = message / scales[:, site : site + 1]
        return forward, scales

    def log_prob(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return natural-log probabilities, shape ``(batch,)``."""
        _, scales = self._forward(sequences)
        return torch.log(scales).sum(dim=1)

    def baum_welch_step(
        self, sequences: torch.Tensor, pseudo_count: float
    ) -> HiddenMarkovModel:
        """Return the parameters after one EM iteration on ``sequences``."""
        batch, length = sequences.shape
        forward, scales = self._forward(sequences)
        backward = torch.ones_like(forward)
        for site in range(length - 2, -1, -1):
            weighted = (
                self.emission[:, sequences[:, site + 1]].T * backward[:, site + 1]
            )
            backward[:, site] = (weighted @ self.transition.T) / scales[
                :, site + 1 : site + 2
            ]
        occupancy = forward * backward  # gamma, rows sum to 1
        pair_counts = torch.einsum(
            "zsj,ja,zsa->ja",
            forward[:, :-1],
            self.transition,
            self.emission[:, sequences[:, 1:]].permute(1, 2, 0)
            * backward[:, 1:]
            / scales[:, 1:, None],
        )
        emission_counts = torch.zeros_like(self.emission)
        emission_counts.index_add_(
            1, sequences.reshape(-1), occupancy.reshape(-1, occupancy.shape[-1]).T
        )
        initial = occupancy[:, 0].sum(dim=0) + pseudo_count
        transition = pair_counts + pseudo_count
        emission = emission_counts + pseudo_count
        return HiddenMarkovModel(
            initial=initial / initial.sum(),
            transition=transition / transition.sum(dim=1, keepdim=True),
            emission=emission / emission.sum(dim=1, keepdim=True),
        )
