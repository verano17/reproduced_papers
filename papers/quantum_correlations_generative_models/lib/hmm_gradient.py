"""Phase 5 control B2: a translation-invariant HMM trained by minibatch gradients.

Same model class as the Baum-Welch HMM (``lib/hmm.py``), parameterised by
softmax logits so it can be trained by the same minibatch protocol as the
basis-enhanced model (batch 8, the paper's epochs, momentum 0.5). It
separates the model class from the training algorithm (LOG.md, Fair Baseline
Plan B2 and Phase 5 Comparison Plan).
"""

from __future__ import annotations

import torch

from .hmm import HiddenMarkovModel


class GradientHMM(torch.nn.Module):
    """Translation-invariant HMM with softmax-parameterised rows.

    Parameters
    ----------
    bond_dim : int
        Number of hidden states h.
    alphabet_size : int
        Number of symbols M.
    generator : torch.Generator
        Source of the initialisation: flat-Dirichlet rows, as for Baum-Welch,
        mapped to logits by ``log``.
    """

    def __init__(
        self, bond_dim: int, alphabet_size: int, generator: torch.Generator
    ) -> None:
        super().__init__()
        start = HiddenMarkovModel.random(bond_dim, alphabet_size, generator)
        self.initial_logits = torch.nn.Parameter(torch.log(start.initial))
        self.transition_logits = torch.nn.Parameter(torch.log(start.transition))
        self.emission_logits = torch.nn.Parameter(torch.log(start.emission))

    def as_hmm(self) -> HiddenMarkovModel:
        """Return the row-stochastic parameters (differentiable)."""
        return HiddenMarkovModel(
            initial=torch.softmax(self.initial_logits, dim=0),
            transition=torch.softmax(self.transition_logits, dim=1),
            emission=torch.softmax(self.emission_logits, dim=1),
        )

    def log_prob(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return natural-log probabilities of a ``(batch, n)`` batch, shape ``(batch,)``.

        Scaled forward algorithm written without in-place updates so autograd
        can differentiate it (``HiddenMarkovModel.log_prob`` fills a
        preallocated tensor in place); equal to it to rounding (tests).
        """
        hmm = self.as_hmm()
        message = hmm.initial[None, :] * hmm.emission[:, sequences[:, 0]].T
        scale = message.sum(dim=1)
        log_prob = torch.log(scale)
        message = message / scale[:, None]
        for site in range(1, sequences.shape[1]):
            message = (message @ hmm.transition) * hmm.emission[:, sequences[:, site]].T
            scale = message.sum(dim=1)
            log_prob = log_prob + torch.log(scale)
            message = message / scale[:, None]
        return log_prob
