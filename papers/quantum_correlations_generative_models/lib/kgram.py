"""Phase 5 reference B5: translation-invariant order-k counting model.

``p(x) = prod_i p(x_i | x_{i-c} .. x_{i-1})`` with context length
``c = min(i, k - 1)``: one shared conditional table per context length
0..k-1, so the first k - 1 sites use the lower-order tables. Each table is
estimated by counting over every position of the training rows where that
context length applies, with add-alpha smoothing. k = 1 is the i.i.d.
marginal, k = 2 a first-order Markov chain (LOG.md, Phase 5 Comparison Plan,
amendment 3).
"""

from __future__ import annotations

import numpy as np


class KGramModel:
    """Smoothed translation-invariant k-gram model fitted by counting.

    Parameters
    ----------
    order : int
        k >= 1; the longest context has k - 1 symbols.
    alphabet_size : int
        Number of symbols M.
    smoothing : float
        Add-alpha pseudo-count, alpha > 0.
    """

    def __init__(self, order: int, alphabet_size: int, smoothing: float) -> None:
        if order < 1:
            raise ValueError(f"k-gram order must be >= 1, got {order}")
        if smoothing <= 0:
            raise ValueError(f"smoothing must be > 0, got {smoothing}")
        self.order = int(order)
        self.alphabet_size = int(alphabet_size)
        self.smoothing = float(smoothing)
        # tables[c][context index, symbol] for context length c = 0..k-1.
        self.tables: list[np.ndarray] = []

    def _context_index(self, contexts: np.ndarray) -> np.ndarray:
        """Map ``(N, c)`` symbol contexts to integer rows (base M)."""
        index = np.zeros(len(contexts), dtype=np.int64)
        for column in range(contexts.shape[1]):
            index = index * self.alphabet_size + contexts[:, column]
        return index

    def fit(self, sequences: np.ndarray) -> KGramModel:
        """Estimate every conditional table from ``(N, n)`` training rows."""
        M = self.alphabet_size
        length = sequences.shape[1]
        self.tables = []
        for context_length in range(self.order):
            counts = np.full((M**context_length, M), self.smoothing)
            # Sites where this context length is used: exactly site c for
            # c < k - 1, every site >= k - 1 for the longest context.
            sites = (
                [context_length]
                if context_length < self.order - 1
                else range(context_length, length)
            )
            for site in sites:
                rows = self._context_index(sequences[:, site - context_length : site])
                np.add.at(counts, (rows, sequences[:, site]), 1.0)
            self.tables.append(counts / counts.sum(axis=1, keepdims=True))
        return self

    def log_prob(self, sequences: np.ndarray) -> np.ndarray:
        """Return natural-log probabilities of ``(N, n)`` rows, shape ``(N,)``."""
        if not self.tables:
            raise RuntimeError("KGramModel.fit must be called before log_prob")
        log_probs = np.zeros(len(sequences))
        for site in range(sequences.shape[1]):
            context_length = min(site, self.order - 1)
            rows = self._context_index(sequences[:, site - context_length : site])
            log_probs += np.log(self.tables[context_length][rows, sequences[:, site]])
        return log_probs

    def free_parameter_count(self) -> int:
        """Return sum over context lengths c of M^c (M - 1)."""
        M = self.alphabet_size
        return sum(M**c * (M - 1) for c in range(self.order))
