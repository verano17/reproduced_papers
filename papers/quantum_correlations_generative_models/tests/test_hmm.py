from __future__ import annotations

import itertools

import pytest
import torch
from common import PROJECT_DIR  # noqa: F401
from quantum_correlations_generative_models.lib.hmm import HiddenMarkovModel


def test_hmm_distribution_is_normalised():
    hmm = HiddenMarkovModel.random(3, 2, torch.Generator().manual_seed(0))
    sequences = torch.tensor(
        list(itertools.product(range(2), repeat=6)), dtype=torch.long
    )
    assert float(hmm.log_prob(sequences).exp().sum()) == pytest.approx(1.0, abs=1e-12)


def test_baum_welch_never_decreases_likelihood():
    generator = torch.Generator().manual_seed(1)
    data = torch.randint(0, 3, (40, 8), generator=generator)
    hmm = HiddenMarkovModel.random(4, 3, generator)
    previous = float(hmm.log_prob(data).sum())
    for _ in range(10):
        hmm = hmm.baum_welch_step(data, pseudo_count=0.0)
        current = float(hmm.log_prob(data).sum())
        assert current >= previous - 1e-9
        previous = current


def test_kgram_is_hmm_with_l_to_k_minus_1_states():
    """E9 (App. A, p16): a translation-invariant 3-gram over L symbols is an HMM with L^2 states.

    The hidden state is the pair (x_{i-1}, x_i) and emits x_i deterministically;
    the first state's phantom predecessor x_0 is marginalised.
    """
    alphabet, length = 2, 5
    generator = torch.Generator().manual_seed(2)
    start = HiddenMarkovModel.random(1, alphabet * alphabet, generator).emission[
        0
    ]  # p0(x0, x1)
    trigram = HiddenMarkovModel.random(
        alphabet * alphabet, alphabet, generator
    ).emission  # T[(a, b), c]
    states = list(itertools.product(range(alphabet), repeat=2))
    transition = torch.zeros(len(states), len(states), dtype=torch.float64)
    emission = torch.zeros(len(states), alphabet, dtype=torch.float64)
    for index, (_previous, b) in enumerate(states):
        emission[index, b] = 1.0
        for c in range(alphabet):
            transition[index, states.index((b, c))] = trigram[index, c]
    hmm = HiddenMarkovModel(start.clone(), transition, emission)
    for sequence in itertools.product(range(alphabet), repeat=length):
        expected = 0.0
        for x0 in range(alphabet):
            probability = float(start[states.index((x0, sequence[0]))])
            window = (x0, sequence[0])
            for symbol in sequence[1:]:
                probability *= float(trigram[states.index(window), symbol])
                window = (window[1], symbol)
            expected += probability
        observed = float(hmm.log_prob(torch.tensor([sequence])).exp())
        assert observed == pytest.approx(expected, abs=1e-12)
