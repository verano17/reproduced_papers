from __future__ import annotations

import itertools
import math

import pytest
import torch
from common import PROJECT_DIR  # noqa: F401
from quantum_correlations_generative_models.lib.basis_enhanced import (
    BasisEnhancedTwoGram,
    RiemannianMomentum,
    free_parameter_count,
    hmm_parameter_count,
    unitarity_error,
)
from quantum_correlations_generative_models.lib.hmm import HiddenMarkovModel


def _all_sequences(length, alphabet):
    return torch.tensor(
        list(itertools.product(range(alphabet), repeat=length)), dtype=torch.long
    )


@pytest.mark.parametrize("dtype", [torch.complex128, torch.float64])
def test_distribution_is_normalised(dtype):
    model = BasisEnhancedTwoGram(3, 2, torch.Generator().manual_seed(1), dtype)
    total = model.log_prob(_all_sequences(5, 2)).exp().sum().detach()
    assert float(total) == pytest.approx(1.0, abs=1e-12)


def test_contains_classical_hmm_exactly():
    generator = torch.Generator().manual_seed(2)
    hmm = HiddenMarkovModel.random(3, 4, generator)
    model = BasisEnhancedTwoGram(3, 4, generator)
    model.from_hmm(hmm.initial, hmm.transition, hmm.emission)
    sequences = _all_sequences(4, 4)
    assert unitarity_error(model) < 1e-12
    assert torch.allclose(
        model.log_prob(sequences), hmm.log_prob(sequences), atol=1e-10
    )


def test_gradient_matches_finite_difference_along_unitary_curve():
    """E8: autograd gradient G = 2 dL/dconj(U) gives dL = Re tr(G^dagger A U) for U(eps) = exp(eps A) U."""
    generator = torch.Generator().manual_seed(3)
    model = BasisEnhancedTwoGram(2, 3, generator)
    data = torch.randint(0, 3, (6, 5), generator=generator)
    loss = -model.log_prob(data).mean()
    loss.backward()
    raw = torch.complex(
        torch.randn(6, 6, generator=generator, dtype=torch.float64),
        torch.randn(6, 6, generator=generator, dtype=torch.float64),
    )
    skew = raw - raw.conj().T
    predicted = float(
        torch.real(
            torch.trace(model.output.grad.conj().T @ (skew @ model.output.detach()))
        )
    )
    epsilon = 1e-6
    with torch.no_grad():
        values = []
        original = model.output.detach().clone()
        for sign in (1, -1):
            model.output.copy_(
                torch.linalg.matrix_exp(sign * epsilon * skew) @ original
            )
            values.append(float(-model.log_prob(data).mean()))
        model.output.copy_(original)
    assert (values[0] - values[1]) / (2 * epsilon) == pytest.approx(
        predicted, rel=1e-6, abs=1e-9
    )


def test_riemannian_step_descends_and_stays_unitary():
    generator = torch.Generator().manual_seed(4)
    model = BasisEnhancedTwoGram(3, 2, generator)
    data = torch.randint(0, 2, (16, 6), generator=generator)
    optimizer = RiemannianMomentum(model.unitary_parameters(), lr=1e-3, momentum=0.0)
    with torch.no_grad():
        before = float(-model.log_prob(data).mean())
    optimizer.zero_grad()
    (-model.log_prob(data).mean()).backward()
    optimizer.step()
    with torch.no_grad():
        after = float(-model.log_prob(data).mean())
    assert after < before
    assert unitarity_error(model) < 1e-12


def test_bond_dimension_one_is_iid_marginal():
    model = BasisEnhancedTwoGram(1, 3, torch.Generator().manual_seed(5))
    marginal = model.output.detach()[:, 0].abs() ** 2
    sequence = torch.tensor([[0, 2, 1, 1]])
    expected = sum(math.log(float(marginal[x])) for x in sequence[0])
    assert float(model.log_prob(sequence).detach()) == pytest.approx(
        expected, abs=1e-12
    )
    assert free_parameter_count(1, 3) == hmm_parameter_count(1, 3) == 2


def _amplitude_formula_probability(model, sequence):
    """p(m) = sum_t |Psi(m, t)|^2 by explicit sums over j and t (Fig. 6a, p12).

    Psi(m, t) = sum_j U_p[j_1, 0] prod_{i>=2} U_t^{(j_{i-1})}[j_i, 0]
                prod_i U_o[(t_i, m_i), (j_i, 0)]; every top wire, the last one
    included, is traced after U_o.
    """
    prior = model.prior.detach()
    transfer = model.transfer.detach()
    output = model.output.detach()
    h, M, n = model.bond_dim, model.alphabet_size, len(sequence)
    total = 0.0
    for traced in itertools.product(range(h), repeat=n):
        amplitude = 0.0
        for hidden in itertools.product(range(h), repeat=n):
            term = prior[hidden[0], 0]
            for site in range(1, n):
                term = term * transfer[hidden[site - 1]][hidden[site], 0]
            for site in range(n):
                term = (
                    term * output[traced[site] * M + sequence[site], hidden[site] * M]
                )
            amplitude = amplitude + term
        total += float(abs(amplitude) ** 2)
    return total


@pytest.mark.parametrize("dtype", [torch.complex128, torch.float64])
@pytest.mark.parametrize("bond_dim,alphabet", [(2, 3), (3, 2)])
def test_log_prob_equals_the_amplitude_formula(dtype, bond_dim, alphabet):
    # Guards the last-site readout: a trace there instead of the sum of all
    # entries is a different (still normalised) model (LOG.md, V1).
    model = BasisEnhancedTwoGram(
        bond_dim, alphabet, torch.Generator().manual_seed(5), dtype
    )
    sequences = _all_sequences(4, alphabet)
    with torch.no_grad():
        contracted = model.log_prob(sequences).exp()
    brute = torch.tensor(
        [_amplitude_formula_probability(model, s) for s in sequences.tolist()],
        dtype=torch.float64,
    )
    torch.testing.assert_close(contracted, brute, atol=1e-13, rtol=0)


def test_free_parameter_counts():
    # Jacobian-rank values (LOG.md, V3): complex (M + 1) h^2 - 2, real as B4.
    assert free_parameter_count(2, 2) == 10
    assert free_parameter_count(3, 3) == 34
    assert free_parameter_count(2, 3, "real") == 9
    assert free_parameter_count(3, 2, "real") == 14
