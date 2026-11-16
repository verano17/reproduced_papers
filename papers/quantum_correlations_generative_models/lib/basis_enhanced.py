"""Translation-invariant basis-enhanced 2-gram model of Sec. V A (Fig. 6a, p12).

Reading of Fig. 6a used here (LOG.md, R2). Each site i has a top wire of
dimension h (the bond dimension) and a bottom wire of dimension M:

1. a prior unitary ``U_p`` prepares the top wire of site 1 from ``|0>``;
2. a uniformly controlled unitary ``CU_t`` (control: computational basis
   state ``j`` of the top wire of site i-1; target: top wire of site i,
   initially ``|0>``) applies ``U_t^{(j)}``;
3. an output unitary ``U_o`` acts on (top, bottom) of every site, bottom
   initially ``|0>``;
4. the bottom wire is measured in the computational basis (outcome m_i) and
   the top wire is traced out.

``U_t`` and ``U_o`` are shared by all sites (translation invariance). The
amplitude for outcomes m and traced top-wire values t is

    Psi(m, t) = sum_j U_p[j_1, 0] prod_{i>=2} U_t^{(j_{i-1})}[j_i, 0]
                       prod_i U_o[(t_i, m_i), (j_i, 0)],

and p(m) = sum_t |Psi(m, t)|^2. Because the control reads ``j_i`` before
``U_o`` acts, the same index ``j_i`` enters the emission at site i and the
control of site i+1. The contraction below carries an operator ``rho`` over
``j`` (a double-layer MPS with bond dimension h):

    rho_1 = u_p u_p^dagger,                       u_p = U_p[:, 0]
    emission: rho <- rho * G_m,                   G_m[j, k] = sum_t K[t, j] conj(K[t, k])
    transfer: rho'[a, b] = sum_{j,k} rho[j, k] T[j, a] conj(T[k, b]),
                                                  T[j, a] = U_t^{(j)}[a, 0]

where ``*`` is the elementwise product and ``K[t, j] = U_o[(t, m), (j, 0)]``.
After the last emission p(m) is the sum of all entries of rho (the last
top wire is traced after ``U_o``, so its coherences contribute; Fig. 6a/6b,
p12, read at 300 dpi). Summing G_m over m gives the identity, so p is
normalised whenever the parameters are unitary (Z = 1, p12). An earlier
version read out only the trace at the last site (a different model; LOG.md,
Pre-handoff verification, V1). If ``U_o`` keeps j and writes
``sqrt(e(m|j))`` on the bottom wire, rho stays diagonal and the model is the
classical HMM with transition ``|U_t^{(j)}[a, 0]|^2`` and emission ``e``
(``from_hmm``).
"""

from __future__ import annotations

import math

import torch


def haar_unitary(
    dim: int, generator: torch.Generator, dtype: torch.dtype
) -> torch.Tensor:
    """Draw a Haar-random unitary (complex dtype) or orthogonal (real dtype) matrix.

    The paper does not state the initialisation (LOG.md, R4); Haar-random
    initialisation is the assumption recorded there.
    """
    real_dtype = (
        torch.float64 if dtype in (torch.complex128, torch.float64) else torch.float32
    )
    if dtype.is_complex:
        matrix = torch.complex(
            torch.randn(dim, dim, generator=generator, dtype=real_dtype),
            torch.randn(dim, dim, generator=generator, dtype=real_dtype),
        )
    else:
        matrix = torch.randn(dim, dim, generator=generator, dtype=real_dtype)
    q_factor, r_factor = torch.linalg.qr(matrix)
    diagonal = torch.diagonal(r_factor)
    return q_factor * (diagonal / diagonal.abs())


def complete_to_unitary(columns: torch.Tensor) -> torch.Tensor:
    """Return a unitary whose leading columns equal ``columns`` (orthonormal)."""
    dim, num_columns = columns.shape
    filler = torch.eye(dim, dtype=columns.dtype)
    stacked = torch.cat([columns, filler], dim=1)
    q_factor, r_factor = torch.linalg.qr(stacked)
    signs = torch.diagonal(r_factor)[:dim]
    q_factor = q_factor * (signs / signs.abs())
    q_factor[:, :num_columns] = columns
    return q_factor


class BasisEnhancedTwoGram(torch.nn.Module):
    """Basis-enhanced 2-gram model (Fig. 6a, p12) with unitary parameters.

    Parameters
    ----------
    bond_dim : int
        Top-wire dimension h.
    alphabet_size : int
        Bottom-wire dimension M.
    generator : torch.Generator
        Source of the Haar-random initialisation.
    dtype : torch.dtype
        ``torch.complex128`` for the paper model; ``torch.float64`` restricts
        every unitary to be real orthogonal (planned ablation B4).
    """

    def __init__(
        self,
        bond_dim: int,
        alphabet_size: int,
        generator: torch.Generator,
        dtype: torch.dtype = torch.complex128,
    ) -> None:
        super().__init__()
        self.bond_dim = int(bond_dim)
        self.alphabet_size = int(alphabet_size)
        h, M = self.bond_dim, self.alphabet_size
        self.prior = torch.nn.Parameter(haar_unitary(h, generator, dtype))
        self.transfer = torch.nn.Parameter(
            torch.stack([haar_unitary(h, generator, dtype) for _ in range(h)])
        )
        self.output = torch.nn.Parameter(haar_unitary(h * M, generator, dtype))

    def unitary_parameters(self) -> list[torch.nn.Parameter]:
        """Return the parameters that must stay unitary (all of them)."""
        return [self.prior, self.transfer, self.output]

    def _emission_columns(self) -> torch.Tensor:
        """Return ``E[t, m, j] = U_o[(t, m), (j, 0)]`` (row index = t*M + m)."""
        h, M = self.bond_dim, self.alphabet_size
        return self.output.reshape(h, M, h, M)[:, :, :, 0]

    def log_prob(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return natural-log probabilities of a batch of sequences.

        Parameters
        ----------
        sequences : torch.Tensor
            Integer tensor of shape ``(batch, n)`` with values in ``0..M-1``.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``(batch,)``.
        """
        batch, length = sequences.shape
        prior_column = self.prior[:, 0]
        transfer_columns = self.transfer[:, :, 0]  # T[j, a] = U_t^{(j)}[a, 0]
        emission = self._emission_columns()  # (t, m, j)
        rho = torch.outer(prior_column, prior_column.conj()).expand(batch, -1, -1)
        log_prob = torch.zeros(batch, dtype=torch.float64, device=sequences.device)
        for site in range(length):
            if site > 0:
                rho = torch.einsum(
                    "zjk,ja,kb->zab", rho, transfer_columns, transfer_columns.conj()
                )
            kraus = emission[:, sequences[:, site], :].permute(1, 0, 2)  # (batch, t, j)
            gram = torch.einsum("ztj,ztk->zjk", kraus, kraus.conj())
            rho = rho * gram
            # The chain is linear in rho, so any per-site rescaling is exact as
            # long as the scales are accumulated; the trace is used as a
            # positive scale (it is not the conditional p(m_i | m_<i): the
            # coherent transfer does not preserve the trace). At the last site
            # the top wire is traced after U_o (Fig. 6a/6b, p12), so p(m) is the
            # sum of ALL entries of rho * G_m, coherences included.
            if site == length - 1:
                norm = rho.sum(dim=(1, 2)).real
            else:
                norm = torch.einsum("zjj->z", rho).real
            log_prob = log_prob + torch.log(norm)
            rho = rho / norm[:, None, None]
        return log_prob

    @torch.no_grad()
    def from_hmm(
        self, initial: torch.Tensor, transition: torch.Tensor, emission: torch.Tensor
    ) -> None:
        """Set the unitaries so the model equals a classical HMM exactly.

        Parameters
        ----------
        initial : torch.Tensor
            Initial hidden distribution, shape ``(h,)``.
        transition : torch.Tensor
            Row-stochastic transition matrix ``A[j, a] = p(a | j)``, shape ``(h, h)``.
        emission : torch.Tensor
            Row-stochastic emission matrix ``e[j, m] = p(m | j)``, shape ``(h, M)``.
        """
        h, M = self.bond_dim, self.alphabet_size
        dtype = self.prior.dtype
        self.prior.copy_(complete_to_unitary(initial.sqrt().to(dtype)[:, None]))
        for j in range(h):
            self.transfer[j].copy_(
                complete_to_unitary(transition[j].sqrt().to(dtype)[:, None])
            )
        # Column (j, 0) of U_o is |t = j> (x) sum_m sqrt(e(m|j)) |m>.
        columns = torch.zeros(h * M, h, dtype=dtype)
        for j in range(h):
            columns[j * M : (j + 1) * M, j] = emission[j].sqrt().to(dtype)
        full = complete_to_unitary(columns)
        # complete_to_unitary puts the h given columns first; U_o needs them at
        # input indices (j, 0) = j*M. Permute the remaining columns into place.
        target_positions = [j * M for j in range(h)]
        other_positions = [
            index for index in range(h * M) if index not in target_positions
        ]
        reordered = torch.empty_like(full)
        reordered[:, target_positions] = full[:, :h]
        reordered[:, other_positions] = full[:, h:]
        self.output.copy_(reordered)


def free_parameter_count(
    bond_dim: int, alphabet_size: int, field: str = "complex"
) -> int:
    """Count the generic number of real parameters p(m) depends on.

    p(m) depends on ``U_o`` only through the Gram matrices
    ``G_m = K_m^dagger K_m`` (an M-outcome POVM on C^h), on the transfer rows
    ``T[j, :]`` (unit vectors) and on the prior column (a unit vector). The
    counts below are the generic rank of the Jacobian of p over all
    sequences with respect to the unitary tangent directions, computed for
    (h, M) in {2, 3} x {2, 3} (LOG.md, Pre-handoff verification, V3):

    - complex: (M + 1) h^2 - 2. The coherent last-site readout (Fig. 6a/6b)
      breaks the row-phase gauge of ``T`` that a trace readout would have
      (that readout gives (M + 1) h^2 - h - 1);
    - real (B4): (M - 1) h (h + 1) / 2 + h (h - 1) + (h - 1).

    At h = 1 both are M - 1, the same as the HMM; the classical HMM has
    h (h + M - 1) - 1. The Fig. 8 test does not use this count: its degrees
    of freedom are inferred as h^2 - 1 (LOG.md, R7).
    """
    h, M = int(bond_dim), int(alphabet_size)
    if field == "complex":
        return (M + 1) * h * h - 2
    if field == "real":
        return (M - 1) * h * (h + 1) // 2 + h * (h - 1) + (h - 1)
    raise ValueError(f"Unknown field {field!r}")


def hmm_parameter_count(bond_dim: int, alphabet_size: int) -> int:
    """Return the free parameters of a translation-invariant HMM: h (h + M - 1) - 1."""
    h, M = int(bond_dim), int(alphabet_size)
    return h * (h + M - 1) - 1


def parameter_matched_hmm_bond_dim(parameter_count: int, alphabet_size: int) -> int:
    """Return the smallest HMM bond dimension h' with at least ``parameter_count`` parameters.

    Phase 5 control B3 (LOG.md, Phase 5 Comparison Plan): the HMM matched to
    a model with ``parameter_count`` free parameters.
    """
    bond_dim = 1
    while hmm_parameter_count(bond_dim, alphabet_size) < parameter_count:
        bond_dim += 1
    return bond_dim


class RiemannianMomentum:
    """Riemannian descent with momentum on the unitary group (Eqs. 22-26, p13).

    Update for each unitary U with Euclidean gradient G = autograd gradient
    (PyTorch returns 2 dL/dconj(U) for a real loss, the convention of Eq. 20):

        v <- beta v + alpha G                          (Eq. 23)
        U <- exp(-(v U^dagger - U v^dagger)) U          (descent form of Eq. 26)

    Eq. 26 as printed reads exp(-(U v^dagger - v U^dagger)) U. With the
    elementwise convention for dL/dconj(U) that is an ascent step; the sign
    used here is the descent direction of Abrudan et al. (paper Ref. [82]),
    verified by ``tests/test_basis_enhanced.py`` (LOG.md, Strategy and Key
    Decisions). The exponent is skew-Hermitian, so U stays unitary up to
    round-off.
    """

    def __init__(
        self, parameters: list[torch.nn.Parameter], lr: float, momentum: float
    ) -> None:
        self.parameters = list(parameters)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.velocities = [torch.zeros_like(parameter) for parameter in self.parameters]

    @torch.no_grad()
    def step(self) -> None:
        for parameter, velocity in zip(self.parameters, self.velocities):
            if parameter.grad is None:
                continue
            velocity.mul_(self.momentum).add_(parameter.grad, alpha=self.lr)
            adjoint = velocity.conj().transpose(-1, -2)
            generator = (
                velocity @ parameter.conj().transpose(-1, -2) - parameter @ adjoint
            )
            parameter.copy_(torch.linalg.matrix_exp(-generator) @ parameter)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None


def unitarity_error(model: BasisEnhancedTwoGram) -> float:
    """Return max |U^dagger U - I| over all unitary parameters."""
    worst = 0.0
    for parameter in model.unitary_parameters():
        matrices = parameter.detach().reshape(
            -1, parameter.shape[-2], parameter.shape[-1]
        )
        identity = torch.eye(matrices.shape[-1], dtype=matrices.dtype)
        deviation = matrices.conj().transpose(-1, -2) @ matrices - identity
        worst = max(worst, float(deviation.abs().max()))
    return worst


def bits(nats: float) -> float:
    """Convert nats to bits."""
    return nats / math.log(2.0)
