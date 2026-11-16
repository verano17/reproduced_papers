# Insights — Enhancing Generative Models via Quantum Correlations (Gao et al., PRX 2022)

Durable lessons from reproducing the basis-enhanced (quantum-inspired MPS)
sequence model against classical HMMs, and from its photonic adaptation.
Evidence: `LOG.md`, `README.md`, `results/`.

## Baseline fairness

- **"Equal bond dimension" is not a fair capacity match.**
  - At equal h, the basis-enhanced 2-gram has (M + 1)h^2 - 2 free
    parameters (the rank of its parameter Jacobian), against
    h(h + M - 1) - 1 for an HMM: 2-6 times more (Promoters h = 8: 318 vs
    87).
  - An HMM given at least as many parameters (h' = 17 for h = 8) fits
    training data better on every data set tested.
  - Any "quantum model beats HMM at equal h" claim needs a
    parameter-matched control.
- **Check the classical baseline's convergence before reading a gap.**
  - The paper's HMM points match a Baum-Welch run stopped after 50-100
    iterations.
  - Converging it (up to 2000 iterations) gains 0.007-0.057 bits/site,
    growing with h. The SPECT training gap then stays at or below 0.010
    bits/site and never reaches 5 sigma.
  - Logging per-iteration histories made this diagnosable without new runs.
- **Gradient training is a weaker way to fit HMMs than Baum-Welch.**
  Minibatch SGD on softmax logits, with BE's batch, epochs and a tuned lr,
  was never better than converged Baum-Welch (worse in 111 of 176
  cells). An "optimizer-matched" HMM is a useful control, but not the
  strongest baseline.
- **Simple references are informative.** A first-order Markov chain (an
  order-2 counting model) beats the quantum-inspired model up to h = 7 on
  biofam and ties it beyond. Include counting models to show how much memory a data set
  needs.
- **What survives a fair comparison can be small and split-dependent.** The
  remaining Promoters held-out advantage is 0.004-0.019 bits/site on a
  26-sequence test quarter. SEMs over seeds leave out split variance, so
  repeat the comparison on other splits before calling it general.
  Here, five more quarters (h = 8, 12, hyperparameters not retuned) kept
  the advantage in 8 of 10 cases and removed it on one quarter; h = 8,
  not separated on the original quarter, held on 4 of the 5 new ones. One
  quarter is not enough either way.
- **A real-amplitude ablation is not parameter-matched by construction.**
  Real orthogonal matrices have about half the free parameters of
  unitaries (53-67% here), so a train gap between the two mixes the field
  with parameter count. Here the ablation was not separated from the
  complex model on held-out data at most h.

## Metric ambiguities

- **The plotted "KL divergence" is a calibration problem.** Gao et al.
  plot NLL - log2(N)/n per site, i.e. KL from the uniform distribution on
  the N evaluated rows. This is why biofam training values are negative.
  Calibrate unstated metrics against closed-form points (h = 1 is the
  i.i.d. marginal) before comparing curves.
- **Unstated significance tests can be recovered from the threshold
  curves.** Fig. 8's 5-sigma line is a one-sided Wilks test with
  df = h^2 - 1, independent of M, and the training-set size as N. The
  "natural" df = M h(h - 1) misses by up to 4 bits.
- **Data-split statements can contradict the paper's own numbers.** The
  text names the standard SPECT split, but the h = 1 values and Fig. 8
  thresholds match a pooled 267-row split. Run both, and label the inferred
  one.

## Implementation pitfalls

- **Check the likelihood against the circuit diagram by brute force.**
  - The first implementation took the trace of rho G_m at the last site.
    The figures (6a, 6b, 12a) trace every top wire after U_o, so the
    correct readout is the sum of all entries.
  - Both versions are normalised and both contain the HMM, so neither the
    normalisation test nor the HMM-embedding test could catch it. It
    shifted results by up to 0.03 bits/site and flipped several verdicts.
  - A test that sums the explicit amplitude formula over all paths for a
    small (h, M, n) catches it at once. Write that test first.
- **Count free parameters numerically.** The rank of the Jacobian of the
  distribution with respect to the parameters gave (M + 1) h^2 - 2; the
  hand count had been (M + 1) h^2 - h - 1. Parameter matching depends on
  this number.

- **Printed optimiser signs can be wrong.** Eq. 25 as printed ascends
  under the elementwise dL/dconj(U) convention. Use the Abrudan et al.
  descent form, and test that one step lowers the loss and stays unitary.
- **Constrained models can produce invalid likelihoods silently.**
  - The basis-enhanced likelihood is normalised only while its matrices
    stay unitary.
  - At learning rates of 0.01-0.03 (on biofam, and at small h), Riemannian
    updates drifted off the unitary group. Runs reported validation NLLs of
    -62 bits/site (probabilities above 1) without raising an error.
  - Log a validity metric (here, unitarity error) at every epoch, and
    exclude runs above a tolerance (1e-6) from aggregates and from
    hyperparameter selection.
- **In-place forward passes break autograd.** The scaled HMM forward
  pass, written in place for Baum-Welch, cannot be differentiated. The
  gradient-trained control needs its own out-of-place forward pass, tested
  equal to the original.
- **Tie-breaks in log scale need rounding.** Equal log distances (0.001
  and 0.1 around 0.01) differ by 1 ulp. Round before comparing.

## Optimizer and trainability behaviour

- **Learning rates tuned at mid-range sizes may diverge at the extremes.**
  - A learning rate chosen on h = 4, 8, 12 was stable there but diverged
    at h = 1-2 on biofam (1200 training sequences after the validation
    carve-out, 150 steps per epoch).
  - Tune or stability-check at both ends of a capacity axis.
- **Several tuned learning rates landed on the top grid value (0.03)**, for
  the quantum-inspired model as for the gradient HMM. Grids for Riemannian
  optimisers on small data sets should extend above the paper's alpha.
- **The model contains the HMM exactly** (`from_hmm`), so it is never
  worse at the global optimum. Where it lost to a converged HMM, the
  optimiser had not reached the HMM-equivalent point within the paper's
  epochs.

## Photonic translation considerations

- **The minimal heralded photonic memory has a structural ceiling.**
  - One memory photon in h modes, plus one probe photon per site through a
    shared mesh, gives Kraus maps K_m = U[h+m, h] W + U[:h, h] U[h+m, :h].
  - Because W and the probe column are shared by all symbols, the memories
    reachable after one emission span at most two dimensions, whatever h
    is.
  - Increasing h adds parameters but not memory. Check this rank before
    running a sweep.
- **Herald cost scales exponentially with sequence length.** Measured:
  2^-18 to 2^-124 per sequence (n = 16 to 57). Report herald success with
  every heralded sequential model; it decides hardware feasibility.
- **Lifting the two-dimensional limit needs symbol-dependent operations**:
  more probe photons per site, or a mesh conditioned on the symbol. Both
  multiply the herald cost.
- **Photonic parameter-matched baselines matter too.** The MerLin
  adaptation has (h + M)(h + M - 1) + 2h parameters. Compare it with an
  HMM of matching size, not only at equal h.

## Useful MerLin patterns

- Extract heralded linear maps from a `QuantumLayer` as follows:
  1. use `MeasurementStrategy.amplitudes(ComputationSpace.FOCK)`;
  2. feed one-hot inputs over `layer.output_keys` occupations;
  3. slice the heralded output occupations;
  4. contract the sequence in PyTorch with an exact normaliser.
- Two-photon amplitudes equal the 2x2 permanents of the mesh unitary
  (checked to 1e-12 in tests), which gives an exact unit test.
- Do not use `UNBUNCHED` for heralded maps: it renormalises each input row
  after dropping bunched states, by an input-dependent factor
  1/sqrt(1 - p_bunched) (verified on MerLin 0.4.1).

## Debugging heuristics

- When a baseline looks *better* than in the paper, diagnose it as
  seriously as a worse result. Test the stronger model's normalisation
  exactly (here, summing over all 2^23 SPECT sequences), then vary one
  component, such as the iteration cap.
- Recalculate selected summaries (best-of-10, tie rules) independently from
  raw per-run metrics. This caught an unimplemented tie rule.

## Promising future directions

- Test symbol-dependent photonic memories and measure the trade-off between
  memory rank and herald cost.
- Separate "complex amplitudes" from "parameter count" further, e.g. with a
  real-orthogonal model matched in parameters to the complex one.
