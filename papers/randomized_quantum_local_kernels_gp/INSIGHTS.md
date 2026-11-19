# Insights — Randomized Quantum Local Kernels for GP Learning

Durable lessons from reproducing Aaraba et al. (QCNC 2026). Useful for any
future reproduction involving quantum or photonic kernels, random feature
dictionaries, GP marginal-likelihood learning, or classical shadows.

## Metric pitfalls

- **Centered kernel alignment is a weak proxy for predictive quality.**
  - A size-matched random-Fourier-feature dictionary had the *lowest* test
    alignment (about 0.92) yet tied the true-family GP oracle on RMSE and
    NLPD.
  - In the authors' own run, the kernel with the *highest* Table I
    alignment (Matérn, 0.992) had by far the worst predictions.
  - Widening the encoding improved Matérn RMSE from 0.091 to 0.054 while
    alignment fell from 0.98 to 0.87.
  - Always report predictive metrics (RMSE, NLPD, coverage) next to
    alignment.
- **Why alignment misleads here.** On a 125-point test Gram, alignment is
  dominated by long-range structure. Spurious long-range ripple (e.g. RFF
  with few frequencies) lowers it without hurting prediction, while missing
  short-scale structure barely lowers it but hurts prediction.
- **Centered vs uncentered.** "Alignment" in a paper may be centered even
  when the printed equation is uncentered. Check the code. Centered
  alignment is invariant to affine rescaling (min-max scaling), and
  uncentered alignment is inflated for kernels with positive entries.
- **Shot noise can inflate alignment.** 10 000-shot features gave higher
  alignment than exact features (+0.02) but worse prediction: the fitted
  noise variance was underestimated, causing overfitting.

## Optimizer and trainability

- **Sanity-check any learned mixture against trivial initializations.** The
  authors' MKL protocol ascends the softmax logits at the wrong σ²:
  - round 1 at σ² = 1, about 500× the true noise;
  - later rounds at another GP's σ², through a shared-variable quirk.

  It then ends with a lower train log marginal likelihood than a uniform
  mixture given the same σ² search (14/15 runs for RBF and non-stationary).
  A converged L-BFGS-B over logits and log σ² jointly fixes this.
- **Converged MKL with 129 hyperparameters overfits** when the uniform
  average of the dictionary is already well matched (RBF, non-stationary):
  equal alignment, worse NLPD, 2σ coverage about 0.88-0.90. MKL helps only
  when the dictionary average is mismatched (Matérn here, or the paper's
  printed ℓ = 0.2 data).
- **Relative-improvement stopping rules depend on additive constants.**
  The authors divide by |L| computed without the −(M/2) log 2π term;
  reproducing the stopping point required the same convention.

## Classical shadows: cost accounting

- **Single-qubit Pauli shadows have per-snapshot variance 3^k − μ² for a
  k-local Pauli,** not 1 − μ². At N = 10 000 with median-of-means over 16
  groups, the sup-norm error over 1.5 M features is about 0.18 (k = 2),
  not < 0.06.
- **The authors' "shadow" simulation was Gaussian direct-measurement noise**
  (variance (1 − μ²)/batch, independent per feature). Check what a "shadow
  simulation" actually samples.
- **Simulating shadows exactly and cheaply.** Sample basis-setting counts
  (multinomial over 3^n settings), then outcome counts per setting from Born
  probabilities, then contract with a precomputed estimator table of shape
  (settings × outcomes × observables). This is exact in distribution
  because the estimator is a sum over snapshots. Chunk over circuit draws:
  the counts tensor is the memory bottleneck.
- **Eq. 11-style bounds:** the standard Pauli shadow norm is 3^ν, and the
  median-of-means needs about 2 log(2M/δ) groups. A (3/2)^ν factor is a red
  flag.

## Expressivity of re-uploading feature maps

- **Data re-uploading with bounded angles is band-limited.** With
  |ξ| < π, |δ| < 1, L = 4 and n = 4, the maximum feature frequency is about
  12-23 rad per unit of x.
  - A convex mixture of such kernels cannot represent short-scale
    (Matérn 3/2) structure; the GP absorbs it as noise (σ² ≈ 0.008 vs
    0.002 true).
  - The fix is a wider encoding range (input scale ×2.5), at the cost of
    alignment on smooth structure.
  - Before blaming "quantum features", check the encoding bandwidth.
- **The uniform average of a random dictionary behaves like one smooth
  kernel** (effective lengthscale about 0.3-0.45 here). Tasks whose true
  lengthscale matches it look "solved" without any learning.

## Photonic translation (MerLin)

- **Phase encodings on a Fock input state are a global phase: a silent
  no-op.** Put at least one mixing element (e.g. a fixed 50:50 brickwork,
  the analogue of |+⟩^n) before the first data phase shifters. Phases right
  before detection are also irrelevant. Test that every encoding layer
  changes the output probabilities.
- **`ComputationSpace.UNBUNCHED` = Fock distribution postselected on
  collision-free outcomes.** Report the acceptance probability, here
  0.44-0.48 on average (as low as 0.18), computed with a parallel `FOCK`
  layer, and convert "N kept events" into raw events (about 2.2× on
  average).
- **Click parities (−1)^click are the natural analogue of Z.** Pair
  parities are the analogue of ZZ. With exactly n clicks, Σ_q⟨Π_q⟩ is
  constant, so one single-mode feature is redundant.
- **One detection basis gives Z-like features only.** Comparing its
  estimation noise with X/Y/Z qubit shadows is not apples to apples.
- **Useful MerLin pattern:** a fixed-value `pcvl.Circuit` with only named
  input phases, `trainable_parameters=[]`, zero-padded input names
  (`in0000`…), `probs()` and `output_keys` to build observables. Verify
  against Perceval SLOS with numerically substituted parameters (agreement
  about 1e-10).

## Baseline fairness

- **A size-matched random-feature dictionary with the same wrapper and MKL
  is the right counterpart for a "quantum kernel generator".** Fix its
  frequency range a priori (here from the RBF dictionary's lengthscale
  range) and say so.
- **Remove estimation-noise asymmetry before attributing differences to the
  kernel family:** run the quantum side with exact features too. Here
  shot noise explained all losses on smooth kernels.
- **Crossed seed designs are pseudo-replication** when some systems are
  deterministic in one seed dimension. Add a data-seed-level robustness
  table. Here 4 of 117 verdicts (all alignment) changed.

## Reproducing from paper vs code

- **When a paper's printed hyperparameters cannot produce its own figure,
  the released notebook's saved outputs are the ground truth for metric
  agreement.** Here ℓ = 0.3 and c = 2.0 (code) vs 0.2 and 0.2 (text); the
  printed non-stationary kernel is nearly stationary. Run both and label
  them.
- **Matching the order of noise draws matters for single-seed agreement.**
  Reordering observables shifted Table I values by up to 0.015, the same
  size as the seed spread.

## Promising directions

- Joint tuning of encoding bandwidth and MKL (alignment vs prediction
  trade-off).
- Photonic measurement settings beyond the Fock basis (interfering with
  reference modes for X/Y-like features).
- Loss and distinguishability models for the photonic generator.
- Multivariate or real datasets where non-stationarity is not tied to a
  1D lengthscale.
