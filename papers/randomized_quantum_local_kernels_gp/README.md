# Learning Gaussian Processes with Randomized Quantum Local Kernels — Reproduction

> Status (2026-11-19): reproduction, photonic (MerLin) adaptation and fair
> classical comparison complete; all results below come from completed and
> validated runs. Two independent reviews of the discrepancies were carried
> out and their corrections are included. Workflow state, decisions and the
> full run ledger are in [`LOG.md`](LOG.md); durable lessons are in
> [`INSIGHTS.md`](INSIGHTS.md).

## Reference and Attribution

- **Paper:** *Learning Gaussian Processes with Randomized Quantum Local
  Kernels*, 2026 International Conference on Quantum Communications,
  Networking, and Computing (QCNC), IEEE, pp. 660-667.
- **Authors:** Abdallah Aaraba, Soumaya Cherkaoui, Ola Ahmad, Jean-Frédéric
  Laprade, Shengrui Wang.
- **DOI:** [10.1109/QCNC69040.2026.00108](https://doi.org/10.1109/QCNC69040.2026.00108)
  (no arXiv version found).
- **Original code:**
  [abdo-aary/randomized-quantum-local-kernels](https://github.com/abdo-aary/randomized-quantum-local-kernels),
  commit `0411494f540f24cc29010defdaf2e3fcc9f497a4`. The repository has no
  license file, so it was read as a reference only (never executed, no
  code copied). This folder is an independent NumPy/MerLin
  reimplementation.
- **License and attribution:** the paper is © 2026 IEEE and is not
  redistributed.

## Original Paper

**Problem.** Choosing a Gaussian-process (GP) covariance kernel is hard when
the correlations are unknown or non-stationary.

**Method.**

- A quantum device is used only as a *generator* of R = 128 kernels. Each
  kernel comes from a shallow 4-qubit ring circuit with 4 data re-uploading
  layers, whose parameters are drawn from a Sobol sequence.
- The circuit's 24 few-body local Pauli expectations (X, Y, Z on each qubit
  and XX, YY, ZZ on ring edges) are estimated with "classical shadows".
  They are wrapped into a kernel by a normalized cosine,
  κ = ½(1 + cos∠(φ(x), φ(x'))).
- A classical multiple-kernel-learning (MKL) layer learns a softmax-weighted
  mixture of the 128 kernels and the noise variance by maximizing the GP
  marginal likelihood.

**Experiments.** 1D synthetic regression (M = 500 inputs on [−1, 1],
375 train / 125 test, noise σ² = 0.002). The labels are samples from three
query kernels: RBF, Matérn 3/2, and a Paciorek-Schervish non-stationary
kernel.

**Reported results.**

- Table I: test kernel alignment of 0.9791 / 0.9920 / 0.9538.
- Fig. 4: test predictions with "well calibrated" uncertainties.
- Fig. 5: Gram heatmaps.
- Sec. V: N = 10 000 shadow snapshots keep the feature error below 0.06.

**What the paper does not report.** No baseline, no seeds or error bars, no
hardware run. It claims no empirical advantage over classical methods; its
motivation says the feature spaces are "difficult to emulate with shallow
classical models".

## Reproduction Scope, Claims, and Deviations

**Reproduced:**

- Data generation (Fig. 3).
- The full Table I pipeline, including Fig. 4 and Fig. 5.
- The shadow-error statement.

All of these were run with 15 runs per configuration (5 data seeds × 3 Sobol
seeds).

**Added:**

- Three feature estimators:
  - exact features;
  - the authors' noise model;
  - a true single-qubit Pauli classical-shadow simulation.
- A converged MKL optimizer (L-BFGS-B) next to the authors' protocol.
- Classical fair baselines.
- A MerLin photonic adaptation.
- A bandwidth diagnostic.

**Claims and verdicts:**

| ID | Claim | Verdict |
|---|---|---|
| C1/C2 | The learned quantum mixture approximates stationary and non-stationary covariances with high test alignment (Table I) | **Supported**: quantitatively reproduced with the authors' exact protocol |
| C3 | Test predictions follow the targets with high fidelity and well-calibrated uncertainty (Fig. 4) | **Partially supported**: near-oracle on RBF and non-stationary with exact features; Matérn RMSE is about 1.8× the oracle, caused by the paper's encoding bandwidth; the published run is mildly overconfident |
| C4 | N = 10 000 classical-shadow snapshots give feature sup-norm error < 0.06 | **Unsupported**: true shadows give 0.178; the 0.059 comes from a direct-measurement noise model |
| — | Usefulness relative to classical alternatives (implicit) | **No quantum or photonic advantage**: classical baselines match or beat every quantum or photonic configuration on predictive metrics |

**Validity level and confidence.**

- **Validity level: V1** (full-data, multi-seed, paper-accurate). The paper's
  task is itself synthetic. We regenerate it at full size (M = 500, R = 128,
  N = 10 000) with the authors' generator and settings, and run 15 runs per
  configuration.
- **Caveat.** The regenerated arrays are not bit-identical to the authors'
  (their arrays are unpublished), so agreement is judged at metric and
  figure level.
- **Reproducibility confidence:**
  - high for Table I and Fig. 3;
  - high for the shadow-error finding;
  - medium for the calibration part of C3 and for the photonic assessment.

A step-by-step walkthrough of all this evidence is in
[`notebook.ipynb`](notebook.ipynb).

**Deviations from the paper text** (the code behind the paper's results was
followed, because it reproduces Table I and Fig. 3):

- **Data settings.** Lengthscale 0.3 and non-stationary c = 2.0 with
  ℓ(x) = ℓ_min + (ℓ_max − ℓ_min)·sigmoid(+cx) were used, as in the authors'
  notebook. The paper prints 0.2, c = 0.2 and 1/(1 + e^{cx}). Both were run
  (`table1_papertext_data.json`).
- **Alignment.** Table I is *centered* kernel alignment, as computed by the
  code. Eq. 25 is uncentered. Both are reported.
- **Optimizer.** The authors' protocol is reproduced exactly: backtracking
  ascent on softmax logits, a low-rank + Hutchinson *gradient* with an exact
  objective, a 9-point log grid for σ² starting at σ² = 1, 3 rounds, and a
  quirk where every GP's weights are ascended at the last GP's σ². The paper
  describes stochastic gradient ascent with an approximated objective.
- **Gates.** The R_ZZ(β) = exp(−iβZZ/2) convention of the code was used
  (Eq. 9 omits the ½).
- **Shadows.** The authors' "shadow" noise is reproduced as `code_gaussian`;
  real classical shadows are simulated as `pauli_shadow`.
- **Simulation.** NumPy statevectors instead of Qiskit Aer (verified equal
  to Qiskit to 1e-10).

## Project Layout

```text
papers/randomized_quantum_local_kernels_gp/
├── lib/            circuits.py (qubit ring circuits, Sobol layout), features.py (exact /
│                   code_gaussian / pauli_shadow), photonic_features.py (MerLin adaptation),
│                   classical_baselines.py (Phase 5 baselines), kernels.py, data.py, gp.py,
│                   mkl.py (authors' protocol + L-BFGS), evaluation.py, plotting.py,
│                   runner.py, experiment_logging.py
├── configs/        defaults.json (smoke) and one JSON per experiment variant (see Configuration)
├── utils/          run_sweep.py, summarize_sweep.py, compare_sweeps.py, compare_baselines.py,
│                   summarize_bandwidth.py, plot_*.py
├── tests/          65 tests (Qiskit cross-checks, gradients, shadows, MerLin vs Perceval, runners)
├── results/        curated tables and figures (generated from outdir/ by the utils scripts)
├── outdir/         raw runs and sweeps (disposable, not committed)
└── LOG.md, INSIGHTS.md, FEEDBACK.md, CONFLUENCE.md, VISITED_URLS.md, notebook.ipynb
```

## Install and How to Run

```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib, merlinquantum, perceval-quandela, torch, pytest, qiskit, pandas, jupyter
```

Quick smoke run (M = 60, R = 8; about 5 s), from the repository root:

```bash
python implementation.py --paper randomized_quantum_local_kernels_gp
```

Table I exactly as the authors' code ran it (about 1 min):

```bash
python implementation.py --paper randomized_quantum_local_kernels_gp --config configs/table1_original_sharednoise.json
```

Converged optimizer, the photonic adaptation, and a classical baseline:

```bash
python implementation.py --paper randomized_quantum_local_kernels_gp --config configs/table1_lbfgs.json
python implementation.py --paper randomized_quantum_local_kernels_gp --config configs/table1_merlin_mixed_lbfgs.json
python implementation.py --paper randomized_quantum_local_kernels_gp --config configs/table1_classical_rff_lbfgs.json
```

A 15-run sweep and its summary (from the paper folder):

```bash
python utils/run_sweep.py --sweep-id S5_table1_original_sharednoise --config configs/table1_original_sharednoise.json --data-seeds 12345 1 2 3 4 --sobol-seeds 12345 1 2
```

Regenerate the comparison tables:

```bash
python utils/compare_baselines.py --out results/phase5_comparison
```

Export every run's per-kernel metrics (the notebook's stored data), then run
the walkthrough notebook in a fresh kernel (about 30 s on CPU):

```bash
python utils/export_run_table.py --out results/all_runs_metrics.csv
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

`notebook.ipynb` walks through the claim, data, method, known-answer checks,
a reduced live experiment (R = 32, exact features, paper seed), the Table I
result, fair baselines, controls, robustness, and the photonic adaptation.
Every decisive number in it is loaded from the stored 15-run sweeps in
`results/all_runs_metrics.csv`; the live cells are labelled reductions.

Each run writes `outdir/run_YYYYMMDD-HHMMSS/` with `run.log`,
`run_status.json`, `config_snapshot.json`, `metrics.json`, `predictions.npz`,
optional `features.npz`, and Fig. 3-5 style PNGs. Sweeps write
`outdir/sweeps/<ID>/` with `sweep.log`, `sweep_status.json` and
`summary.{json,md}`.

## Configuration

`cli.json` is the authoritative schema. Paper flags:

- `--task` (`gp_mkl` or `gp_parametric`)
- `--feature-backend` (`qubit_statevector`, `merlin_photonic`,
  `classical_rff`, `rbf_dictionary`)
- `--feature-mode` (`exact`, `code_gaussian`, `pauli_shadow`,
  `photon_sampled`)
- `--num-snapshots`, `--feature-seed`, `--num-draws`, `--sobol-seed`,
  `--data-seed`, `--input-scale`
- `--optimizer` (`authors_protocol` or `lbfgs`), `--noise-coupling`
  (`per_kernel` or `authors_shared`), `--gradient`, `--max-iterations`
- `--detector` (`threshold` or `pnr`), `--initial-mixing`

Global options (`--config`, `--outdir`, `--seed`, `--dtype`, `--device`,
`--log-level`, `--data-root`) come from the shared runtime. The code is
NumPy/CPU and float64 only and rejects other values.

| Config | Purpose |
|---|---|
| `table1_original_sharednoise.json` | **Table I protocol exactly as in the authors' code** (code data settings, Gaussian noise model N = 10 000, approximate gradient, shared-σ² quirk) |
| `table1_original.json` | Same with per-GP σ² (no quirk) |
| `table1_lbfgs.json` | Same features, converged L-BFGS-B MKL (not in the paper) |
| `table1_exact_features.json`, `table1_exact_features_lbfgs.json` | Shot-noise-free qubit features |
| `table1_pauli_shadow.json` | True classical-shadow features |
| `table1_papertext_data.json` | Data settings as printed in the paper (ℓ = 0.2, c = 0.2) |
| `table1_exact_features_lbfgs_x2p5.json` | Bandwidth diagnostic: inputs × 2.5 before encoding (characterization) |
| `table1_merlin_mixed.json`, `table1_merlin_mixed_lbfgs.json` | MerLin adaptation, corrected encoding (use these) |
| `table1_merlin.json`, `table1_merlin_lbfgs.json`, `table1_merlin_exact.json` | MerLin adaptation, first version (layer-0 encoding inert; kept for provenance) |
| `table1_classical_{rff,rff_lbfgs,rbfdict,rbfdict_lbfgs,gp_true,gp_rbf}.json` | Classical fair baselines |
| `table1_reduced.json`, `defaults.json` | Quick and smoke runs (not results) |

## Data

The data are synthetic and fully procedural: no download and no pretrained
weights.

- **Generation.** `lib/data.py` reimplements the authors' generator from
  its specification. With `default_rng(seed)` it draws X ~ U[−1, 1]^500,
  then one child seed per query kernel. Each label vector is
  `default_rng(child).multivariate_normal(0, K* + 0.002 I)`, with K*
  clipped to [0, 1] and given a unit diagonal. The first 375 points are
  train and the last 125 test.
- **Seeds.** The data are regenerated on the fly from `data.seed` (paper
  seed 12345; data seeds 1-4 for repetitions). Optional caching writes to
  `<data_root>/randomized_quantum_local_kernels_gp/synthetic_gp/`.
- **Check against the authors.** The authors' arrays were never published,
  so bit-level equality cannot be checked. The regenerated paper-seed data
  matches the paper's Fig. 3 at pixel level (independently verified). Our
  paper-seed run is still not bit-identical to the authors': train log
  marginal likelihood 922.3 vs 930.5, probably from NumPy/LAPACK differences
  in `multivariate_normal`.
- **Sources.** The paper PDF, 220-dpi page renders and the reference
  snapshot of the authors' repository are under
  `data/randomized_quantum_local_kernels_gp/sources/` (git-ignored).
  Recreate the snapshot with:

```bash
git clone https://github.com/abdo-aary/randomized-quantum-local-kernels.git data/randomized_quantum_local_kernels_gp/sources/repo_snapshot
```

```bash
git -C data/randomized_quantum_local_kernels_gp/sources/repo_snapshot checkout 0411494f540f24cc29010defdaf2e3fcc9f497a4
```

## Results Obtained and Comparison with the Paper

All local values are `synthetic-data`. Unless stated otherwise, each value
is the mean ± sample std over 15 runs: data seeds {12345, 1, 2, 3, 4} ×
Sobol seeds {12345, 1, 2}, feature-noise seed 123. This is a crossed design
with fixed noise seeds, so the confidence intervals are optimistic.

### Paper items

| Paper item | Claim tested | Paper value | Reproduced value | Delta | Metric agreement | Trend agreement | Claim support | Seeds | Comment |
|---|---|---:|---:|---:|---|---|---|---:|---|
| Table I, RBF | C1 | 0.9791 | 0.9717 ± 0.0192 | −0.0074 | yes (within 0.4 std) | yes | supported | 15 | Authors' exact protocol (S5); paper value is one run |
| Table I, Matérn 3/2 | C1 | 0.9920 | 0.9879 ± 0.0092 | −0.0041 | yes (within 0.45 std) | yes | supported | 15 | Without the shared-σ² quirk: 0.9758 (systematically lower) |
| Table I, non-stationary | C2 | 0.9538 | 0.9585 ± 0.0293 | +0.0047 | yes (within 0.2 std) | yes | supported | 15 | — |
| Table I, paper-seed single run | C1/C2 | 0.9791 / 0.9920 / 0.9538 | 0.9781 / 0.9878 / 0.9528 | ≤ 0.004 | yes | yes | supported | 1 | `preliminary`; data not bit-identical |
| Fig. 3 (data) | — | figure | identical at pixel level | — | yes | yes | — | 1 | Shows code settings ℓ = 0.3, c = 2.0 were used |
| Fig. 4 (test RMSE, not reported in the paper) | C3 | notebook: 0.049 / 0.083 / 0.047 | 0.047 / 0.098 / 0.050 | +0.002 / +0.015 / +0.003 | partial (Matérn) | yes | partially supported | 15 | Noise floor 0.045; oracle 0.045 / 0.050 / 0.047 |
| Fig. 4 (2σ coverage, "well calibrated") | C3 | notebook: 0.912 / 0.904 / 0.920 | 0.945 / 0.929 / 0.953 | +0.03 / +0.03 / +0.03 | n/a (qualitative claim) | yes | partially supported | 15 | Published run detectably overconfident (joint p ≈ 0.001); Matérn low in general |
| Fig. 5 (Gram heatmaps) | C1 | figure | same block pattern, raised floor | — | qualitative | yes | supported | 1 | `results/table1_original_paperseed_fig5_*.png` |
| Sec. V shadow error | C4 | < 0.06 (notebook 0.059) | authors' model 0.0586 ± 0.0007; **true shadows 0.1780 ± 0.0089** | +0.119 (true shadows) | **no** | — | **unsupported** | 15 | Authors' estimator is direct-measurement noise, not shadows |

### All sweeps

There is no candidate selection: every sweep is a repetition sweep. Metric:
test centered alignment, primary; test split; final optimizer state; mean ±
std. Every sweep is **COMPLETE** with all planned runs successful (15/15,
except S8 with 5/5). Learned-mixture alignment for rbf / matern32 /
nonstat_ls:

| Sweep | System | Alignment | Summary |
|---|---|---|---|
| S1 | Authors' protocol, per-GP σ² | 0.973 / 0.976 / 0.959 | [`results/sweep_S1_table1_original_summary.md`](results/sweep_S1_table1_original_summary.md) |
| S2 | Exact features (authors' protocol, exact gradient) | 0.970 / 0.954 / 0.958 | [`results/sweep_S2_table1_exact_features_summary.md`](results/sweep_S2_table1_exact_features_summary.md) |
| S3 | True Pauli shadows | 0.964 / 0.980 / 0.956 | [`results/sweep_S3_table1_pauli_shadow_summary.md`](results/sweep_S3_table1_pauli_shadow_summary.md) |
| S4 | Paper-text data settings | 0.964 / 0.950 / 0.948 | [`results/sweep_S4_table1_papertext_data_summary.md`](results/sweep_S4_table1_papertext_data_summary.md) |
| S5 | **Authors' exact protocol** | 0.972 / 0.988 / 0.959 | [`results/sweep_S5_table1_original_sharednoise_summary.md`](results/sweep_S5_table1_original_sharednoise_summary.md) |
| S6 | Converged L-BFGS | 0.981 / 0.986 / 0.977 | [`results/sweep_S6_table1_lbfgs_summary.md`](results/sweep_S6_table1_lbfgs_summary.md) |
| S7 | Exact features, L-BFGS | 0.959 / 0.969 / 0.957 | [`results/sweep_S7_table1_exact_features_lbfgs_summary.md`](results/sweep_S7_table1_exact_features_lbfgs_summary.md) |
| S8 | Inputs × 2.5, exact, L-BFGS (5 runs) | see characterization | [`results/s8_bandwidth_characterization.md`](results/s8_bandwidth_characterization.md) |
| M1-M4 | MerLin adaptation | see Photonic section | `results/sweep_M*_summary.md` |
| C1-C6 | Classical baselines | see Fair Baselines | `results/sweep_C*_summary.md` |

Every per-sweep summary lists all three methods (learned mixture, uniform
weights, best single kernel) with alignment, RMSE, NLPD and coverage.

### Other findings on the paper's method

- **Optimizer.** The authors' optimizer under-converges: it ascends the
  weights at the wrong σ². For RBF and non-stationary data it ends with a
  lower train likelihood than a uniform mixture given the same σ² search
  (14/15 runs each).
  - A converged optimizer (S6) is above uniform in every run.
  - Even converged, learned weights give the same alignment as uniform
    weights for RBF and non-stationary data and worse test NLPD
    (overfitting). Learning clearly helps Matérn.
  - The paper never compares with uniform weights; this is a missing
    ablation rather than a contradiction.
- **Shot noise inflates alignment.** Exact features give lower alignment
  than 10 000-shot features (S7 vs S6: −0.02 on every kernel) but better
  predictions. The noise acts as extra roughness.
- **Printed data settings.** With the paper's printed settings (S4), uniform
  weights drop to 0.90 / 0.89 alignment on RBF / Matérn and learning adds
  about 0.065. The printed non-stationary kernel is nearly stationary
  (ℓ ∈ [0.415, 0.485]).

### Where the paper and the evidence disagree (independently re-checked twice)

- **Contradicted.** "N = 10 000 classical-shadow snapshots give feature
  error < 0.06": true shadows give 0.178. The paper's own Eq. 11 would need
  about 3 × 10⁵ snapshots, and its (3/2)^ν factor is inconsistent with the
  standard 3^ν shadow norm.
- **Paper text vs the code that produced the results.**
  - Settings: ℓ = 0.3 and c = 2.0, not 0.2 and 0.2.
  - Non-stationary lengthscale: increases left to right, not decreases.
  - Table I metric: centered alignment, not the uncentered Eq. 25 (stated
    to lie in [0, 1]).
  - σ²: chosen by a grid search, not gradient ascent.
  - Objective: exact, with only the gradient approximated, contrary to
    Sec. IV-D. The paper itself says "exact" twice.
  - Optimizer: deterministic, not "stochastic".
- **Overstated.**
  - "Uncertainties are well calibrated": the published run is detectably
    overconfident.
  - In the authors' own run, the kernel with the best Table I alignment
    (Matérn, 0.992) has by far the worst predictions: RMSE 0.083 vs noise
    floor 0.045, NLPD −1.00.
  - Table I is a single run given to 4 decimals, while the run-to-run std
    is 0.01-0.03.

Figures: `results/table1_alignment_code_settings.png`,
`results/table1_alignment_optimizers.png`,
`results/table1_original_paperseed_fig{3,4,5}_*.png`.

## Fair Baselines

The paper reports none, so the baselines declared in the Phase 1 plan were
run on the same data, split, metrics and 15 seed pairs:

- **B3:** a size-matched random-Fourier-feature dictionary: 128 kernels ×
  24 features, per-kernel lengthscale log-uniform on [0.02, 2], same cosine
  wrapper and MKL.
- **B4:** a dictionary of 128 RBF lengthscales on [0.02, 2].
- **B1:** a true-family GP (the oracle).
- **B2:** a plain RBF GP.

Hyperparameters are fitted by marginal likelihood (ML-II, 10 restarts) or
MKL. Nothing is tuned on test data. **Decision rule** (fixed before any
baseline run): a difference counts only if its paired 95% bootstrap CI
excludes 0 and it exceeds 0.01 alignment, 5% RMSE or 0.05 NLPD.

| System (converged MKL or ML-II) | Alignment rbf / matern32 / nonstat | RMSE rbf / matern32 / nonstat | Median s/run* |
|---|---|---|---:|
| Qubit dictionary, 10 000-shot features (S6) | 0.981 / 0.986 / 0.977 | 0.049 / 0.098 / 0.051 | 189 |
| Qubit dictionary, exact features (S7) | 0.959 / 0.969 / 0.957 | 0.045 / 0.091 / 0.047 | — |
| Photonic adaptation, corrected (M4, simulated) | 0.981 / 0.964 / 0.965 | 0.047 / 0.084 / 0.049 | — |
| B3 RFF dictionary (C2) | 0.921 / 0.925 / 0.895 | 0.045 / 0.052 / 0.047 | 111 |
| B4 RBF dictionary (C4) | 0.978 / 0.980 / 0.931 | 0.045 / 0.050 / 0.047 | 85 |
| B1 true-family GP (C5) | 0.998 / 0.993 / 0.984 | 0.045 / 0.050 / 0.047 | 8 |
| B2 RBF GP (C6) | 0.998 / 0.777 / 0.945 | 0.045 / 0.053 / 0.047 | 5 |

\*Indicative: Apple M2, 8 cores, NumPy CPU, with up to four sweeps running
concurrently. The full table with ± std and every pairwise verdict is in
[`results/phase5_comparison.md`](results/phase5_comparison.md); the
data-seed robustness check is in
[`results/phase5_verdict_robustness.md`](results/phase5_verdict_robustness.md).

- **Prediction.** No quantum or photonic configuration beats any classical
  baseline on RMSE or NLPD, and no prediction verdict changes when the runs
  are aggregated by data seed.
  - With exact features (S7), RBF and non-stationary predictions **tie**
    the matched RFF dictionary and the oracle. The S6 losses there are a
    cost of 10 000-shot estimation.
  - Matérn remains worse (RMSE about 0.09 vs 0.05) because the paper's
    encoding range limits the circuit features to low frequencies.
    Widening the encoding ×2.5 (S8, 5 runs, characterization) brings
    Matérn RMSE to 0.054 but drops its alignment to 0.87
    ([`results/s8_bandwidth_characterization.md`](results/s8_bandwidth_characterization.md)).
- **Alignment.** The quantum dictionaries beat B3 on centered alignment in
  most cells and tie B4 except on non-stationary data. That edge is weak:
  - it comes mostly from long-range ripple in the RFF kernels;
  - it is partly produced by shot noise;
  - 4 of the alignment wins become ties at the data-seed level.
- **Alignment as a metric.** Centered alignment is a weak proxy for
  predictive quality: the RFF dictionary has the lowest alignment yet ties
  the oracle on RMSE and NLPD.
- **Outcome: no quantum or photonic advantage.** Commensurability: every
  dictionary-vs-baseline comparison is commensurable (same data, split,
  metrics, evaluation code). Paper Table I vs local runs is only partially
  commensurable: one paper-reported run, alignment only.

Figure: [`results/phase5_alignment_vs_rmse.png`](results/phase5_alignment_vs_rmse.png).

## Photonic Assessment and MerLin Extension

**Status: `MERLIN_IMPLEMENTATION`** (an adaptation, not the paper's method).

- **Scientific role preserved.** A random dictionary of shallow, local,
  data re-uploading circuits whose few-body local observables are the
  features, plus an unchanged classical MKL layer.
- **Mapping** (`lib/photonic_features.py`). Per draw, a fixed random
  linear-optical circuit on 12 modes with 4 photons:
  - a fixed 50:50 beam-splitter brickwork (corrected version), then
  - 4 layers of [PS(δ·ξ·x) on every mode (data), random PS(α),
    nearest-neighbour beam-splitter brickwork, random PS(ω)];
  - all settings Sobol-sampled.
- **Features.** Click parities ⟨Π_q⟩ and ⟨Π_qΠ_(q+1)⟩ (23 features), the
  analogue of Z / ZZ only, estimated jointly from 10 000 postselected events
  per input.
- **Rejected mappings.** Dual-rail emulation of the qubit gates (needs
  probabilistic CZ), a global fidelity kernel (breaks locality), and
  trainable universal meshes (not shallow or random).
- **Design flaw found in review.** Without the initial brickwork, the
  layer-0 data phases act on a Fock state and have no effect (M1/M2: only 3
  effective layers). M3/M4 use the corrected circuit.

Alignment / RMSE, 15 runs each, paired with the qubit sweeps:

| Kernel | Qubit, converged (S6) | Photonic, converged (M4) | Qubit, authors' protocol (S5) | Photonic, authors' protocol (M3) |
|---|---|---|---|---|
| RBF | 0.981 / 0.049 | 0.981 / 0.047 | 0.972 / 0.047 | 0.938 / 0.047 |
| Matérn 3/2 | 0.986 / 0.098 | 0.964 / 0.084 | 0.988 / 0.098 | 0.950 / 0.087 |
| Non-stationary | 0.977 / 0.051 | 0.965 / 0.049 | 0.959 / 0.050 | 0.913 / 0.049 |

- **Pre-declared tolerance** (alignment within 0.02, RMSE within 20%):
  - M4 passes on RBF and non-stationary. Matérn misses on alignment by
    0.002, while its RMSE is better than qubit.
  - M3 fails on alignment for all kernels; RMSE is within tolerance.
  - The uncorrected M1/M2 passed in 2 of 6 cells.
  - Details: [`results/merlin_tolerance_check.md`](results/merlin_tolerance_check.md).
- **Predictive metrics.** Photonic test NLPD is better than qubit by
  0.10-0.15 under converged MKL, but it does not beat the classical
  baselines.
- **Estimation noise.** Feature error at N = 10 000 is about 0.049, similar
  to the qubit features under the authors' noise model (0.059). This is not
  an estimation advantage over qubit shadows: the photonic features are
  Z-like only.
- **Assessment confidence:** medium. The main modelling choices are a
  single detection basis and n-click postselection.

## Hardware-Aware Settings (MerLin results)

| Field | Value |
|---|---|
| Computation space | `ComputationSpace.UNBUNCHED` (sweeps); `FOCK` for one PNR ablation run |
| Detector model | threshold (PNR in the ablation) |
| Photon number / modes | 4 photons, 12 modes |
| Input state | \|1,0,0,1,0,0,1,0,0,1,0,0⟩ |
| Encoding | phase shifters PS(δ_ℓq·ξ_ℓ·x) on every mode, 4 layers; initial fixed 50:50 brickwork (M3/M4) |
| Measurement | `MeasurementStrategy.probs(UNBUNCHED)`; parities from one detection basis |
| Postselection | 4 distinct clicks; success probability 0.437 on average (M3/M4; 0.475 for M1/M2), lowest single (draw, input) about 0.18-0.22; about 2.2× raw events per kept event on average |
| Simulator / QPU | MerLin 0.4.1 analytic probabilities (CPU, float64), NumPy multinomial sampling; **simulation only, no hardware** |
| Shot count | 10 000 postselected events per (draw, input) |
| Wall-clock | M3: 22 min for 15 runs (median 69 s/run); M4: 50 min (median 188 s/run); concurrent with other sweeps |
| Seeds | data {12345, 1, 2, 3, 4} × Sobol {12345, 1, 2}; shot seed 123 |
| Gradients | no photonic parameter is trained; MKL gradients are classical (approximate for M1/M3, exact for M2/M4) |
| Loss / distinguishability | ideal (not modelled) |

## Limitations

- **Data.** Synthetic 1D tasks only (as in the paper). The data are not
  bit-identical to the authors' (LAPACK).
- **Statistics.** 15 runs form a crossed 5 × 3 design with fixed noise
  seeds, so the CIs are optimistic. Some alignment verdicts change at the
  5-data-seed level.
- **B3 baseline.** Its frequency range was fixed a priori, not tuned. The
  qubit and photonic encodings use the paper's ranges; S8 is a 5-run
  characterization, not a tuned alternative.
- **Simulation scope.** Classical-shadow and photonic results are
  simulated; the photonic simulation is idealized.
- **Not run.** No hardware runs and no larger qubit or mode counts.
- **Timing.** Wall-clock comes from concurrent sweeps on a laptop and is
  indicative only.

## Tests

```bash
cd papers/randomized_quantum_local_kernels_gp && pytest -q
```

65 tests cover:

- the circuit vs Qiskit (statevectors and Sobol parameter order);
- exact features vs Qiskit expectations;
- unbiasedness and variance of the shadow simulation;
- GP likelihood vs SciPy and gradients vs finite differences;
- MKL recovery of a generating lengthscale, the shared-σ² coupling, and
  L-BFGS optimality;
- MerLin vs an independent Perceval SLOS simulation, and the layer-0
  encoding effect;
- classical baselines (RFF → RBF limit, ML-II recovery);
- end-to-end runs for every backend and task.

## Citation and License

```bibtex
@inproceedings{aaraba2026randomized,
  title     = {Learning Gaussian Processes with Randomized Quantum Local Kernels},
  author    = {Aaraba, Abdallah and Cherkaoui, Soumaya and Ahmad, Ola and
               Laprade, Jean-Fr{\'e}d{\'e}ric and Wang, Shengrui},
  booktitle = {2026 International Conference on Quantum Communications,
               Networking, and Computing (QCNC)},
  pages     = {660--667},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/QCNC69040.2026.00108}
}
```

The code in this folder follows the repository license (MIT). The paper
remains © 2026 IEEE.
