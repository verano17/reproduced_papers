# Action Required: Send to MerLin Team

## Summary

Two items from building a heralded, sequential photonic model with MerLin
0.4.1.

1. **Medium:** with `MeasurementStrategy.amplitudes`, `ComputationSpace.UNBUNCHED`
   renormalises each input's output amplitudes after dropping bunched
   states. The scale factor differs from input to input. Heralded
   (post-selected) linear maps read from UNBUNCHED amplitudes are therefore
   silently distorted, and FOCK must be used instead. This is plausibly
   intended behaviour, but it is not obvious from the API and it is easy to
   get wrong.
2. **Low:** there is no primitive for a *sequential heralded* circuit, i.e.
   keep some modes coherent, detect others, re-inject a photon, and repeat.
   It had to be assembled by hand from `output_keys`. Herald success
   probabilities are not reported.

No MerLin bug is claimed. The two-photon amplitudes were exact (they equal
the 2x2 permanents of the mesh unitary to 1.2e-16), and gradients flowed to
every mesh phase.

## Context

Reproduction of Gao et al., "Enhancing Generative Models via Quantum
Correlations", PRX 12, 021037 (2022), in
`papers/quantum_correlations_generative_models`.
- **Paper's model:** a matrix-product-state sequence model with a coherent
  memory of dimension h.
- **Photonic adaptation:** a heralded two-photon "photonic quantum HMM".
  - One photon spread over h hidden modes is the memory.
  - At each site a probe photon is injected, and one shared trainable MZI
    mesh acts on h + M modes.
  - Photon-number-resolving detection of exactly one photon in the M data
    modes heralds the emitted symbol, while the hidden photon stays
    coherent.
  - The per-symbol Kraus maps K_m (h x h) are read from the layer's FOCK
    amplitudes for the h basis inputs. The sequence likelihood is the
    globally post-selected one.
- **Runs:** 480 runs (h = 1..12, 10 seeds, 4 data arms), exact simulation,
  shots = 0, float64, CPU.

Heralded and post-selected linear-optical maps are the natural way to build
memories, feed-forward and recurrent photonic models. Both items affect
anyone doing that with MerLin.

## Items to Send

### Medium — ComputationSpace / MeasurementStrategy.amplitudes: UNBUNCHED renormalises per input, which distorts heralded maps

- Type: missing documentation / API confusion
- What happened:
  - With `MeasurementStrategy.amplitudes(computation_space=UNBUNCHED)`,
    every input's output amplitude vector has unit norm over the unbunched
    states only.
  - Compared with the same layer in FOCK space, each unbunched amplitude is
    multiplied by 1/sqrt(1 - p_bunched(input)), and p_bunched depends on the
    input state.
  - For a heralded map K_m[b, a] built from these amplitudes, each column a
    is therefore rescaled by a different factor. The resulting channel is
    wrong, and it is no longer linear in the input when the input is a
    superposition.
  - In FOCK space the amplitudes are the plain unitary projections, which
    is what a heralding (post-selection) construction needs.
- Evidence:
  - Probe (MerLin 0.4.1, Python 3.13, CPU): h = 3 hidden modes, M = 2 data
    modes, 2 photons, `CircuitBuilder(n_modes=5).add_entangling_layer(seed=0)`.
    Inputs are |hidden a, probe in mode 3> for a = 0, 1, 2; the same layer is
    built in FOCK and in UNBUNCHED.
  - Output:
    ```text
    FOCK n_states 15 row norms [1. 1. 1.]
    UNBUNCHED n_states 10 row norms [1. 1. 1.]
    bunched mass per input (FOCK): [0.100167 0.032939 0.025994]
    UNBUNCHED/FOCK amplitude ratio per input row (min,max): [(1.05419, 1.05419), (1.016888, 1.016888), (1.013256, 1.013256)]
    expected 1/sqrt(1-bunched): [1.05419  1.016888 1.013256]
    ```
  - The ratio is constant within each row and different across rows, which
    confirms per-input renormalisation.
  - Probe code (abridged):
    ```python
    b = ml.CircuitBuilder(n_modes=5)
    b.add_entangling_layer(seed=0)
    layer = ml.QuantumLayer(
        builder=b,
        n_photons=2,
        measurement_strategy=ml.MeasurementStrategy.amplitudes(computation_space=space),
        dtype=torch.float64,
    )
    # input rows: one-hot on the occupation (a, probe) for a = 0..h-1, using layer.output_keys
    ```
  - Same observation in the adaptation's API assessment, made before
    implementing (`LOG.md`, "Photonic Translation Assessment (Phase 4)").
    The final implementation therefore pins `ComputationSpace.FOCK`
    (`lib/photonic_hmm_merlin.py`).
  - FOCK correctness is tested: K_m equals the 2x2 permanents of the mesh
    unitary to 1e-12, the model sums to 1 over all sequences, and the herald
    success equals the enumerated mass (`tests/test_photonic_merlin.py`).
- Why it matters:
  - UNBUNCHED is the recommended default for threshold-detector models, so
    users building heralded gates, photonic memories or feed-forward blocks
    are likely to start there.
  - The resulting error is silent: the rows are unit-norm and gradients
    flow, so training runs normally on a wrong model.
  - The effect here is a 1-5% amplitude distortion that varies with the
    input (two photons, measured). The factor is 1/sqrt(1 - p_bunched), so
    it should grow with the bunching probability, i.e. with photon number
    and mode overlap; only the two-photon case was measured.
- Suggested action:
  - Document that UNBUNCHED amplitudes are renormalised per input after
    post-selection, and say when FOCK is required: heralding, Kraus-map
    extraction, any use of amplitudes as a linear map.
  - Optionally offer an un-renormalised (projected) amplitude output for
    UNBUNCHED, or return the discarded bunched probability alongside it.
- Uncertainty: the renormalisation looks deliberate (UNBUNCHED models
  threshold detection with post-selection). The item is about
  discoverability and documentation, not a defect.

### Low — Circuit building: no primitive for sequential heralded circuits (partial detection plus re-injection), and no herald-success output

- Type: improvement
- What happened:
  - The adaptation needs, at every site, the steps below; MerLin offers no
    construct for this loop.
    1. inject one probe photon into a fixed mode;
    2. apply the mesh;
    3. detect only the data modes, keeping the hidden modes coherent;
    4. herald on "exactly one photon detected";
    5. repeat with the same mesh.
  - Workaround:
    - map Fock occupations to indices through `layer.output_keys`;
    - feed one-hot inputs for each hidden basis state;
    - slice out the heralded amplitudes into K[m, b, a];
    - contract the sequence and the herald-success normaliser Z in plain
      PyTorch.
- Evidence: `lib/photonic_hmm_merlin.py` (`__init__` builds the index maps
  from `output_keys`; `kraus_operators`, `log_prob`, `log_herald_success`
  do the contraction). Herald success per sequence, log2 mean over seeds:
  - biofam: -26 to -18 (n = 16);
  - Promoters: -124 to -26 (n = 57);
  - SPECT: -47 to -33 (n = 23).

  These numbers are in `results/photonic_comparison.json`.
- Why it matters:
  - Recurrent and memory-based photonic models (QRNN-style, photonic HMMs,
    feed-forward circuits) all need this pattern.
  - Building it by hand relies on the ordering of `output_keys` and on
    manual normalisation. It also hides the herald cost, which is the main
    hardware-feasibility number: here 10^8 to 10^37 attempts per heralded
    57-symbol sample.
- Suggested action:
  - Consider a heralded-map helper, e.g. one returning the Kraus operators
    for given kept modes, detected modes, herald pattern and ancilla input.
  - Or a cookbook recipe for sequential heralding, which reports the herald
    success probability.
  - The `partial` measurement strategy may already cover part of this; a
    cookbook example would show it.
- Uncertainty:
  - Only the public API was inspected (`inspect.signature`,
    `output_keys`); no MerLin source was read.
  - An existing MerLin feature may cover this pattern without our having
    found it.
