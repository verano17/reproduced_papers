# Action Required: Send to MerLin Team

## Summary

Building a photonic local-kernel generator with MerLin 0.4.1 (fixed-value
Perceval circuits, threshold detection, many draws) surfaced two silent
behaviours that change scientific results:

- `ComputationSpace.UNBUNCHED` is a postselection whose success
  probability is not exposed.
- Phase encodings applied directly to a Fock input state are a no-op, and
  nothing warns about it.

Two smaller documentation gaps (input-parameter ordering, seeding of
`shots=` sampling) cost time. No MerLin computation bug was found: MerLin's
probabilities matched independent Perceval SLOS and permanent calculations
to about 1e-16.

## Context

The paper (Aaraba et al., QCNC 2026) builds GP kernels from local
observables of R = 128 random shallow 4-qubit circuits with data
re-uploading. As a MerLin adaptation we replaced each circuit with a fixed
random linear-optical circuit:

- 12 modes, 4 photons, input |1,0,0,1,0,0,1,0,0,1,0,0⟩;
- 4 layers of data phase shifters, random phases and a nearest-neighbour
  beam-splitter brickwork;
- a `QuantumLayer(circuit=…, input_parameters=["in"],
  trainable_parameters=[])`;
- `MeasurementStrategy.probs(ComputationSpace.UNBUNCHED)` (threshold
  detection) or `FOCK` (photon-number-resolving);
- local click-parity features computed from `output_keys`.

The pattern ("a dictionary of fixed random circuits as feature
generators") is common in reservoir and kernel papers, so these points
matter beyond this reproduction. Code:
`papers/randomized_quantum_local_kernels_gp/lib/photonic_features.py`.

## Items to Send

### Medium — Encoding / circuit design: phase encoding on a Fock input state silently has no effect

- **Type:** missing documentation / improvement
- **What happened:** our first circuit applied the data-dependent phase
  shifters `PS(δ·ξ·x)` as the first operation on the Fock input state.
  Diagonal phases on a single Fock state only add a global phase, so
  the first of four re-uploading layers had no effect on any output
  probability. Likewise, the random phases applied just before detection
  have no effect. Nothing warned us. The circuit ran, trained, and produced
  plausible but different results (3 effective encoding layers instead of
  4). Prepending a fixed 50:50 beam-splitter brickwork fixed it.
- **Evidence:**
  - Test
    `tests/test_photonic.py::test_first_layer_encoding_effect_depends_on_initial_mixing`.
    Perturbing the layer-0 data phases changes the probabilities by
    < 1e-12 without an initial mixing layer, and by > 1e-3 with one.
  - An independent review measured changes of 1.4e-16 (layer-0 data
    phases), 1.9e-16 (layer-0 fixed phases) and 8e-17 (last-layer phases).
  - Scientific impact, 15-run sweeps, uncorrected (M1/M2) vs corrected
    (M3/M4) circuit (`results/merlin_tolerance_check.md`): Matérn test RMSE
    0.106 → 0.087 (authors' protocol) and 0.099 → 0.084 (converged
    optimizer).
  - MerLin 0.4.1, Perceval 1.2.4.
- **Why it matters:** angle or phase encodings are the default way to load
  classical data, and users porting qubit circuits (which start from |0⟩ or
  |+⟩) can easily put the first encoding before any interferometer. The
  failure is silent and changes expressivity.
- **Suggested action:**
  - Add a cookbook or user-guide warning: phase encodings must follow at
    least one mixing element, and phases right before detection are
    irrelevant.
  - Optionally, have `QuantumLayer` (or `CircuitBuilder`) warn when an
    input parameter appears only in phase shifters that act on an
    unmixed Fock input state, or that are followed only by detection.

### Medium — Measurement / ComputationSpace: `UNBUNCHED` is a postselection, but this and its success probability are undocumented

- **Type:** missing documentation / improvement
- **What happened:** with `MeasurementStrategy.probs(ComputationSpace.UNBUNCHED)`
  the output sums to 1 over the C(m, n) collision-free patterns. We
  verified it equals the `FOCK` distribution restricted to collision-free
  outcomes and renormalized. In other words, it is threshold detection
  with **postselection on n distinct clicks**, not a model of threshold
  detectors that keeps all events. The postselection success probability
  is not returned. We needed a second `FOCK`-space layer to compute it, and
  it is substantial: on average 0.44-0.48, as low as 0.18 for some inputs.
- **Evidence:**
  - Probe: max |UNBUNCHED − renormalized FOCK| = 0.0 (`LOG.md`, "Phase 4
    Photonic Assessment").
  - Test
    `tests/test_photonic.py::test_merlin_matches_independent_perceval_simulation[threshold]`
    agrees with Perceval SLOS, restricted and renormalized, to 1e-10.
  - An independent permanent-based check agrees to 1.3e-16.
  - Postselection rates are logged in every photonic `metrics.json`
    (`photonic.postselection_probability_mean` and `_min`).
- **Why it matters:**
  - Hardware-aware reporting needs the raw-event cost (here about 2.2× on
    average and 5.6× in the worst case).
  - Users comparing "N shots" across spaces may be misled.
  - The cookbook currently describes `UNBUNCHED` as "threshold detectors,
    hardware-aligned" without mentioning postselection.
- **Suggested action:**
  - State explicitly in the docs and cookbook that `UNBUNCHED` is
    postselected on collision-free outcomes.
  - Expose the postselection (acceptance) probability, for example as an
    optional output or attribute, so users do not need a parallel `FOCK`
    simulation.
  - Clarify what `shots=` means in `UNBUNCHED` (kept events vs raw
    events).

### Low — QuantumLayer inputs: ordering rule for `input_parameters` prefixes is not documented

- **Type:** missing documentation
- **What happened:** with `input_parameters=["in"]` and 48 phase
  parameters, we could not find a stated rule for how the input tensor's
  columns map to the parameters (name-sorted vs circuit order). To be
  safe we zero-padded the names (`in0000` … `in0047`) so that both
  orderings coincide. We did not observe a failure; this is a
  documentation gap, not a confirmed issue.
- **Evidence:** `lib/photonic_features.py::_input_name` (comment). The
  correctness of our ordering is confirmed by the test that matches
  Perceval SLOS with numerically substituted parameters
  (`tests/test_photonic.py::test_merlin_matches_independent_perceval_simulation`).
- **Why it matters:** with unpadded names (`in2` vs `in10`), a name-sorted
  mapping would silently scramble the inputs.
- **Suggested action:** document the mapping rule in the `QuantumLayer`
  docstring, and ideally expose the resolved parameter order (e.g.
  `layer.input_parameter_names`).

### Low — Sampling: no documented seed control for `layer(x, shots=N)`

- **Type:** missing documentation
- **What happened:** we needed reproducible finite-shot features for
  128 × 500 (draw, input) pairs. The docs do not say how the sampling in
  `layer(x, shots=N)` is seeded (e.g. through `torch.manual_seed` or a
  generator argument). We therefore sampled with our own seeded NumPy
  `multinomial` from MerLin's analytic probabilities. We did not test
  whether `torch.manual_seed` controls MerLin's sampler; this is
  unverified.
- **Evidence:** `lib/photonic_features.py::sample_frequencies` and the
  `shot_sampling` field in each photonic `metrics.json`; API probe in
  `LOG.md`, "Phase 4 Photonic Assessment".
- **Why it matters:** reproducible shot-noise experiments are standard in
  reproductions, and users may otherwise implement their own sampling
  (as we did).
- **Suggested action:** document the RNG used by `shots=` sampling and
  accept an explicit `generator` or `seed` argument.
