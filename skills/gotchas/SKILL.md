---
name: gotchas
description: >
  LightStim known pitfalls, bugs, and non-obvious API contracts. Read this
  whenever a build produces unexpected results, the tracker raises a RuntimeError,
  noiseless-check detections appear, LER is suspiciously high, or the user reports
  "it was working and now it isn't." Each entry states the symptom, the root cause,
  and the fix pattern.
user-invocable: false
---

# LightStim Gotchas & Common Pitfalls

A living catalogue of non-obvious bugs and API contracts discovered during
development. When something goes wrong, scan the symptom descriptions first —
most bugs fall into one of these categories.

---

## 1. Coupler Pitfalls

### 1-A  `system.data_indices` includes coupler qubits after `register_coupler()`

**Symptom**: Tracker raises `RuntimeError: Logical Count Mismatch! Expected N, Found M`
where M ≫ N, often during the *first* `apply_syndrome_extraction` after batch
initialization.

**Root cause**: `register_coupler()` allocates new data qubits (the ancilla corridor
between patches) and adds them to `system.data_indices` and `system.index_to_owner_map`
(owner = the coupler name, e.g. `'coupler_23'`). If you batch-initialize all of
`system.data_indices` you also initialize the coupler's corridor qubits, injecting
M − N phantom "logical" rows into the tracker.

**Fix**: Filter by owner when doing batch init:

```python
coupler_names = set(system.coupler_patches.keys())
patch_data = {q: 'X' for q in system.data_indices
              if system.index_to_owner_map.get(q) not in coupler_names}
builder.initialize(init_dict=patch_data, n=system.num_qubits)
```

Or equivalently, filter for a known set of patch names:

```python
valid_patches = {'W1', 'W2', 'W3', 'W4', 'W5'}
patch_data = {q: 'X' for q in system.data_indices
              if system.index_to_owner_map.get(q) in valid_patches}
```

The sequential per-patch init pattern (`wd = {q: 'X' for q in system.data_indices if system.index_to_owner_map[q] == patch_name}`) avoids this naturally, which is why it works even when couplers are pre-registered.

---

### 1-B  Coupler data qubits must be measured *immediately* after `deactivate_coupler()`

**Symptom**: Noiseless check passes but LER is much higher than theory predicts.
Detslice diagrams show coupler data qubits sitting idle for one or more SE rounds
after the coupler is deactivated.

**Root cause**: After `builder.deactivate_coupler(coupler_name)`, the coupler's
corridor data qubits are no longer part of the active stabilizer set. They sit idle
during subsequent SE rounds, accumulating depolarizing errors with no stabilizer
measurement to catch them. Those errors propagate back to the patches when the
qubits are finally measured.

**Fix**: Measure coupler data qubits immediately at deactivation:

```python
builder.activate_coupler("coupler_23")
cp = system.coupler_patches["coupler_23"]
cp_data = sorted(system.local_to_global_map["coupler_23"][q] for q in cp.data_indices)

builder.initialize(init_dict={q: 'X' for q in cp_data}, n=system.num_qubits)
builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds_ls)

# Measure coupler data qubits HERE — not in the final readout
builder.apply_data_readout(final_measurements={q: 'X' for q in cp_data})
builder.deactivate_coupler("coupler_23")
```

Reference implementations: `lightstim/protocols/cnot_ls.py`,
`lightstim/protocols/bell_teleportation.py` (BellTeleportZZLS / BellTeleportXXLS).

---

### 1-C  `activate_coupler()` must come before `initialize()` for coupler qubits

**Symptom**: `builder.initialize` raises a KeyError or the coupler qubits don't
appear in `system.data_indices`.

**Root cause**: The coupler's global qubit indices are only allocated and added to
the system maps after `activate_coupler()` (or after `register_coupler()` for
pre-registered couplers). Calling `initialize` on coupler qubit indices before
activation references indices that don't exist yet.

**Fix**: Always `activate_coupler()` → `initialize(coupler qubits)` → `apply_syndrome_extraction` → `apply_data_readout` → `deactivate_coupler()`.

---

### 1-D  Define-by-run couplers: expand tracker and write coordinates

**Symptom**: Tracker array-index error or missing qubit coordinates when couplers
are registered inside a loop (define-by-run pattern).

**Root cause**: When `register_coupler()` is called after the tracker and builder
are already constructed, the tracker's internal matrices are too small and no
`QUBIT_COORDS` instructions are emitted for the new qubits.

**Fix**:

```python
system.register_coupler(...)
n_new = system.num_qubits
if n_new > tracker.num_qubits:
    tracker.expand(n_new - tracker.num_qubits)
builder.write_coordinates()   # emits QUBIT_COORDS for new indices only
```

---

## 2. Circuit Construction Pitfalls

### 2-A  `apply_unitary_block()` auto-inserts a TICK; `initialize()` does not

**Symptom**: Unexpected TICK boundaries in the stim circuit; diagram looks wrong.

**Root cause**:
- `builder.apply_unitary_block(circuit)` prepends a `TICK` if the last instruction
  in the builder's circuit is not already a `TICK`.
- `builder.initialize(init_dict, n)` does NOT add any TICK — it emits RX/R/RY
  instructions directly.

This means: if you call `initialize` followed by another `initialize`, no TICK
separates them. If you call `initialize` then `apply_unitary_block`, one TICK is
inserted between them automatically.

---

### 2-B  Patch objects must be stored from `system.add_patch()`, not retrieved by name

**Symptom**: `No LogicalOpSet registered for str` or `KeyError` when calling
`executor.apply_logical_operation(op, [patch])`.

**Root cause**: `apply_logical_operation` expects the actual `QECPatch` object, not
a string name. There is no `system.patch_objects["name"]` accessor.

**Fix**: Store the return value at `add_patch()` time:

```python
p1 = system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch1', offset=(0, 0))
p2 = system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch2', offset=(dx, 0))
# Later:
executor.apply_logical_operation("transversal_cnot", [p1, p2])
```

---

### 2-C  `stabilizer_canonicalization()` must be called before SE if logicals are set manually

**Symptom**: Tracker finds wrong number of logicals; OBSERVABLE_INCLUDE instructions
reference measurement indices that don't correspond to the intended logical.

**Root cause**: After encoding (initialization + unitary gates), the tracker's
internal basis for logicals may not match the code's canonical logical operators.
If you call `logical_canonicalization()` with custom logical vectors, you must also
call `stabilizer_canonicalization()` first so that the stabilizer and logical
subspaces are properly separated.

---

### 2-D  SE `rounds` argument: `rounds_init` vs `rounds_gate` distinction (TG distillation)

**Symptom**: Simulated LER doesn't match paper results for TG 7-to-1 distillation.

**Root cause**: The TG protocol has two distinct SE phases with different optimal
round counts:
- `rounds_init` (state preparation SE): paper value = `d`. Controls how well the
  initial logical |Y⟩ states are prepared.
- `rounds_gate` (post-gate SE): paper value = `1`. Applied after each transversal
  CNOT layer; using `d` rounds here wastes time and biases error statistics.

**Fix**: Use `build_distillation_circuit(d, rounds_init=d, rounds_gate=1)` from
`lightstim.protocols.tg_distillation`. Never conflate the two parameters.

---

## 3. Simulation & Decoding Pitfalls

### 3-A  CrossLS / PQRM circuits cannot use PyMatching — use MWPF

**Symptom**: PyMatching raises a hyperedge error or produces incorrect LER when
decoding a CrossLS or PQRM circuit.

**Root cause**: PQRM X-stabilizers are high-weight (weight 7–15) and cannot be
decomposed into 2-body edges by `decompose_errors=True`. PyMatching (MWPM) is
fundamentally restricted to edges (weight ≤ 2). MWPF handles arbitrary hyperedges
natively and is the correct decoder for these circuits.

**Fix**:

```python
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig('mwpf'),   # not 'pymatching'
    max_shots=200_000, max_errors=50,
    batch_size=10_000, num_workers=4,
)
```

This applies to: `CrossLSExperiment`, any circuit built around `PQRMCode`,
and any custom circuit whose DEM contains hyperedges with weight > 2.

---

### 3-B  PyMatching requires `decompose_errors=True` in the DEM

**Symptom**: PyMatching raises an error about hyperedges, or decoding performance
is very poor.

**Root cause**: PyMatching (MWPM) only handles edges (weight-1 and weight-2 errors).
Multi-qubit faults produce hyperedges in the raw DEM. `decompose_errors=True` splits
hyperedges into their constituent edges so MWPM can process the graph.

**Fix**:

```python
dem = circuit.detector_error_model(decompose_errors=True)
# SimulationPipeline does this internally when decoder='pymatching'
```

When using `SimulationPipeline` with `decoder_config=DecoderConfig('pymatching')`,
decomposition is applied automatically — you don't need to do it manually.

---

### 3-B  Always run noiseless check before simulation

**Symptom**: LER ≈ 50% or simulation hangs because every shot has detection events.

**Root cause**: A silent build bug (wrong initialization, missing SE round, wrong
observable) causes detection events even without noise.

**Fix**: Before adding any noise, confirm zero detection events on the clean circuit:

```python
dets, obs = circuit.compile_detector_sampler().sample(100, separate_observables=True)
assert not np.any(dets), "Noiseless circuit has detection events — build bug!"
assert not np.any(obs), "Noiseless circuit has observable flips — build bug!"
```

---

### 3-C  GPU decoder: use custom simulation loop, not `sinter`

**Symptom**: GPU decoder (`nv-qldpc-decoder` / `cudaq_qec`) is slower than CPU or
produces wrong results when called through `sinter.collect`.

**Root cause**: `sinter` launches one Python process per shot batch, incurring GPU
kernel launch overhead for every batch. The GPU decoder amortizes this over large
batches and requires a persistent process.

**Fix**: Use the `SimulationPipeline` with `backend='gpu'` and large `batch_size`
(≥ 50 000). Do not use `sinter` for GPU decoding.

---

### 3-D  Post-selection logic: `post_select_corrected_observable_indices` vs `post_select_detector_coords`

**Symptom**: Post-selection discards too many or too few shots; distillation
post-selection rate is nonsensical.

**Root cause**: There are two independent post-selection mechanisms:
- `post_select_corrected_observable_indices`: post-selects after decoding, on the
  *decoder-corrected* observable bits. Used for magic state distillation.
- `post_select_detector_coords`: post-selects on raw detection events at specific
  space-time coordinates (e.g. state injection ancilla).

Confusing these (or using raw observables instead of decoder-corrected) gives wrong
post-selection rates.

---

## 4. Environment & Tooling Pitfalls

### 4-A  Always use `venv/bin/python`, never system Python

**Symptom**: `cudaq_qec` import fails, or GPU decoder silently falls back to a
random-guess decoder giving LER ≈ 50%.

**Root cause**: `cudaq_qec` and other GPU-side packages are installed in the
project's virtual environment, not the system Python. The system Python doesn't
have them and may silently import a stub or nothing at all.

**Fix**: Every Python invocation in this repo must use `venv/bin/python`:

```bash
venv/bin/python benchmarks/...
venv/bin/python -m pytest tests/
```

Never use `python`, `python3`, or `pip` without the `venv/bin/` prefix.

---

### 4-B  Check GPU availability before launching GPU experiments

**Symptom**: GPU experiment hangs or errors because all GPUs are in use.

**Fix**: Run `nvidia-smi` and confirm at least one GPU is free before launching.
Ask the user to confirm `num_workers` matches available GPU memory.

```bash
nvidia-smi
```

---

### 4-C  CPU benchmark experiments: always wrap with timeout and limit concurrency

**Symptom**: Server runs at 100% CPU for hours; other users are affected.

**Root cause**: Multi-process CPU experiments with many workers × large shot counts
can saturate all cores for days.

**Fix**: Wrap experiments with `timeout`, limit workers to ≤ 8 for shared servers,
and ask the user before launching any experiment expected to take > 1 hour.

```bash
timeout 3600 venv/bin/python benchmarks/... --num-workers 8
```

---

## 5. Tracker Internal Invariants

These are for debugging tracker `RuntimeError` messages.

| Error message | Likely cause |
|---|---|
| `Logical Count Mismatch! Expected N, Found M` with M ≫ N | Coupler data qubits accidentally initialized (see 1-A) |
| `Logical Count Mismatch! Expected N, Found M` with M = 0 | Wrong SE circuit used (ancilla indices don't match the system) or coupler not activated before SE |
| `Measurement X commutes with all rows … is linearly independent` | A data qubit was not initialized before SE (row missing from tableau) |
| `Mirror qubit at … not found` | `fold_transversal_s` called on a non-square patch |
| `Circuit chunk involves qubit K, exceeding system size N` | `apply_unitary_block` called with a circuit that references qubits outside the system |

---

## 6. Observable & Detector Pitfalls

### 6-A  Multi-patch distillation: use `identify_distillation_observables()` to find target/PS observable indices

**Symptom**: Simulated LER is for the wrong logical qubit; post-selection is on the
output qubit instead of the input magic states.

**Root cause**: With 4–8 observables in a distillation circuit, the observable
ordering depends on which patches were added and measured first. Never hard-code
observable index 0 as the target.

**Fix**:

```python
from lightstim.simulation.observable_analysis import (
    build_obs_patch_matrix, identify_distillation_observables
)
matrix, patch_names = build_obs_patch_matrix(circuit, system)
T, target_obs, ps_obs = identify_distillation_observables(matrix, patch_names, ["W4"])
```

This returns the correct `target_obs` and `ps_obs` indices regardless of circuit
construction order.

---

## 7. Decoder Selection Pitfalls

### 7-A  Check for hyperedges before choosing a decoder

**Symptom**: PyMatching reports 0 errors (silently wrong) or raises an error on
circuits with hyperedges (fold-transversal gates, transversal CNOT, TG Bell teleportation,
CrossLS, PQRM). PyMatching is fundamentally restricted to edges (weight ≤ 2).

**Root cause**: Fold-transversal H/S gates (via SWAP + CZ operations), transversal CNOT
across patches, and multi-patch syndrome correlations produce weight > 2 hyperedges in
the DEM. PyMatching silently decomposes or drops these, giving wrong LER.

**Detect hyperedges before choosing a decoder**:

```python
dem = noisy_circuit.detector_error_model(decompose_errors=False)
max_weight = max(
    len([t for t in inst.targets_copy() if t.is_relative_detector_id()])
    for inst in dem.flattened() if inst.type == 'error'
)
# max_weight > 2 → hyperedges present → do NOT use PyMatching
print(f"Max DEM edge weight: {max_weight}")
```

**Decoder selection by circuit type** (empirical benchmarks):

| Circuit type | d=3,5 | d=7, p ≤ 2e-3 | d=7, p ≥ 5e-3 |
|---|---|---|---|
| Memory / Lattice Surgery (no hyperedges) | `pymatching` | `pymatching` | `pymatching` |
| Fold-transversal H/S (hyperedges) | `bposd` CPU | `nv-qldpc-decoder` GPU | `nv-qldpc-decoder` GPU |
| Transversal CNOT (hyperedges) | `bposd` or GPU | `nv-qldpc-decoder` GPU | `nv-qldpc-decoder` GPU |
| TG Bell teleportation (hyperedges) | `bposd` CPU | `mwpf` CPU | `nv-qldpc-decoder` GPU |
| CrossLS / PQRM (hyperedges) | `mwpf` | `mwpf` | `nv-qldpc-decoder` GPU |

**num_workers by decoder**:

| Decoder | Recommended num_workers |
|---|---|
| `pymatching` | 8–16 |
| `bposd` | 8–16 |
| `mwpf` | 8 |
| `nv-qldpc-decoder` | 1 (GPU manages parallelism internally) |

---

### 7-B  CPU decoders hang at high p and large d

**Symptom**: `bposd` or `mwpf` decoding stalls indefinitely on a single shot. The
pipeline appears frozen with no output.

**Root cause**: Dense syndromes (≥ 200 detection events/shot for d=7 TG circuits)
create "hard instances" where BP fails to converge and OSD explores exponentially
many configurations. One shot can take hours.

**Concrete case**: d=7, p=5e-3, TG Bell teleportation — DEM has 48,249 error
mechanisms, 72% hyperedges. MWPF decoded first ~80k shots (21s), then hung
indefinitely on shot 80,001.

**Fix**: Switch to `nv-qldpc-decoder` GPU when CPU decoders hang. GPU parallelism
prevents individual-shot blowup. CUDA_VISIBLE_DEVICES controls which GPU to use:

```bash
nvidia-smi                                           # check free GPUs first
CUDA_VISIBLE_DEVICES=2 venv/bin/python run_tg.py    # use GPU 2
```

---

## 8. Multi-Patch Coupler: Perpendicular Boundary Limitation

### 8-A  `UnrotatedMultiPatchCoupler` only supports side-connected patches

**Symptom**: `circuit.detector_error_model()` raises an error about non-deterministic
detectors, e.g.:

```
The circuit contains non-deterministic detectors.
R on qubit 132 [coords (9, 8)] anti-commuted with detector D223
```

**Root cause**: The 6-tick interleaved SE schedule (Li 2014, arXiv:1410.7808) applies
X and Z syndrome CNOTs simultaneously at ticks 3-4. At a **perpendicular** boundary
(patch boundary perpendicular to the corridor axis), a data qubit is shared by both
a corridor X-syndrome and an endpoint Z-syndrome in the same tick. The tracker's
back-propagated Pauli includes a syndrome-to-syndrome cross-talk term that makes the
corresponding detector non-deterministic.

At **parallel** boundaries (patch boundary parallel to the corridor), X and Z CNOT
directions are fully separated in time, so no cross-talk occurs.

**What works** (N patches all side-connected, verified):
- 2-patch ZZ, 3-patch ZZZ, 4-patch ZZZZ
- 5-patch selective Z₄ with idle patch
- Mixed distances (d=3 + d=4)

**What fails**: Any patch whose boundary is perpendicular to the corridor axis
(i.e., centered below or above the corridor, like an "endpoint" patch).

**Current workaround**: Place all interacting patches as side patches (left/right of
the vertical corridor). For layouts that would naturally use an endpoint position,
shift the patch to a side position:

```python
# Fails: endpoint position (perpendicular boundary)
system.add_patch(p5, name='p5', offset=(4, 16))   # centered below corridor

# Works: side position (parallel boundary)
system.add_patch(p5, name='p5', offset=(-2, 16))  # left side of corridor
```

**Known fix paths** (not yet implemented):
- **8-tick fully separated schedule**: separate X and Z CNOT rounds completely.
  No cross-talk at any orientation. Cost: +2 ticks/round (33% more depth).
- **ZX interleaving schedule** (Gidney et al. arXiv:2603.01628): 4-tick minimum,
  handles arbitrary boundary orientations.
- **Diagonal schedule** (Fowler & Kishony arXiv:2602.09099): period-7 schedule
  that eliminates hook errors at arbitrary boundaries.

---

## 9. Open Bugs

### 9-A  Sequential coupler reuse blocks detectors (tracker stabilizer transition)

**Status**: Open. Blocks multi-cycle lattice surgery distillation protocols
(e.g., Steane 4 × ZZZZ sequential measurements).

**Symptom**: For 4 sequential coupler cycles (each activating a different patch
subset), the tracker's logical count explodes after the first cycle transitions to
the second. `RuntimeError: Logical Count Mismatch` during the second
`apply_syndrome_extraction`.

**Root cause**: `SyndromeTracker.process_mid_measurement` assumes a stable stabilizer
structure. After `deactivate_coupler()` + `activate_coupler()` for a new patch
subset, the stabilizer tableau rows from cycle k no longer decompose against cycle
k+1's SE measurements. Old rows become independent → classified as logicals →
logical count explodes.

**What works**:
- Single measurement cycle: DEM OK
- Multiple cycles with `if_detector=False`: circuit compiles (no detector tracking)
- 4 cycles with Z-init (trivial): DEM OK

**What doesn't work**: 4 cycles with Y/X init + proper detector tracking.

**Potential solutions** (not yet implemented):
1. **Stabilizer canonicalization mid-circuit**: new tracker method that re-aligns the
   tableau with the current active stabilizers mid-circuit, preserving logicals.
2. **Tracker reset at cycle boundaries**: after each deactivation, reset stabilizer
   rows to current active stabilizers (loses historical records, starts fresh).
3. **Unified coupler**: register one coupler with all patches upfront, selectively
   mask/unmask boundary stabilizers per cycle (avoids stabilizer structure changes).

**Related files**: `src/ir/tracker.py` (`process_mid_measurement`,
`stabilizer_canonicalization`), `src/ir/builder.py` (`apply_data_readout`),
`tests/test_back_propagated_pauli.py`.
