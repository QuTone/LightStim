---
name: builder-tracker-api
description: >
  Direct use of CircuitBuilder and SyndromeTracker to construct a custom QEC
  circuit from scratch — without a pre-built experiment class. Use this skill
  whenever the user wants to: implement a novel protocol from a paper, chain
  multiple logical operations with full control over each step, build a circuit
  that no existing XXXExperiment class covers, or understand how detectors and
  observables are generated internally.
user-invocable: true
---

# CircuitBuilder + SyndromeTracker Direct API

All pre-built experiment classes (`MemoryExperiment`, `CNOTLSExperiment`, etc.) are
thin wrappers around this layer. Use this skill when you need to go below them.

## The mental model

```
QECPatch(s) → QECSystem → SyndromeTracker + CircuitBuilder → stim.Circuit
                                          ↑
                             Auto-generates DETECTOR / OBSERVABLE_INCLUDE
```

`CircuitBuilder` emits circuit instructions. `SyndromeTracker` maintains the symplectic
tableau (stabilizers + logicals in GF(2)) and decides, at each measurement, whether a
detection event is deterministic (→ DETECTOR) or consumes a logical degree of freedom
(→ OBSERVABLE_INCLUDE). You never write DETECTOR by hand.

## Minimum viable setup

```python
import stim
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock
)
from lightstim.noise.config import NoiseConfig

# 1. Build patches and system
system = QECSystem()
p = system.add_patch(UnrotatedSurfaceCode(distance=3), name="p", offset=(0, 0))

# 2. Build tracker and builder
tracker = SyndromeTracker(
    num_qubits=system.num_qubits,
    expected_num_logicals=system.num_logicals,
)
builder = CircuitBuilder(tracker, system)

# 3. Build extraction block (one SE round)
se = UnrotatedSurfaceCodeExtractionBlock(system)

# 4. Emit circuit
builder.write_coordinates()
builder.initialize({q: "X" for q in system.data_indices}, n=system.num_qubits)
builder.apply_syndrome_extraction(se.circuit, rounds=3)
builder.apply_data_readout({q: "X" for q in system.data_indices})

# 5. Noise
noisy = builder.build_noisy_circuit(
    NoiseConfig(p_2q=1e-3, p_meas=1e-3), noise_model="circuit_level"
)
```

## Call sequence and invariants

```
write_coordinates()
initialize(init_dict, n)                  ← no TICK added automatically
[stabilizer_canonicalization()]           ← only after encoding gates
apply_syndrome_extraction(chunk, rounds)  ← adds SHIFT_COORDS + chunk (+ REPEAT)
[apply_unitary_block(gate_circuit)]       ← adds TICK + gate
[activate_coupler / deactivate_coupler]   ← changes active stabilizer set
apply_data_readout(final_measurements)    ← resolves remaining rows → DET/OBS
noisy = build_noisy_circuit(noise, model)
```

**Invariants you must preserve:**
1. Every qubit in `init_dict` must have a global index in `system`.
2. `circuit_chunk` passed to `apply_syndrome_extraction` must end with `M` or `MX` on
   syndrome qubits. The last instruction is the measurement that the tracker analyzes.
3. `apply_unitary_block` prepends a `TICK` if the last instruction isn't one.
   `initialize` does NOT prepend a `TICK`. Plan your TICK boundaries accordingly.
4. Between `activate_coupler` and `deactivate_coupler`, the active stabilizer set
   changes — the next `apply_syndrome_extraction` uses the coupler's joint stabilizers.

## Multi-phase protocol pattern

For protocols with distinct phases (init → gate → gate → readout):

```python
# Phase A: initialize control and target patches
builder.write_coordinates()
ctrl_data = {q: "Z" for q in system.data_indices if system.index_to_owner_map[q] == "ctrl"}
tgt_data  = {q: "X" for q in system.data_indices if system.index_to_owner_map[q] == "tgt"}
builder.initialize({**ctrl_data, **tgt_data}, n=system.num_qubits)

# Phase B: SE rounds before gate
builder.apply_syndrome_extraction(se.circuit, rounds=d)

# Phase C: transversal gate (updates symplectic tableau for both patches)
gate_circuit = build_transversal_cx(system, ctrl_patch, tgt_patch)
builder.apply_unitary_block(gate_circuit)

# Phase D: SE rounds after gate
builder.apply_syndrome_extraction(se.circuit, rounds=d)

# Phase E: readout in chosen basis
builder.apply_data_readout({q: "Z" for q in system.data_indices})
```

## How the tracker decides: DETECTOR vs OBSERVABLE_INCLUDE

At each syndrome measurement, the tracker back-propagates the measured Pauli to the
data qubit layer (via tableau inversion), then checks the current full tableau:

- **Anti-commutes** with an existing row → that row is "consumed" by the measurement.
  The measurement *updates* the stabilizer/logical; no DETECTOR emitted.
- **Commutes** with all rows AND is a linear combination of *stabilizers only* →
  deterministic → **DETECTOR** emitted.
- **Commutes** with all rows AND requires a *logical* component to decompose →
  can't be a detector (logical DOF still free) → flagged for observable construction later.
- **Commutes but linearly independent** → `RuntimeError: linearly independent` →
  means a qubit was not initialized.

At data readout (`process_data_measurement`):
- Stabilizer rows that are fully determined by the measurements → **DETECTOR**
- Logical rows (and gauge-measurement rows with logical components) → **OBSERVABLE_INCLUDE**

## Z-only / X-only memory trick

For Z-memory experiments you only need Z-basis detectors (X-ancilla detectors are
irrelevant to the logical Z observable). Pass `z_only=True` to suppress X-ancilla
DETECTORs from the DEM while still updating the tableau correctly:

```python
builder.apply_syndrome_extraction(se.circuit, rounds=d, z_only=True)
builder.apply_data_readout({q: "Z" for q in system.data_indices}, z_only=True)
```

## Define-by-run: adding patches after tracker/builder construction

For protocols that add patches dynamically (e.g. growing a lattice):

```python
system.register_tracker(tracker)   # must be called before add_patch
system.register_builder(builder)   # must be called before add_patch

# Later, during circuit construction:
new_patch = system.add_patch(SomePatch(distance=3), name="p2", offset=(dx, dy))
# → tracker.expand() called automatically
# → QUBIT_COORDS inserted automatically at correct position in circuit

# Now initialize the new patch's qubits:
new_data = {q: "X" for q in new_patch.data_indices}
builder.initialize(new_data, n=system.num_qubits)
```

## Tracker case studies: expected behavior for common protocols

Use these to verify the tracker is doing the right thing. Print
`tracker.logicals.count` after each `apply_syndrome_extraction` call to debug.

### Surface-Surface ZZ measurement, both patches initialized |+⟩ (X basis)

Logicals: X₁ on patch1, X₂ on patch2. Coupler: Z-stabilizers.

Both X logicals **anti-commute** with coupler Z-stabs (X·Z anti-commute).
→ Case A: both logicals are absorbed (`expected_num_logicals` drops 2→0).
→ No gauge measurements (no `stabilizer_with_logical_components`).
→ After SE: `tracker.logicals.count == 0`. ✓

The joint ZZ observable is automatically emitted as `OBSERVABLE_INCLUDE` at final readout.

### Surface-Surface ZZ measurement, both patches initialized |0⟩ (Z basis)

Logicals: Z₁ on patch1, Z₂ on patch2. Coupler: Z-stabilizers.

Both Z logicals **commute** with coupler Z-stabs (Z·Z commute).
→ No Case A for logicals.
→ Some measurements decompose to include both Z₁ and Z₂ simultaneously (gauge).
→ `gauge_logical_vectors = [[1,1], ...]`, GF(2) rank = 1 → 1 logical DOF consumed.
→ After SE: `tracker.logicals.count == 1` (the merged Z₁⊗Z₂ product logical remains). ✓

### CrossLS: PQRM patch (Z state) + Surface patch (|+⟩)

Logicals: Z_PQRM on {25,26,27}, X_surface on {2,12,22}. Coupler: Z-stabs.

X_surface **anti-commutes** with coupler Z-stab on qubit 2.
→ Case A: X_surface absorbed (`expected` 2→1).

Z_PQRM **commutes** with ALL measurements (Z·Z commute, no X overlap).
→ 8 measurements each decompose to include Z_PQRM.
→ `gauge_logical_vectors = [[1],[1],...,[1]]` (8 rows), GF(2) rank = 1 → 1 DOF consumed.
→ After SE: `tracker.logicals.count == 0`. ✓

The PQRM logical is measured via the gauge measurement span, correctly.

---

## Noiseless sanity check (always do this first)

```python
# Before adding noise, verify zero detection events and zero observable flips:
sampler = builder.circuit.compile_detector_sampler()
dets, obs = sampler.sample(shots=100, separate_observables=True)
assert not dets.any(),  f"Noiseless circuit has detection events — build bug!"
assert not obs.any(),   f"Noiseless circuit has observable flips — build bug!"
```

## Working examples

- `lightstim/protocols/two_patch_ls.py` — two-patch ZZ/XX lattice surgery, full pipeline
- `lightstim/protocols/cnot_ls.py` — three-patch CNOT via lattice surgery
- `notebooks/LogicalOps/multi_patch_LS.ipynb` — N-patch Z-product measurement with `build_zz_circuit()` helper
