# LightStim IR Layer API Reference

The IR layer (`lightstim/ir/`) is the engine of LightStim. It provides three interlocking
abstractions: **QECPatch** (code geometry + physics), **QECSystem** (global canvas), and
**CircuitBuilder + SyndromeTracker** (circuit construction + automatic detector generation).

---

## 1. `QECPatch` — Code Definition

**Module:** `lightstim.ir.qec_patch`

Abstract base class for all QEC codes. Subclass it to define a new code family.

### Key Attributes

| Attribute | Type | Description |
|---|---|---|
| `qubit_coords` | `Dict[int, (float,float)]` | `uid → (x, y)` for all qubits |
| `index_map` | `Dict[(float,float), int]` | `(x, y) → uid` reverse lookup |
| `data_indices` | `Set[int]` | UIDs of data qubits |
| `syndrome_indices` | `Set[int]` | UIDs of all syndrome (ancilla) qubits |
| `syndrome_indices_x` | `Set[int]` | UIDs of X-syndrome qubits |
| `syndrome_indices_z` | `Set[int]` | UIDs of Z-syndrome qubits |
| `stabilizers` | `List[Dict]` | See stabilizer record format below |
| `logical_ops` | `List[Dict]` | See logical op record format below |
| `num_logicals` | `int` | Number of logical qubits (must set in `build()`) |

### Stabilizer record format

```python
{
    "pauli":       {uid: "X"/"Z"/"Y", ...},  # data qubit support
    "type":        "X" | "Z" | "Mixed",
    "data_indices": [uid, ...],              # same as pauli keys, sorted
    "syn_coord":   (float, float),           # coordinate of ancilla
    "syn_idx":     int,                      # uid of ancilla
}
```

### Methods to call in `build()`

```python
uid = self.add_qubit(x, y, role)
# role: 'data' | 'syndrome_x' | 'syndrome_z'
# Returns the assigned integer uid.

self.create_stim_stabilizer(
    target_dict,     # {(x, y): "X"/"Z"/"Y", ...}  — data qubit coords + Pauli type
    syn_coord,       # (x, y) of the ancilla
    type,            # "X" or "Z"
)
# Appends one entry to self.stabilizers.

self.create_stim_logical(
    target_dict,     # {(x, y): "X"/"Z"/"Y", ...}
    op_type,         # "X" or "Z"
)
# Appends one entry to self.logical_ops.
```

### Abstract methods to implement

```python
def _process_params(self):
    # Validate self.params (called by __init__ before build).
    ...

def build(self):
    # Register all qubits, stabilizers, logical_ops, set num_logicals.
    ...
```

### Geometry helpers

```python
patch.shift_coords(dx, dy)         # Translate all qubit coords in place.
patch.transpose_coords()           # Reflect across y=x (swaps x↔y).
patch.rotate_coords(theta, center) # Rotate by theta radians (CCW).
patch._get_bounds()                # → (min_x, max_x, min_y, max_y)
patch.get_info()                   # → dict of code metadata
```

---

## 2. `QECSystem` — Global Canvas

**Module:** `lightstim.ir.qec_system`

Aggregates multiple patches into one globally indexed system. Manages stabilizer
activation/deactivation and coupler lifecycle.

### Construction

```python
from lightstim.ir.qec_system import QECSystem

system = QECSystem()
```

### Adding patches

```python
global_patch = system.add_patch(
    patch,          # QECPatch instance
    offset=(0, 0),  # (dx, dy) translation applied before registration
    name="p1",      # Unique string key. Auto-generated if None.
    is_active=True, # If True, patch's stabilizers immediately go active.
)
# Returns: deep copy of patch with all indices converted to global.
# Side effect: allocates global qubit indices; expands tracker/builder if registered.
```

### Key attributes (read after all patches added)

| Attribute | Description |
|---|---|
| `system.num_qubits` | Total qubit count (property) |
| `system.num_logicals` | Total logical qubit count |
| `system.data_indices` | Set of all data qubit global indices |
| `system.syndrome_indices` | Set of all syndrome qubit global indices |
| `system.qubit_coords` | `{global_idx: (x, y)}` |
| `system.index_map` | `{(x, y): global_idx}` |
| `system.index_to_owner_map` | `{global_idx: patch_name}` |
| `system.local_to_global_map` | `{patch_name: {local_idx: global_idx}}` |
| `system.stabilizers` | Master list of all stabilizer records (global indices) |
| `system.active_stabilizer_indices` | Set of stabilizer UIDs currently ON |

### Active stabilizer properties

```python
system.active_stabilizers         # List of active stabilizer records
system.active_stabilizers_x       # X-type only
system.active_stabilizers_z       # Z-type only
system.active_syndrome_indices    # Global indices of active syndrome qubits
system.active_syndrome_indices_x  # X-syndrome only
system.active_syndrome_indices_z  # Z-syndrome only
```

### Define-by-run (dynamic patch addition)

Register tracker and builder first; subsequent `add_patch` calls auto-expand both:

```python
system.register_tracker(tracker)   # tracker.expand() called on add_patch
system.register_builder(builder)   # QUBIT_COORDS appended on add_patch
```

### Coupler management

```python
# Register a coupler (inactive by default)
coupler_patch = system.register_coupler(
    protocol,          # LogicalCouplerProtocol instance
    patch_names,       # List[str] — names of patches to couple
    name="coupler_12", # Optional explicit name
    **kwargs,          # Passed to protocol._build_coupler_geometry()
)

# Activate: replaces boundary stabilizers with joint coupler stabilizers
system.activate_coupler("coupler_12")

# Deactivate: restores original boundary stabilizers
system.deactivate_coupler("coupler_12")

# Remove: frees coordinate space for re-registration (call after deactivate)
system.remove_coupler("coupler_12")
```

---

## 3. `CircuitBuilder` — Circuit Construction

**Module:** `lightstim.ir.builder`

Constructs the Stim circuit step by step. Every method both emits circuit instructions
and updates the `SyndromeTracker`.

### Construction

```python
from lightstim.ir.builder import CircuitBuilder

builder = CircuitBuilder(tracker, system, if_detector=True)
```

### A. Setup

```python
builder.write_coordinates()
# Emits QUBIT_COORDS instructions for all qubits in system.
# Call once at the start, before any other instructions.
```

### B. Initialization

```python
builder.initialize(
    init_dict,     # {global_idx: "X"/"Y"/"Z"}  — which qubits, in which basis
    n,             # system.num_qubits (total qubit count for tableau padding)
    noiseless=False, # If True, tags resets as 'noiseless' (noise injector skips them)
)
# Emits: RX / R / RY for the respective qubits.
# Tracker: calls process_initialization() — adds rows to stabilizer tableau.
# Does NOT add a TICK automatically.
```

### C. Syndrome Extraction

```python
builder.apply_syndrome_extraction(
    circuit_chunk,  # stim.Circuit — exactly ONE round of stabilizer measurement
                    # (last instruction MUST be M or MX on syndrome qubits)
    rounds=1,       # Number of rounds. Round 1: full tracker analysis. Rounds 2+: REPEAT block.
    noiseless=False,# Tag all instructions as 'noiseless'
    z_only=False,   # Suppress X-ancilla DETECTOR instructions (for Z-basis memory)
)
# Emits: SHIFT_COORDS + circuit_chunk + (REPEAT block if rounds > 1)
# Tracker:
#   Round 1 — tableau inversion to find back-propagated Paulis; calls process_mid_measurement()
#   Rounds 2+ — repeats detectors as rec[-k] ^ rec[-k-n_syn]; updates record offsets
```

**Critical constraint:** The `circuit_chunk` must use global qubit indices.
Build it from `system.active_stabilizers` (or `_x` / `_z` subsets) and
`system.index_map` to look up global indices.

### D. Unitary Blocks (gates between SE rounds)

```python
builder.apply_unitary_block(
    unitary_block,  # stim.Circuit — unitary gates only (no measurements/resets)
    noiseless=False,
)
# Emits: TICK (if not already present) + the unitary block.
# Tracker: conjugates stabilizer/logical tableau by the unitary's symplectic matrix.
```

### E. Coupler activation

```python
builder.activate_coupler("coupler_12")   # Wrapper for system.activate_coupler()
builder.deactivate_coupler("coupler_12") # Wrapper for system.deactivate_coupler()
# Changes system.active_stabilizer_indices — takes effect at next apply_syndrome_extraction.
```

### F. Data Qubit Readout

```python
builder.apply_data_readout(
    final_measurements,  # {global_idx: "X"/"Y"/"Z"} — defaults to all data in Z
    noiseless=False,
    z_only=False,        # Must match z_only used in apply_syndrome_extraction
)
# Emits: MX / MY / M for the specified qubits.
# Tracker: calls process_data_measurement() — resolves remaining stabilizers into
#          DETECTOR instructions and logicals into OBSERVABLE_INCLUDE.
```

### G. Canonicalization (before SE, after encoding)

```python
builder.stabilizer_canonicalization()
# Re-organizes tracker tableau into stabilizer vs. logical subspaces
# using the code's canonical stabilizer basis.
# Call after encoding (unitary gates), before first SE round.

builder.logical_canonicalization(canonical_logicals)
# canonical_logicals: {logical_index: pauli_vector (2n,)}
# Call after stabilizer_canonicalization() to choose preferred logical representatives.
```

### H. Noise Injection

```python
noisy_circuit = builder.build_noisy_circuit(
    noise_params,        # NoiseConfig
    noise_model,         # str — 'circuit_level' | 'phenomenological' | 'code_capacity' | 'XZ_biased'
)
# Returns a new stim.Circuit with noise channels injected.
# Does NOT modify builder.circuit in place.
```

### Full call sequence

```
write_coordinates()
initialize(data_qubits, n)
[stabilizer_canonicalization()]          ← only needed after encoding gates
apply_syndrome_extraction(se_block, rounds=d)
[apply_unitary_block(gate_circuit)]      ← only for logical gates mid-circuit
[activate_coupler / deactivate_coupler]  ← only for lattice surgery
apply_data_readout(final_measurements)
noisy_circuit = build_noisy_circuit(noise_params, noise_model)
```

---

## 4. `SyndromeTracker` — Automatic Detector Generation

**Module:** `lightstim.ir.tracker`

Maintains the symplectic tableau (stabilizers + logicals) and generates DETECTOR /
OBSERVABLE_INCLUDE instructions automatically. You rarely call tracker methods directly —
`CircuitBuilder` wraps all of them.

### Construction

```python
from lightstim.ir.tracker import SyndromeTracker

tracker = SyndromeTracker(
    num_qubits,               # Initial system size
    expected_num_logicals=0,  # k — the tracker validates against this at every SE round
    post_select_detector_coords=None,  # Set[Tuple[float,...]] — coords where detectors get post-select tag
)
```

### Key state

| Attribute | Type | Description |
|---|---|---|
| `tracker.stabilizers` | `PauliTableau` | Rows = current stabilizer generators (GF(2) symplectic) |
| `tracker.logicals` | `PauliTableau` | Rows = current logical operators |
| `tracker.total_measurements` | `int` | Running count of all measurements so far |
| `tracker.expected_num_logicals` | `int` | Decremented when a logical is consumed by a gate |
| `tracker.num_qubits` | `int` | System size (expanded by `expand()`) |

### Methods called by `CircuitBuilder` (you usually don't call these directly)

```python
tracker.process_initialization(init_tableau)
# Adds rows from |0⟩/|+⟩/|Y⟩ initialization to stabilizer tableau.

tracker.process_mid_measurement(circuit, back_propagated_paulis, syn_coords, no_detector_mask)
# Handles one SE round: updates tableau + emits DETECTOR instructions to circuit.

tracker.process_data_measurement(circuit, final_paulis, idx_to_coord_map, syndrome_qubit_indices)
# Final readout: resolves remaining rows into DETECTOR + OBSERVABLE_INCLUDE.

tracker.process_unitary_block(circuit_chunk)
# Conjugates tableau by the unitary's symplectic matrix (forward evolution).
```

### Methods you may call directly

```python
tracker.expand(delta)
# Add delta new qubits to the tracker (for define-by-run).
# Call BEFORE using new qubit indices in any circuit instruction.

tracker.set_expected_logicals(k)
# Update the expected logical count mid-circuit (e.g. after a logical is consumed).

tracker.reset_records_for_qubits(qubit_indices)
# Clean tracker state for qubits being re-initialized mid-circuit.
# Preserves operators with mixed support; removes operators supported only on target qubits.
```

### The Logical Count Guardrail

After every `process_mid_measurement`, the tracker checks:

```
num_absorbed_by_gauge + tracker.logicals.count == expected_num_logicals
```

If this fails, it raises `RuntimeError: Logical Count Mismatch!`. Common causes:

| Error pattern | Likely cause |
|---|---|
| `Found M ≫ N` | Coupler data qubits accidentally initialized (see gotchas 1-A) |
| `Found 0` | Wrong SE circuit (ancilla indices don't match system) |
| `commutes with all rows but is linearly independent` | Data qubit not initialized before SE |

### Post-selection

Mark specific detectors for post-selection at construction time:

```python
tracker = SyndromeTracker(
    num_qubits=n,
    post_select_detector_coords={(x, y, t), ...}  # Detectors at these coords get POST_SELECT tag
)
```

The `SimulationPipeline` reads tagged detectors and discards shots with any flagged detection event.

---

## 5. `LogicalCouplerProtocol` — Coupler Factory

**Module:** `lightstim.ir.coupler`

Abstract factory that takes a list of `QECPatch` objects and produces a new `LogicalCouplerPatch`
containing the boundary and corridor stabilizers for a logical joint measurement.

### Interface

```python
from lightstim.ir.coupler import LogicalCouplerProtocol, LogicalCouplerPatch

class MyProtocol(LogicalCouplerProtocol):
    EXPECTED_PATCH_COUNT = 2  # or None for variable

    def __init__(self):
        super().__init__(name_prefix="my_coupler")

    def _build_coupler_geometry(self, coupler_patch: LogicalCouplerPatch,
                                patches: List[QECPatch], **params):
        # 1. Analyze geometry of patches (positions, boundary structure)
        # 2. Register new corridor qubits: coupler_patch.add_qubit(x, y, role)
        # 3. Register stabilizers: coupler_patch.stabilizers.append(record)
        # 4. Mark conflicting boundary stabilizers:
        #    coupler_patch.conflicting_stabilizer_coords.add(syn_coord)
        ...
```

### `LogicalCouplerPatch` extras

Beyond `QECPatch`, `LogicalCouplerPatch` adds:

```python
coupler_patch.conflicting_stabilizer_coords  # Set[(x,y)] — syndrome coords whose
                                              # original stabilizer gets paused on activate.
```

### Stabilizer record format for couplers

Couplers use coordinate keys (not local index keys) in the pauli dict, because
corridor qubits are not yet in the global index map when the coupler patch is built:

```python
{
    "pauli":    {(x, y): "X"/"Z", ...},  # coord keys (translated to global by QECSystem)
    "type":     "X" | "Z",
    "syn_coord": (float, float),
}
# Note: "syn_idx" and "data_indices" are resolved by QECSystem.add_patch().
```

### Lifecycle

```python
# 1. Build protocol (stateless)
protocol = MyProtocol()

# 2. Register coupler (allocates qubits, inactive by default)
coupler_patch = system.register_coupler(protocol, ["p1", "p2"], name="c12", interaction_type="ZZ")

# 3. Activate (boundary stabs paused, joint stabs active)
builder.activate_coupler("c12")
# → initialize coupler data qubits → apply_syndrome_extraction → apply_data_readout (coupler only)

# 4. Deactivate (restores original boundary stabs)
builder.deactivate_coupler("c12")

# 5. (Optional) Remove for reuse
system.remove_coupler("c12")
```
