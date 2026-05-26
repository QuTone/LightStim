---
name: logical-coupler-design
description: >
  Design a new LogicalCouplerProtocol in LightStim to implement inter-patch
  joint measurements (lattice surgery). Use this skill when the user wants to:
  implement a new logical coupler for a code that doesn't have one yet, understand
  how the Unrotated Surface Code coupler works internally, design a coupler for
  a new code family (e.g. BB code logical measurement), or extend an existing
  coupler with a new boundary configuration. This skill is the key to replicating
  papers that describe custom "logical coupler" constructions without open-sourcing
  their implementation.
user-invocable: true
---

# Logical Coupler Design

## The core idea

A joint measurement between two logical patches (e.g. measuring ZZ = Z_L ⊗ Z_L) is
implemented by temporarily activating a "coupler patch" whose stabilizers span the
boundary between both patches. The coupler's joint stabilizers let the syndrome qubits
at the boundary "see" data qubits from both patches simultaneously, effectively measuring
the product of the two logical operators.

```
Before coupler:
  [ctrl patch] | gap | [tgt patch]
   Z_ctrl stabs   ←gap→  Z_tgt stabs   (boundary Z stabs face the gap)

After activate_coupler("zz"):
  [ctrl patch] | corridor | [tgt patch]
   (boundary stabs paused)
   joint ZZ stabs now span both patch boundaries + corridor
```

The coupler patch is a factory output — it contains only the **corridor qubits and
boundary-redefined stabilizers**. It is registered inactive and activated/deactivated
during circuit construction.

---

## The `LogicalCouplerProtocol` interface

```python
from lightstim.ir.coupler import LogicalCouplerProtocol, LogicalCouplerPatch
from lightstim.ir.qec_patch import QECPatch
from typing import List

class MyProtocol(LogicalCouplerProtocol):
    EXPECTED_PATCH_COUNT = 2   # or None for variable N

    def __init__(self):
        super().__init__(name_prefix="my_coupler")

    def _build_coupler_geometry(self, coupler_patch: LogicalCouplerPatch,
                                patches: List[QECPatch], **params):
        # This is the only method you implement.
        # params are passed from system.register_coupler(..., **kwargs).
        ...
```

`create_coupler_patch()` (public, inherited) calls:
1. `_validate_patch_count(patches)` — checks `EXPECTED_PATCH_COUNT`
2. Constructs `LogicalCouplerPatch(name=name)` — an empty container
3. Calls `_build_coupler_geometry(coupler_patch, patches, **params)` — your code
4. Returns the filled `coupler_patch`

---

## Three things `_build_coupler_geometry` must do

### 1. Register corridor qubits

```python
# The coupler patch owns NEW qubits that fill the gap between patches.
# Use the same role inference as the neighboring patches (data/syndrome_x/syndrome_z).
coupler_patch.add_qubit(x, y, role='data')           # corridor data qubit
coupler_patch.add_qubit(x, y, role='syndrome_z')     # corridor Z-syndrome qubit
```

The role pattern must be consistent with both neighboring patches so SE blocks
can be built using the same extraction logic (same `_infer_role_from_anchor` parity
convention).

### 2. Register stabilizers (including redefined boundary ones)

Coupler stabilizers use **coordinate keys** in the `pauli` dict (not local integer
keys), because corridor qubits don't have global indices yet when the coupler patch
is built. `QECSystem.add_patch()` translates coords → global indices automatically.

```python
coupler_patch.stabilizers.append({
    "pauli":     {(x1, y1): "Z", (x2, y2): "Z", ...},  # coord → Pauli
    "type":      "Z",
    "syn_coord": (sx, sy),   # coordinate of the ancilla measuring this stabilizer
    # "syn_idx" and "data_indices" resolved by QECSystem.add_patch()
})
```

**Key: boundary stabilizers are redefined here** — a boundary Z-stabilizer on the
ctrl patch normally touches 2 data qubits. As a coupler stabilizer, it is replaced
by a joint stabilizer touching 3+ qubits (2 from the ctrl patch + 1+ from the corridor).

### 3. Mark conflicting boundary stabilizers for pausing

```python
# Any original patch stabilizer whose syndrome sits at the boundary must be paused
# when the coupler is active. Add its syndrome coord to the set:
coupler_patch.conflicting_stabilizer_coords.add((sx, sy))
```

When `system.activate_coupler()` is called, it finds all stabilizers in
`system.active_stabilizer_indices` whose `syn_coord` is in `conflicting_stabilizer_coords`
and removes them, replacing them with the coupler's redefined joint stabilizers.

---

## How the Unrotated Surface Code coupler works (concrete example)

The `UnrotatedTwoPatchCoupler` implements ZZ or XX lattice surgery between two
unrotated surface code patches.

### Step 1: Geometry analysis (`_analyze_geometry`)

Determines **logical operator orientation**:
- ZZ interaction → logical Z runs horizontally → patches must be vertically stacked
  (left edge of one patch at the same x as the left edge of the other)
- XX interaction → logical X runs vertically → patches must be horizontally stacked

Checks:
- Both patches have the same rotation/transposition
- One patch's extent along the non-gap axis **contains** the other's (the smaller
  is the "anchor")
- Gap size in both dimensions is even (required for consistent parity tiling)

Returns: `(logical_op_orientation, anchor_patch, gap_bounds)` as a 4-tuple
`(x_min, x_max, y_min, y_max)`.

### Step 2: Geometry construction (`_construct_coupling_region`)

Sweeps from the anchor patch toward the target patch, filling the gap column by column
(or row by row for horizontal orientation). At each position, `_infer_role_from_anchor`
determines whether the qubit should be data, syndrome_x, or syndrome_z based on
**parity distance** from the anchor's reference qubits:

```python
# Parity distance from anchor data qubit determines role:
# Same parity on both axes  → data
# Sync with X-ancilla grid  → syndrome_x
# Sync with Z-ancilla grid  → syndrome_z
data_delta = (x - anchor_data[0], y - anchor_data[1])
is_data = (data_delta[0] % 2 == 0 and data_delta[1] % 2 == 0) or \
          (data_delta[0] % 2 == 1 and data_delta[1] % 2 == 1)
```

The last column/row adjacent to the target patch adapts its y-range (or x-range)
to match the target patch's extent (handles asymmetric sizes).

### Step 3: Stabilizer construction (`_init_stabilizers`)

**Phase 1 — New corridor syndrome qubits:**
Each new syndrome qubit probes its 4 neighbors (up/down/left/right). If a neighbor
is a data qubit (in the corridor or in either patch), it is added to the stabilizer support.
This auto-detects the stabilizer weight: boundary ancillas near one patch get 3 neighbors,
interior ancillas get 4.

**Phase 2 — Existing boundary syndrome qubits:**
`_find_boundary_syndrome_candidates` scans existing syndrome qubits at the patch
boundary (x or y coordinate equal to `gap_min` or `gap_max`). For each:
1. Resolves the syndrome type from the patch it belongs to
2. Calls `_probe_and_create_stabilizer` to redefine the stabilizer with the
   extended support (now including corridor data qubits)
3. Adds to `conflicting_stabilizer_coords`

The result: boundary syndrome qubits get a new stabilizer record that covers
both patch data qubits AND corridor data qubits — this is the "joint" stabilizer
that makes the measurement non-trivial.

---

## Decision checklist for designing a new coupler

### For any code pair:

1. **What logical observable are you measuring?**
   - ZZ → need Z-type joint stabilizers along the logical Z boundary of both patches
   - XX → need X-type joint stabilizers along the logical X boundary

2. **Where do the logical operators live?**
   - In an unrotated SC: Z logical runs along a row of data qubits (horizontal strip)
   - In a rotated SC: Z logical runs diagonally
   - In a BB code: Z logical is a sparse operator (weight ~12) on a subset of data qubits
   → The coupler's joint stabilizers must be supported on the *same qubits* as the
     logical operator, so measuring them projects onto the logical ZZ eigenspace.

3. **What corridor geometry do you need?**
   - For patch codes (SC, CC): a strip of data + syndrome qubits bridging the gap
   - For LDPC codes (BB, PQRM): there may be no geometric gap — the "coupler" defines
     new syndrome qubits whose support overlaps both patches' logical operator support

4. **What boundary stabilizers need to be redefined?**
   - Any boundary syndrome qubit that previously measured only one patch's data qubits
     must now also include corridor data qubits (and possibly target patch data qubits)
   - Mark these with `conflicting_stabilizer_coords`

5. **Parity consistency:**
   - Corridor qubit roles must follow the same parity convention as the anchor patch
   - All new syndrome qubit coords must be at half-integer steps from data qubit coords
     (for unrotated SC: data at integer coords, syndromes at integer+0.5)

---

## Multi-patch coupler (`UnrotatedMultiPatchCoupler`)

For N-patch joint measurements (ZZZ, ZZZZ, ...):

```python
from lightstim.qec_code.surface_code.unrotated.multi_patch_coupler import (
    UnrotatedMultiPatchCoupler
)

protocol = UnrotatedMultiPatchCoupler()
coupler = system.register_coupler(
    protocol, ["p1", "p2", "p3"],
    name="zzz_coupler",
    path_axis="vertical",    # corridor orientation
    center_axis=8.0,         # x-coord splitting patches into left/right groups
)
```

Internally creates a single corridor connecting all N patches. Each side patch
contributes boundary data qubits to the joint stabilizers running along the corridor.

---

## Protocol lifecycle in circuit construction

```python
# 1. Register (allocates qubits, inactive)
system.register_coupler(protocol, ["ctrl", "tgt"], name="zz", interaction_type="ZZ")

# 2. Activate (pauses boundary stabs, enables joint stabs)
builder.activate_coupler("zz")

# 3. Initialize coupler data qubits (they don't exist until after registration)
cp = system.coupler_patches["zz"]
cp_data = sorted(system.local_to_global_map["zz"][q] for q in cp.data_indices)
builder.initialize({q: "X" for q in cp_data}, n=system.num_qubits)

# 4. Run SE with joint stabilizers active
se = make_se(system)  # SE block reads system.active_stabilizers — includes joint stabs
builder.apply_syndrome_extraction(se.circuit, rounds=rounds_ls)

# 5. Measure coupler data qubits IMMEDIATELY — before deactivation (gotchas 1-B)
builder.apply_data_readout({q: "X" for q in cp_data})
builder.deactivate_coupler("zz")

# 6. Continue with original stabilizers restored
builder.apply_syndrome_extraction(se_original.circuit, rounds=rounds_after)
```

## Reference script

Read `scripts/template.py` for a complete minimal `LogicalCouplerProtocol` subclass
that implements a custom one-dimensional ZZ coupler from scratch, with geometry
analysis, qubit registration, stabilizer construction, and circuit integration.
