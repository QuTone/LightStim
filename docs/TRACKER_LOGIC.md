# SyndromeTracker: Measurement Processing Logic

## Overview

`SyndromeTracker` maintains a stabilizer/logical tableau and automatically generates detectors and logical observables as measurements are processed. The core method is `process_mid_measurement`, which handles each round of syndrome extraction.

## Data Structures

```
stabilizers: PauliTableau   — (k, 2n) GF(2) matrix + measurement records
logicals:    PauliTableau   — (l, 2n) GF(2) matrix + records (initially empty)
expected_num_logicals: int   — total logical DOF of the system
```

## `process_mid_measurement` — Three-Step Flow

### Input

- `back_propagated_paulis`: (m, 2n) GF(2) matrix. Each row is the Pauli operator that a syndrome measurement effectively measures, obtained by back-propagating `Z_syndrome` (or `X_syndrome`) through the inverse of the SE circuit's unitary part.
- `syn_coords`: coordinates for each measurement (used for detector annotations).

### Step 1: Combine into Full Tableau

```
full_matrix = [stabilizers.matrix]   (k rows)
              [logicals.matrix  ]   (l rows)
              ─────────────────
              total = k + l rows

num_stabs = k    (indices 0..k-1 are stabilizer rows)
                 (indices k..k+l-1 are logical rows)
```

### Step 2: Process Each Measurement

For each back-propagated Pauli `M_i`, check symplectic commutativity against every row in `full_matrix`:

**Case A — Anti-commutes** (state update):
- Pick the first anti-commuting row as pivot.
- Update all other anti-commuting rows: `row[j] ← row[j] ⊕ row[pivot]`.
- Replace pivot row with `M_i`, recording the measurement index.
- If pivot was a logical (index ≥ `num_stabs`): `expected_num_logicals -= 1`.
- Physically: this measurement collapses a degree of freedom. If it anti-commutes with a logical operator, that logical is now fixed by the measurement outcome.

**Case B — Commutes** (detector or gauge):
- Decompose `M_i` as a GF(2) linear combination of the full tableau rows.
- If purely stabilizer components (all indices < `num_stabs`): construct a **detector** from the measurement records.
- If any logical component (some index ≥ `num_stabs`): this is a **gauge measurement** — it contains logical information and cannot be a detector. Flag it in `stabilizer_with_logical_components`.

### Step 3: Clean Basis Reorganization

After processing all measurements, decompose `full_matrix` against `back_propagated_paulis`:

```python
_, _, new_basis_indices = solve_linear_decomposition(
    basis=back_propagated_paulis,
    targets=full_matrix
)
```

- Rows **dependent** on measurements → become new stabilizers (with fresh measurement records).
- Rows **independent** of measurements → become logicals (they survive this round).
- Old stabilizer rows with existing records (from previous rounds, not measured this round) are preserved as stabilizers.

### Sanity Check

```
num_absorbed = GF(2) rank of gauge_logical_vectors
assert num_absorbed + logicals.count == expected_num_logicals
```

**Why rank, not count**: Multiple gauge measurements can share the same logical component. For example, in ZZ lattice surgery where the PQRM logical Z commutes with all coupler Z-stabs, 8 different measurements may decompose to include the same logical Z — but they collectively absorb only 1 logical degree of freedom. Similarly, a single measurement can involve multiple logicals (e.g., XX coupler with both patches in |+⟩ measures the product X₁⊗X₂), absorbing only 1 DOF, not 2.

The `_gauge_logical_vectors` list stores, for each gauge measurement, a binary vector indicating which logical indices appear in its decomposition. The GF(2) rank of these vectors gives the true number of independent logical degrees of freedom consumed.

## `stabilizer_canonicalization` — Pre-SE Reorganization

Called **after encoding, before syndrome extraction** to properly separate stabilizers from logicals.

- Takes the canonical stabilizer set from `system.active_stabilizer_indices`.
- Decomposes the tracker's full tableau against this canonical basis.
- Rows dependent on the canonical basis → stabilizers (marked with `UNMEASURED_STAB_RECORD`).
- Independent rows → logicals.
- Verifies `logicals.count == expected_num_logicals`.

This is necessary after non-trivial encoding (e.g., PQRM hypercube encoding) where the tracker's rows have been evolved through unitaries and need to be re-aligned with the system's stabilizer definitions.

## `process_final_measurement` — End of Circuit

Processes the final data qubit measurements:

- Stabilizer rows **not** in `stabilizer_with_logical_components` → detectors.
- Stabilizer rows **in** `stabilizer_with_logical_components` → `OBSERVABLE_INCLUDE` (logical measurement outcomes).

## Case Studies

### Surface-Surface Lattice Surgery (ZZ, both |+⟩)

```
Logicals: X₁ on patch1, X₂ on patch2
Coupler:  Z-stabs connecting patches

Step 2: Both X logicals anti-commute with coupler Z-stabs (X·Z = anti-commute)
        → Case A: both absorbed, expected 2→0
        → No gauge measurements (swlc = 0, rank = 0)
Step 3: 0 logicals remain
Check:  0 + 0 = 0 ✓
```

### Surface-Surface Lattice Surgery (ZZ, both |0⟩)

```
Logicals: Z₁ on patch1, Z₂ on patch2
Coupler:  Z-stabs connecting patches

Step 2: Both Z logicals commute with coupler Z-stabs (Z·Z = commute)
        → No Case A for logicals
        1 measurement decomposes to include both Z₁ and Z₂
        → gauge_vector = [1, 1], rank = 1
        → expected decreases by 0 (no Case A on logicals)
        But wait: some measurement anti-commutes with a stabilizer that
        overlaps with a logical → Case A on that stabilizer → expected stays 2
Step 3: 1 logical remains (the merged Z₁⊗Z₂ product)
Check:  1 + 1 = 2 ✓
```

### CrossLS: PQRM(1,2,4) state=Z + Surface |+⟩

```
Logicals: Z_PQRM on {25,26,27}, X_surface on {2,12,22}
Coupler:  Z-stabs on {2,25,53}, {3,28,53,54}, {4,29,54}

Step 2: X_surface anti-commutes with coupler Z on {2,25,53} (X on qubit 2 vs Z on qubit 2)
        → Case A: X_surface absorbed, expected 2→1
        Z_PQRM commutes with ALL measurements (Z·Z = commute, no X overlap)
        → Case B: 8 measurements decompose to include Z_PQRM
        → 8 gauge_vectors, all = [1] (same logical), rank = 1
Step 3: 0 logicals remain (Z_PQRM dependent on measurement span)
Check:  1 + 0 = 1 ✓
```

## Known Assumptions

1. **Single PMM call per SE phase**: `_gauge_logical_vectors` and `stabilizer_with_logical_components` are reset at the start of each `process_mid_measurement` call. This is correct because each call corresponds to a distinct measurement round with its own tableau state.

2. **Gauge vectors have consistent dimensions**: Within a single PMM call, all gauge vectors have dimension `num_logs` (the logical count at the start of that call). This is guaranteed since `num_logs` is fixed for the duration of one call.

3. **`process_final_measurement` uses last PMM's state**: The `stabilizer_with_logical_components` set from the most recent PMM call determines which final measurements become observables vs detectors.
