# Perpendicular Boundary Limitation in Multi-Patch Coupler

## Summary

The `UnrotatedMultiPatchCoupler` supports **side-connected patches** (left/right of vertical corridor) but NOT **endpoint patches** (centered below/above corridor). This document explains the root cause and potential fixes.

## What Works

Side-connected patches interface with the corridor via **parallel boundaries** — the patch boundary edge runs parallel to the corridor axis. Example:

```
   left patch │ corridor │ right patch
   ───────────┤          ├────────────
              │          │
   ───────────┤          ├────────────
   left patch │          │ right patch
```

All verified configurations:
- 2-patch ZZ (178 det, DEM OK)
- 3-patch ZZZ (292/324 det, DEM OK)
- 4-patch ZZZZ (396 det, DEM OK)
- 5-patch selective Z₄ with idle patch (528 det, DEM OK)
- Mixed distances d=3 + d=4 (620 det, DEM OK)

## What Fails

Endpoint patches interface via **perpendicular boundaries** — the boundary edge is perpendicular to the corridor axis. Example:

```
   left patch │ corridor │ right patch
   ───────────┤          ├────────────
              │          │
              │──────────│
              │ endpoint │
```

## Root Cause

### The 6-Tick CNOT Schedule

The SE block uses a 6-tick interleaved schedule (from Li, arXiv:1410.7808):

```
Tick 1: Z(-1,0)       — Z syndromes probe left
Tick 2: Z(+1,0)       — Z syndromes probe right
Tick 3: X(0,+1) Z(0,+1) — both probe down
Tick 4: X(0,-1) Z(0,-1) — both probe up
Tick 5: X(-1,0)       — X syndromes probe left
Tick 6: X(+1,0)       — X syndromes probe right
```

At Ticks 3-4, **both X and Z syndromes** apply CNOTs simultaneously. This interleaving is what makes 6 ticks possible (vs 8 if fully separated).

### The Conflict at Perpendicular Boundaries

At the corridor-endpoint boundary, a data qubit is shared between:
1. A **corridor X-syndrome** (applying CNOT at Tick 3 or 5)
2. An **endpoint Z-syndrome** (applying CNOT at Tick 2 or 4)

In the **bulk lattice**, this same pattern exists everywhere and works correctly. Stim's error analysis validates it because the R (reset) at the start of each round makes syndrome qubits' states deterministic.

At the **perpendicular boundary**, the data qubit at the junction (e.g., `(8,8)`) is shared by:
- Corridor X-syn `(6,7)` via `dx_x=(0,+1)` at Tick 3: `CX syn(6,7) → data(6,8)` (p5 data)
- Endpoint Z-syn `(5,8)` via `dx_z=(+1,0)` at Tick 2: `CX data(6,8) → syn(5,8)`

The back-propagated Pauli for the corridor X-syndrome picks up a term on the endpoint Z-syndrome through this shared data qubit. While the tracker can handle this (via anti-commutation), **Stim's DEM validation** catches that the resulting detectors are non-deterministic:

```
The circuit contains non-deterministic detectors.
R on qubit 132 [coords (9, 8)] anti-commuted with detector D223
D223's backward-propagating error sensitivity includes X132 [coords (9, 8)]
```

This means the R (reset) of a coupler syndrome qubit at the boundary flips a detector — indicating the detector incorrectly depends on the syndrome qubit's state, which should be deterministic after R.

### Why Parallel Boundaries Don't Have This Issue

At parallel (side) boundaries, the corridor X-syndromes and patch Z-syndromes don't share data qubits through the interleaved ticks. The boundary runs along the corridor axis, so the CNOT directions that cross the boundary (horizontal, Ticks 1-2 for Z, Ticks 5-6 for X) are fully separated in time.

## Debugging Evidence

Concrete trace for the failing measurement:

```
FAIL at measurement 24
  syn qubit 50: coord=(8.0, 9.0) owner=p5
  Data terms: X@(8,8), X@(8,10), X@(7,9) — correct stabilizer support
  Syndrome cross-talk: X@(9,8) — coupler Z-syndrome (should not appear)
```

The `X@(9,8)` term is the syndrome cross-talk from the shared data qubit at `(8,8)`.

## Potential Fixes

### Option 1: 8-Tick Fully Separated Schedule
Separate X and Z CNOT rounds completely:
- Ticks 1-4: Z-syndromes only (left, right, down, up)
- Ticks 5-8: X-syndromes only (down, up, left, right)

No X-Z interleaving → no cross-talk at ANY boundary orientation. Cost: +2 ticks per round (33% more circuit depth).

### Option 2: ZX Interleaving Schedule (arXiv:2603.01628)
Recent work by Gidney et al. (2026) achieves 4-tick minimum depth while preserving full fault distance for **arbitrary layouts**. This would be ideal for lattice surgery with mixed boundary orientations.

### Option 3: Diagonal Schedule (arXiv:2602.09099)
Fowler & Kishony (2026) propose a period-7 schedule that orients hook errors along diagonals, working with arbitrary boundary geometries.

### Option 4: Custom Boundary Schedule
Use the standard 6-tick schedule for the bulk, and a modified schedule at perpendicular boundary regions. This is more complex but preserves the 6-tick efficiency for most of the circuit.

## Current Workaround

Place all interacting patches as **side patches** (left/right of the vertical corridor). For layouts that would naturally have an endpoint (like the bottom of a bus), shift the patch to a side position:

```python
# Instead of endpoint (fails):
system.add_patch(p5, name='p5', offset=(4, 16))  # centered below corridor

# Use side patch (works):
system.add_patch(p5, name='p5', offset=(-2, 16))  # left side
```

## References

- Li, Y. "A magic state's fidelity can be superior to the operations that created it." arXiv:1410.7808 (2014). — Source of the 6-tick schedule.
- Fowler & Kishony. "Surface code off-the-hook: diagonal syndrome-extraction scheduling." arXiv:2602.09099 (2026).
- Gidney et al. "No More Hooks in the Surface Code." arXiv:2603.01628 (2026).
- Zhang et al. "Dense packing of the surface code." arXiv:2511.06758 (2025).
