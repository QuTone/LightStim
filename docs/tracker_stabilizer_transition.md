# Tracker Stabilizer Transition Problem

## Status: Open — Blocks Sequential Coupler Reuse with Detectors

## Problem Statement

When multiple lattice surgery measurements are performed sequentially (e.g., 4 ZZZZ measurements in a Steane distillation circuit), each measurement cycle changes the system's active stabilizer set (different boundary stabilizers for different patch subsets). The tracker's internal stabilizer tableau reflects the PREVIOUS cycle's stabilizer structure. When the NEXT cycle's SE round runs, the tracker can't decompose its old rows against the new SE measurements → stabilizer rows explode into logicals.

## What Works

- Single measurement cycle: DEM OK (568 det, 1 obs for Y/X init)
- 4 sequential cycles with `if_detector=False`: circuit compiles correctly
- 4 sequential cycles with Z init (trivial case): DEM OK (2114 det, 4 obs)
- Qubit index reuse: `add_patch` correctly reuses dormant qubit indices

## What Doesn't Work

- 4 sequential cycles with Y/X init + `if_detector=True`: tracker logical count explodes after cycle 1→2 transition
- Any approach to "clean up" tracker state between cycles breaks in different ways:
  - `reset_records_for_qubits` (UNMEASURED_STAB_RECORD): old rows still can't decompose against new SE
  - `reset_records_for_qubits` + column zeroing: changes Pauli operators AFTER detectors are frozen → non-deterministic DEM
  - `process_data_measurement` mid-circuit: aggressively prunes tracker state → next SE can't decompose
  - Skip re-stabilization SE (direct coupler-to-coupler): qubit active set conflicts

## Root Cause

The tracker's `process_mid_measurement` assumes a STABLE stabilizer structure. Each SE round's back-propagated Paulis are decomposed against the existing stabilizer tableau. When the stabilizer structure changes (coupler activation/deactivation), the old tableau rows no longer match the new SE measurements:

1. Old rows reference stabilizer patterns from cycle k
2. New SE measures stabilizer patterns from cycle k+1
3. Old rows are independent of new measurements → classified as logicals
4. Logical count explodes → sanity check fails

## Confirmed Non-Issues

- **No syndrome cross-talk**: Verified via inverse tableau analysis for all configurations (d=3, d=5, 2-patch, 4-patch). All back-propagated Pauli weights = stabilizer_weight + 1. See `tests/test_back_propagated_pauli.py`.
- **Qubit index reuse works**: `add_patch` correctly reuses dormant indices via `active_qubit_indices` tracking.
- **`process_data_measurement` works for single cycle**: Mid-circuit data measurement + partial reduction correctly cleans up logicals for single-cycle case.

## Potential Solutions

1. **Stabilizer Canonicalization mid-circuit**: A new tracker method that re-aligns the stabilizer tableau with the system's current active stabilizers, while preserving logicals and their measurement records. Similar to existing `stabilizer_canonicalization` but designed for mid-circuit use (after detectors have been constructed).

2. **Tracker reset at cycle boundaries**: After each coupler deactivation, fully reset the tracker's stabilizer rows to match the current active stabilizers. This would lose historical measurement records but start fresh for the next cycle.

3. **Unified coupler approach**: Register ONE coupler with ALL patches upfront, then selectively mask/unmask boundary stabilizers for each measurement subset. This avoids stabilizer structure changes entirely.

## Related Files

- `src/ir/tracker.py`: `process_mid_measurement`, `reset_records_for_qubits`, `stabilizer_canonicalization`
- `src/ir/qec_system.py`: `activate_coupler`, `deactivate_coupler`, `active_qubit_indices`
- `src/ir/builder.py`: `apply_data_readout` → `process_data_measurement`
- `tests/test_back_propagated_pauli.py`: syndrome cross-talk verification
- `plan/qubit_lifecycle_redesign.md`: full design plan
