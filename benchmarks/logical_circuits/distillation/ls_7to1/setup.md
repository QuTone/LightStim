# Steane 7-to-1 |Y⟩ Distillation — Experiment Setup

## Reference

Litinski, "A Game of Surface Codes" (arXiv:1812.01238), Figure 46(c):
Steane [7,1,3] code distills 7 noisy |Y⟩ states into 1 higher-fidelity |Y⟩.

The protocol uses 4 sequential ZZZZ parity measurements on (4 choose 3) = 4 subsets of the input states. Three of the resulting logical observables encode Steane syndrome checks (post-select on these being +1); the fourth is the distilled output.

## Patch Layout

5 unrotated surface code patches, transposed (Z boundaries face the corridor), connected by a vertical lattice surgery corridor:

```
 W1 (0, 0)          W3 (right_x, 0)
     \                   /
      ---- corridor ----
     /                   \
 W2 (0, y_sp)       W4 (right_x, y_sp)
     \
      ---- corridor ----
     /
 W5 (0, 2*y_sp)
```

Layout parameterization (from d):
```python
patch_size = 2 * (d - 1)       # patch extent in each dimension
gap = 2 * d + 2                # gap between patch edges
right_x = patch_size + gap      # right column offset
y_spacing = gap                 # vertical spacing between patch rows
center = patch_size + gap / 2   # corridor midpoint (center_axis)
# corridor internal width = 2*d + 1 columns
```

| d | qubits | detectors (rounds=d) | corridor width |
|---|--------|---------------------|----------------|
| 3 | 272 | 1,732 | 7 |
| 5 | 768 | 8,552 | 11 |
| 7 | 1,520 | 24,044 | 15 |

## Circuit Structure

### Step 1: State Preparation

Prepare |Y⟩ on W1, W2, W3 (noisy magic states), |+⟩ on W4 (output register), and |Y⟩ on W5 (reusable ancilla).

Two preparation methods are supported:

**Method A — Fold-transversal S (default):**
```
RX on all data qubits → fold-transversal S gate
```
This produces |Y⟩ = S|+⟩ deterministically (no post-selection needed for state prep). The fold-transversal S is a single-layer gate (S/S† on diagonal qubits + CZ on mirror pairs), following arXiv:2406.17653v1 Extended Data Fig. 1c.

**Method B — Corner state injection (not yet supported in multi-patch):**
```
Corner qubit: RY  (target |Y⟩ state)
Upper diagonal (rel_y >= rel_x): RX  (|+⟩)
Lower diagonal (rel_y < rel_x): R   (|0⟩)
```
This uses the standard state injection protocol with post-selection on the first SE round's detectors. The injection is non-deterministic — the injected state has a physical error rate, and post-selection filters out faulty preparations.

**Current limitation:** Corner injection creates a non-code-space state (mixed X/Z/Y basis). The tracker's `process_mid_measurement` requires every syndrome measurement to be decomposable in terms of existing tableau rows. With a diagonal X/Z split, some stabilizers (e.g., ZZZZ spanning both X-initialized and Z-initialized qubits) are linearly independent of the initial rows — the tracker raises a RuntimeError. This is specific to multi-patch circuits where SE covers all patches simultaneously; single-patch state injection works correctly. Enabling this requires either per-patch SE isolation or tracker support for "free" measurements on non-code-space states.

### Step 2: Pre-coupler Syndrome Extraction

Run `rounds` SE rounds on all 5 patches (coupler inactive). This establishes the baseline detector history for error correction.

### Step 3: Sequential ZZZZ Measurements (x4)

Four sequential lattice surgery operations, each measuring ZZZZ on a 4-patch subset:

| Measurement | Patches | Logical observable |
|---|---|---|
| 1 | W1, W2, W3, W5 | L0 (Steane check) |
| 2 | W1, W2, W4, W5 | L1 (Steane check) |
| 3 | W1, W3, W4, W5 | L2 (output) |
| 4 | W2, W3, W4, W5 | L3 (Steane check) |

Each measurement cycle:
1. **Register coupler** (if new patch subset) — `UnrotatedMultiPatchCoupler` creates a vertical corridor connecting the 4 patches
2. **Activate coupler** — merge coupler boundary stabilizers into the active stabilizer set
3. **Initialize coupler data** — RX on corridor data qubits
4. **Syndrome extraction** — `rounds` SE rounds with coupler active
5. **Mid-circuit MX** — measure coupler data + W5 data qubits in X basis (closes the measurement chain)
6. **Deactivate coupler** — restore original code patch stabilizers
7. **Re-inject W5** — prepare fresh |Y⟩ for next cycle (except after last measurement)

The coupler reuses qubit indices across measurements (corridor geometry is the same for all subsets that share the same corridor position), keeping the total qubit count constant.

### Step 4: Final Readout

1. **S† on W4** — rotate |Y⟩ → |+⟩ (noiseless, Pauli frame correction)
2. **MX on W1-W4** — destructive X-basis measurement of all working patches

This produces 4 logical observables. L0, L1, L3 are post-selected (must be +1 for valid Steane syndrome). L2 is the distilled output whose logical error rate we measure.

## Simulation Pipeline

```python
SimulationPipeline(
    max_shots=10_000_000,                    # shot budget
    max_errors=200,                          # early stop for statistical significance
    post_select_observable_indices=[0, 1, 3], # Steane checks
    target_observable_indices=[2],            # distilled output LER
)
```

The pipeline runs: sample → post-select (discard shots where any check observable is flipped) → decode → count errors on target observable.

## Files

- `LS_distillation_7_to_1.py` — experiment script (configurable d, p, state prep method)
- `LS_distillation_7_to_1_results.csv` — raw numerical results
- `LS_distillation_7_to_1_results.json` — detailed results with metadata
- `summary.md` — results analysis

## LightStim + AI: Future Vision

This experiment demonstrates a powerful pattern: **a human describes the protocol logic (this document), and an AI agent translates it into a working simulation script using LightStim's API**.

The key insight is that LightStim provides the right level of abstraction:
- **Patches** = logical qubits (no manual qubit indexing)
- **Couplers** = lattice surgery operations (no manual stabilizer bookkeeping)
- **Tracker** = automatic detector generation (no manual DETECTOR annotation)
- **Builder** = circuit construction (handles coordinates, initialization, readout)
- **Pipeline** = simulation + decoding (post-selection, error counting, progress)

With these abstractions, an experiment's setup.md becomes a **specification** that is close enough to executable code that an AI can bridge the gap. The workflow becomes:

```
Human writes setup.md (protocol logic, patch layout, measurement sequence)
    ↓
AI reads setup.md + LightStim API docs
    ↓
AI generates experiment script (LS_distillation_7_to_1.py)
    ↓
Pipeline runs simulation and produces results
    ↓
AI analyzes results and writes summary.md
```

This means new experiments — different distillation protocols ([[15,1,3]], [[8,3,2]]), different codes (rotated SC, color codes), different gate sequences — can be prototyped rapidly by writing a setup.md and letting the AI handle the implementation details.

The bottleneck shifts from "how do I construct this Stim circuit" to "what experiment do I want to run" — which is where the human physicist's expertise is most valuable.
