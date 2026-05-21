# Logical Operation Benchmark — Experiment Setup

## Overview

Two figures benchmarking logical gate quality for the unrotated surface code.


| Figure | Title              | X-axis                  | Y-axis             | Story                                                                                                                |
| ------ | ------------------ | ----------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| Fig 1  | Logical Operations | Physical Error Rate (p) | Logical Error Rate | H, S, Trans-CNOT, LS-CNOT vs Memory baseline — how much overhead does each gate add comparing to memory experiments? |
| Fig 2  | State Injection    | Physical Error Rate (p) | LER + PS rate      | Full sweep over state, post-selection scheme, protocol, distance                                                     |


---

## Plotting Style

Consistent with `eval/memory_benchmark/setup.md`:

```python
PALETTE_DIST = {3: "#a63603", 5: "#1b9e77", 7: "#7570b3"}

PAPER_RC = {
    "font.family": "sans-serif",
    "font.weight": "bold",
    "font.size": 14,
    "axes.labelsize": 17,
    "axes.titlesize": 20,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "lines.markersize": 9,
}
```

- **Color → distance** (3/5/7 → `PALETTE_DIST`)
- **Line style → gate type** (solid = gate, dashed = memory baseline)
- **Marker → gate** (o = H, s = S, ^ = Trans-CNOT, D = LS-CNOT, x = memory)

---

## Figure 1: Logical Operations — LER vs PER

**Code:** Unrotated Surface Code, distances d = 3, 5, 7
**Noise model:** circuit_level (uniform depolarizing, all rates = p)
**Physical error rates:** `[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]`
**Decoder:** BP+OSD (CPU, 32 workers) — PyMatching MUST NOT be used for S and CNOT experiments; fold/CNOT correlations produce high-weight DEM hyperedges that PyMatching cannot handle, causing d-scaling to break (tested: PyMatching LER at d=5 is 31x too high for S·S†)
**max_shots:** 1e9
**max_errors:** 100

**SE rounds:** `rounds = 1` before and after each gate section (TBD — pending discussion; current script uses rounds=2 matching the original baseline notebook)

**Goal:** Compare LER of each logical gate to the memory experiment baseline at the same (d, p). A gate's LER significantly above the memory baseline indicates gate-introduced overhead.

---

### (1) Fold-Transversal H Gate

**Protocol:** Two complementary experiments; LER is the average.


| Sub-experiment | Init basis | SE rounds | Gate | SE rounds | Measure basis |
| -------------- | ---------- | --------- | ---- | --------- | ------------- |
| A (H: |0⟩→|+⟩) | Z (|0⟩_L)  | `rounds`  | H_L  | `rounds`  | X             |
| B (H: |+⟩→|0⟩) | X (|+⟩_L)  | `rounds`  | H_L  | `rounds`  | Z             |


```
LER_H = (LER_A + LER_B) / 2
```

Each sub-experiment has one logical observable. H_L = fold (transversal H on all data qubits + SWAP mirror pairs).

**Rationale:** Averaging Z→X and X→Z channels gives a symmetric estimate of H gate fidelity unbiased by basis choice.

---

### (2) Fold-Transversal S Gate

**Protocol:** Single experiment — S roundtrip to keep circuit fault-tolerant.

```
|+⟩_L → SE(rounds) → S_L → SE(rounds) → S†_L → SE(rounds) → MX (transversal)
```

```
LER_S = LER_total / 2      # circuit applies 2 S-type gate applications
```

**Rationale for roundtrip:** Direct `|+⟩ → S_L → MY` (measuring Y via unencode) creates weight-0 DEM errors because the unencode round has incomplete detector coverage. The S·S† roundtrip restores X-basis readout, achieving full fault tolerance. Dividing by 2 recovers per-gate LER.

**One logical observable** (X_L parity after roundtrip = +1 if no logical error).

---

### (3) Transversal CNOT

**Protocol:** 4 sub-experiments sweeping input/output logical basis combinations. The transversal CNOT maps logical Paulis as:

```
Z_C ⊗ I_T  →  Z_C ⊗ Z_T
I_C ⊗ Z_T  →  I_C ⊗ Z_T
X_C ⊗ I_T  →  X_C ⊗ I_T
I_C ⊗ X_T  →  X_C ⊗ X_T
```

CNOT logical action: Z_C→Z_C⊗Z_T, Z_T→Z_T, X_C→X_C, X_T→X_C⊗X_T


| Label | init(C,T) | meas(C,T) | Observables              | Noiseless value |
| ----- | --------- | --------- | ------------------------ | --------------- |
| ZZ_ZZ | ZZ        | ZZ        | 2 (Z_C, Z_T independent) | both +1         |
| ZX_ZX | ZX        | ZX        | 2 (Z_C, X_T independent) | both +1         |
| XZ_XX | XZ        | XX        | 1 (X_C⊗X_T joint)        | +1              |
| XZ_ZZ | XZ        | ZZ        | 1 (Z_C⊗Z_T joint)        | +1              |
| XX_XX | XX        | XX        | 2 (X_C, X_T independent) | both +1         |


XZ_XX and XZ_ZZ are separate circuits (different readout basis) from the same input state.

```
LER_CNOT = mean(LER_ZZ_ZZ, LER_ZX_ZX, LER_XZ_XX + LER_XZ_ZZ, LER_XX_XX) (each LER correspond to two logical observables)
```

Each LER is averaged over its observables by the simulation pipeline.

**Source:** `experiments/CNOT_trans.py` — `CNOTTransExperiment` with `initial_basis_control/target` and `measure_basis_control/target`.

---

### (4) LS CNOT (Lattice Surgery)

**Protocol:** Same 4-experiment evaluation strategy as transversal CNOT; only the gate implementation changes.

LS CNOT uses a 3-patch protocol with an ancilla patch A, control C, and target T:

```
Layout:  A ─── T   (XX joint measurement)
         |
         C         (ZZ joint measurement)
```

Two valid initializations for ancilla:

- **Protocol A:** ancilla initialized in |+⟩_L → ZZ(C-A) → XX(T-A) → measure Z on A
- **Protocol B:** ancilla initialized in |0⟩_L → XX(T-A) → ZZ(C-A) → measure X on A

Choose one protocol for the benchmark (recommend Protocol A, ancilla |+⟩).

**Source:** `experiments/CNOT_LS.py` — `CNOTLSExperiment`.

```
LER_LS_CNOT = mean(LER_1, LER_2, LER_3a + LER_3b, LER_4)
```

---

### (5) Memory Baseline

**Must be rerun.** The existing memory benchmark (`eval/memory_benchmark/`) covers p ∈ [1e-3, 1.5e-2], but the logical-op sweep starts at p = 1e-4. A dedicated memory run at matching p values is needed.

**Protocol:** Standard Z-basis memory experiment.

```
|0⟩_L → SE(d rounds) → measure Z (transversal)
```

**Parameters to match Fig 1 gate experiments:**

- Code: Unrotated Surface Code, d = 3, 5, 7
- Noise model: circuit_level
- Rounds: d (= code distance)
- Basis: Z
- p: `[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]`
- Decoder: PyMatching (CPU, 32 workers)

**Source:** `experiments/memory.py` — `MemoryExperiment`

Plot as dashed reference lines on the same axes as gate LER curves.

---

### Data Storage

```
results/fig1_H_raw.csv             # H gate sub-experiments
results/fig1_S_raw.csv             # S gate (S·S† roundtrip)
results/fig1_CNOT_trans_raw.csv    # Transversal CNOT sub-experiments
results/fig1_CNOT_LS_raw.csv       # LS CNOT sub-experiments
results/fig1_memory_raw.csv        # Memory baseline (new run at matching p)
```

CSV columns: `gate, sub_experiment, d, p, rounds, shots, errors, logical_error_rate`

All averaging (H: mean of 2, S: divide by 2, CNOT: mean of 5) is done in post-processing.

---

## Figure 2: State Injection — Unrotated Surface Code

**Goal:** Characterize injection fidelity and post-selection cost across all states, post-selection schemes, and protocols.

### Sweep Parameters (Full)


| Parameter            | Values                                                    |
| -------------------- | --------------------------------------------------------- |
| **inject_state**     | Z, X, Y                                                   |
| **post_select_mode** | `full_postselection`, `full_qec`, `hybrid`                |
| **distance**         | 3, 5, 7                                                   |
| **rounds**           | 2, 3                                                      |
| **p**                | 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2                                    |
| **protocol**         | `corner` (unrotated SC); `corner` + `middle` (rotated SC) |


**Total combinations (unrotated, corner only):** 3 × 3 × 3 × 2 × 7 = 378 tasks
**Total combinations (rotated, corner + middle):** 2 × 3 × 3 × 3 × 2 × 7 = 756 tasks

### Post-Selection Modes


| Mode                 | Description                                                         |
| -------------------- | ------------------------------------------------------------------- |
| `full_postselection` | Discard any shot with a detector flip during injection rounds       |
| `full_qec`           | No post-selection; treat injection errors like any other QEC errors |
| `hybrid`             | Post-select on the injection round only; full QEC for SE rounds     |


### Metrics

- **LER (logical_error_rate):** fraction of surviving shots with logical error
- **PS rate (post_selection_rate):** fraction of shots that survive post-selection (1.0 for `full_qec`)

### What to Present

**Plot A — fix d=5, LER + PS rate vs PER (dual y-axis)**

- x-axis: PER (log scale)
- Left y-axis: LER (log scale)
- Right y-axis: PS survival rate (linear, 0–1)
- 1 subplot per state (Z, X, Y) or fix state=Y with 3 subplots for modes
- Solid lines = LER; dashed lines = PS rate; color = post-selection mode (full_ps / hybrid / full_qec)
- Story: at what p does each mode's PS overhead become acceptable vs its LER benefit?

**Plot B — fix p=1e-3 + hybrid mode, LER & PS rate vs d**

- x-axis: d = 3, 5, 7
- Left y-axis: LER (log scale)
- Right y-axis: PS survival rate (linear)
- 3 curves = |Z⟩, |X⟩, |Y⟩
- Story: Y hardest to inject; how does both LER and PS cost scale with code distance?

### Source Scripts

- `eval/logical_op_benchmark/state_injection/run_unrotated_sc.py` — full pipeline (build → simulate → CSV → plots)
- `eval/logical_op_benchmark/state_injection/run_rotated_sc.py` — rotated SC with corner + middle protocol comparison

### Data Storage

```
results/state_injection/unrotated_sc_eval.csv          # full sweep
results/state_injection/unrotated_sc_ler_wide.csv      # pivot by post_select_mode
results/state_injection/rotated_sc_eval.csv
results/state_injection/fig2_panel_*.png
```

CSV columns: `code, inject_state, post_select_mode, protocol, d, rounds, p, shots, errors, logical_error_rate, post_selection_rate`

---

## Running

```bash
# Fig 1: all gates + memory baseline
python eval/logical_op_benchmark/run_logical_ops.py

# Fig 1: single gate
python eval/logical_op_benchmark/run_logical_ops.py --gate H
python eval/logical_op_benchmark/run_logical_ops.py --gate S
python eval/logical_op_benchmark/run_logical_ops.py --gate CNOT_trans
python eval/logical_op_benchmark/run_logical_ops.py --gate CNOT_LS
python eval/logical_op_benchmark/run_logical_ops.py --gate memory

# Fig 1: quick mode
python eval/logical_op_benchmark/run_logical_ops.py --quick

# Fig 2: state injection — unrotated SC
python eval/logical_op_benchmark/state_injection/run_unrotated_sc.py

# Fig 2: state injection — rotated SC (corner + middle)
python eval/logical_op_benchmark/state_injection/run_rotated_sc.py
```

---

## TODO

- [ ] **Decide SE rounds for gate experiments:** gate experiments currently use `rounds=2` fixed; memory uses `rounds=d`. Options: (a) use `rounds=1` for all gates (minimum FT), (b) use `rounds=d` to match memory, (c) sweep rounds and show separately (see below).
- [ ] **Rounds sweep experiment (Fig 1 extension):** For each gate at fixed p, plot LER vs number of SE rounds (1, 2, 3, d). Shows how much LER improves as more syndrome information is collected around the gate. Motivates choosing rounds=d as the "fair" comparison to memory.
- [ ] **Switch run script decoder to BP+OSD** (see note above — PyMatching fails for S/CNOT).

---

## Status


| Component                          | Status                                                                              |
| ---------------------------------- | ----------------------------------------------------------------------------------- |
| Fold-transversal H/S circuits      | Done (`experiments/fold_transversal.py`)                                            |
| Transversal CNOT circuits          | Done (`experiments/CNOT_trans.py`)                                                  |
| LS CNOT circuits                   | Done (`experiments/CNOT_LS.py`)                                                     |
| Fig 1 run script                   | Done (`eval/logical_op_benchmark/run_logical_ops.py`)                               |
| Fig 1 results                      | **TODO**                                                                            |
| State injection scripts            | Done (`eval/logical_op_benchmark/state_injection/`)                                 |
| State injection results            | Running                                                                             |
| Memory baseline (matching p range) | **TODO** — must run separately (existing memory_benchmark covers different p range) |


