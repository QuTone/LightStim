# Surface-PQRM Lattice Surgery — Experiment Setup

## Protocol Overview

**CrossLS** is a lattice surgery protocol that teleports a non-Clifford gate from a Punctured Quantum Reed-Muller (PQRM) code onto a surface code qubit. The key idea: PQRM supports transversal non-Clifford gates (T, P8, P16) natively via its hypercube structure, but has poor error-correcting properties. By merging PQRM with a surface code via lattice surgery, the logical state of the surface code acquires the non-Clifford gate without ever applying it directly.

### Why PQRM?

The Punctured Quantum Reed-Muller code is a CSS code with:
- **Transversal T gate**: transversal physical T on all data qubits implements logical T
- **Hypercube encoding**: diagonal initialization + m CNOT layers prepares any logical state
- **X stabilizers for post-selection only**: no syndrome ancilla for X checks; errors detected at readout
- **Supported parameters**: (rx, rz, m) ∈ {(1,2,4), (1,3,5), (1,4,6)}, giving [[15,1,3]], [[31,1,5]], [[63,1,7]] codes

### The CrossLS Protocol

```
Surface code (left)    Coupler (middle)    PQRM (right)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  |+⟩ init          ZZ merge column    hypercube encode
  Surface SE                               PQRM SE (Z only)
  ─────────────────────────────────────────────────────
  Lattice surgery SE rounds (ZZ coupler joins X_L across boundary)
  ─────────────────────────────────────────────────────
  MZ on surface      MX on coupler        MX on PQRM
```

The ZZ coupler at x=−1 performs a ZZ joint measurement between surface and PQRM. Depending on the PQRM input state (Z/X/Y), this teleports a logical Clifford correction or non-Clifford gate to the surface code.

**Physical mechanism**: The surface is rotated 90° so its logical Z operator runs along the right boundary (x-direction). The PQRM logical Z runs along its left column. The ZZ coupler connects these boundaries, implementing an XX-type logical merge: `X_L(surface) × X_L(PQRM)` is stabilized, teleporting the PQRM logical state onto the surface.

### Post-Selection

PQRM X stabilizers have no syndrome ancilla — they are high-weight parity checks verified at final transversal MX. Any shot that violates an X-stabilizer is discarded:

| Mode | Description |
|------|-------------|
| `pqrm_only` | Post-select on PQRM X-stab violations at final MX only |
| `hybrid` | Post-select on PQRM X-stabs + boundary-adjacent surface detectors in first round |

---

## Code Locations

| Component | Path |
|-----------|------|
| PQRM patch (geometry, stabilizers, logicals) | `src/qec_code/PQRM/pqrm_patch.py` |
| PQRM SE schedule (6-tick bulk+boundary deltas) | `src/qec_code/PQRM/pqrm_se_config.py` |
| PQRM logical ops (hypercube encoding) | `src/qec_code/PQRM/pqrm_operation.py` |
| PQRM SE block (Z-stab only) | `src/qec_code/PQRM/pqrm_se_block.py` |
| ZZ coupler geometry | `experiments/cross_ls/surface_pqrm_coupler.py` |
| Synchronized Surface+PQRM SE block | `experiments/cross_ls/surface_pqrm_se_block.py` |
| CrossLS experiment (main) | `experiments/cross_ls/cross_ls_experiment.py` |
| Full sweep run script | `eval/new_protocol/surface-PQRM-LS/run_sweep.py` |
| Rounds sweep script | `eval/new_protocol/surface-PQRM-LS/run_rounds_sweep.py` |
| CrossLS notebook | `notebooks/test_CrossLS.ipynb` |
| PQRM memory notebook | `notebooks/memory_experiment.ipynb` (PQRM section appended) |

---

## Experiment Design

### Figure 1: LER vs PER — PQRM × d_surf × state

**Goal:** Show that CrossLS successfully teleports the PQRM logical state to the surface code with LER that scales favorably with d_surf.

**Sweep parameters:**

| Parameter | Values |
|-----------|--------|
| PQRM code | (1,2,4) → [[15,1,3]]; (1,3,5) → [[31,1,5]]; (1,4,6) → [[63,1,7]] |
| d_surf | 3, 4, 5, 6, 7 |
| state | Z, X, Y |
| rounds | d_surf |
| p | 1e-4 to 1e-2 (log-spaced) |
| decoder | MWPF or BP+OSD |
| post-selection | hybrid |

**Key story:** The surface code distance d_surf controls how well the lattice surgery round is protected. The PQRM code size (m) sets the non-Clifford gate fidelity from the magic state injection side. Together, they give a 2D trade-off space: larger PQRM → better gate, larger d_surf → better error correction.

**Expected:**
- LER decreases with d_surf (surface code suppresses surgery errors)
- Y state has higher LER than X/Z (sensitive to both X and Z channels)
- Post-selection rate decreases with p (PQRM X-stab violations scale as O(p))

### Figure 2: LER vs rounds — d_surf sweep

**Goal:** Show how LER changes with the number of SE rounds during lattice surgery.

```bash
python eval/new_protocol/surface-PQRM-LS/run_rounds_sweep.py
```

### Figure 3: Post-selection rate vs p

Show the fraction of shots surviving post-selection as a function of p, for each PQRM code size. This determines the overhead of the CrossLS protocol in practice.

---

## Running

```bash
# Full LER vs PER sweep (MWPF CPU, PQRM(1,2,4))
python eval/new_protocol/surface-PQRM-LS/run_sweep.py \
    --pqrm 1,2,4 --decoder mwpf --workers 32 \
    --p 2e-3 --max-shots 2000000 --max-errors 200 \
    --output eval/new_protocol/surface-PQRM-LS/results/sweep_mwpf_124.csv

# Multiple PQRM codes
python eval/new_protocol/surface-PQRM-LS/run_sweep.py \
    --pqrm 1,2,4 1,3,5 1,4,6 --decoder mwpf --workers 32 \
    --p 2e-3 --output eval/new_protocol/surface-PQRM-LS/results/sweep_all.csv

# GPU BP+OSD
python eval/new_protocol/surface-PQRM-LS/run_sweep.py \
    --pqrm 1,2,4 --decoder bposd --backend gpu --gpu-id 0 --workers 4 \
    --p 2e-3 --output eval/new_protocol/surface-PQRM-LS/results/sweep_gpu.csv

# Rounds sweep (PQRM(1,2,4), p=1e-3)
python eval/new_protocol/surface-PQRM-LS/run_rounds_sweep.py
```

Both scripts support resume: rows already in the output CSV are skipped on restart.

---

## PQRM Code Parameters

| (rx, rz, m) | Code | Data qubits | Z stabs | X stabs (PS) | Logical Z weight |
|-------------|------|-------------|---------|--------------|-----------------|
| (1, 2, 4)   | [[15,1,3]] | 15 | 10 | 4 | 3 |
| (1, 3, 5)   | [[31,1,5]] | 31 | 24 | 10 | 7 |
| (1, 4, 6)   | [[63,1,7]] | 63 | 52 | 20 | 7 |

The PQRM code has distance t+1 against Z errors (protected by Z stabilizers) but only detects X errors via post-selection (no X syndrome ancilla). The CrossLS protocol's LER is thus limited by the PQRM X-error rate and the surface code's ability to correct surgery errors.

---

## SE Schedule

The synchronized SE block (`SurfacePQRMSEBlock`) runs in 11 ticks per round:

```
R (reset syndromes) → TICK
H (X-type syndromes) → TICK
[6 CNOT ticks: surface uses canonical deltas, PQRM uses 6-tick bulk/boundary schedule]
H (X-type syndromes) → TICK
M (measure all syndromes)
```

Surface + coupler syndromes use the same canonical tick deltas as `UnrotatedSurfaceCodeExtractionBlock`. PQRM Z-stabilizers use a separate bulk/boundary schedule tuned to the hypercube geometry.
