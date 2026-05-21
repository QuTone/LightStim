# Memory Experiment Benchmark — Experiment Setup

## Figures Overview

| Figure | Title | X-axis | Y-axis | Story |
|--------|-------|--------|--------|-------|
| Fig 1 | Surface Code Family | Physical Error Rate (p) | Logical Error Rate | Same code family, distance scaling. Rotated > Unrotated > Toric qubit overhead → lower LER |
| Fig 2 | BB Codes | Physical Error Rate (p) | Logical Error Rate | [[72,12,6]], [[108,8,10]], [[144,12,12]] × GPU BP+OSD + MWPF — 6 lines |
| Fig 3 | Qubit Efficiency | Physical Qubits per Logical Qubit (N_total/k) | LER per Logical Qubit | Cross-code comparison at fixed p=1e-3. N_total from circuit.num_qubits (includes syndrome qubits) |
| Fig 4 | Scheduling Impact | Physical Error Rate (p) | Logical Error Rate | Perpendicular (FT) vs Parallel (non-FT) zigzag. Demonstrates modular SE block replacement |

## Plotting Style

Consistent across all figures:

```python
PALETTE_DIST = {3: "#a63603", 5: "#1b9e77", 7: "#7570b3", 9: "#d95f02"}

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

- **Color → distance** (3/5/7/9 → `PALETTE_DIST`)
- **Line style → code/decoder** (solid/dashed/dotted)
- **Marker → decoder type** (o = pymatching/GPU BPOSD, X = MWPF)
- Reference style: `eval/memory_benchmark/results/fig4_scheduling_v2.png`

## Figure 1: Surface Code Family — LER vs PER

**Codes:**
| Code | Class | Distances | Decoder |
|------|-------|-----------|---------|
| Rotated Surface Code | `RotatedSurfaceCode` | 3, 5, 7 | PyMatching (CPU, 32 workers) |
| Unrotated Surface Code | `UnrotatedSurfaceCode` | 3, 5, 7 | PyMatching (CPU, 32 workers) |
| Toric Code | `ToricCode` | 3, 5, 7 | PyMatching (CPU, 32 workers) |

**Physical error rates:** `[1e-3, 2e-3, 5e-3, 7e-3, 1e-2, 1.2e-2, 1.5e-2]`
**Noise model:** circuit_level (uniform depolarizing, all rates = p)
**Rounds:** d (= code distance)
**Basis:** Z
**Decoder:** `DecoderConfig(name="pymatching", backend="cpu")`
**num_workers:** 32 (CPU)
**max_shots:** 1e9
**max_errors:** 200
**Total data points:** 3 codes × 3 distances × 7 p = 63
**Note:** Toric code has 2 logical qubits; compute LER/k for fair comparison.

## Figure 2: BB Codes — LER vs PER

**Codes and decoders (6 lines total):**
| Code | n | k | d | l | m | A | B |
|------|---|---|---|---|---|---|---|
| [[72,12,6]] | 72 | 12 | 6 | 6 | 6 | [[3,0],[0,1],[0,2]] | [[0,3],[1,0],[2,0]] |
| [[108,8,10]] | 108 | 8 | 10 | 9 | 6 | [[3,0],[0,1],[0,2]] | [[0,3],[1,0],[2,0]] |
| [[144,12,12]] | 144 | 12 | 12 | 12 | 6 | [[3,0],[0,1],[0,2]] | [[0,3],[1,0],[2,0]] |

Each code runs with **both** decoders:
- **GPU BP+OSD** (solid line, marker `o`)
- **MWPF** (dashed line, marker `X`) — `cluster_node_limit=50`

**Physical error rates:** `[1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3]` (high→low: fast points first)
**Noise model:** circuit_level (uniform depolarizing)
**Rounds:** d (code distance)
**Basis:** Z

**GPU BP+OSD decoder:**
```python
DecoderConfig(
    name="nv-qldpc-decoder",
    backend="gpu",
    params={
        "max_iterations": 1000,
        "osd_order": 10,
        "bp_method": "min_sum",
        "ms_scaling_factor": 0,
        "osd_method": "osd_cs",
        "use_osd": True,
    },
)
```

**MWPF decoder:**
```python
DecoderConfig(name="mwpf", backend="cpu", params={"cluster_node_limit": 50})
```

**num_workers:** 32 (both GPU BP+OSD and MWPF)
**batch_size:** 10,000
**max_shots:** 1e8 per task (worst case [[144,12,12]] at p=1e-3 needs ~42M shots)
**max_errors:** 100
**Total data points:** 3 codes × 2 decoders × 6 p = 36
**Note:** Must run with `venv/bin/python` for GPU (cudaq_qec). Compute LER/k for y-axis.

**MWPF scope:** [[72,12,6]] only. Larger codes ([[108,8,10]], [[144,12,12]]) omitted — the MWPF paper (HyperBlossom) reports BPOSD is more accurate than MWPF for large BB codes under circuit-level noise, so there is no scientific motivation to run them.

## Figure 3: Qubit Efficiency — LER/k vs N_total/k

**Fixed p = 1e-3.** Each code instance is one data point.

**X-axis:** Physical Qubits per Logical Qubit = `circuit.num_qubits / k`
(includes both data qubits and syndrome qubits — total qubit count in the circuit)

**Data sources:**
| Code | Distances | Decoder | Source |
|------|-----------|---------|--------|
| Rotated SC | 3, 5, 7 | PyMatching | from Fig 1 |
| Unrotated SC | 3, 5, 7 | PyMatching | from Fig 1 |
| Toric | 3, 5, 7 | PyMatching | from Fig 1 |
| Color (6-6-6) | 3, 5, 7 | MWPF (c=50) | run separately |
| BB [[72,12,6]] | — | GPU BP+OSD | from Fig 2 |
| BB [[108,8,10]] | — | GPU BP+OSD | from Fig 2 |
| BB [[144,12,12]] | — | GPU BP+OSD | from Fig 2 |

**Color code decoder:**
```python
DecoderConfig(name="mwpf", backend="cpu", params={"cluster_node_limit": 50})
```

**num_workers:** 32
**max_shots:** 1e6
**max_errors:** 200
**Total data points:** 12 topological codes + 3 BB codes = 15 data points

## Figure 4: Scheduling Impact — LER vs PER

**Code:** Rotated Surface Code only (standard orientation, no coordinate rotation)

**Schedules:**
| Schedule | FT? | Effective distance | Description |
|----------|-----|-------------------|-------------|
| `perpendicular` | Yes | d | X and Z stabilizers probe in perpendicular directions per tick |
| `parallel` | No | ~d/2 | X and Z probe in same direction → hook errors halve effective distance |

**Distances:** 3, 5, 7
**Physical error rates:** `[5e-3, 2e-3, 1e-3, 7e-4, 5e-4, 2e-4, 1e-4]`
**Rationale:** Lower p range needed to see clear separation between FT and non-FT schedules; threshold is well below 1e-2 for this effect.
**Noise model:** circuit_level
**Decoder:** `DecoderConfig(name="pymatching", backend="cpu")`
**num_workers:** 32 (CPU)
**max_shots:** 1e8
**max_errors:** 100
**Total data points:** 2 schedules × 3 distances × 7 p = 42

## Data Storage

Each figure saves its own CSV:
- `results/fig1_surface_codes.csv`
- `results/fig2_bb_codes.csv`
- `results/fig3_color_extra.csv` (color code extra runs)
- `results/fig3_efficiency.csv` (combined fig3 data)
- `results/fig4_scheduling.csv`

Figures can be reproduced from CSV without rerunning experiments.

## Formal Results

Final paper-ready figures and data go to `arxiv/memory_benchmark/` in the project root.
Working/scratch results stay in `eval/memory_benchmark/results/`.

## Running

```bash
# Full run (all figures, GPU for Fig 2 and Fig 3 Color):
/home/xiang/workspace/LightStim/venv/bin/python -m eval.memory_benchmark.run_all

# Quick test (2 p values, fewer shots):
/home/xiang/workspace/LightStim/venv/bin/python -m eval.memory_benchmark.run_all --quick

# Single figure:
/home/xiang/workspace/LightStim/venv/bin/python -m eval.memory_benchmark.run_all --figure 1
/home/xiang/workspace/LightStim/venv/bin/python -m eval.memory_benchmark.run_all --figure 4

# Custom params:
/home/xiang/workspace/LightStim/venv/bin/python -m eval.memory_benchmark.run_all --max-shots 2000000 --max-errors 500 --num-workers 32

# Fig 2 needs venv for GPU (cudaq_qec):
/home/xiang/workspace/LightStim/venv/bin/python -m eval.memory_benchmark.run_all --figure 2 --num-workers 32
```

**Output:** `eval/memory_benchmark/results/` — CSV data + PNG plots per figure.

## Baseline Data (TODO)

For Fig 2 (BB codes), overlay Bravyi et al. original data for validation.
Source: digitize from paper figures or use published data tables.
