# notebooks/

Demonstration notebooks for LightStim protocols. Each notebook corresponds to a
protocol in `lightstim/protocols/` and shows:

1. A **circuit visualization** (small scale: d=3, 1-2 SE rounds)
2. A **small numerical result** (hardcoded d=3/5, not a full sweep)

Sweep experiments belong in `benchmarks/`, not here.

## Directory structure

```
notebooks/
├── CrossLS/          Surface–PQRM lattice surgery
├── LogicalCircuits/  Bell teleportation, GHZ prep, magic-state distillation
├── LogicalOps/       Single-qubit logical gates, lattice surgery, state injection
├── Memory/           Memory experiments across all supported QEC codes
└── System/           Framework internals: QECSystem API, code info
```

## Notebook index

### CrossLS/

| Notebook | Protocol | Description |
|---|---|---|
| `cross_ls.ipynb` | `lightstim/protocols/cross_ls/` | Surface ↔ PQRM lattice surgery; detector slices and small LER sweep |

### LogicalCircuits/

| Notebook | Protocol | Description |
|---|---|---|
| `bell_teleportation.ipynb` | `protocols/bell_teleportation.py` | Bell-state teleportation via TG, ZZ-LS, XX-LS |
| `ghz_state_prep.ipynb` | `protocols/ghz.py` | Multi-patch GHZ state preparation |
| `ls_distillation.ipynb` | `protocols/ls_distillation.py` | Steane 7-to-1 \|Y⟩ distillation (LS variant) |
| `tg_distillation.ipynb` | `protocols/tg_distillation.py` | 7-to-1 \|Y⟩ distillation (TG/PQRM hypercube variant) |

### LogicalOps/

| Notebook | Protocol | Description |
|---|---|---|
| `logical_CNOT_LS.ipynb` | `protocols/cnot_ls.py` | Logical CNOT via lattice surgery (ZZ + XX) |
| `logical_CNOT_trans.ipynb` | `protocols/cnot_trans.py` | Transversal CNOT between two surface-code patches |
| `logical_H_S.ipynb` | `protocols/fold_transversal.py` | Fold-transversal H and S gates |
| `state_injection.ipynb` | `protocols/state_injection.py` | Non-FT magic-state injection (rotated SC) |
| `two_patch_LS.ipynb` | `protocols/two_patch_ls.py` | Two-patch ZZ lattice surgery coupler |
| `multi_patch_LS.ipynb` | *(in-progress)* | Multi-patch lattice surgery; unrotated SC N-patch coupler |

### Memory/

Memory notebooks use the `lightstim` QECPatch API directly (no separate protocol file).
They compare LER vs PER and show distance scaling for each code family.

| Notebook | Code family |
|---|---|
| `memory_surface_family.ipynb` | Rotated SC, unrotated SC, toric code |
| `memory_BB.ipynb` | Bivariate Bicycle codes ([[72,12,6]] … [[288,12,18]]) |
| `memory_color.ipynb` | Triangular color code (6-6-6) |
| `memory_PQRM.ipynb` | PQRM codes (1,2,4), (1,3,5), (1,4,6) |
| `memory_repetition.ipynb` | Repetition code (sanity check) |
| `memory_4D_hadamard.ipynb` | 4D geometric code (Hadamard-encoded) |

### System/

| Notebook | Description |
|---|---|
| `qec_system_intro.ipynb` | QECSystem multi-patch API: define-by-run pattern, global index space |
| `define_by_run.ipynb` | Deep-dive into dynamic patch addition and coupler registration |
| `qec_code_info.ipynb` | Display code properties: stabilizers, logicals, distance, qubit layout |

## Development workflow

See [`skills/notebook-workflow/SKILL.md`](../skills/notebook-workflow/SKILL.md) for the
full protocol development lifecycle: prototype → package → benchmark → demo.
