# Logical Circuit Benchmark — Experiment Setup

Benchmarks for multi-patch **logical circuits**: Bell-state teleportation and magic-state distillation.
Complements `eval/logical_op_benchmark/` (single logical gates) by testing sequential lattice surgery
and transversal interactions.

---

## Directory structure

```
eval/logical_circuit_benchmark/
├── setup.md                         ← this file
├── bell-teleportation/
│   ├── run_tg.py                    ← transversal CNOT teleportation (CPU BP+OSD)
│   ├── run_ls_zz.py                 ← ZZ-merge teleportation, vertical stack (CPU PyMatching)
│   ├── run_ls_xx.py                 ← XX-merge teleportation, horizontal row (CPU PyMatching)
│   ├── start_tmux_sweeps.sh         ← optional: start 3 detached tmux sessions (no REPO/BT env)
│   └── results/
│       ├── tg_results.csv
│       ├── ls_zz_results.csv
│       └── ls_xx_results.csv
└── distillation/
    ├── ls_7to1/                     ← Steane 7-to-1 (LS layout)
    └── tg_7to1/                     ← Steane 7-to-1 (transversal)
```

**Reference notebook:** `notebooks/test_bell_teleport.ipynb` — circuit structure matches the three
`bell-teleportation/` scripts.

---

## Plotting style

Align with `eval/memory_benchmark/setup.md` and `eval/logical_op_benchmark/setup.md`:

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

- **Color → distance** (3 / 5 / 7)
- **Subplot or line style → protocol** (TG vs ZZ-LS vs XX-LS)
- **Marker or secondary grouping → teleported state** (|X⟩_L vs |Z⟩_L)

---

## Figure 1: Bell-state teleportation — LER vs PER

**Goal:** Compare logical error rate when teleporting |X⟩_L or |Z⟩_L from patch1 onto patch3 via
an entangled patch2, for three protocols (transversal CNOT vs two lattice-surgery variants).

### Protocols

| Protocol | Script | Layout | Mechanism | Ancilla-style prep on patch2 / patch3 |
| -------- | ------ | ------ | --------- | -------------------------------------- |
| **TG** | `run_tg.py` | 3 patches in a row (`dx = 2(2d−1)−2`) | Two `transversal_cnot` gates | patch2 = \|+⟩, patch3 = \|0⟩ |
| **ZZ-LS** | `run_ls_zz.py` | Vertical stack, step = `2d` | Two ZZ merges (coupler_23 then coupler_12) | patch2, patch3 in X eigenstate |
| **XX-LS** | `run_ls_xx.py` | Horizontal row, step = `2d` | Two XX merges (coupler_23 then coupler_12) | patch2, patch3 in Z eigenstate |

### States teleported

Only **|X⟩_L** and **|Z⟩_L** (no |Y⟩_L sweep for this benchmark).

### Sweep parameters

| Parameter | Value |
| --------- | ----- |
| Code | Unrotated surface code, **d = 3, 5, 7** |
| Noise | `circuit_level` (uniform depolarizing, all rates = **p**) |
| **p** | `5e-4, 1e-3, 2e-3, 5e-3` |
| TG SE rounds | `rounds_pre = d`, `rounds_mid = 1`, `rounds_post = 1` |
| LS SE rounds | `rounds_pre = d`, **`rounds_ls = d`** per merge block |
| **max_shots** | `1e9` (script default) |
| **max_errors** | `100` (early stop per (d, p, state) task) |
| **num_workers** | `8` (parallel decode workers per script process) |

### Decoders (by protocol)

| Protocol | Decoder | Rationale |
| -------- | ------- | --------- |
| **TG** | **BP+OSD (CPU)** | Same class of issue as fold / transversal CNOT in `logical_op_benchmark`: correlated DEM structure is a poor fit for MWPM-only decoders; BP+OSD is the safe default. |
| **ZZ-LS, XX-LS** | **PyMatching (CPU, 8 workers)** | Surgery DEM for these teleport circuits is handled well by MWPM in practice. |

Scripts still accept `--decoder` to override for debugging.

### Data columns (CSV)

Each script appends one row per (d, state, p, decoder) after that task **finishes**. On restart, any row already in the CSV is skipped (key = `d`, `p`, `state`, `decoder`). If the process dies **during** one (d, p, state) task, that task has **no row yet** — the next run **re-runs that task from scratch** (no within-task resume).

**Output paths (from repo root `LightStim/`):**

- `eval/logical_circuit_benchmark/bell-teleportation/results/tg_results.csv`
- `eval/logical_circuit_benchmark/bell-teleportation/results/ls_zz_results.csv`
- `eval/logical_circuit_benchmark/bell-teleportation/results/ls_xx_results.csv`

Shared semantics: `logical_error_rate` is from the simulation pipeline (post-selected shots if applicable).

**TG** (`results/tg_results.csv`):  
`d, state, rounds_pre, rounds_mid, rounds_post, p, decoder, num_qubits, num_detectors, num_observables, shots, errors, logical_error_rate, build_time_sec, decoding_time_sec`

**ZZ-LS / XX-LS** (`ls_zz_results.csv`, `ls_xx_results.csv`):  
`d, state, rounds_pre, rounds_ls, p, decoder, ...` (same trailing columns as TG).

### Plot (suggested)

LER vs PER (log–log): one subplot per protocol **or** one panel with protocol as line style; **two curves**
per series for |X⟩ and |Z⟩; color = **d**. Optional: dashed memory baseline from
`eval/logical_op_benchmark/` or `eval/memory_benchmark/` at matching (d, p, rounds).

---

## Figure 2: 7-to-1 |Y⟩ distillation — output fidelity vs PER

Unchanged from the earlier plan; see `eval/logical_circuit_benchmark/distillation/ls_7to1/setup.md`
and `distillation/tg_7to1/` for scripts and prior sweeps.

---

## Running (tmux-friendly)

From repo root:

```bash
cd eval/logical_circuit_benchmark/bell-teleportation

# Sanity: build + noiseless sample only
python run_tg.py --build-only
python run_ls_zz.py --build-only
python run_ls_xx.py --build-only

# Full sweeps (defaults: p list above, d=3,5,7, states X Z)
python run_tg.py
python run_ls_zz.py
python run_ls_xx.py
```

Optional: `-d 3`, `-p 1e-3`, `--max-shots`, `--num-workers`, `--decoder …`.

### Three detached tmux sessions (parallel)

**推荐：** 在仓库里直接跑脚本，**不依赖** 你先 `export REPO` / `export BT`（忘记设置 `BT` 时 `cd $BT` 为空，子 shell 会立刻失败，tmux 会话马上退出，`tmux ls` 看不到新 session）。

```bash
# 在任意 cwd 执行均可
/home/xiang/workspace/LightStim/eval/logical_circuit_benchmark/bell-teleportation/start_tmux_sweeps.sh
```

若你的 clone 不在上述路径，先 `cd` 到该目录再执行：

```bash
cd /path/to/LightStim/eval/logical_circuit_benchmark/bell-teleportation
./start_tmux_sweeps.sh
```

脚本会：用 **本脚本所在目录** 作为 `BT`、自动 `../../..` 找到仓库根并 `source venv/bin/activate`（存在时）、创建 `results/*.log`、跳过已存在的同名 tmux session。

**手动一行版（必须把路径写满，勿单独复制 `cd $BT` 那行）：**

```bash
BT=/home/xiang/workspace/LightStim/eval/logical_circuit_benchmark/bell-teleportation
mkdir -p "$BT/results"
tmux new-session -d -s bell-tg  bash -lc "source /home/xiang/workspace/LightStim/venv/bin/activate 2>/dev/null; cd '$BT' && python run_tg.py 2>&1 | tee '$BT/results/run_tg.log'"
tmux new-session -d -s bell-zz bash -lc "source /home/xiang/workspace/LightStim/venv/bin/activate 2>/dev/null; cd '$BT' && python run_ls_zz.py 2>&1 | tee '$BT/results/run_ls_zz.log'"
tmux new-session -d -s bell-xx bash -lc "source /home/xiang/workspace/LightStim/venv/bin/activate 2>/dev/null; cd '$BT' && python run_ls_xx.py 2>&1 | tee '$BT/results/run_ls_xx.log'"
```

Attach: `tmux attach -t bell-tg`（或 `bell-zz` / `bell-xx`）。列表: `tmux ls`。

---

## TODO

- [ ] Generate Fig 1 plots from the three CSVs (+ optional memory baseline).
- [ ] Distillation Fig 2a: noiseless-Clifford noise model (inject noise only on |Y⟩ inputs).
