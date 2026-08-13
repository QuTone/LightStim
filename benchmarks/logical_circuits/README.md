# Logical Circuits Benchmark

Benchmarks for multi-patch logical circuit protocols: Bell-state teleportation,
S-gate teleportation, routing overhead, and magic state distillation (LS and TG 7-to-1).

All experiments share a single unified runner with per-task checkpointing.

## Experiments

| `--experiment` | Protocol | Output CSV |
|---|---|---|
| `bell_tele` | Bell-state teleportation via TG / ZZ-LS / XX-LS | `results/bell_tele_results.csv` |
| `s_gate_tele` | Logical S-gate teleportation via ZZ-LS / transversal CNOT | `results/s_gate_tele_results.csv` |
| `distill_ls` | LS 7-to-1 \|Y⟩ distillation (Steane protocol) | `results/distill_ls_results.csv` |
| `distill_tg` | TG 7-to-1 \|Y⟩ distillation (PQRM hypercube) | `results/distill_tg_results.csv` |
| `all` | All of the above | all three CSVs |

Protocol implementations:
- Bell teleportation: `lightstim/protocols/bell_teleportation.py`
- S-gate teleportation: `lightstim/protocols/gate_teleport.py`
- LS distillation: `lightstim/protocols/ls_distillation.py`
- TG distillation: `lightstim/protocols/tg_distillation.py`

## How to run

```bash
# Quick smoke test (d=3,5, 2 p-values):
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py --quick

# Bell teleportation sweep:
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py \
    --experiment bell_tele \
    --distances 3 5 7 \
    --p-values 5e-4 1e-3 2e-3 5e-3

# S-gate teleportation sweep:
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py \
    --experiment s_gate_tele \
    --codes unrotated_sc \
    --s-gate-methods ZZ cnot_trans \
    --state-preps logical_gate \
    --distances 3 5 7 \
    --p-values 5e-4 1e-3 2e-3 5e-3

# LS 7-to-1 distillation, circuit-level noise:
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py \
    --experiment distill_ls \
    --distances 3 5 7 \
    --p-values 1e-3 3e-3 5e-3

# LS distillation, injection-only noise (post-selection overhead analysis):
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py \
    --experiment distill_ls \
    --noise-mode injection \
    --p-injected 1e-3 5e-3 2e-2

# TG 7-to-1 distillation (GPU recommended for d=7):
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py \
    --experiment distill_tg \
    --distances 3 5 7 \
    --p-values 1e-3 3e-3 5e-3 \
    --decoder gpu_bposd --num-workers 1

# All experiments:
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py --experiment all
```

### Key CLI options

| Flag | Default | Description |
|---|---|---|
| `--experiment` | `bell_tele` | Experiment to run (see table above) |
| `--distances` | `3 5 7` | Code distances |
| `--p-values` | `5e-4 1e-3 2e-3 5e-3` | Physical error rates |
| `--codes` | `unrotated_sc` | Codes for S-gate teleportation |
| `--s-gate-methods` | all | S-gate teleportation methods |
| `--state-preps` | `logical_gate` | S-gate resource-state preparation modes (`logical_gate` or `inject`) |
| `--noise-mode` | `circuit_level` | `circuit_level` or `injection` |
| `--p-injected` | — | Injection noise p (injection mode only) |
| `--decoder` | auto | `pymatching`, `mwpf`, `cpu_bposd`, `gpu_bposd`, `mle-ilp` (`bposd` and `nv-qldpc-decoder` are accepted aliases) |
| `--mle-time-limit` | `0` | Soft seconds-per-shot limit for `mle-ilp`; zero is unlimited |
| `--on-decode-failure` | `error` | Count, discard, or ignore decoder timeout/failure shots |
| `--num-workers` | `8` | Parallel CPU workers (use 1 for GPU) |
| `--max-shots` | `1e9` | Max shots per task |
| `--max-errors` | `100` | Stop after N logical errors |
| `--batch-size` | `10000` | Simulation batch size |
| `--quick` | off | Smoke test: d=3,5, 2 p-values |

The formal S-gate benchmark defaults to `unrotated_sc` with `logical_gate`
resource-state preparation. `inject` is available for explicit comparisons, but
it is a non-FT resource-state preparation path. Rotated/injection variants are
kept as protocol/notebook smoke demos until rotated logical S/S† provides a
fault-tolerant verification path.

> **Decoder guidance**: Bell teleportation (TG protocol) and TG distillation produce
> hyperedge DEMs. Use `mle-ilp` for an exact small-instance reference, `mwpf`
> (CPU), `cpu_bposd` (CPU BP+OSD), or `gpu_bposd`
> (CUDA BP+OSD via `cudaq_qec`) for these.
> LS-based experiments (ZZ-LS, XX-LS, LS distillation) use `pymatching`.
> `gpu_bposd` requires a visible NVIDIA GPU (`nvidia-smi -L`).

Exact MLE is unlimited by default. Large DEMs can have long solve tails, so a
bounded benchmark should set both `--mle-time-limit` and an explicit
`--on-decode-failure` policy, for example:

```bash
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py \
    --experiment distill_tg --distances 3 --p-values 1e-3 \
    --decoder mle-ilp --mle-time-limit 2 --on-decode-failure discard
```

## Output format

### bell_tele / routing

```
gate, protocol, state, routing_mult, d, rounds, p,
decoder, decoder_time_limit, on_decode_failure,
shots, errors, logical_error_rate, seconds
```

### s_gate_tele

```
experiment, code, method, state_prep, d, rounds, p,
decoder, decoder_time_limit, on_decode_failure,
shots, errors, logical_error_rate, seconds
```

### distill_ls / distill_tg

```
experiment, d, rounds, p_injected, noise_mode, p, p_in,
decoder, decoder_time_limit, on_decode_failure,
shots, post_selected_shots, post_selection_rate,
errors, logical_error_rate, seconds
```

## How to plot

```bash
# Bell teleportation:
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/bell-teleportation/plot_bell_tele.py

# S-gate teleportation:
PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/s-gate-teleportation/plot_s_gate_tele.py
```

> Distillation plot scripts are not yet implemented. Results CSV can be inspected directly.
