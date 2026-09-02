# LightStim Simulation API Reference

**Module:** `lightstim.simulation.decoder_backend`

The simulation layer takes a built `stim.Circuit` and runs it through a
sample → post-select → decode → stats pipeline.

---

## `SimulationPipeline`

The unified entry point. One instance, one `.run()` call.

```python
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

pipeline = SimulationPipeline(
    # ── Decoder ─────────────────────────────────────────────────────────────
    decoder_config=DecoderConfig("pymatching"),  # see DecoderConfig below
    
    # ── Stopping criteria ───────────────────────────────────────────────────
    max_errors=200,        # Stop after this many logical errors (primary)
    max_shots=1_000_000,   # Hard cap on total shots
    
    # ── Parallelism ─────────────────────────────────────────────────────────
    batch_size=10_000,     # Shots per batch
    num_workers=4,         # Parallel worker processes (CPU only)
    
    # ── Post-selection ──────────────────────────────────────────────────────
    # (all optional; None = no post-selection on that dimension)
    post_select_detector_indices=None,              # List[int] — raw detector indices
    post_select_observable_indices=None,            # List[int] — raw observable indices
    post_select_corrected_observable_indices=None,  # List[int] — after decoding correction
    
    # ── Logical observable targeting ─────────────────────────────────────────
    target_observable_indices=None,  # List[int] — which observables count as errors; None = all
    
    # ── Progress ────────────────────────────────────────────────────────────
    print_progress=True,
)

stats: SimulationStats = pipeline.run(circuit)
```

### Post-selection modes

| Parameter | What it filters | Typical use |
|---|---|---|
| `post_select_detector_indices` | Shots with any detection event at raw index *i* | State injection ancilla |
| `post_select_observable_indices` | Shots with a raw (pre-decode) observable flip | Rarely used |
| `post_select_corrected_observable_indices` | Shots where decoder-corrected observable is 1 | Magic state distillation — post-select on output qubits |

`post_select_detector_indices` can also be auto-detected from circuit tags (detectors
tagged with `post-select` are discovered by `pipeline._resolve_post_select_indices(circuit)`).

### `allow_gauge_detectors`

Set `True` only for circuits where some detectors are intentionally non-deterministic
(e.g. mixed-boundary measurements). Otherwise the pipeline warns about gauge detectors.

---

## `DecoderConfig`

```python
from lightstim.simulation.decoder_backend import DecoderConfig

config = DecoderConfig(
    name="pymatching",    # Decoder name — see table below
    backend="cpu",        # "cpu" | "gpu" | "fpga"
    params={},            # Decoder-specific kwargs (e.g. {"max_bp_iters": 100})
)
```

### Available decoders

| `name` | Backend | Notes |
|---|---|---|
| `"pymatching"` | cpu | MWPM. Always available. Use `decompose_errors=True` internally. |
| `"bposd"` | cpu | BP+OSD via `stimbposd`. Good for LDPC. |
| `"mwpf"` | cpu | Minimum-Weight Parity Factor. Handles hyperedges natively. Required for PQRM/CrossLS. |
| `"nv-qldpc-decoder"` | gpu | GPU BP+OSD via `cudaq_qec`. Use large `batch_size` (≥50 000). |
| `"ldpc-bp"` | cpu | Plain BP (no OSD) via `ldpc.BpDecoder`. High LER floor alone; pair with `"bposd"` or chain into `"relay-bp"` for competitive accuracy on QLDPC codes. |
| `"relay-bp"` | cpu | Relay-BP via `relay_bp`. Handles DEM hyperedges natively (no `decompose_errors`). |
| `"tesseract"` | cpu | Beam-search MLE via `tesseract_decoder`. Lazy import — a mismatched prebuilt wheel only affects this decoder. |
| `"chain"` | cpu | Multi-level escalation over other registered decoders (see below). |

### Multi-level decoder chain (hierarchical decoding)

Escalate non-converged shots from a fast decoder to a slower/more accurate
one — e.g. plain BP → relay-BP → MLE — via the `"chain"` decoder. Stage
*k+1* only re-decodes the shots stage *k* flagged as failed.

```python
# Shorthand: a plain list of DecoderConfigs becomes a chain.
# The *last* config's on_decode_failure is the chain-level policy.
pipeline = SimulationPipeline(decoder_config=[
    DecoderConfig("ldpc-bp", params={"max_iter": 200}),
    DecoderConfig("relay-bp", params={"num_sets": 300}, on_decode_failure="error"),
])

# Equivalent explicit form.
pipeline = SimulationPipeline(decoder_config=DecoderConfig(
    "chain", params={"stages": ["ldpc-bp", "relay-bp"]}, on_decode_failure="error",
))
```

A stage's own `on_decode_failure` has no effect — inside the chain,
"failure" means "escalate," and only the final policy applies to shots
every stage fails on.

---

## `SimulationStats`

Returned by `pipeline.run(circuit)`.

```python
stats.shots                  # int — total shots attempted
stats.post_selected_shots    # int — shots that passed post-selection
stats.errors                 # int — logical errors observed
stats.seconds                # float — wall-clock time

stats.logical_error_rate     # float = errors / post_selected_shots
stats.post_selection_rate    # float = post_selected_shots / shots

stats.ler_error_bar(z=1.96)  # float — half-width of z-sigma Wilson CI
stats.ler_error_bar(z=1.0)   # 1-sigma half-width

stats.decoder                # str — decoder name used
stats.json_metadata          # Dict — from ExperimentTask.json_metadata
```

---

## `ExperimentTask`

Optional wrapper for passing metadata alongside a circuit:

```python
from lightstim.simulation.decoder_backend.pipeline import ExperimentTask

task = ExperimentTask(circuit, json_metadata={"d": 5, "p": 1e-3, "code": "rotated_sc"})
stats = pipeline.run(task.circuit)
# stats.json_metadata == {"d": 5, "p": 1e-3, "code": "rotated_sc"}
```

---

## Common usage patterns

### Threshold sweep

```python
import numpy as np
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),
    max_errors=200,
    max_shots=500_000,
    print_progress=False,
)

results = []
for d in [3, 5, 7]:
    for p in np.logspace(-3, -1, 10):
        circuit = build_memory_circuit(d=d, p=p)  # your build function
        stats = pipeline.run(circuit)
        results.append({"d": d, "p": p, "ler": stats.logical_error_rate,
                         "eb": stats.ler_error_bar()})
```

### Distillation with post-selection on output

```python
from lightstim.simulation.observable_analysis import (
    build_obs_patch_matrix, identify_distillation_observables
)

# Find which observable index is the target qubit
matrix, patch_names = build_obs_patch_matrix(circuit, system)
T, target_obs, ps_obs = identify_distillation_observables(matrix, patch_names, ["output_patch"])

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("bposd"),
    max_errors=50,
    post_select_corrected_observable_indices=ps_obs,
    target_observable_indices=target_obs,
)
stats = pipeline.run(circuit)
```

### GPU decoding

```python
pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("nv-qldpc-decoder", backend="gpu"),
    batch_size=100_000,   # Large batches amortize GPU launch overhead
    num_workers=1,        # GPU decoder uses one process
    max_errors=200,
)
# Always use venv/bin/python — system Python lacks cudaq_qec.
# Check nvidia-smi before launching.
```

---

## `NoiseConfig` quick reference

**Module:** `lightstim.noise.config`

```python
from lightstim.noise.config import NoiseConfig

noise = NoiseConfig(
    p_1q=1e-3,    # Depolarizing after single-qubit gates (H, S, ...)
    p_2q=1e-3,    # Depolarizing after two-qubit gates (CX, CZ, ...)
    p_meas=1e-3,  # Measurement bit-flip probability
    p_reset=1e-3, # State-preparation bit-flip probability
    p_idle=0,     # Idle placement depends on the selected noise model
    custom_params={},  # Arbitrary extra rates for custom NoiseInjector rules
)
```

Pass to `builder.build_noisy_circuit(noise, noise_model)` where `noise_model` is one of:
`'circuit_level'`, `'uniform_circuit_level'`, `'phenomenological'`,
`'code_capacity'`, `'XZ_biased'`.

`circuit_level` preserves LightStim's historical convention: `p_idle` is
applied to active data qubits at each `TICK[SE_start]`. Use
`uniform_circuit_level` when every `TICK`-delimited moment should apply
`p_idle` to the complement of that moment's operated qubits. Empty, reset-only,
and measurement-only moments count; a non-empty final moment without a trailing
`TICK` also counts. Moments whose physical operations are all tagged
`noiseless` do not. `TICK[noiseless]` suppresses an otherwise empty moment;
it does not hide untagged physical operations in a mixed moment.

Here, *uniform* describes idle-noise coverage, not equal error rates. To use
the common paper convention with one physical error probability, explicitly
set `p_1q = p_2q = p_meas = p_reset = p_idle = p`. Gate and SPAM placement
otherwise remains the same as LightStim's existing `circuit_level` model.

Tick-aligned `REPEAT` blocks remain compressed. To preserve exact moment
semantics, other repeat layouts (including nested or iteration-spanning
moments) are expanded during injection; normalize very large repeat bodies
with explicit `TICK` boundaries to avoid a large expansion.

---

## MWPF Decoder Configuration

MWPF (Minimum-Weight Parity Factor, arXiv:2508.04969) handles hyperedges natively —
required for fold-transversal gates, transversal CNOT, TG distillation, CrossLS, and
color codes. Surface code / LS circuits without hyperedges should use `pymatching`.

### Key parameter: `cluster_node_limit` (c)

Controls the maximum number of dual variables per cluster. The most important tuning knob.

| `c` | Behavior | When to use |
|-----|----------|-------------|
| `0` | Unlimited (full HyperBlossom) | Highest accuracy, slowest. Code-capacity benchmarks. |
| `50` | Good tradeoff | **Surface code circuit-level** (default) |
| `200` | More aggressive | **Color codes, BB/qLDPC codes** — complex hypergraph structures |

### Recommended per code family

| Code | `c` | BP pre-processing | Notes |
|------|-----|-------------------|-------|
| Rotated/Unrotated Surface Code | 50 | No | PyMatching faster for no-hyperedge circuits |
| Toric Code | 50 | No | Same as surface |
| Color Code (6-6-6) | **200** | No | Hyperedges essential — MWPF outperforms PyMatching significantly |
| BB Code (qLDPC) | **200** | Optional | BP pre-processing can further improve accuracy |

### Configuration examples

```python
from lightstim.simulation.decoder_backend import DecoderConfig

# Surface code — use defaults (c=50, SolverSerialJointSingleHair)
config = DecoderConfig(name="mwpf")

# Color code or BB code — larger clusters
config = DecoderConfig(name="mwpf", params={"cluster_node_limit": 200})

# qLDPC with BP pre-processing (requires ldpc package)
config = DecoderConfig(name="mwpf", params={
    "cluster_node_limit": 200,
    "bp": True,
    "max_iter": 100,
    "bp_method": "ms",            # min-sum
    "ms_scaling_factor": 0.625,
    "bp_weight_mix_ratio": 1.0,
})

# Fast mode (Hypergraph Union-Find — lowest accuracy, fastest)
config = DecoderConfig(name="mwpf", params={
    "decoder_type": "SolverSerialUnionFind",
    "cluster_node_limit": 0,
})
```

**Critical**: always use `decompose_errors=False` with MWPF (LightStim's pipeline does
this by default). `decompose_errors=True` destroys hyperedge information and degrades
MWPF to MWPM-equivalent performance.

**Version requirement**: `mwpf >= 0.2.8`. Import via
`from mwpf import SinterMWPFDecoder` rather than sinter's built-in decoder name.
