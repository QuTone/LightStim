# LightStim Simulation API Reference

**Module:** `lightstim.simulation.decoder_backend`

`SimulationPipeline` takes a built `stim.Circuit` and runs it through a
sample → post-select → decode → stats pipeline. The optional `ClifftSampler`
below also accepts Clifford+T circuit text for standalone sampling.

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

### Enumerate physical faults

Use the same pipeline to enumerate faults through a fixed number of independent
physical locations. The pipeline applies its detector, observable, target and
decoder settings. Each section of the report contains the probability of
exactly that fault order, including the probability that all other locations
were quiet.

```python
report = pipeline.enumerate_faults(circuit, max_order=2)
for row in report.per_order:
    print(row.order, row.configurations, row.accepted_mass, row.loss_mass)
print(report.conditional_loss, report.omitted_probability, report.full_loss_bounds)
```

`conditional_loss` conditions on **both** acceptance and the enumerated
orders. The bounds include the remaining probability mass. A conditional
quantity with zero possible acceptance is `None`. Each two-qubit depolarizing
channel is one location with 15 mutually exclusive alternatives. Measurement
faults flip the reported result, including its use in feedback, while preserving
the quantum state after measurement. The catalogue supports independent
`X/Y/Z_ERROR`, `DEPOLARIZE1/2`, `PAULI_CHANNEL_1/2` and noisy single-qubit,
paired, and Pauli-product readouts. Each `MXX`, `MYY` or `MZZ` pair must
address two distinct qubits, including when a target is inverted. Correlated-error
chains, heralded noise, and noisy `MPAD` raise an error.

Clifford inputs use Stim's deterministic detector and observable flip signatures,
relative to the noiseless reference. Physical T/T† text or an `EXP_VAL` probe
selects exhaustive Born-branch propagation. The Born evaluator applies gates
to sparse complex amplitudes and integrates the supported measurement, reset,
and feedback branches for each **joint** fault set. Arithmetic is floating
point. Ordinary Clifford input can also request `backend="born"` to use this
calculation with Stim's reference parities. Extended inputs use raw record
parities with zero reference by default. Supply `detector_reference` and
`observable_reference` when the expected parities differ.

```python
import math
from lightstim.simulation.decoder_backend import SimulationPipeline

# A single T prepares (|0> + exp(i*pi/4)|1>)/sqrt(2) from |+>.
# EXP_VAL returns the conditional Pauli expectations in declaration order.
physical = "H 0\nX_ERROR(0.02) 0\nT 0\nX_ERROR(0.02) 0\nEXP_VAL X0 Y0"

def magic_state_infidelity(corrected_target_bits, expectations):
    x, y = expectations
    return (1 - (x + y) / math.sqrt(2)) / 2

report = SimulationPipeline().enumerate_faults(physical, max_order=2, loss=magic_state_infidelity)
assert math.isclose(report.conditional_loss, 0.01)
```

With a decoder, pass `decoder_model=...` for extended physical text containing
detectors or observable frame bits. The caller is responsible for matching
detector order, logical labels and references to that model. The enumerator
caches each observed syndrome's prediction and success flag, so the decoder
must return the same decision and flag for repeated inputs. The optional `loss(corrected_target_bits,
expectations)` receives corrected bits in target-index order and returns
a loss in [0, 1]. The default is one if any targeted observable is flipped.
An `EXP_VAL` probe requires an explicit loss so that the target is defined.
For a deterministic Clifford circuit the pipeline builds its usual decoder
model automatically. An explicit model also works for channels whose exact
physical enumeration is supported but whose DEM Stim cannot directly produce
for the configured decoder.

For identity recovery, or to replay individual witnesses, use the standalone
analyzer. Pass the same decoder configuration, detector selection and reference
bits to reproduce a decoded pipeline experiment.

```python
from lightstim.analysis.faults import FaultEnumerator

audit = FaultEnumerator(circuit)
report = audit.enumerate(max_order=2)
for witness in report.witnesses:
    for site, alternative in witness.faults:
        location = report.locations[site]
        print(location.instruction_index, location.target_start, location.alternatives[alternative])
    assert audit.evaluate(witness.faults) == witness.response
```

Witness indices refer to this flattened circuit's positive-rate physical
catalogue. Witnesses include up to `witness_limit` harmful configurations per
order. Born witnesses must have conditional loss above
`QuantumLimits.roundoff_tolerance`. The report retains the computed numeric losses.
`max_configurations` bounds the number of physical configurations through
`max_order`. `QuantumLimits` bounds circuit expansion, live state sizes,
branches and work for each fixed-fault propagation. The generic Born backend is
intended for small circuits and bounded witnesses. An exceeded limit raises
an exception. Enumeration works with the core LightStim dependencies.

### Sampling source

The sampler draws from the supplied noisy circuit. The detector error model
(DEM) configures the decoder separately. With `sampler_backend="stim"` (the
default), each worker uses `circuit.compile_detector_sampler()`.

For Clifford circuits whose instructions Clifft supports, choose
`sampler_backend="clifft"` to draw the measurement records with Clifft.
Install Clifft in the Python 3.12+ environment running LightStim. The sampler
uses that environment by default. The [installation instructions](#clifftsampler)
also cover running Clifft in a separate environment.
The pipeline converts those records to detector events and observable flips
relative to Stim's noiseless reference. Both backends then use the same
post-selection, decoder, and error accounting.

```python
pipeline = SimulationPipeline(
    sampler_backend="clifft",
    decoder_config=DecoderConfig("pymatching"),
    num_workers=2,
)
stats = pipeline.run(clifford_circuit)
```

Samples follow the same model across worker counts. Work scheduling can change
the exact draws. A failed worker raises `RuntimeError`. The Clifft pipeline requires
measurement record targets in `OBSERVABLE_INCLUDE`. Clifft 0.7.0 reports
unsupported instructions such as heralded channels at compilation. For extended
Clifford+T circuit text and `EXP_VAL`, use the standalone sampler below.

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
(e.g. intentionally retained gauge measurements). The default `False` rejects
non-deterministic detectors during DEM construction.

For compatibility, `True` also enables DEM decomposition with
`ignore_decomposition_failures=True`. Stim retains any undecomposed hyperedges;
the chosen decoder must support them. These settings apply to both single- and
multi-worker execution and do not change the circuit used for sampling. The
pipeline emits a warning when this option is enabled.

---

## `ClifftSampler`

`lightstim.simulation.clifft_backend.ClifftSampler` samples physical Clifford+T
circuit text in batches. Install the tested Clifft version in the Python 3.12+
environment running LightStim.

```bash
python -m pip install clifft==0.7.0
```

```python
from lightstim.simulation.clifft_backend import ClifftSampler

circuit = "RX 0\nT 0\nEXP_VAL X0\nEXP_VAL Y0\nMX 0\n"
with ClifftSampler(circuit, seed=42) as sampler:
    batch = sampler.sample_batch(1024)

print(batch.expectations.mean(axis=0))  # [1/sqrt(2), 1/sqrt(2)]
```

If LightStim runs on Python 3.10 or 3.11, install Clifft in a separate Python
3.12+ environment and set `CLIFFT_PYTHON` to that environment's Python executable
before running LightStim. This setting applies to both the sampler and the
pipeline. Standalone sampler callers can also specify `python_executable`
directly if they need a per-instance override.

The sampler compiles once per instance. Each `ClifftBatch` array has one row per
attempted shot and zero columns if that output is absent from the circuit.

| Field | Meaning |
| --- | --- |
| `measurements` | Classical measurement records in circuit order |
| `detectors` | Raw XOR parities of the declared detector records |
| `observables` | Raw XOR parities at the declared observable indices |
| `expectations` | Values of the explicitly supplied `EXP_VAL` probes, in circuit order |

Raw parities can be nonzero on ideal circuits. An ideal true-T circuit can also
have intrinsically random logical readout. Interpret results using the
experiment's reference, acceptance policy, and metric. This low-level sampler
leaves that interpretation to the caller, independently of the Stim LER pipeline.
Instruction tags are stripped from the transport text; gates, noise, and record
targets are preserved.

The `threads` option defaults to one worker thread. `max_peak_rank` limits the
peak rank reported by Clifft (default `24`), and `timeout` limits each compile or
sample request in seconds (default `300`). On a failed request the sampler
closes its worker. Use the same seed, batch sizes, thread count, and runtime to
replay a run.

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
| `"ionq-beam-search"` | cpu | Optional IonQ BP-guided beam search; [installation, configuration, and reference](../decoder/ionq_beam_search.md). |
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
`'circuit_level'`, `'circuit_level_with_idling'`, `'phenomenological'`,
`'code_capacity'`, `'XZ_biased'`.

`circuit_level` preserves LightStim's historical convention: `p_idle` is
applied to active data qubits at each `TICK[SE_start]`. Use
`circuit_level_with_idling` when every `TICK`-delimited moment should apply
`p_idle` to the complement of that moment's operated qubits. Empty, reset-only,
and measurement-only moments count; a non-empty final moment without a trailing
`TICK` also counts. Moments whose physical operations are all tagged
`noiseless` do not. `TICK[noiseless]` suppresses an otherwise empty moment;
it does not hide untagged physical operations in a mixed moment.

The five error rates remain independently configurable. To use the common
paper convention with one physical error probability, explicitly
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
