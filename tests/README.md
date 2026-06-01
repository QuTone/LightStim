# LightStim Test Suite

## Quick start

```bash
# Install test dependencies
pip install pytest pytest-timeout

# Run all CI tests (~89 tests, ~30s)
pytest tests/ -m "not slow" --timeout=90 -q

# Run only the core smoke tests
pytest tests/ -m smoke --timeout=60 -q
```

## Test tiers

| Marker | Purpose | When it runs |
|---|---|---|
| `smoke` | Core invariants, < 30s | Every commit |
| *(no marker)* | Medium-speed integration tests | Before PR merge |
| `slow` | Long-running (> 1 min) | Manual / pre-release |

CI runs only `not slow` (~89 tests, ~30s).

## Files

| File | What it covers | Speed |
|---|---|---|
| `test_protocols.py` | Noiseless build of all 17 protocols; checks `num_detectors > 0`, zero detection events, DEM is constructible. Includes two mathematical invariants (a 2-patch system has 2 logical qubits; S⁴ = I). | Fast |
| `test_pipeline.py` | Noisy circuit → `SimulationPipeline` → LER > 0; verifies the noise-injection and decoding chain is intact. | Fast |
| `test_export.py` | Schema validation for `lightstim.frontend.export_all()` output; locks the front-end/back-end JSON contract. | Fast |
| `test_api.py` | FastAPI endpoint smoke tests (via `TestClient`, no live server). | Fast |
| `test_back_propagated_pauli.py` | Unit tests for the Clifford tracking in `SyndromeTracker`. | Fast |
| `test_simulation_backend_quality.py` | Decoder-backend edge cases: unknown decoder errors, post-selection, `list_decoders()` deduplication. | Fast |
| `color_code/test_color_code.py` | Color-code math: stabilizer commutation, logical operator weight, qubit counts. | Fast |
| `test_run_memory.py` | End-to-end CLI tests for `benchmarks/memory/run_memory.py`; most are marked `slow`. | Slow |

## The core invariant

LightStim's central value is **automatic, correct detector generation** (via `SyndromeTracker`). The most important invariant is:

> A correctly constructed noiseless circuit must produce zero detection events and zero observable errors when sampled.

Every protocol in `test_protocols.py` verifies this. Any change to the tracker, coupler geometry, or builder SE construction that breaks this invariant will be caught.

## Why we don't test LER accuracy

- LER values are **experimental results**, not a code contract. Improvements to detector construction may lower LER — such changes should be allowed to merge.
- Verifying LER accuracy needs minute-scale shot counts, which doesn't fit CI.
- Threshold checks (e.g. `d=5 < d=3`) and raw-vs-decomposed consistency are better suited to **manual pre-release review** rather than CI.

## Adding tests for a new protocol

When you add a new protocol, add a noiseless build test to the matching class in `test_protocols.py`:

```python
def test_my_new_protocol(self):
    from lightstim.protocols.my_protocol import MyProtocol
    exp = MyProtocol(distance=3, rounds=2, noise_params=None)
    c = build_quiet(exp.build)
    assert_valid_circuit(c)
    assert_noiseless(c)
    assert_dem_valid(c)
```
