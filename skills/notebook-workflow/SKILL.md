# Notebook Workflow

How to use LightStim notebooks: from protocol prototyping through packaging,
benchmarking, and final demo format.

---

## 1. Development lifecycle

```
[PROTOTYPE] notebook
      │  implement circuit logic, visualize, debug
      ↓
lightstim/protocols/<name>.py    ← package the protocol
      │
      ↓
benchmarks/<category>/           ← large-scale numerical sweep
      │  run_<name>.py  (CSV output, checkpointing)
      │  plot_<name>.py (reads CSV, saves PNG)
      ↓
[DEMO] notebook                  ← import from protocols, strip raw code
```

---

## 2. Notebook status labels

Add one of these labels at the top of every notebook (as a Markdown cell):

```
**[PROTOTYPE]** — protocol code lives here; not yet packaged into lightstim/protocols/
```

```
**[DEMO]** — protocol is in lightstim/protocols/; this notebook only imports and demonstrates
```

A DEMO notebook should contain **no raw circuit-building logic** — only imports,
a small visualization, and a small hardcoded numerical result.

---

## 3. Verification before packaging

Before moving a protocol from notebook to `lightstim/protocols/`, validate with a
small-scale run (d=3, a handful of p-values):

**Pass criteria:**
- LER ≤ 10× PER at the target operating point → circuit is connected, decoding is working
- LER < PER (breakeven) → ideal, but not required at d=3 (color code needs d≥5)

**Fault-tolerant check** (applies to memory, gate, distillation):
- Compare LER at d=3 vs d=5 under the same p
- FT protocol: d=5 should be ~10× better than d=3
- Non-FT (e.g., state injection): no distance suppression expected — LER is
  dominated by injection error probability, not code distance

**What to look for if verification fails:**
- LER ≈ 50%: detector or observable wiring error — check tracker output
- LER ≈ PER (no suppression on FT circuit): noise is not on the right qubits,
  or boundary stabilizers are missing
- Huge LER variance: too few shots — increase or use a higher p for smoke test

---

## 4. Circuit visualization tricks

### Use fewer rounds for readable diagrams

Detslice diagrams with many rounds are hard to read. Use 1–2 SE rounds:

```python
circuit = MyProtocol(d=3, rounds=2).build()
circuit.without_noise().diagram("detslice-with-ops-svg")
```

This keeps the diagram to 2-3 columns and makes detector connections legible.

### Comment out large diagram cells before committing

Inline SVG from `.diagram(...)` is embedded as a base64 blob in the `.ipynb` file.
A single detslice diagram for d=7 can add 1–3 MB to the file.

**Before `git add` or `git commit`, comment out or clear output from cells like:**

```python
# Comment out before committing:
# circuit.without_noise().diagram("detslice-with-ops-svg")

# Alternatively, clear all outputs with: Kernel → Restart & Clear Output
```

If you want to keep the visualization result, export it to a PNG/SVG separately
and reference it from a Markdown cell instead.

### Filter to a subset of detectors or observables

Use `filter_coords` to isolate a specific stabilizer type or spatial region:

```python
# Show only Z-type detectors (coords where z-component matches)
circuit.without_noise().diagram(
    "detslice-with-ops-svg",
    filter_coords={2: 1},   # example: filter on the 3rd coord = 1
)
```

This is useful when the full diagram is too dense to read.

### Zoom into a specific time segment

Use `tick=range(start, end)` to show only a slice of the circuit in time:

```python
# Show only the second SE round (ticks 5–10, for example)
circuit.without_noise().diagram(
    "detslice-with-ops-svg",
    tick=range(5, 11),
)
```

Useful for inspecting the boundary between rounds, or the final measurement.

---

## 5. Demo notebook structure

After packaging, a notebook should follow this layout:

```python
# Cell 1: imports
from lightstim.protocols.my_protocol import MyProtocol
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig
import stim

# Cell 2: circuit visualization (small scale, 1-2 rounds)
circuit = MyProtocol(d=3, rounds=2).build()
# circuit.without_noise().diagram("detslice-with-ops-svg")  # ← commented out before commit

# Cell 3: small numerical result (hardcoded, no sweep loop)
import lightstim.noise.config as nc
noisy = ...  # inject noise
pipeline = SimulationPipeline(DecoderConfig("pymatching"), max_shots=10_000, max_errors=100)
stats = pipeline.run(noisy)
print(f"d=3, p=1e-3: LER = {stats.logical_error_rate:.2e}")
```

No sweep loops, no CSV output, no argparse. Those belong in `benchmarks/`.

---

## 6. Protocol → notebook mapping

See `notebooks/README.md` for the full table of notebooks and their corresponding
protocols.
