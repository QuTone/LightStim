# LightStim

LightStim is a modular Quantum Error Correction (QEC) framework built on [Stim](https://github.com/quantumlib/Stim). It provides high-level abstractions for constructing fault-tolerant circuits with **automatic detector generation**, running simulation and decoding pipelines, and comparing logical error rates across codes and protocols.

## What this repo is for

- Build QEC experiments from reusable abstractions (`QECPatch`, `QECSystem`, `CircuitBuilder`, `SyndromeTracker`)
- Support multi-patch workflows (transversal gates, lattice surgery, state injection)
- Inject standardized noise models (`code_capacity`, `phenomenological`, `circuit_level`, `XZ_biased`)
- Decode with a unified backend (PyMatching, BP+OSD CPU/GPU, MWPF, Relay-BP, Tesseract)
- Analyze and visualize logical error rates
- Inspect circuits interactively in a browser (DEM 3D, Circuit Timeline, DetSlice animator)

## Repository layout

```text
LightStim/
├── lightstim/                  # Main library package
│   ├── ir/                     # Core abstractions (QECPatch, QECSystem, CircuitBuilder, SyndromeTracker)
│   ├── qec_code/               # Code implementations (surface, toric, BB, color, PQRM, repetition)
│   ├── noise/                  # Noise config and injectors
│   ├── protocols/              # Experiment orchestration (memory, CNOT, lattice surgery, ...)
│   ├── simulation/             # Decoder backend and simulation pipeline
│   ├── frontend/               # Circuit → JSON exporters (powers the web UI)
│   └── plot/                   # Plotting helpers
├── server/                     # Optional FastAPI HTTP server (powers the web UI)
├── docs/
│   ├── api/                    # Library API reference (ir.md, simulation.md)
│   ├── getting_started.md      # Installation, quick start, QEC concepts
│   └── vision.md               # Design philosophy and LightStim + AI workflow
├── skills/                     # Claude Code skill definitions
│   ├── builder-tracker-api/    # Direct CircuitBuilder + SyndromeTracker usage
│   ├── logical-coupler-design/ # Design a new LogicalCouplerProtocol
│   ├── simulate-decode/        # Run SimulationPipeline and get LER
│   ├── custom-noise/           # Configure noise models
│   ├── extend-new-code/        # Add a new QEC code
│   ├── extend-new-decoder/     # Add a new decoder backend
│   └── gotchas/                # Known pitfalls and debugging patterns
├── notebooks/                  # Jupyter notebooks by topic
├── benchmarks/                 # Simulation scripts and paper artifacts
├── requirements.txt
└── Dockerfile
```

## Quick start

### 1) Install

```bash
git clone https://github.com/QuTone/LightStim.git
cd LightStim

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -e .                   # core library (PyMatching included)
pip install -e ".[decoders]"       # optional CPU decoders: BP+OSD, MWPF, Relay-BP, Tesseract
pip install -e ".[server]"         # optional FastAPI server for the web UI
pip install -e ".[dev]"            # development / notebook environment
pip install -e ".[gpu]"            # optional NVIDIA GPU decoder (requires CUDA)
```

> **GPU decoder** (`nv-qldpc-decoder`) requires NVIDIA GPU + CUDA 12.x. Install it only with
> `pip install -e ".[gpu]"` on compatible systems.
>
> **Tesseract**: a prebuilt `tesseract-decoder` wheel may not match every CPU. If it fails to
> import on your machine, build it from source (the repo's `CMakeLists.txt` uses `-march=native`).
> LightStim imports `tesseract_decoder` lazily, so this only affects the Tesseract decoder — not
> `import lightstim` or the other decoders.

### 2) Optional: Jupyter kernel

```bash
python -m ipykernel install --user --name=lightstim --display-name="LightStim"
```

## Usage examples

### Build and simulate a memory experiment

```python
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock,
)
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

# Build circuit
system = QECSystem()
system.add_patch(RotatedSurfaceCode(distance=5), name='main')
tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
builder = CircuitBuilder(tracker, system)
se = RotatedSurfaceCodeExtractionBlock(system)

builder.write_coordinates()
builder.initialize({q: 'Z' for q in system.data_indices}, n=system.num_qubits)
builder.apply_syndrome_extraction(se.circuit, rounds=5)
builder.apply_data_readout({q: 'Z' for q in system.data_indices})

noisy = builder.build_noisy_circuit(
    NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3),
    noise_model='circuit_level',
)

# Simulate and decode
pipeline = SimulationPipeline(
    decoder_config=DecoderConfig('pymatching'),
    max_shots=100_000,
    max_errors=200,
    num_workers=4,
)
stats = pipeline.run(noisy)
print(f"LER: {stats.logical_error_rate:.3e} ± {stats.ler_error_bar():.3e}")
```

### Decoder options

| Decoder | Config | Notes |
|---|---|---|
| PyMatching (MWPM) | `DecoderConfig('pymatching')` | Default. Surface / LS circuits. |
| BP+OSD CPU | `DecoderConfig('bposd')` | LDPC codes, hyperedge circuits. |
| MWPF | `DecoderConfig('mwpf')` | Hyperedges (CrossLS, PQRM, color code). |
| Relay-BP | `DecoderConfig('relay-bp')` | LDPC / hyperedge circuits. Needs `.[decoders]`. |
| Tesseract | `DecoderConfig('tesseract', params={'det_beam': 50})` | Beam-search MLE. Needs `.[decoders]`. |
| GPU BP+OSD | `DecoderConfig('nv-qldpc-decoder')` | NVIDIA GPU. Large d or high p. |

### Available QEC codes

| Code | Patch class | Notes |
|---|---|---|
| Rotated surface | `RotatedSurfaceCode` | Primary code, most complete |
| Unrotated surface | `UnrotatedSurfaceCode` | Lattice surgery coupler available |
| Toric | `ToricCode` | Periodic boundary conditions |
| Color (6-6-6) | `ColorCode` | Fold-transversal H/S gates |
| Bivariate bicycle | `BBCode` | [[144,12,12]] gross code etc. |
| PQRM | `PQRMPatch` | Transversal T gate; CrossLS protocol |
| Repetition | `RepetitionCode` | Classical benchmark |

All codes are in `lightstim/qec_code/`.

## Interactive web UI (optional)

LightStim ships with a small FastAPI server (`server/`) that exposes every
protocol over HTTP. Pair it with the
[LightStim-front-end](https://github.com/QuTone/LightStim-front-end) React
app (needs Node.js 18+) to inspect circuits in your browser: 3D
detector-error-model viewer, circuit timeline, detslice animator, etc.

```bash
# Terminal 1 — backend (this repo; venv activated)
pip install -e ".[server]"          # one-time: installs FastAPI + uvicorn
venv/bin/uvicorn server.main:app --port 9999

# Terminal 2 — front-end (separate repo — clone it next to LightStim)
git clone https://github.com/QuTone/LightStim-front-end.git ../LightStim-front-end
cd ../LightStim-front-end
npm install
npm run dev                         # serves http://localhost:8080
```

Open the dev URL that Vite prints (it picks the next free port if 8080 is
taken), choose a protocol from the sidebar, and hit *Build Circuit*. The
front-end calls `http://localhost:9999` by default; override it with
`VITE_API_URL` in `.env.local`. See [`server/README.md`](server/README.md)
for the full endpoint list and `LightStim-front-end`'s README for the UI side.

> **Running on a remote server (not your laptop)?** `localhost` in your
> browser means *your own* machine, so it won't reach a dev server running on
> the remote host. Forward the ports over SSH from your laptop — e.g.
> `ssh -L 8080:localhost:8080 -L 9999:localhost:9999 you@your-server` (use
> whatever port Vite actually printed) — then open `http://localhost:8080`
> locally. Most IDEs (VS Code, Cursor) auto-forward ports: just click the
> link in their *Ports* panel instead of typing the URL by hand.

> The backend has **no UI of its own** — `localhost:9999` returns plain
> JSON. The visual rendering lives entirely in the front-end repo.

## Try it with your AI coding agent

No tutorial needed. If you use Claude Code, Codex, Cursor, or any AI coding assistant,
paste the prompt below and let it build your first experiment:

> Read `skills/SKILL.md` in this repo first. Then build an unrotated surface code
> ZZ lattice surgery experiment: first patch d=3, second patch d=5 placed at
> offset (0, 10). Prepare the first patch in |+⟩ and the second in |0⟩; measure
> them in Z and X basis respectively. Use circuit-level noise with all error rates
> 1e-3. Choose an appropriate decoder and run a simulation with enough shots to
> get a reliable LER estimate. Print the result.

The skills system routes your AI to the right protocol class, decoder, and
simulation parameters — the same way a collaborator who knows the codebase would.

## Documentation

- `docs/api/ir.md` — `QECPatch`, `QECSystem`, `CircuitBuilder`, `SyndromeTracker`, `LogicalCouplerProtocol`
- `docs/api/simulation.md` — `SimulationPipeline`, `DecoderConfig`, `SimulationStats`, MWPF configuration
- `docs/getting_started.md` — installation, quick start, and QEC concepts
- `docs/vision.md` — design philosophy, key abstractions, and the LightStim + AI workflow
- `lightstim/simulation/README.md` — decoder backend architecture
- `skills/` — task-oriented instructions for AI-assisted development

## Testing

```bash
pip install pytest pytest-timeout
pytest tests/ -m "not slow" --timeout=90 -q   # 89 tests, ~30s
```

See [`tests/README.md`](tests/README.md) for the full test structure, and
[`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, PR conventions,
and how to extend each layer (codes, protocols, decoders, noise models).

## Contact

For collaboration, security disclosures, or anything not suited for a public
issue, email Xiang Fang at <x8fang@ucsd.edu>. For bugs and feature requests,
open a [GitHub issue](https://github.com/QuTone/LightStim/issues).

### Citation

```bibtex
@article{fang2026lightstim,
  author       = {Fang, Xiang and Wang, Ming and Wu, Yue and Prabhu, Sharanya and Tullsen, Dean and Miniskar, Narasinga Rao and Mueller, Frank and Humble, Travis and Ding, Yufei},
  title        = {LightStim: A Framework for QEC Protocol Evaluation and Prototyping with Automated DEM Construction},
  year         = {2026},
  eprint       = {2604.21472},
  archivePrefix= {arXiv},
  primaryClass = {quant-ph}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
