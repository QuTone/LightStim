# Contributing to LightStim

Thanks for your interest in contributing! LightStim is a QEC research framework
under active development. The notes below are specific to this repo — read them
before opening a PR so you don't get caught by the gotchas we hit during
development.

If you're using an AI coding assistant (Claude Code, Codex, Cursor), point it
at `skills/SKILL.md` first — that catalog routes the assistant to the right
abstractions for each kind of contribution.

## 1. Development setup

```bash
git clone https://github.com/QuTone/LightStim.git
cd LightStim
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"        # core + decoders + notebook tooling
pip install -e ".[gpu]"        # optional NVIDIA GPU decoder (requires CUDA 12.x)
```

**Always work inside the venv.** The GPU decoder (`cudaq_qec` /
`nv-qldpc-decoder`) is venv-only — running with system Python silently
falls back to garbage results (LER ≈ 99%) instead of erroring out.

For notebook contributions:
```bash
python -m ipykernel install --user --name=lightstim --display-name="LightStim"
```
and **always pick the "LightStim" kernel** in Jupyter — same reason.

## 2. Running tests

The suite is layered by speed marker:

| Marker | Use | When |
|---|---|---|
| `smoke` | Core invariants, < 30s | Every commit |
| *(none)* | Full integration tests | Before opening a PR |
| `slow` | > 1 min (e.g. end-to-end CLI) | Manual / pre-release |

```bash
# What CI runs (must pass before merge):
venv/bin/python -m pytest tests/ -m "not slow" --timeout=90 -q   # ~30s, 89 tests

# Quick local sanity check:
venv/bin/python -m pytest tests/ -m smoke -q                     # ~15s

# Everything including slow tests:
venv/bin/python -m pytest tests/ -q
```

CI runs on every push and PR (`.github/workflows/ci.yml`). PRs whose CI is
red won't be merged — fix locally first.

## 3. Adding a new QEC code

1. Subclass `lightstim.ir.qec_patch.QECPatch` in
   `lightstim/qec_code/<your_code>/code_patch.py` — define qubit coordinates,
   stabilizers, and logical operators.
2. Implement an SE block (`SE_block.py`) that defines syndrome extraction
   circuits for your code's stabilizers.
3. Read `skills/extend-new-code/SKILL.md` for the conventions
   (coordinate system, stabilizer encoding, what the auto-detector pipeline
   needs from you).
4. Add a noiseless-build test for memory experiments using your code in
   `tests/test_protocols.py` (the `TestNoiselessMemory` class).

The bar to merge: noiseless circuits produce **zero detection events and
zero observable errors**. If your code passes this, the auto-detector
pipeline accepts it.

## 4. Adding a new protocol / HTTP endpoint

Protocols live in `lightstim/protocols/` and orchestrate `QECPatch` +
`QECSystem` + `CircuitBuilder` into a buildable experiment (e.g.
`MemoryExperiment`, `BellTeleportTG`).

**If you expose your protocol over HTTP** (so it shows up in the web UI):

1. Add a Pydantic request model + endpoint to `server/main.py`.
2. **Add a smoke test in `tests/test_api.py`** — the test class
   `TestEndpoints` has a pattern to copy. CI gates new endpoints on this.
3. **Cross-repo coordination** — the front-end repo
   ([LightStim-front-end](https://github.com/QuTone/LightStim-front-end))
   has an `ENDPOINT_MAP` in `src/lib/api.ts` and protocol catalog in
   `src/data/experiments.ts`. Add your endpoint there too, or the UI
   won't see it.

If you change the **JSON output shape** (e.g. add a new field to
`export_all()` in `lightstim/frontend/export.py`):
- Update `tests/test_export.py::test_export_top_level_keys` to match
- Update `CircuitPayload` interface in
  `LightStim-front-end/src/lib/api.ts`
- Mention it in your PR description — the front-end repo needs a
  coordinated PR

## 5. Adding a new decoder

Decoders plug into `lightstim/simulation/decoder_backend/` via the
registry pattern. Every decoder is a `sinter.Decoder` subclass
registered under `(name, backend)` via `register_decoder()`.

1. Add `lightstim/simulation/decoder_backend/decoders/<your_decoder>.py`
   with a `sinter.Decoder` subclass implementing
   `compile_decoder_for_dem(*, dem) -> sinter.CompiledDecoder`. The
   compiled decoder's `decode_shots_bit_packed` does the work.
2. Call `register_decoder("<name>", YourDecoder, aliases=[...],
   backend="cpu"|"gpu"|"fpga")` at module bottom so it's discoverable
   via `DecoderConfig(name="<name>", backend="...")`.
3. Add a soft-import hook in
   `lightstim/simulation/decoder_backend/decoders/__init__.py` behind a
   `importlib.util.find_spec` guard so missing optional dependencies
   don't break startup.
4. Add a smoke test in `tests/test_simulation_backend_quality.py` —
   verify `list_decoders()` includes your name and the pipeline produces
   nonzero LER on the trivial single-qubit observable circuit.

The full walkthrough — including three concrete patterns (thin wrapper,
parameter translation, custom DEM-matrix decode) drawn from the existing
PyMatching / BPOSD / MWPF / cudaqx implementations — lives in
[`skills/extend-new-decoder/SKILL.md`](skills/extend-new-decoder/SKILL.md).
Read `skills/simulate-decode/SKILL.md` for how decoders are *used* by
`SimulationPipeline`, and `lightstim/simulation/README.md` for the
backend architecture.

## 6. Adding a new noise model

Noise is applied as a post-processing pass on a clean circuit
(`lightstim/noise/injector.py`), driven by composable `NoiseRule`s.

1. Define new rules in `lightstim/noise/rules.py` if you need new
   error channels (e.g. amplitude damping, leakage). Each rule maps gate
   names to stim noise instructions.
2. Register the model in `NoiseInjector` so it can be selected by name.
3. Add any new noise parameters to `lightstim/noise/config.py`
   (`NoiseConfig`).
4. Add a smoke test in `tests/test_pipeline.py` — at minimum verify
   `stats.errors > 0` at `p=0.05` (proves the noise actually injects).

Read `skills/custom-noise/SKILL.md` for the design rationale (rule
composition, why post-processing instead of inline).

## 7. Notebook policy

Notebooks in `notebooks/` are tracked **with outputs** for reproducibility
— a reader should be able to see the result without running anything. But:

- **Trim oversized SVG outputs** (single cell > 10 MB) — stim circuit
  diagrams can balloon; comment out the `.diagram(...)` call and leave a
  note (e.g. `# uncomment to render, large SVG`) instead of shipping it
- **Always run with the `LightStim` kernel** (venv) — system kernel
  produces wrong results for any cudaq_qec code
- **Don't commit `.ipynb_checkpoints/`** (already gitignored)
- For a new feature, add or update a notebook in the matching subdir
  (`Memory/`, `LogicalOps/`, `LogicalCircuits/`, `System/`, `CrossLS/`)

If you add a new notebook, make sure `git ls-files notebooks/ | xargs du -ck`
total stays under ~60 MB. Bigger artifacts should go to the Zenodo archive
referenced in `benchmarks/*/README.md`.

## 8. CI requirements

Every PR must:

1. Pass `pytest tests/ -m "not slow" --timeout=90` (the CI command).
2. Include tests for new code paths:
   - New QEC code → `tests/test_protocols.py`
   - New HTTP endpoint → `tests/test_api.py`
   - New decoder → `tests/test_simulation_backend_quality.py` or `test_pipeline.py`
   - New noise model → `tests/test_pipeline.py`
3. Not regress existing tests.

You can simulate CI locally with:
```bash
venv/bin/python -m pytest tests/ -m "not slow" --timeout=90 -q
```

## 9. Commit message style

We use Conventional Commits. Common prefixes:

| Prefix | When |
|---|---|
| `feat(<area>):` | New functionality |
| `fix(<area>):` | Bug fix |
| `refactor(<area>):` | Restructuring without behavior change |
| `test:` / `test(<area>):` | Adding or fixing tests |
| `docs:` / `docs(<area>):` | Documentation only |
| `chore(<area>):` | Maintenance, deps, formatting |

`<area>` is one of: `api`, `ir`, `noise`, `protocols`, `simulation`,
`qec_code`, `tests`, `notebooks`, `frontend`. Examples:

```
feat(api): add /api/circuit/multi-patch-ls endpoint with init_basis override
fix(tracker): handle paused stabilizer reclassification for N>2 coupler
refactor(simulation): split decoder backend into per-decoder modules
test(protocols): cover noiseless build for all 17 protocols
```

## 10. PR scope and paper artifacts

**Don't modify** these directories without coordinating — they are
paper-reproducibility artifacts:

- `paper_artifact/*/precompute/` — canonical reference data committed
  by the authors. Reviewers compare their local re-runs against these.
  Changing them invalidates the paper's reproducibility claim.
- `LICENSE`, `CITATION.cff` (if present)

**These are gitignored, don't worry about them in PRs**:

- `paper_artifact/*/results/` (local re-runs)
- `benchmarks/*/results/` (archived on Zenodo separately)
- `archive/`, `processing/`, `eval/` (development scratchpads)

**Keep PRs scoped.** One feature / one fix per PR. If you're tempted to
"clean up while I'm here," split that into a separate `chore: ...` PR.
Mixed PRs are harder to review and harder to revert if needed.

## License

By contributing, you agree that your contributions will be licensed under
the Apache License 2.0, the same as the rest of LightStim.

## Questions?

Open an issue with the `question` label. For larger design discussions
(new code family, new protocol, architecture change), open an issue first
to align on direction before writing the PR.

For private inquiries (collaboration, security disclosures, anything not
suited for a public issue), email Xiang Fang at <x8fang@ucsd.edu>.
