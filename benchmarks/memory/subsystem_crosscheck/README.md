# Subsystem integration cross-check (review workspace)

This is a local scientific benchmark, not a claim that SHYPS paper performance
has been reproduced. Results, source snapshots, raw counts, circuit assets, and
the detailed Chinese report are in `results/2026-09-06/`. They are intentionally
ignored by git. The scripts and review notebook are retained as research assets.

## What is being compared

- **Bacon–Shor capacity:** Napp and Preskill, *Optimal Bacon-Shor codes*, QIC 13,
  490–510 (2013), equations (10), (12), (19)–(20). Each data qubit sees ONE
  independent `X_ERROR(p)` and ONE independent `Z_ERROR(p)` between ideal gauge
  extraction rounds. Preparation, extraction, and final readout are perfect.
  Reported failure is the single protected Z-memory observable (X error sector),
  not the sum of X and Z logical errors. `validate.py` enumerates every DEM fault
  pattern for d=3,5,7,9 and compares the exact degenerate-ML probability to the
  published formula. The d=173 numerical anchor is analytic only.
- **Bacon–Shor circuit noise:** native `BaconShorCode`, `MemoryExperiment`, d XZ
  gauge pairs, all LightStim-generated detectors. Reset flips, ancilla readout
  flips, and CNOT depolarization each have probability p; final data readout is
  perfect and there is no idle noise. This second model is not the capacity
  model and is not compared to the capacity formula.
- **SHYPS reference:** the authors' exact public `.stim` files, Z-memory and
  Z detectors. These contain one preparation SE pair plus d repeated pairs
  (5 total for r=3, 9 for r=4), with reset flips, ancilla measurement flips, and
  `DEPOLARIZE2(p)`. The final data measurement is perfect. Missing p values are
  obtained by changing the common p in the published p=0.001 circuit; all
  locations remain unchanged. Raw logical block failures per shot are compared
  directly to the public figure-2 CSVs.
- **SHYPS regenerated:** a verified qubit permutation maps the reference gauge
  graph to the native `SHYPSCode`. Only the authors' reset/CNOT/measurement blocks
  are supplied to `CircuitBuilder`; reference detectors and observables never
  enter Tracker. LightStim generates all annotations. The full physical and
  noisy operation streams match exactly after ignoring timing/coordinate
  annotations. Every generated annotation passes an exact signed Stim flow
  check. Pure Z-record detectors are selected *after* running the full Tracker.
- **SHYPS generic:** native `MemoryExperiment(SHYPSCode(r=...))`, default XZ
  scheduling, d+1 total pairs and the same explicit noise adapter. A ZX order
  control is also saved. These circuits are not asserted to be the authors'
  gate schedule. The patch's display coordinates do not impose local hardware
  connectivity.

## Decoder and statistics conventions

All decoding uses LightStim's decoder registry, and capacity also has a public
`SimulationPipeline` smoke check. The sampler harness uses deterministic batch
seeds and saves a checkpoint after each batch. Configurations sharing a circuit,
seed offset, and batch size reuse the same samples for paired decoder comparisons.
Independent r=4 shards use disjoint seed offsets; aggregation checks seed uniqueness.

- BPOSD baseline: 100 BP iterations, minimum sum, OSD-CS order 10, dynamic scaling
  (`ms_scaling_factor=0`), parallel schedule. This explicitly overrides LightStim's
  1000-iteration default for benchmark compute cost. Additional variants use serial
  scheduling, r=3 scaling 0.1, or r=4 serial scheduling with scaling 0.85. All
  variants are retained, including poor-performing ones. Scaling parameters 0.1
  and 0.85 are motivated by Table VI, but the paper uses a different decoder.
- MWPF: `SolverSerialJointSingleHair`, cluster-node limit 50. This is a bounded
  optimization setting, not exact maximum-likelihood class decoding. The installed
  mwpf 0.2.12 accepts `timeout=1` but its `config` property does not forward it to
  the solver; the actual outer safeguard is the benchmark supervisor. Native
  panics are configured to raise. `audit_mwpf.py` replays earlier jobs with the
  strict policy and checks every batch's count to audit silent fallback behavior.
- MLE-ILP: exact **most-likely fault pattern**, not degenerate maximum-likelihood
  logical class. A 0.2 s soft per-shot limit, 25 cut rounds, and no RPC rounds are
  used. A timeout is counted as an operational failure and reported separately.
  Timeout-dominated SHYPS points are not evidence of its unconstrained LER.
- The paper's main SHYPS results use a **proprietary sliding-window BP+LSD**
  decoder: (2,1), BP max 100/scaling 0.1/LSD order 1 for r=3;
  BP max 2000/scaling 0.85/LSD order 4 for r=4. The reference repository contains
  circuits and data, not that implementation. Global BPOSD/MWPF runs do not
  reproduce it, and no threshold reproduction is claimed.

Any flip among the k output observables counts as one block failure; there is no
postselection. Most points target 200 errors, subject to shot/time caps. Confidence
intervals are pointwise nominal 95% exact Clopper–Pearson intervals; zero counts
are plotted only as upper bounds. These are not simultaneous confidence bands or
time-uniform confidence sequences. Per-shot rates are the primary metric. The
CSV also exports equation (52) under s=d and s=d+1 to expose the preparation-round
convention rather than silently choosing one. It does not assume independence
of the logical observables when computing raw block failure.

## Reproduce

Run from the LightStim worktree root with its dependencies installed. On this
workspace the interpreter is `../LightStim/venv/bin/python`; set `PYTHONPATH=.`.
The public reference repository is expected at
`../../External/ComputingEfficientlyInQLDPCCodes` relative to the worktree root.
Its pinned revision and the installed library versions are in `manifest.json`.
Every worker uses one numerical thread. Keep concurrent workers across commands
at or below 48. The review run used several independent suites concurrently.
To restore the archive, place its `subsystem_crosscheck/` directory under
`benchmarks/memory/` in the matching LightStim checkout and place
`review_notebook.ipynb` at `playground/subsystem/subsystem_crosscheck.ipynb`. If the external reference clone
is absent, circuit loading falls back to the included `reference_snapshot/`.
The implementation patch in the results directory and its base revision in
`manifest.json` preserve the exact production-code changes used in this run.

```bash
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/lightstim-crosscheck-mpl MPLBACKEND=Agg
PY=../LightStim/venv/bin/python
$PY benchmarks/memory/subsystem_crosscheck/validate.py --part capacity
$PY benchmarks/memory/subsystem_crosscheck/validate.py --part shyps3
$PY benchmarks/memory/subsystem_crosscheck/validate.py --part shyps4
$PY benchmarks/memory/subsystem_crosscheck/circuit_audit.py
$PY benchmarks/memory/subsystem_crosscheck/run.py --suite initial --workers 8
$PY benchmarks/memory/subsystem_crosscheck/run.py --suite alternatives --workers 4
$PY benchmarks/memory/subsystem_crosscheck/prepare_comparisons.py --r 3
$PY benchmarks/memory/subsystem_crosscheck/prepare_comparisons.py --r 4
```

Additional actual suites are saved as JSON in `results/2026-09-06/`:
`bacon_decoder_control_suite.json`, `capacity_precision_extension.json`,
`comparison_suite_r3.json`, `r3_decoder_control_suite.json`,
`r4_reference_shards.json`, and `r4_native_decoder_suite.json`.
The later `r4_bp_osd_diagnostic_suite.json` isolates BP iteration count on the
large automatic-detector circuit; it is explicitly a small-sample diagnostic.
Pass each path to `run.py --suite PATH --workers N`. Every individual actual
configuration is also saved under `configs/` and can be resumed with `--job PATH`.
Resuming retains previous batches and seeds; to restart independently use a new
variant/job ID and seed offset. A time-capped job needs an increased total budget
to collect additional batches.

```bash
$PY benchmarks/memory/subsystem_crosscheck/audit_mwpf.py --workers 8
$PY benchmarks/memory/subsystem_crosscheck/analyze.py
```

`results.csv` contains aggregated counts; `comparison_to_paper.csv` adds the
authors' counts, intervals, and rate ratios. `jobs/` contains every batch's seed,
count and timing. `assets/circuits/` contains circuits and DEMs; `assets/samples/`
contains the first batch's packed detector/observable samples and decoder
predictions for each job. This is sufficient to inspect bit ordering and paired
results; subsequent batches can be regenerated exactly using their recorded
seeds, batch size, Stim version, and circuit hash. Every curve is saved as PNG,
SVG and PDF. The reference snapshot retains the authors' MIT license.
