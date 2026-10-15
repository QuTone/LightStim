# Rotated surface-code enlargement benchmark

Measures decoded logical failure probability for source preparation, source
syndrome extraction, retained-data enlargement in the `+x,+y` direction, and
target readout. X and Z preparations are separate experiments. The result is a
failure probability for the **whole circuit**, not a rate per extraction round
or a measurement of target-code distance.

The reusable circuit comes from `RotatedSurfaceEnlargeExperiment`; sampling and
decoding use LightStim's native `SimulationPipeline`. PyMatching is the default.
`p_1q`, `p_2q`, `p_meas`, and `p_reset` equal the swept `p`. Idle noise defaults to
zero and is controlled independently by `--p-idle`.

Run from the repository root:

```bash
# Fast pipeline smoke test: 3->5 and 3->7, X/Z, two p values.
.venv/bin/python benchmarks/logical_ops/run_rotated_enlarge.py --quick

# A modest error-targeted sweep.
.venv/bin/python benchmarks/logical_ops/run_rotated_enlarge.py \
  --source-distance 3 --target-distances 5 7 --bases X Z \
  --rounds 3 --source-rounds 1 --p-values 0.001 0.003 0.01 \
  --max-shots 1000000 --max-errors 100 --workers 4

# Tiny noiseless end-to-end check; explicit arguments override --quick.
.venv/bin/python benchmarks/logical_ops/run_rotated_enlarge.py \
  --quick --target-distances 5 --p-values 0 --max-shots 32 \
  --batch-size 16 --output /tmp/rotated_enlarge_smoke.csv
```

| Option | Default | Meaning |
|---|---|---|
| `--source-distance` | `3` | Odd source distance, at least 3 |
| `--target-distances` | source + 2, source + 4 | Odd distances larger than source |
| `--bases` / `--basis` | `X Z` | Prepared and measured logical bases |
| `--rounds` | `1` | Total target extraction rounds, including the transition round |
| `--source-rounds` | `1` | Source extraction rounds before enlargement |
| `--p-values` | `0.0005 0.001 0.003 0.01` | Gate, reset, and measurement error probabilities |
| `--p-idle` | `0` | Absolute idle probability, independent of `p` |
| `--decoder` | `pymatching` | `pymatching`, `bposd`, or `mwpf`; optional backends require installation |
| `--max-shots` | `1000000` | Attempted-shot cap per task |
| `--max-errors` | `100` | Error stopping target; batch completion can overshoot |
| `--batch-size` | `1000` | Shots per batch |
| `--workers` / `--num-workers` | `1` | Native pipeline workers |
| `--quick` | off | Defaults to two targets, `p=0.001 0.003`, 1000 shots, 10 errors, batches of 250 |
| `--output` | `benchmarks/logical_ops/results/rotated_enlarge_results.csv` | Append destination |

Each task appends and flushes one CSV row immediately after completion. Every
invocation samples all requested tasks again: there is **no implicit resume or
skip**. Reusing a path retains previous rows with distinct `run_id` and `row_id`,
including when shot budgets change. Use a fresh path for a separate dataset;
do not run simultaneous writers against one path. An incompatible CSV header
is rejected before simulation.

CSV rows include:

- `run_id`, `row_id`, `task_index`, and UTC `completed_at`: run and row identity.
- `source_distance`, `target_distance`, `basis`, `rounds`, `source_rounds`,
  `direction`, `p`, `p_idle`, `noise_model`, `decoder`: experimental settings.
- `max_shots`, `max_errors`, `batch_size`, `num_workers`: requested collection budget.
- `shots`, `accepted`, `errors`: raw attempted, accepted, and failed-shot counts.
  This experiment has no intended postselection, so `accepted` equals `shots`.
- `logical_error_rate`, `ci95_low`, `ci95_high`: errors / accepted and the
  two-sided Wilson binomial interval (`z=1.96`). Zero observed errors retain a
  positive upper bound; zero accepted shots would produce `nan`.
- `seconds`: pipeline sampling/decoding wall time, excluding circuit construction.
- `circuit_sha256`: hash of the generated noisy Stim circuit.
- `config_json`: complete effective per-task protocol, noise, decoder, and
  sampling configuration, including decoder defaults and the quick-mode flag.

Quick runs check construction and pipeline integration; their small counts
are not precision estimates or evidence for a threshold. The runner does not
expose a sampling seed because the native pipeline has no seed parameter.
The current native pipeline uses seed 0 for one worker and process-based seeds
for multiple workers; this policy is recorded in `config_json`. Repeated
single-worker runs are not independent samples and should not be pooled as such.
