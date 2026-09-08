# IonQ Beam Search: low-LER paper-point check

The p=0.001 point was run until **at least 20 combined X+Z logical errors**,
then all in-flight jobs were included. The final result is **20 errors in
12,220,000 total shots**, consistent with the paper plot at this point.

## Configuration and result

- LightStim adapter: commit `e33100f` on `Ming/ionq-beam-BP-beta`.
- Unmodified IonQ native algorithm: upstream commit
  `084a475b05fb64308103317a1ce5a0c4b0be58aa`.
- Authors' BB[[144,12,12]] X/Z memory circuits; 12 syndrome rounds; p=0.001.
- `beam8_230iters`: max_rounds=10, beam_width=8, num_results=1,
  initial_iters=30, iters_per_round=20.
- 12 process workers; 5,000 shots per basis per job; 1,222 completed jobs.
- Completed job IDs form the entire prefix **0 through 1221**, with no missing
  or duplicated IDs. Seeds are distinct within each basis.
- Elapsed sampling/coordinator time: **4,059.5 s (67.7 minutes)**.

| Basis | Shots | Observable errors (paper policy) | Invalid corrections | Errors under LightStim's default strict policy |
|---|---:|---:|---:|---:|
| X | 6,110,000 | 12 | 9 | 12 |
| Z | 6,110,000 | 8 | 9 | 10 |
| Total | 12,220,000 | **20** | 18 | **22** |

Using the paper's sum-of-basis error rate per syndrome extraction round:

`(12/6110000 + 8/6110000) / 12 = 2.7277686852e-7`.

The archived paper figure's orange p=0.001 marker is approximately **2.8e-7**
(read from the plot; no numerical CSV or reference error counts are provided).
The result agrees at the resolution supported by this 20-error experiment.
This is one parameter point, not a reproduction of the complete curve or of
other beam widths.

A conservative, stopping-valid **95% confidence sequence** for the paper LER
is **[3.68e-8, 9.24e-7]**. It combines separate 97.5% beta-mixture Bernoulli
confidence sequences for X and Z, divided by 12. The interval is deliberately
wider than a fixed-sample interval: sampling stopped based on observed errors.
Bounds are evaluated on the complete job prefix after all in-flight jobs finish.
Twenty events still leave substantial statistical uncertainty.

### Failure-policy distinction

The paper wrapper scores the returned observable prediction even when the
correction does not satisfy the syndrome. In this run, two Z shots had invalid
corrections but correct observable predictions. Therefore:

- `on_decode_failure="ignore"`: **20 errors**, LER **2.72777e-7/round**;
  this is the policy used for the paper comparison and stopping target.
- `on_decode_failure="error"` (LightStim default): **22 errors**,
  LER **3.00055e-7/round**.

No shots were discarded. The error target counts observable errors, not just
non-converged/invalid corrections.

## Upstream agreement and validation

For both circuits, each worker verified identical H, observable matrix, priors,
and column ordering between LightStim and upstream. All **39,124** compared
shot predictions matched exactly (19,564 X and 19,560 Z). This includes **every
one of the 20 observed logical-error shots**, plus the first 16 shots of every
job/basis. Remaining shots were decoded only through the LightStim adapter.

The final audit checked job continuity and uniqueness, seed uniqueness, error
shot indices against counts, aggregate totals, and zero upstream mismatches.
Eight targeted statistical/accounting tests passed. An isolated interrupted
checkpoint test recovered a missing job with identical seeds/counts and no
extra jobs or duplicates. A completed checkpoint also resumed without sampling
additional shots.

## Reproduce or extend

Build the optional upstream extension as described in
[the decoder guide](../../docs/ionq_beam_search.md), then run from LightStim:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
python benchmarks/decoding/ionq_beam_search_low_ler.py \
    --upstream /path/to/BeamSearchDecoder \
    --p 0.001 --target-errors 20 --workers 12 --shots-per-job 5000 \
    --seed 20260910
```

The local test used `/tmp/lightstim-ionq-beam-upstream`. The generated
`benchmarks/decoding/results/ionq_low_ler_p001/` directory holds:

- `configuration.json`: revision, parameters, versions, and circuit hashes;
- `jobs.jsonl`: every completed job's seeds, counts, error-shot indices, and
  upstream comparisons;
- `summary.json`: final aggregation and stopping-aware uncertainty.

These generated checkpoints remain local and are ignored by Git. This report
is committed. Repeating the command resumes the same checkpoint; increasing
`--target-errors` extends it. A fresh run with a different seed/configuration
must use a different `--output-dir`. In-flight completion timing can change the
final job-prefix length on a fresh run. Each job's sampling seed is fixed by
`SeedSequence([20260910, job_id, basis_index])`, with X=0 and Z=1, taking one
uint64 generated state value. Replaying the exact prefix 0..1221 reproduces the
recorded sample set on the same Stim version/machine and sampling call pattern.

Versions: Python 3.13.5; numpy 2.2.6; scipy 1.17.1; stim/sinter 1.15.0;
ldpc 2.4.1; stimbposd 0.1.0. The run did not change the native algorithm, core
adapter, default dependencies, or distribution configuration.

SHA-256:

- X circuit: `6628d4483489cd8ea792da846274e8e245c7168c3560bf4c5e1cd2b3fb736dc3`
- Z circuit: `106f45471003ecdf6d843ea3b9da45d2f0a766716569a9fababaacbcdeb356db`
- Completed jobs JSONL: `da8f7434801bea0560f7252d050ff12110c2453f3a2793799b84e8c8572752e6`

References: [paper Fig. 2](https://arxiv.org/html/2512.07057v1),
[archived plot](https://github.com/ionq-publications/BeamSearchDecoder/blob/084a475b05fb64308103317a1ce5a0c4b0be58aa/simulation_results/beam_144_12.pdf),
[upstream LER scoring](https://github.com/ionq-publications/BeamSearchDecoder/blob/084a475b05fb64308103317a1ce5a0c4b0be58aa/simulation_functions.py),
[time-uniform confidence sequences](https://arxiv.org/abs/1810.08240).
