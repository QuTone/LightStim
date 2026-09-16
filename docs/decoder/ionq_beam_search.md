# IonQ BP-guided beam search (beta)

`DecoderConfig("ionq-beam-search")` adapts IonQ's original C++/Cython
BeamSearchDecoder to the LightStim simulation pipeline. It is distinct from
the separately packaged `tesseract` decoder.

LightStim contains only the Python adapter. It does not copy, bundle, or build
the IonQ implementation, so this backend is not part of LightStim's default or
`decoders` dependencies.

## License and support status

The upstream implementation is licensed separately under CC BY-NC-SA 4.0,
including a restriction to non-commercial use unless IonQ grants other terms.
Installing or using that implementation remains subject to its
[upstream license](https://github.com/ionq-publications/BeamSearchDecoder/blob/main/LICENSE).
LightStim's Apache-2.0 license does not replace or relax those terms.

This integration is beta because upstream currently requires a manual native
build and does not publish a standard Python wheel. A C++20 compiler is needed.

## Install the upstream extension

Use the same Python environment to build upstream and run LightStim. The tested
upstream revision is `084a475b05fb64308103317a1ce5a0c4b0be58aa`.

```bash
python -m pip install numpy scipy ldpc Cython setuptools
git clone https://github.com/ionq-publications/BeamSearchDecoder.git /path/to/BeamSearchDecoder
cd /path/to/BeamSearchDecoder
git checkout 084a475b05fb64308103317a1ce5a0c4b0be58aa
cd decoder
python setup.py build_ext --inplace
export PYTHONPATH="/path/to/BeamSearchDecoder/decoder${PYTHONPATH:+:$PYTHONPATH}"
python -c "from beam_search_decoder import BeamSearchDecoder"
```

The `PYTHONPATH` setting must also be visible to worker processes and notebook
kernels. An unavailable or unloadable extension raises an installation hint
only when `ionq-beam-search` is selected; other LightStim decoders remain usable.
Build once for the Python environment and platform you will use. A change of
Python ABI, NumPy ABI, or platform can require rebuilding the extension.
Users only import LightStim in their experiment code; the adapter loads the
separately built extension on their behalf when selected.

## Use with LightStim workloads

The decoder works with supported Stim detector error models regardless of
whether the circuit implements memory, logical operations, or logical circuits.
All five benchmark runners accept `--decoder ionq-beam-search`:

| Runner | Example workload arguments |
|---|---|
| `benchmarks/memory/run_memory.py` | `--codes rotated_sc --distances 3` |
| `benchmarks/logical_ops/run_logical_ops.py` | `--gate TwoPatchLS_rotated_ZZ --distances 3` |
| `benchmarks/logical_circuits/run_logical_circuits.py` | `--experiment bell_tele --protocols tg --states Z --distances 3` |
| `benchmarks/state_injection/run_state_injection.py` | `--inject-states Z --inject-protocols corner --inject-modes hybrid --distances 3` |
| `benchmarks/cross_ls/run_cross_ls.py` | `--experiment sweep --states Z --pqrm 1,2,4 --distances 3 --backend cpu` |

For example, after exporting `PYTHONPATH` as above, run a small logical-operation
check from the LightStim repository root:

```bash
python benchmarks/logical_ops/run_logical_ops.py \
    --gate TwoPatchLS_rotated_ZZ --distances 3 --p-values 0.001 \
    --decoder ionq-beam-search --max-shots 100 --max-errors 100 \
    --batch-size 100 --num-workers 1 --output /tmp/rotated_ls_ionq.csv
```

Or select BB memory:

```bash
PYTHONPATH=/path/to/BeamSearchDecoder/decoder \
python benchmarks/memory/run_memory.py \
    --codes bb_72_12_6 \
    --p-values 0.003 \
    --basis X Z \
    --decoder ionq-beam-search \
    --max-shots 1000
```

The logical-circuits runner also accepts `--experiment distill_ls distill_tg`.
Both distillation paths forward the selected decoder to their input calibration
and output evaluation. This changes the decoder only; the protocol's existing
observable transformations and post-selection rules still apply. The memory
runner's `--mode full_postselection` intentionally uses no decoder.

For programmatic use, the defaults reproduce the paper's `beam8_230iters`
configuration:

```python
from lightstim.simulation.decoder_backend import DecoderConfig, SimulationPipeline

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig(
        "ionq-beam-search",
        params={
            "max_rounds": 10,
            "beam_width": 8,
            "num_results": 1,
            "initial_iters": 30,
            "iters_per_round": 20,
        },
        on_decode_failure="error",
    ),
    max_shots=2000,
    num_workers=2,
)
stats = pipeline.run(circuit)
```

Here `circuit` is any circuit supported by LightStim's DEM-based pipeline,
including one returned by a logical-operation or logical-circuit experiment.
This backend does not add a non-Clifford simulator or support circuits that
cannot be converted into a supported detector error model. Nonempty models
require finite error-mechanism priors strictly between zero and one; empty
noiseless DEMs are supported.

`max_rounds` counts search-branching rounds, not syndrome-extraction rounds or
decoding windows. All five decoder parameters must be positive integers.

## Detector error models and failures

The adapter consumes undecomposed DEM hyperedges and returns corrections over
the DEM's error mechanisms. It checks each returned correction against the
input syndrome instead of trusting upstream's `converge` field, which describes
the last explored branch and can disagree with the selected result.

There are two different checks: whether the predicted logical observables
match the sampled observables, and whether the proposed error-mechanism
correction satisfies `H @ correction == syndrome` over GF(2). Passing one does
not imply passing the other.

LightStim's existing failure policies apply:

- `error` (default) counts an observable mismatch OR an invalid correction as
  a failure, counting their overlap only once.
- `discard` removes invalid corrections from the accepted-shot denominator.
- `ignore` scores the returned observable prediction even when the correction
  is invalid. This matches the upstream paper repository's Sinter wrapper and
  is appropriate when reporting that observable-prediction metric.

For example, 20 observable mismatches plus two additional shots with correct
observable predictions but invalid corrections give 20 errors with `ignore`
and 22 with `error`, on exactly the same samples. The default is a conservative
failure metric, not evidence that two more observable flips occurred. Neither
policy modifies the native decoder's output. Specify the policy when comparing
results; a discard-conditioned LER has a different denominator as well.

## Batching and workers

The adapter copies data across the native boundary, supports empty DEMs and
empty shot batches, and can be used as a stage in `DecoderConfig.chain(...)`.
LightStim already accepts batches of shots: the adapter loops over the batch
and calls upstream `decode(syndrome)` once per shot. Each call runs the existing
C++ algorithm; it does not rebuild the decoder or launch a process per shot.
The decoder model is constructed once per DEM per worker.

`batch_size` controls LightStim sampling and dispatch. `num_workers` parallelizes
the standard pipeline across processes, each with its own decoder instance.
The existing transformed-observable TG distillation loop remains serial.
For small smoke runs, also lower `--batch-size`: the standard pipeline checks
stopping limits at batch boundaries, so its default batch can exceed a small
`--max-shots` budget.

Upstream currently exposes a single-shot Python/Cython API. A native batch
entry point could reduce Python call overhead, but would require work on the
native binding and careful handling of mutable decoder state for parallel
calls. It is not required for this integration, and increasing `batch_size`
alone does not turn the native decoder into a vectorized or GPU implementation.

## Test

Contract tests run without the optional extension; native, multiprocessing,
and workload CLI smoke tests run automatically when it is importable:

```bash
python -m pytest tests/test_ionq_beam_search_decoder.py -q
```

Each CLI smoke case uses eight shots per workload point, including injection
and full-noise LS/TG distillation. These test integration, not LER scaling.

The adapter was validated against the pinned upstream implementation on
identical parity-check matrices, priors, syndromes, and observable predictions.
Performance numbers depend strongly on the compiler, CPU, and build settings
and are therefore not treated as package-level guarantees.

References: [IonQ implementation](https://github.com/ionq-publications/BeamSearchDecoder),
[beam-search decoder paper](https://arxiv.org/abs/2512.07057).
