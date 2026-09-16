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

## Use with LightStim workloads

The general memory runner accepts the backend directly:

```bash
PYTHONPATH=/path/to/BeamSearchDecoder/decoder \
python benchmarks/memory/run_memory.py \
    --codes bb_72_12_6 \
    --p-values 0.003 \
    --basis X Z \
    --decoder ionq-beam-search \
    --max-shots 1000
```

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

`max_rounds` counts search-branching rounds, not syndrome-extraction rounds or
decoding windows. All five decoder parameters must be positive integers.

## Detector error models and failures

The adapter consumes undecomposed DEM hyperedges and returns corrections over
the DEM's error mechanisms. It checks each returned correction against the
input syndrome instead of trusting upstream's `converge` field, which describes
the last explored branch and can disagree with the selected result.

LightStim's existing failure policies apply:

- `error` (default) counts an invalid correction as a logical error.
- `discard` removes invalid corrections from the accepted-shot denominator.
- `ignore` scores the returned observable prediction even when the correction
  is invalid. This matches the upstream paper repository's Sinter wrapper and
  should be used only when reproducing its reported metric.

The adapter copies data across the native boundary, supports empty DEMs and
empty shot batches, and can be used as a stage in `DecoderConfig.chain(...)`.
It currently decodes shots one at a time because upstream exposes no batch API.

## Test

Contract tests run without the optional extension; native and multiprocessing
tests run automatically when it is importable:

```bash
python -m pytest tests/test_ionq_beam_search_decoder.py -q
```

The adapter was validated against the pinned upstream implementation on
identical parity-check matrices, priors, syndromes, and observable predictions.
Performance numbers depend strongly on the compiler, CPU, and build settings
and are therefore not treated as package-level guarantees.

References: [IonQ implementation](https://github.com/ionq-publications/BeamSearchDecoder),
[beam-search decoder paper](https://arxiv.org/abs/2512.07057).
