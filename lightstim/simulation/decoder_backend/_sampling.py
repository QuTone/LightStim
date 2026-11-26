"""Compile shot samplers that return Stim-relative detector and observable flips."""

from contextlib import contextmanager
from functools import partial

import numpy as np
import stim

from ..clifft_backend import ClifftSampler


def _validate_record_observables(circuit: stim.Circuit) -> None:
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            _validate_record_observables(instruction.body_copy())
        elif instruction.name == "OBSERVABLE_INCLUDE" and any(
            not target.is_measurement_record_target for target in instruction.targets_copy()
        ):
            raise ValueError("Clifft pipeline sampling does not support Pauli targets in OBSERVABLE_INCLUDE")


@contextmanager
def _compile_sampler(
    circuit: stim.Circuit,
    *,
    backend: str,
    seed: int,
    stream_offset: int | None = None,
):
    if stream_offset is not None or backend == "clifft":
        # Workers and Clifft batches consume adjacent seeds. Mix the run seed
        # before reserving those ranges so adjacent runs do not reuse samples.
        start = int(np.random.SeedSequence(seed).generate_state(1, dtype=np.uint64)[0])
        seed = (start + (stream_offset or 0)) % 2**64
    if backend == "stim":
        sampler = circuit.compile_detector_sampler(seed=seed)
        yield partial(sampler.sample, bit_packed=False, separate_observables=True)
        return

    if backend != "clifft":
        raise ValueError(f"Unknown sampler backend {backend!r}")

    # Hidden Pauli-frame contributions cannot be reconstructed from records.
    _validate_record_observables(circuit)
    converter = circuit.compile_m2d_converter()
    with ClifftSampler(str(circuit), seed=seed) as sampler:
        def sample(shots: int) -> tuple[np.ndarray, np.ndarray]:
            batch = sampler.sample_batch(shots)
            # Clifft returns unpacked uint8 records. Stim interprets uint8 as bit packed.
            return converter.convert(
                measurements=batch.measurements.astype(np.bool_, copy=False),
                separate_observables=True,
            )

        yield sample
