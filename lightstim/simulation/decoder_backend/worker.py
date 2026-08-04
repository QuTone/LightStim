"""
Worker functions for parallel simulation (CPU and GPU).

Used when post-selection is required; otherwise sinter.collect handles parallelism.
"""

import os
from multiprocessing import Manager
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import stim


def _decode_worker_cpu(
    circuit: stim.Circuit,
    decoder_name: str,
    decoder_params: Dict[str, Any],
    decoder_backend: str,
    batch_size: int,
    max_shots: int,
    max_errors: int,
    post_select_indices: List[int],
    post_select_observable_indices: Optional[List[int]],
    post_select_corrected_observable_indices: Optional[List[int]],
    target_observable_indices: Optional[List[int]],
    shots_counter,
    post_counter,
    errors_counter,
    lock,
    worker_id: int = 0,
    gpu_id: Optional[int] = None,
    on_decode_failure: str = "error",
    completed_counter=None,
    base_seed: Optional[int] = None,
) -> None:
    """
    Single worker process: reserve shots -> sample -> post-select -> decode.
    Updates shared counters (shots_counter, post_counter, errors_counter) under lock.

    ``shots_counter`` reserves work units up front (before decoding) to cap total
    shots and prevent large overshoot when many workers race near max_shots.
    ``completed_counter`` (optional) counts shots only *after* they have been
    sampled and decoded, so progress reporting reflects finished — not merely
    reserved — work. At termination the two are equal.
    """
    from ._accounting import count_batch
    from .registry import get_decoder
    from .post_select import apply_post_selection

    if gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    decoder = get_decoder(decoder_name, backend=decoder_backend, **decoder_params)
    dem = circuit.detector_error_model(
        decompose_errors=getattr(decoder, "decompose_errors", False),
    )
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    sampler = dem.compile_sampler(
        seed=(base_seed + worker_id) if base_seed is not None
        else os.getpid() + worker_id * 10000)

    while True:
        with lock:
            if shots_counter.value >= max_shots or errors_counter.value >= max_errors:
                break
            remaining = max_shots - shots_counter.value
            shots_to_take = min(batch_size, remaining)
            shots_counter.value += shots_to_take

        det_data, obs_data, _ = sampler.sample(
            shots=shots_to_take,
            bit_packed=False,
        )

        det_filtered, obs_filtered = apply_post_selection(
            det_data, obs_data, post_select_indices,
            post_select_observable_indices=post_select_observable_indices,
        )
        kept = det_filtered.shape[0]
        if kept == 0:
            # These shots were still sampled/processed — count them as completed
            # (they just contribute nothing after post-selection).
            if completed_counter is not None:
                with lock:
                    completed_counter.value += shots_to_take
            continue

        # sinter.Decoder expects little-endian bit packing.
        det_packed = np.packbits(det_filtered, axis=1, bitorder="little")
        pred_packed = compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=det_packed,
        )

        batch_kept, batch_errors = count_batch(
            obs_filtered=obs_filtered,
            pred_packed=pred_packed,
            post_select_corrected_observable_indices=post_select_corrected_observable_indices,
            target_observable_indices=target_observable_indices,
            flags=getattr(compiled, "last_flags", None),
            on_decode_failure=on_decode_failure,
        )
        with lock:
            post_counter.value += batch_kept
            errors_counter.value += batch_errors
            if completed_counter is not None:
                completed_counter.value += shots_to_take
