"""
Worker functions for parallel simulation (CPU and GPU).

Used when post-selection is required; otherwise sinter.collect handles parallelism.
"""

import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    allow_gauge_detectors: bool,
    error_path: Optional[str] = None,
    worker_id: int = 0,
    gpu_id: Optional[int] = None,
) -> None:
    try:
        _decode_worker_cpu_impl(
            circuit=circuit,
            decoder_name=decoder_name,
            decoder_params=decoder_params,
            decoder_backend=decoder_backend,
            batch_size=batch_size,
            max_shots=max_shots,
            max_errors=max_errors,
            post_select_indices=post_select_indices,
            post_select_observable_indices=post_select_observable_indices,
            post_select_corrected_observable_indices=post_select_corrected_observable_indices,
            target_observable_indices=target_observable_indices,
            shots_counter=shots_counter,
            post_counter=post_counter,
            errors_counter=errors_counter,
            lock=lock,
            allow_gauge_detectors=allow_gauge_detectors,
            worker_id=worker_id,
            gpu_id=gpu_id,
        )
    except BaseException as exc:
        if error_path is not None:
            Path(error_path).write_text(
                "\n".join(
                    [
                        f"worker_id={worker_id}",
                        f"type={type(exc).__name__}",
                        f"message={exc}",
                        traceback.format_exc(),
                    ]
                )
            )
        raise


def _decode_worker_cpu_impl(
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
    allow_gauge_detectors: bool,
    worker_id: int = 0,
    gpu_id: Optional[int] = None,
) -> None:
    """
    Single worker process: reserve shots -> sample -> post-select -> decode.
    Updates shared counters (shots_counter, post_counter, errors_counter) under lock.
    shots_counter tracks reserved/completed work units, preventing large overshoot
    when many workers race near max_shots.
    """
    from .registry import get_decoder
    from .post_select import apply_post_selection

    if gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    decoder = get_decoder(decoder_name, backend=decoder_backend, **decoder_params)
    dem = circuit.detector_error_model(
        decompose_errors=getattr(decoder, "decompose_errors", False) or allow_gauge_detectors,
        allow_gauge_detectors=allow_gauge_detectors,
        ignore_decomposition_failures=allow_gauge_detectors,
    )
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    sampler = dem.compile_sampler(seed=os.getpid() + worker_id * 10000)

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
            continue

        # sinter.Decoder expects little-endian bit packing.
        det_packed = np.packbits(det_filtered, axis=1, bitorder="little")
        obs_packed = np.packbits(obs_filtered, axis=1, bitorder="little")
        pred_packed = compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=det_packed,
        )

        if post_select_corrected_observable_indices:
            # Post-decode PS: keep only shots where corrected obs == 0.
            n_obs = obs_filtered.shape[1]
            pred_unpacked = np.unpackbits(pred_packed, axis=1, bitorder="little")[:, :n_obs]
            corrected = obs_filtered ^ pred_unpacked
            corr_mask = np.all(corrected[:, post_select_corrected_observable_indices] == 0, axis=1)
            kept_corr = int(corr_mask.sum())
            if kept_corr > 0:
                corrected_kept = corrected[corr_mask]
                if target_observable_indices is not None:
                    batch_errors = int(np.sum(np.any(corrected_kept[:, target_observable_indices] != 0, axis=1)))
                else:
                    batch_errors = int(np.sum(np.any(corrected_kept != 0, axis=1)))
            else:
                batch_errors = 0
            with lock:
                post_counter.value += kept_corr
                errors_counter.value += batch_errors
        elif target_observable_indices is not None:
            n_obs = obs_filtered.shape[1]
            pred_unpacked = np.unpackbits(pred_packed, axis=1, bitorder="little")[:, :n_obs]
            batch_errors = int(np.sum(np.any(
                pred_unpacked[:, target_observable_indices] != obs_filtered[:, target_observable_indices], axis=1
            )))
            with lock:
                post_counter.value += kept
                errors_counter.value += batch_errors
        else:
            batch_errors = int(np.sum(np.any(pred_packed != obs_packed, axis=1)))
            with lock:
                post_counter.value += kept
                errors_counter.value += batch_errors
