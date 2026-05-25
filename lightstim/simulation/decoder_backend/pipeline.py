"""Simulation pipeline: sample -> post-select -> decode -> stats."""

import multiprocessing as mp
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import stim

from .config import DecoderConfig, PipelineConfig, SimulationStats
from .post_select import apply_post_selection, get_post_select_detector_indices
from .progress import ProgressReporter, ProgressSnapshot, get_progress_logger
from .registry import get_decoder
from .worker import _decode_worker_cpu


class ExperimentTask:
    """Standard input for simulation: circuit + metadata."""

    def __init__(self, circuit: stim.Circuit, json_metadata: Optional[Dict[str, Any]] = None):
        self.circuit = circuit
        self.json_metadata = json_metadata or {}


class SimulationPipeline:
    """
    Unified pipeline: sampling, post-selection, decoding, with optional parallel execution.

    Uses a unified custom loop for all paths so progress reporting is consistent
    across CPU/GPU, post-selection on/off, and single/multi-process modes.
    """

    def __init__(
        self,
        decoder_config: Optional[DecoderConfig] = None,
        max_shots: int = 1_000_000,
        max_errors: int = 100,
        batch_size: int = 10_000,
        num_workers: int = 4,
        post_select_detector_indices: Optional[List[int]] = None,
        post_select_observable_indices: Optional[List[int]] = None,
        post_select_corrected_observable_indices: Optional[List[int]] = None,
        target_observable_indices: Optional[List[int]] = None,
        allow_gauge_detectors: bool = False,
        output_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
        output_format: str = "csv",
        save_resume_filepath: Optional[str] = None,
        print_progress: bool = True,
        progress_enabled: Optional[bool] = None,
        progress_interval_sec: float = 10.0,
        progress_min_delta_shots: Optional[int] = None,
        progress_poll_interval_sec: float = 0.5,
        progress_output: str = "print",
        progress_logger_name: str = "lightstim.simulation.progress",
        progress_file_path: Optional[str] = None,
        progress_file_max_bytes: int = 10_000_000,
        progress_file_backup_count: int = 5,
    ):
        self.config = PipelineConfig(
            max_shots=max_shots,
            max_errors=max_errors,
            batch_size=batch_size,
            num_workers=num_workers,
            decoder=decoder_config or DecoderConfig("pymatching", backend="cpu"),
            post_select_detector_indices=post_select_detector_indices,
            post_select_observable_indices=post_select_observable_indices,
            post_select_corrected_observable_indices=post_select_corrected_observable_indices,
            target_observable_indices=target_observable_indices,
            allow_gauge_detectors=allow_gauge_detectors,
            output_dir=output_dir,
            output_filename=output_filename,
            output_format=output_format,
            save_resume_filepath=save_resume_filepath,
            progress_enabled=print_progress if progress_enabled is None else progress_enabled,
            progress_interval_sec=progress_interval_sec,
            progress_min_delta_shots=progress_min_delta_shots,
            progress_poll_interval_sec=progress_poll_interval_sec,
            progress_output=progress_output,
            progress_logger_name=progress_logger_name,
            progress_file_path=progress_file_path,
            progress_file_max_bytes=progress_file_max_bytes,
            progress_file_backup_count=progress_file_backup_count,
            print_progress=print_progress,
        )
        self._status_logger = None
        if self.config.progress_output in ("logging", "both"):
            self._status_logger = get_progress_logger(
                logger_name=self.config.progress_logger_name,
                file_path=self.config.progress_file_path,
                file_max_bytes=self.config.progress_file_max_bytes,
                file_backup_count=self.config.progress_file_backup_count,
            )

    def _resolve_post_select_indices(self, circuit: stim.Circuit) -> List[int]:
        """Get post-select detector indices from config or circuit tags."""
        if self.config.post_select_detector_indices is not None:
            return self.config.post_select_detector_indices
        return get_post_select_detector_indices(circuit)

    def run(
        self,
        circuit: stim.Circuit,
        json_metadata: Optional[Dict[str, Any]] = None,
    ) -> SimulationStats:
        """
        Run simulation on a single circuit via the unified custom pipeline.
        """
        meta = json_metadata or {}
        post_indices = self._resolve_post_select_indices(circuit)
        return self._run_custom(circuit, meta, post_indices)

    def _warn_dem_flags(self, circuit: stim.Circuit, decoder_name: str) -> None:
        """Warn about DEM/decoder combinations that can silently affect LER."""
        if self.config.allow_gauge_detectors:
            warnings.warn(
                "allow_gauge_detectors=True: decomposition failures are silently "
                "ignored (ignore_decomposition_failures=True). Hyperedges that cannot "
                "be decomposed are dropped, which can underestimate LER.",
                stacklevel=4,
            )
            return

        # pymatching now uses decompose_errors=True + enable_correlations=True,
        # so it handles hyperedges correctly — no warning needed.

    def _run_custom(
        self,
        circuit: stim.Circuit,
        json_metadata: Dict[str, Any],
        post_select_indices: List[int],
    ) -> SimulationStats:
        """Custom sampling loop with post-selection (single or multi-process)."""
        decoder_name = self.config.decoder.name
        self._warn_dem_flags(circuit, decoder_name)
        start = time.perf_counter()
        reporter = self._make_progress_reporter()
        has_post_selection = (len(post_select_indices) > 0 or
                              bool(self.config.post_select_observable_indices) or
                              bool(self.config.post_select_corrected_observable_indices))

        if self.config.num_workers <= 1:
            return self._run_custom_single(
                circuit,
                json_metadata,
                post_select_indices,
                decoder_name,
                start,
                reporter,
                has_post_selection,
            )

        # Multi-process
        # Use shared-memory synchronized primitives directly.
        # This avoids Manager proxy IPC overhead under high worker counts.
        shots_counter = mp.Value("q", 0)
        post_counter = mp.Value("q", 0)
        errors_counter = mp.Value("q", 0)
        lock = mp.Lock()

        procs = []
        for wid in range(self.config.num_workers):
            p = mp.Process(
                target=_decode_worker_cpu,
                args=(
                    circuit,
                    decoder_name,
                    self.config.decoder.params,
                    self.config.decoder.backend,
                    self.config.batch_size,
                    self.config.max_shots,
                    self.config.max_errors,
                    post_select_indices,
                    self.config.post_select_observable_indices,
                    self.config.post_select_corrected_observable_indices,
                    self.config.target_observable_indices,
                    shots_counter,
                    post_counter,
                    errors_counter,
                    lock,
                    wid,
                    wid if self.config.decoder.backend != "cpu" else None,
                ),
            )
            p.start()
            procs.append(p)

        while any(p.is_alive() for p in procs):
            snapshot = self._build_snapshot(
                shots=shots_counter.value,
                kept=post_counter.value,
                errors=errors_counter.value,
                start=start,
                has_post_selection=has_post_selection,
            )
            reporter.emit(snapshot)
            time.sleep(self.config.progress_poll_interval_sec)

        for p in procs:
            p.join()

        elapsed = time.perf_counter() - start
        final_snapshot = self._build_snapshot(
            shots=shots_counter.value,
            kept=post_counter.value,
            errors=errors_counter.value,
            start=start,
            has_post_selection=has_post_selection,
        )
        reporter.emit(final_snapshot, final=True)
        return SimulationStats(
            shots=shots_counter.value,
            post_selected_shots=post_counter.value,
            errors=errors_counter.value,
            seconds=elapsed,
            decoder=decoder_name,
            json_metadata=json_metadata,
        )

    def _run_custom_single(
        self,
        circuit: stim.Circuit,
        json_metadata: Dict[str, Any],
        post_select_indices: List[int],
        decoder_name: str,
        start: float,
        reporter: ProgressReporter,
        has_post_selection: bool,
    ) -> SimulationStats:
        """Single-threaded custom loop."""
        decoder_instance = get_decoder(
            decoder_name,
            backend=self.config.decoder.backend,
            **self.config.decoder.params,
        )

        dem = circuit.flattened().detector_error_model(
            decompose_errors=getattr(decoder_instance, "decompose_errors", False) or self.config.allow_gauge_detectors,
            allow_gauge_detectors=self.config.allow_gauge_detectors,
            ignore_decomposition_failures=self.config.allow_gauge_detectors,
        )
        compiled = decoder_instance.compile_decoder_for_dem(dem=dem)
        sampler = circuit.compile_detector_sampler(seed=0)

        total_shots = 0
        post_selected_shots = 0
        errors = 0
        batch_size = self.config.batch_size

        while total_shots < self.config.max_shots and errors < self.config.max_errors:
            det_data, obs_data = sampler.sample(
                shots=batch_size,
                bit_packed=False,
                separate_observables=True,
            )
            total_shots += det_data.shape[0]

            det_filtered, obs_filtered = apply_post_selection(
                det_data, obs_data, post_select_indices,
                post_select_observable_indices=self.config.post_select_observable_indices,
            )
            kept = det_filtered.shape[0]
            if kept == 0:
                reporter.emit(
                    self._build_snapshot(
                        shots=total_shots,
                        kept=post_selected_shots,
                        errors=errors,
                        start=start,
                        has_post_selection=has_post_selection,
                    )
                )
                continue

            # sinter.Decoder expects little-endian bit packing.
            det_packed = np.packbits(det_filtered, axis=1, bitorder="little")
            obs_packed = np.packbits(obs_filtered, axis=1, bitorder="little")
            pred_packed = compiled.decode_shots_bit_packed(
                bit_packed_detection_event_data=det_packed,
            )

            ps_corr_idx = self.config.post_select_corrected_observable_indices
            if ps_corr_idx:
                # Post-decode PS: keep only shots where corrected obs[ps_corr_idx] == 0.
                # corrected[i] = obs[i] XOR pred[i] — the residual after applying the decoder.
                n_obs = obs_filtered.shape[1]
                pred_unpacked = np.unpackbits(pred_packed, axis=1, bitorder="little")[:, :n_obs]
                corrected = obs_filtered ^ pred_unpacked
                corr_mask = np.all(corrected[:, ps_corr_idx] == 0, axis=1)
                post_selected_shots += int(corr_mask.sum())
                if corr_mask.sum() > 0:
                    corrected_kept = corrected[corr_mask]
                    target = self.config.target_observable_indices
                    if target is not None:
                        errors += int(np.sum(np.any(corrected_kept[:, target] != 0, axis=1)))
                    else:
                        errors += int(np.sum(np.any(corrected_kept != 0, axis=1)))
            elif self.config.target_observable_indices is not None:
                # Only count errors on specified observables
                n_obs = obs_filtered.shape[1]
                pred_unpacked = np.unpackbits(pred_packed, axis=1, bitorder="little")[:, :n_obs]
                target = self.config.target_observable_indices
                post_selected_shots += kept
                errors += int(np.sum(np.any(
                    pred_unpacked[:, target] != obs_filtered[:, target], axis=1
                )))
            else:
                post_selected_shots += kept
                errors += int(np.sum(np.any(pred_packed != obs_packed, axis=1)))
            reporter.emit(
                self._build_snapshot(
                    shots=total_shots,
                    kept=post_selected_shots,
                    errors=errors,
                    start=start,
                    has_post_selection=has_post_selection,
                )
            )

        elapsed = time.perf_counter() - start
        reporter.emit(
            self._build_snapshot(
                shots=total_shots,
                kept=post_selected_shots,
                errors=errors,
                start=start,
                has_post_selection=has_post_selection,
            ),
            final=True,
        )
        return SimulationStats(
            shots=total_shots,
            post_selected_shots=post_selected_shots,
            errors=errors,
            seconds=elapsed,
            decoder=decoder_name,
            json_metadata=json_metadata,
        )

    def run_batch(
        self,
        tasks: List[Union[ExperimentTask, tuple]],
    ) -> pd.DataFrame:
        """
        Run simulation on multiple tasks. Returns a DataFrame with all stats.
        """
        # Normalize to ExperimentTask
        normalized = []
        for t in tasks:
            if isinstance(t, ExperimentTask):
                normalized.append(t)
            elif isinstance(t, (list, tuple)) and len(t) >= 2:
                normalized.append(ExperimentTask(t[0], t[1] if len(t) > 1 else {}))
            else:
                raise ValueError(f"Invalid task: {t}")

        records = []
        for i, task in enumerate(normalized):
            self._emit_status(f"Task {i + 1}/{len(normalized)}: {task.json_metadata}")
            stats = self.run(task.circuit, task.json_metadata)
            row = {
                "shots": stats.shots,
                "post_selected_shots": stats.post_selected_shots,
                "post_selection_rate": stats.post_selection_rate,
                "errors": stats.errors,
                "logical_error_rate": stats.logical_error_rate,
                "seconds": stats.seconds,
                "decoder": stats.decoder,
                **stats.json_metadata,
            }
            records.append(row)

        df = pd.DataFrame(records)

        # Save to file if output_dir configured
        if self.config.output_dir:
            self._save_output(df)

        return df

    def _save_output(self, df: pd.DataFrame) -> str:
        """Save DataFrame to output_dir with configured format."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = self.config.output_filename or "sim_{timestamp}.csv"
        fname = fname.replace("{timestamp}", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = out_dir / fname

        fmt = getattr(self.config, "output_format", "csv") or "csv"
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "json":
            df.to_json(path, orient="records", indent=2)
        elif fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

        self._emit_status(f"Saved results to {path}")
        return str(path)

    def _make_progress_reporter(self) -> ProgressReporter:
        return ProgressReporter(
            enabled=self.config.progress_enabled,
            interval_sec=self.config.progress_interval_sec,
            min_delta_shots=self.config.progress_min_delta_shots or max(self.config.batch_size, 1),
            output=self.config.progress_output,
            logger_name=self.config.progress_logger_name,
            file_path=self.config.progress_file_path,
            file_max_bytes=self.config.progress_file_max_bytes,
            file_backup_count=self.config.progress_file_backup_count,
        )

    def _build_snapshot(
        self,
        *,
        shots: int,
        kept: int,
        errors: int,
        start: float,
        has_post_selection: bool,
    ) -> ProgressSnapshot:
        elapsed = max(0.0, time.perf_counter() - start)
        effective_kept = kept if has_post_selection else shots
        return ProgressSnapshot(
            shots_total=shots,
            shots_kept=effective_kept,
            errors_total=errors,
            elapsed_sec=elapsed,
            max_shots=self.config.max_shots,
            max_errors=self.config.max_errors,
        )

    def _emit_status(self, message: str) -> None:
        if not self.config.progress_enabled:
            return
        if self.config.progress_output in ("print", "both"):
            print(message)
        if self.config.progress_output in ("logging", "both") and self._status_logger is not None:
            self._status_logger.info(message)
