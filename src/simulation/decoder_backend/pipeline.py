"""Simulation pipeline: sample -> post-select -> decode -> stats."""

import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import sinter
import stim

from .config import DecoderConfig, PipelineConfig, SimulationStats
from .post_select import apply_post_selection, get_post_select_detector_indices
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

    When post_select_detector_indices is empty/None and no tagged detectors exist,
    delegates to sinter.collect for full compatibility. Otherwise runs a custom
    sampling loop with post-selection.
    """

    def __init__(
        self,
        decoder_config: Optional[DecoderConfig] = None,
        max_shots: int = 1_000_000,
        max_errors: int = 100,
        batch_size: int = 10_000,
        num_workers: int = 4,
        post_select_detector_indices: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
        output_format: str = "csv",
        save_resume_filepath: Optional[str] = None,
        print_progress: bool = True,
    ):
        self.config = PipelineConfig(
            max_shots=max_shots,
            max_errors=max_errors,
            batch_size=batch_size,
            num_workers=num_workers,
            decoder=decoder_config or DecoderConfig("pymatching", backend="cpu"),
            post_select_detector_indices=post_select_detector_indices,
            output_dir=output_dir,
            output_filename=output_filename,
            output_format=output_format,
            save_resume_filepath=save_resume_filepath,
            print_progress=print_progress,
        )

    def _resolve_post_select_indices(self, circuit: stim.Circuit) -> List[int]:
        """Get post-select detector indices from config or circuit tags."""
        if self.config.post_select_detector_indices is not None:
            return self.config.post_select_detector_indices
        return get_post_select_detector_indices(circuit)

    def _use_sinter_directly(self, circuit: stim.Circuit) -> bool:
        """True if we can delegate to sinter (no post-selection)."""
        return len(self._resolve_post_select_indices(circuit)) == 0

    def run(
        self,
        circuit: stim.Circuit,
        json_metadata: Optional[Dict[str, Any]] = None,
    ) -> SimulationStats:
        """
        Run simulation on a single circuit. Uses sinter when no post-selection;
        otherwise runs custom pipeline.
        """
        meta = json_metadata or {}
        post_indices = self._resolve_post_select_indices(circuit)

        if self._use_sinter_directly(circuit):
            return self._run_sinter(circuit, meta)
        return self._run_custom(circuit, meta, post_indices)

    def _run_sinter(
        self,
        circuit: stim.Circuit,
        json_metadata: Dict[str, Any],
    ) -> SimulationStats:
        """Delegate to sinter.collect (no post-selection)."""
        decoder_name = self.config.decoder.name
        decoder_instance = get_decoder(
            decoder_name,
            backend=self.config.decoder.backend,
            **self.config.decoder.params,
        )

        task = sinter.Task(
            circuit=circuit,
            decoder=decoder_name,
            json_metadata=json_metadata,
        )
        tasks = [task]
        dem = circuit.detector_error_model(
            decompose_errors=getattr(decoder_instance, "decompose_errors", False),
            approximate_disjoint_errors=True,
        )
        # Use custom_decoders so we pass our decoder instance
        results = sinter.collect(
            num_workers=self.config.num_workers,
            tasks=tasks,
            max_shots=self.config.max_shots,
            max_errors=self.config.max_errors,
            custom_decoders={decoder_name: decoder_instance},
            save_resume_filepath=self.config.save_resume_filepath,
            print_progress=self.config.print_progress,
        )
        r = results[0]
        return SimulationStats(
            shots=r.shots,
            post_selected_shots=r.shots,  # no post-selection
            errors=r.errors,
            seconds=r.seconds,
            decoder=r.decoder,
            json_metadata={**r.json_metadata} if hasattr(r, "json_metadata") else json_metadata,
        )

    def _run_custom(
        self,
        circuit: stim.Circuit,
        json_metadata: Dict[str, Any],
        post_select_indices: List[int],
    ) -> SimulationStats:
        """Custom sampling loop with post-selection (single or multi-process)."""
        decoder_name = self.config.decoder.name
        start = time.perf_counter()

        if self.config.num_workers <= 1:
            return self._run_custom_single(
                circuit, json_metadata, post_select_indices, decoder_name, start
            )

        # Multi-process
        manager = mp.Manager()
        shots_counter = manager.Value("i", 0)
        post_counter = manager.Value("i", 0)
        errors_counter = manager.Value("i", 0)
        lock = manager.Lock()

        procs = []
        for wid in range(self.config.num_workers):
            p = mp.Process(
                target=_decode_worker_cpu,
                args=(
                    circuit,
                    decoder_name,
                    self.config.decoder.params,
                    self.config.batch_size,
                    self.config.max_shots,
                    self.config.max_errors,
                    post_select_indices,
                    shots_counter,
                    post_counter,
                    errors_counter,
                    lock,
                    wid,
                    None,
                ),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        elapsed = time.perf_counter() - start
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
    ) -> SimulationStats:
        """Single-threaded custom loop."""
        decoder_instance = get_decoder(
            decoder_name,
            backend=self.config.decoder.backend,
            **self.config.decoder.params,
        )

        dem = circuit.detector_error_model(
            decompose_errors=getattr(decoder_instance, "decompose_errors", False),
            approximate_disjoint_errors=True,
        )
        compiled = decoder_instance.compile_decoder_for_dem(dem=dem)
        sampler = dem.compile_sampler(seed=0)

        total_shots = 0
        post_selected_shots = 0
        errors = 0
        batch_size = self.config.batch_size

        while total_shots < self.config.max_shots and errors < self.config.max_errors:
            det_data, obs_data = sampler.sample(
                shots=batch_size,
                separate_observables=True,
                bit_packed=False,
            )
            total_shots += det_data.shape[0]

            det_filtered, obs_filtered = apply_post_selection(
                det_data, obs_data, post_select_indices
            )
            kept = det_filtered.shape[0]
            post_selected_shots += kept
            if kept == 0:
                continue

            det_packed = np.packbits(det_filtered, axis=1)
            obs_packed = np.packbits(obs_filtered, axis=1)
            pred_packed = compiled.decode_shots_bit_packed(
                bit_packed_detection_event_data=det_packed,
            )
            errors += int(np.sum(np.any(pred_packed != obs_packed, axis=1)))

            if self.config.print_progress and total_shots % (10 * batch_size) < batch_size:
                elapsed = time.perf_counter() - start
                ler = errors / post_selected_shots if post_selected_shots else 0
                print(
                    f"shots={total_shots:,} kept={post_selected_shots:,} "
                    f"errors={errors} LER={ler:.2e} {elapsed:.1f}s"
                )

        elapsed = time.perf_counter() - start
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
            if self.config.print_progress:
                print(f"Task {i + 1}/{len(normalized)}: {task.json_metadata}")
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

        if self.config.print_progress:
            print(f"Saved results to {path}")
        return str(path)
