import stim
import pandas as pd
from typing import List, Dict, Any, Literal, Optional

from .decoder import BaseDecoder, SinterMWPMDecoder
from .decoder_backend import SimulationPipeline, ExperimentTask as PipelineExperimentTask, DecoderConfig


class ExperimentTask:
    """Standard input format for our simulator (alias for pipeline)."""
    def __init__(self, circuit: stim.Circuit, json_metadata: Optional[Dict[str, Any]] = None):
        self.circuit = circuit
        self.json_metadata = json_metadata or {}


class QECSimulator:
    """
    Unified entry point for running large-scale QEC simulations.
    Supports both CPU (via Sinter) and GPU (via Custom Worker) backends.
    """

    def __init__(self, backend: Literal['sinter_cpu', 'nvidia_gpu'] = 'sinter_cpu', num_workers: int = 4):
        self.backend = backend
        self.num_workers = num_workers

    def run_batch(self, 
                  tasks: List[ExperimentTask], 
                  max_shots: int = 1_000_000,
                  max_errors: int = 1000,
                  decoder: BaseDecoder = None,
                  gpu_ids: Optional[List[int]] = None,
                  output_dir: Optional[str] = None,
                  post_select_detector_indices: Optional[List[int]] = None) -> pd.DataFrame:
        
        if decoder is None:
            decoder = SinterMWPMDecoder()

        print(f"Starting Simulation Batch | Backend: {self.backend} | Decoder: {decoder.name}")
        
        if self.backend == 'sinter_cpu':
            return self._run_via_pipeline(
                tasks, max_shots, max_errors, decoder,
                output_dir=output_dir,
                post_select_detector_indices=post_select_detector_indices,
            )
        elif self.backend == 'nvidia_gpu':
            raise NotImplementedError(
                "QECSimulator backend='nvidia_gpu' used a placeholder decoder and "
                "has been disabled. Use SimulationPipeline with "
                "DecoderConfig(name='bposd' or 'nv-qldpc-decoder', backend='gpu') instead."
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ==========================================================================
    # Backend 1: Via SimulationPipeline (CPU, supports post-selection)
    # ==========================================================================
    def _run_via_pipeline(
        self,
        tasks: List[ExperimentTask],
        max_shots: int,
        max_errors: int,
        decoder: BaseDecoder,
        output_dir: Optional[str] = None,
        post_select_detector_indices: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        pipeline = SimulationPipeline(
            decoder_config=DecoderConfig(decoder.name, backend="cpu", params=getattr(decoder, "params", {})),
            max_shots=max_shots,
            max_errors=max_errors,
            num_workers=self.num_workers,
            output_dir=output_dir,
            post_select_detector_indices=post_select_detector_indices,
        )
        pipeline_tasks = [
            PipelineExperimentTask(t.circuit, t.json_metadata)
            for t in tasks
        ]
        return pipeline.run_batch(pipeline_tasks)

    # ==========================================================================
    # Backend 2: Custom GPU (NVIDIA)
    # ==========================================================================
    def _run_gpu(self, tasks, max_shots, max_errors, decoder, gpu_ids) -> pd.DataFrame:
        raise NotImplementedError(
            "The legacy QECSimulator NVIDIA backend is disabled because it did "
            "not perform real decoding. Use SimulationPipeline with a GPU "
            "DecoderConfig instead."
        )
