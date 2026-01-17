import stim
import sinter
import pandas as pd
import multiprocessing as mp
import time
from typing import List, Dict, Any, Literal
from dataclasses import dataclass
from datetime import datetime

# Import internal modules
from .decoder import BaseDecoder, SinterMWPMDecoder, NvidiaBpOsdDecoder
from .gpu_worker import _gpu_decode_worker_process

@dataclass
class ExperimentTask:
    """Standard input format for our simulator."""
    circuit: stim.Circuit
    json_metadata: Dict[str, Any]

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
                  decoder: BaseDecoder = None, # 接收我们定义的 Decoder 对象
                  gpu_ids: List[int] = [0]) -> pd.DataFrame:
        
        if decoder is None:
            decoder = SinterMWPMDecoder() # Default

        print(f"Starting Simulation Batch | Backend: {self.backend} | Decoder: {decoder.name}")
        
        if self.backend == 'sinter_cpu':
            return self._run_sinter(tasks, max_shots, max_errors, decoder)
        elif self.backend == 'nvidia_gpu':
            return self._run_gpu(tasks, max_shots, max_errors, decoder, gpu_ids)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ==========================================================================
    # Backend 1: Sinter (CPU)
    # ==========================================================================
    def _run_sinter(self, tasks, max_shots, max_errors, decoder) -> pd.DataFrame:
        # 转换任务格式
        sinter_tasks = [
            sinter.Task(circuit=t.circuit, json_metadata=t.json_metadata)
            for t in tasks
        ]

        # 调用 Sinter
        stats = sinter.collect(
            num_workers=self.num_workers,
            tasks=sinter_tasks,
            decoders=[decoder.name], # e.g. 'pymatching'
            max_shots=max_shots,
            max_errors=max_errors,
            print_progress=True,
            save_resume_filepath=f"sinter_resume_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        
        # 格式化输出
        data = []
        for s in stats:
            row = {
                "shots": s.shots,
                "errors": s.errors,
                "decoder": s.decoder,
                "seconds": s.seconds,
                **s.json_metadata
            }
            if s.shots > 0:
                row["logical_error_rate"] = s.errors / s.shots
            data.append(row)
            
        return pd.DataFrame(data)

    # ==========================================================================
    # Backend 2: Custom GPU (NVIDIA)
    # ==========================================================================
    def _run_gpu(self, tasks, max_shots, max_errors, decoder, gpu_ids) -> pd.DataFrame:
        results = []
        
        for i, task in enumerate(tasks):
            print(f"Processing GPU Task {i+1}/{len(tasks)}: {task.json_metadata}")
            
            # Setup Managers
            manager = mp.Manager()
            shared_sim_counter = manager.Value('i', 0)
            shared_fail_counter = manager.Value('i', 0)
            lock = manager.Lock()
            
            # Distribute Processes
            procs_per_gpu = max(1, self.num_workers // len(gpu_ids))
            workers = []
            dem_str = task.circuit.detector_error_model().__str__()
            
            start_time = time.time()
            
            for rank in range(len(gpu_ids) * procs_per_gpu):
                gpu_id = gpu_ids[rank % len(gpu_ids)]
                p = mp.Process(
                    target=_gpu_decode_worker_process,
                    args=(
                        dem_str,
                        decoder.params, # 传入参数字典
                        max_shots,
                        max_errors,
                        gpu_id,
                        shared_sim_counter,
                        shared_fail_counter,
                        lock
                    )
                )
                p.start()
                workers.append(p)
                
            for p in workers:
                p.join()
                
            duration = time.time() - start_time
            
            # Collect Stats
            row = {
                "shots": shared_sim_counter.value,
                "errors": shared_fail_counter.value,
                "decoder": decoder.name,
                "seconds": duration,
                **task.json_metadata
            }
            if row["shots"] > 0:
                row["logical_error_rate"] = row["errors"] / row["shots"]
            results.append(row)
            
        return pd.DataFrame(results)