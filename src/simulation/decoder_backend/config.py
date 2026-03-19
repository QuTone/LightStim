"""Configuration dataclasses for the decoder backend pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class DecoderConfig:
    """Configuration for a decoder (algorithm + backend)."""

    name: str  # e.g. 'pymatching', 'bposd', 'nv-qldpc-decoder'
    backend: Literal["cpu", "gpu", "fpga"] = "cpu"
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.backend = self.backend.lower()
        if self.backend not in ("cpu", "gpu", "fpga"):
            raise ValueError(f"backend must be 'cpu', 'gpu', or 'fpga', got {self.backend}")


@dataclass
class PipelineConfig:
    """Configuration for the simulation pipeline."""

    max_shots: int = 1_000_000
    max_errors: int = 100
    batch_size: int = 10_000
    num_workers: int = 4
    decoder: Optional[DecoderConfig] = None
    post_select_detector_indices: Optional[List[int]] = None
    post_select_observable_indices: Optional[List[int]] = None
    target_observable_indices: Optional[List[int]] = None  # None = all observables
    output_dir: Optional[str] = None
    output_filename: Optional[str] = None
    output_format: Literal["csv", "json", "parquet"] = "csv"
    save_resume_filepath: Optional[str] = None
    progress_enabled: bool = True
    progress_interval_sec: float = 10.0
    progress_min_delta_shots: Optional[int] = None
    progress_poll_interval_sec: float = 0.5
    progress_output: Literal["print", "logging", "both"] = "print"
    progress_logger_name: str = "lightstim.simulation.progress"
    progress_file_path: Optional[str] = None
    progress_file_max_bytes: int = 10_000_000
    progress_file_backup_count: int = 5
    print_progress: bool = True

    def __post_init__(self):
        if self.decoder is None:
            self.decoder = DecoderConfig("pymatching", backend="cpu")
        if self.output_filename is None and self.output_dir is not None:
            self.output_filename = "sim_{timestamp}.csv"
        if not self.print_progress:
            # Backward compatibility: existing callers use print_progress as master switch.
            self.progress_enabled = False
        if self.progress_min_delta_shots is None:
            self.progress_min_delta_shots = max(self.batch_size, 10_000)
        if self.progress_output not in ("print", "logging", "both"):
            raise ValueError(
                "progress_output must be 'print', 'logging', or 'both', "
                f"got {self.progress_output!r}"
            )
        if self.progress_interval_sec <= 0:
            raise ValueError("progress_interval_sec must be > 0")
        if self.progress_poll_interval_sec <= 0:
            raise ValueError("progress_poll_interval_sec must be > 0")


@dataclass
class SimulationStats:
    """Statistics from a single simulation run."""

    shots: int
    post_selected_shots: int
    errors: int
    seconds: float
    decoder: str
    json_metadata: Dict[str, Any]

    @property
    def post_selection_rate(self) -> float:
        if self.shots == 0:
            return 0.0
        return self.post_selected_shots / self.shots

    @property
    def logical_error_rate(self) -> float:
        if self.post_selected_shots == 0:
            return 0.0
        return self.errors / self.post_selected_shots
