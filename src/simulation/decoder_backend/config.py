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
    output_dir: Optional[str] = None
    output_filename: Optional[str] = None
    output_format: Literal["csv", "json", "parquet"] = "csv"
    save_resume_filepath: Optional[str] = None
    print_progress: bool = True

    def __post_init__(self):
        if self.decoder is None:
            self.decoder = DecoderConfig("pymatching", backend="cpu")
        if self.output_filename is None and self.output_dir is not None:
            self.output_filename = "sim_{timestamp}.csv"


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
