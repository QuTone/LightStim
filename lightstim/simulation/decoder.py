from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseDecoder(ABC):
    """
    Abstract base class for Decoders.
    Serves as a configuration container.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Name used for logging and Sinter integration."""
        pass

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """Parameters required to initialize the decoder."""
        pass

class SinterMWPMDecoder(BaseDecoder):
    """
    Configuration for Sinter's PyMatching decoder.
    """
    def __init__(self):
        self._name = "pymatching"

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> Dict[str, Any]:
        return {}

class NvidiaBpOsdDecoder(BaseDecoder):
    """
    Configuration for NVIDIA's GPU BP+OSD decoder.
    """
    def __init__(self, max_iter: int = 50, osd_order: int = 0, batch_size: int = 10000):
        self._max_iter = max_iter
        self._osd_order = osd_order
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return "nvidia_gpu_bp_osd"

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "max_iterations": self._max_iter,
            "osd_order": self._osd_order,
            "batch_size": self._batch_size
        }