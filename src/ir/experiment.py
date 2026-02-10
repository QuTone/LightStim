# src/ir/experiment.py

import stim
from abc import ABC, abstractmethod
from typing import Type, Optional, Any

from .builder import CircuitBuilder
from .logical_executor import LogicalExecutor
from .tracker import SyndromeTracker
from ..noise.config import NoiseConfig

class QECExperiment(ABC):
    """
    Abstract base class for all QEC experiments.
    
    This class provides common infrastructure for QEC experiments:
    - System, Builder, and Tracker management
    - Noise configuration
    - Common build workflow helpers
    """
    
    def __init__(self,
                 extraction_block_class: Type,
                 rounds: int = 2,
                 noise_params: Optional[NoiseConfig] = None,
                 noise_model: Optional[str] = 'circuit_level',
                 if_detector: bool = True):
        """
        Args:
            extraction_block_class: Class for syndrome extraction block.
            rounds: Number of QEC rounds.
            noise_params: Parameters for noise injection.
            noise_model: Strategy string ('circuit_level', 'code_capacity', etc.).
            if_detector: Whether to generate detectors.
        """
        self.extraction_block_class = extraction_block_class
        self.rounds = rounds
        self.noise_params = noise_params
        self.noise_model = noise_model
        self.if_detector = if_detector
        
        # Internal state - initialized in build() method
        self.system: Optional[Any] = None
        self.logical_executor: Optional[LogicalExecutor] = None
        self.builder: Optional[CircuitBuilder] = None
        self.tracker: Optional[SyndromeTracker] = None
    
    def _setup_experiment(self):
        """
        Helper method to setup tracker and builder from the system.
        Should be called after self.system is initialized.
        """
        if self.system is None:
            raise ValueError("System must be initialized before setting up tracker and builder.")
        
        # Get number of qubits and logicals
        if hasattr(self.system, 'num_qubits'):
            num_qubits = self.system.num_qubits
        elif hasattr(self.system, 'qubit_coords'):
            num_qubits = len(self.system.qubit_coords)
        else:
            raise ValueError("System must have either 'num_qubits' or 'qubit_coords' attribute.")
        
        if hasattr(self.system, 'num_logicals'):
            num_logicals = self.system.num_logicals
        else:
            num_logicals = 0
        
        self.tracker = SyndromeTracker(num_qubits=num_qubits, expected_num_logicals=num_logicals)
        self.builder = CircuitBuilder(
            tracker=self.tracker,
            system_config=self.system,
            if_detector=self.if_detector
        )
        self.logical_executor = LogicalExecutor(builder=self.builder)
    
    def _inject_noise(self, circuit: stim.Circuit) -> stim.Circuit:
        """
        Helper method to inject noise into the circuit if noise_params is provided.
        
        Args:
            circuit: The clean circuit.
            
        Returns:
            Noisy circuit if noise_params is provided, otherwise the original circuit.
        """
        if self.noise_params is not None:
            print("Injecting noise...")
            return self.builder.build_noisy_circuit(
                noise_params=self.noise_params,
                noise_model=self.noise_model
            )
        else:
            return circuit
    
    @abstractmethod
    def build(self) -> stim.Circuit:
        """
        Constructs the full, noisy Stim circuit for the experiment.
        
        Subclasses must implement this method to define the specific experiment workflow.
        """
        pass
