"""Rotated-code memory experiment with a permanent enlargement step.

Prepare an X/Z eigenstate at the source distance, enlarge through the logical
executor, and destructively read out the output patch in the same basis.
When noise is configured it covers preparation, growth and readout. The
resulting LER is an end-to-end experiment metric, not an isolated growth-gate
error rate or a claim that the whole circuit has the output code's distance.
"""

from __future__ import annotations

import stim

from lightstim.ir.experiment import QECExperiment
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
    RotatedSurfaceCodeLogicalOpSet,
)


class RotatedSurfaceEnlargeExperiment(QECExperiment):
    """Prepare -> source SE -> enlarge -> transversal X/Z readout.

    ``source_rounds`` counts preparation SE cycles; ``rounds`` counts all
    target-distance SE cycles, including the first growth projection.
    There is no implicit ideal terminal SE or postselection. After ``build``,
    ``output_patch`` is the enlarged global view; ``source_patch`` is retained
    only for inspecting the input geometry.
    """

    def __init__(
        self,
        distance: int = 3,
        target_distance: int = 5,
        basis: str = "X",
        rounds: int = 1,
        source_rounds: int = 1,
        direction: str = "+x,+y",
        noise_params: NoiseConfig | None = None,
        noise_model: str = "circuit_level",
    ):
        for name, value in (("distance", distance), ("target_distance", target_distance)):
            if type(value) is not int or value < 3 or value % 2 == 0:
                raise ValueError(f"{name} must be an odd integer at least 3")
        if target_distance <= distance:
            raise ValueError("target_distance must be greater than distance")
        for name, value in (("rounds", rounds), ("source_rounds", source_rounds)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if direction != "+x,+y":
            raise ValueError("Only direction='+x,+y' is currently supported")
        basis = str(basis).upper()
        if basis not in {"X", "Z"}:
            raise ValueError("basis must be 'X' or 'Z'")
        super().__init__(
            extraction_block_class=RotatedSurfaceCodeExtractionBlock,
            rounds=rounds,
            noise_params=noise_params,
            noise_model=noise_model,
        )
        self.distance = distance
        self.target_distance = target_distance
        self.basis = basis
        self.source_rounds = source_rounds
        self.direction = direction
        self.source_patch = None
        self.output_patch = None

    def build(self) -> stim.Circuit:
        """Build a fresh clean circuit, then optionally inject configured noise."""
        self.system = QECSystem()
        self.source_patch = self.system.add_patch(
            RotatedSurfaceCode(distance=self.distance), name="patch",
        )
        self._setup_experiment()
        self.logical_executor.register_op_set(
            RotatedSurfaceCode,
            RotatedSurfaceCodeLogicalOpSet(self.extraction_block_class),
        )
        self.builder.write_coordinates()
        self.builder.initialize(
            {q: self.basis for q in sorted(self.source_patch.data_indices)},
            n=self.system.num_qubits,
        )
        source_se = self.extraction_block_class(self.system, scheduling="perpendicular")
        self.builder.apply_syndrome_extraction(source_se.circuit, rounds=self.source_rounds)
        self.output_patch = self.logical_executor.apply_logical_operation(
            "enlarge", [self.source_patch],
            target_distance=self.target_distance,
            direction=self.direction,
            rounds=self.rounds,
        )
        self.builder.apply_data_readout(
            {q: self.basis for q in sorted(self.output_patch.data_indices)},
        )
        return self._inject_noise(self.builder.to_stim_circuit())
