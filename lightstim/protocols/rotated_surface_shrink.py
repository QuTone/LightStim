"""Rotated-code memory experiment with a permanent contraction step.

Prepare an X/Z eigenstate at the source distance, measure the removed data
through the logical executor, and read out the retained patch in the same
basis. The tracker includes the measurement-dependent logical Pauli frame
in the final observable. Configured noise covers the complete experiment.
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


class RotatedSurfaceShrinkExperiment(QECExperiment):
    """Prepare -> source SE -> shrink -> transversal X/Z readout.

    ``source_rounds`` counts preparation SE cycles. ``rounds`` counts optional
    target-distance SE cycles after shrinking; its default of zero adds only
    the removed-data measurements. The minimum-coordinate corner survives.
    There is no postselection or physical logical-frame correction. After
    ``build``, ``output_patch`` is the smaller global view; ``source_patch``
    remains available only for inspecting the input geometry.
    """

    def __init__(
        self,
        distance: int = 5,
        target_distance: int = 3,
        basis: str = "X",
        rounds: int = 0,
        source_rounds: int = 1,
        direction: str = "-x,-y",
        noise_params: NoiseConfig | None = None,
        noise_model: str = "circuit_level",
    ):
        for name, value in (("distance", distance), ("target_distance", target_distance)):
            if type(value) is not int or value < 3 or value % 2 == 0:
                raise ValueError(f"{name} must be an odd integer at least 3")
        if target_distance >= distance:
            raise ValueError("target_distance must be smaller than distance")
        if type(rounds) is not int or rounds < 0:
            raise ValueError("rounds must be a nonnegative integer")
        if type(source_rounds) is not int or source_rounds < 1:
            raise ValueError("source_rounds must be a positive integer")
        if direction != "-x,-y":
            raise ValueError("Only direction='-x,-y' is currently supported")
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
            "shrink", [self.source_patch],
            target_distance=self.target_distance,
            direction=self.direction,
            rounds=0,
        )
        # Construct an ordinary SE block from the updated system, after the
        # measurement-only contraction. Its active checks and physical IDs now
        # describe the smaller patch; no source-distance block is reused.
        if self.rounds:
            target_se = self.extraction_block_class(self.system, scheduling="perpendicular")
            self.builder.apply_syndrome_extraction(target_se.circuit, rounds=self.rounds)
        self.builder.apply_data_readout(
            {q: self.basis for q in sorted(self.output_patch.data_indices)},
        )
        return self._inject_noise(self.builder.to_stim_circuit())
