"""Memory experiment with a disabled rotated-surface-code data qubit."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import FrozenSet, Literal, Optional, Sequence, Tuple

import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
)

Basis = Literal["X", "Z"]


@dataclass(frozen=True)
class DataQubitDefect:
    """System state captured when data qubits are disabled."""

    qubits: FrozenSet[int]
    affected_stabilizers: FrozenSet[int]
    unaffected_stabilizers: FrozenSet[int]


def _alternating_gauge_schedule(
    rounds: int,
    *,
    first_basis: Basis = "Z",
) -> Tuple[Basis, ...]:
    """Return an alternating X/Z gauge schedule."""
    if rounds < 1:
        raise ValueError("rounds must be positive.")
    first_basis = str(first_basis).upper()
    if first_basis not in {"X", "Z"}:
        raise ValueError("first_basis must be X or Z.")
    second_basis = "X" if first_basis == "Z" else "Z"
    return tuple(
        first_basis if round_index % 2 == 0 else second_basis
        for round_index in range(rounds)
    )


def _fully_noiseless_extraction_round(system: QECSystem) -> stim.Circuit:
    """Suppress both gate noise and the SE-boundary idle-noise marker."""
    result = stim.Circuit()
    for instruction in RotatedSurfaceCodeExtractionBlock(system).circuit:
        tag = (
            "noiseless"
            if instruction.name == "TICK" and instruction.tag == "SE_start"
            else instruction.tag
        )
        result.append(
            instruction.name,
            instruction.targets_copy(),
            instruction.gate_args_copy() or None,
            tag=tag,
        )
    return result


class RotatedSurfaceDefectMemoryExperiment:
    """Build a memory experiment that disables one data qubit mid-circuit.

    The experiment prepares a rotated surface-code memory, measures the chosen
    data qubit, removes it from neighboring check supports, and alternates the
    affected X- and Z-type gauge checks. Unaffected checks remain active in
    every post-defect round.

    Args:
        distance: Rotated surface-code distance.
        defect_coord: Coordinate of the data qubit to disable. Defaults to the
            center data qubit.
        memory_basis: Uniform initialization and final-readout basis.
        defect_measure_basis: Basis used to measure the disabled data qubit.
        preparation_rounds: Fully noiseless rounds before the noisy protocol.
        pre_defect_rounds: Rounds before the defect event. Defaults to
            ``distance``.
        post_defect_schedule: Explicit sequence of affected gauge bases.
            Defaults to ``distance`` alternating rounds beginning with Z.
        noise_params: Optional LightStim noise configuration.
        noise_model: Noise-injector strategy name.
    """

    def __init__(
        self,
        distance: int,
        *,
        defect_coord: Optional[Tuple[float, float]] = None,
        memory_basis: Basis = "Z",
        defect_measure_basis: Basis = "Z",
        preparation_rounds: int = 1,
        pre_defect_rounds: Optional[int] = None,
        post_defect_schedule: Optional[Sequence[Basis]] = None,
        noise_params: Optional[NoiseConfig] = None,
        noise_model: str = "circuit_level",
    ):
        self.distance = distance
        self.defect_coord = (
            (distance, distance) if defect_coord is None else tuple(defect_coord)
        )
        self.memory_basis = self._normalize_basis(
            memory_basis,
            name="memory_basis",
        )
        self.defect_measure_basis = self._normalize_basis(
            defect_measure_basis,
            name="defect_measure_basis",
        )
        if preparation_rounds < 0:
            raise ValueError("preparation_rounds must be non-negative.")
        self.preparation_rounds = preparation_rounds
        self.pre_defect_rounds = (
            distance if pre_defect_rounds is None else pre_defect_rounds
        )
        if self.pre_defect_rounds < 0:
            raise ValueError("pre_defect_rounds must be non-negative.")

        if post_defect_schedule is None:
            self.gauge_schedule = _alternating_gauge_schedule(distance)
        else:
            self.gauge_schedule = tuple(
                self._normalize_basis(basis, name="post_defect_schedule")
                for basis in post_defect_schedule
            )
            if not self.gauge_schedule:
                raise ValueError("post_defect_schedule must not be empty.")

        self.noise_params = noise_params
        self.noise_model = noise_model

        self.system: Optional[QECSystem] = None
        self.tracker: Optional[SyndromeTracker] = None
        self.builder: Optional[CircuitBuilder] = None
        self.defect_qubit: Optional[int] = None
        self.defect: Optional[DataQubitDefect] = None
        self.clean_circuit: Optional[stim.Circuit] = None

    @staticmethod
    def _normalize_basis(basis: str, *, name: str) -> Basis:
        normalized = str(basis).upper()
        if normalized not in {"X", "Z"}:
            raise ValueError(f"{name} must be X or Z; got {basis!r}.")
        return normalized

    def build(self) -> stim.Circuit:
        """Construct the detector-annotated clean or noisy Stim circuit."""
        system = QECSystem()
        system.add_patch(
            RotatedSurfaceCode(distance=self.distance),
            name="patch",
        )
        tracker = SyndromeTracker(
            num_qubits=system.num_qubits,
            expected_num_logicals=system.num_logicals,
        )
        builder = CircuitBuilder(
            tracker=tracker,
            system_config=system,
            if_detector=True,
        )
        builder.write_coordinates()

        data_basis = {
            qubit: self.memory_basis
            for qubit in sorted(system.data_indices)
        }
        builder.initialize(data_basis, system.num_qubits, noiseless=True)

        if self.preparation_rounds:
            builder.apply_syndrome_extraction(
                _fully_noiseless_extraction_round(system),
                rounds=self.preparation_rounds,
                noiseless=True,
            )
        if self.pre_defect_rounds:
            builder.apply_syndrome_extraction(
                RotatedSurfaceCodeExtractionBlock(system).circuit,
                rounds=self.pre_defect_rounds,
            )

        try:
            defect_qubit = system.index_map[self.defect_coord]
        except KeyError as ex:
            raise ValueError(
                f"defect_coord {self.defect_coord} is not a qubit coordinate."
            ) from ex
        if defect_qubit not in system.data_indices:
            raise ValueError(
                f"defect_coord {self.defect_coord} is not a data-qubit coordinate."
            )

        builder.apply_mid_data_readout(
            {defect_qubit: self.defect_measure_basis},
        )
        baseline_active = frozenset(system.active_stabilizer_indices)
        affected = frozenset(
            system.disable_data_qubits({defect_qubit})
            & set(baseline_active)
        )
        defect = DataQubitDefect(
            qubits=frozenset({defect_qubit}),
            affected_stabilizers=affected,
            unaffected_stabilizers=baseline_active - affected,
        )

        for basis, phase in groupby(self.gauge_schedule):
            selected_gauges = {
                uid
                for uid in defect.affected_stabilizers
                if system.stabilizers[uid].get("type") == basis
            }
            system.active_stabilizer_indices = (
                set(defect.unaffected_stabilizers) | selected_gauges
            )
            builder.apply_syndrome_extraction(
                RotatedSurfaceCodeExtractionBlock(system).circuit,
                rounds=sum(1 for _ in phase),
            )

        builder.apply_data_readout({
            qubit: self.memory_basis
            for qubit in sorted(
                system.data_indices - system.disabled_data_qubit_indices
            )
        })

        self.system = system
        self.tracker = tracker
        self.builder = builder
        self.defect_qubit = defect_qubit
        self.defect = defect
        self.clean_circuit = builder.circuit.copy()

        if self.noise_params is None:
            return self.clean_circuit
        return builder.build_noisy_circuit(
            noise_params=self.noise_params,
            noise_model=self.noise_model,
        )
