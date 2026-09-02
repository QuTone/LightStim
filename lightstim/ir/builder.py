import copy
import logging
import stim
import numpy as np
from typing import List, Dict, Any, Optional, Union, Literal, Set, Tuple
from dataclasses import dataclass

_log = logging.getLogger(__name__)

from ..ir.tracker import SyndromeTracker, _append_detector
from ..noise.config import NoiseConfig
from ..noise.injector import NoiseInjector

def _make_noiseless(circuit: stim.Circuit) -> stim.Circuit:
    """Return a copy of circuit with all gate instructions tagged 'noiseless'.

    Structural instructions (TICK, SHIFT_COORDS, DETECTOR, etc.) are preserved
    as-is so that detector generation and time-shift logic are unaffected.
    """
    structural = {'TICK', 'SHIFT_COORDS', 'QUBIT_COORDS', 'DETECTOR', 'OBSERVABLE_INCLUDE'}
    result = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(inst.repeat_count, _make_noiseless(inst.body_copy())))
        elif inst.name in structural:
            args = inst.gate_args_copy() or None
            result.append(inst.name, inst.targets_copy(), args, tag=inst.tag)
        else:
            args = inst.gate_args_copy() or None
            result.append(inst.name, inst.targets_copy(), args, tag="noiseless")
    return result


@dataclass(frozen=True)
class _MeasurementBlockAnalysis:
    circuit: stim.Circuit
    forward_symplectic_matrix: np.ndarray
    back_propagated_paulis: np.ndarray
    reset_paulis: Optional[np.ndarray]
    measurement_qubit_indices: List[int]
    measurement_bases: List[str]
    measurement_coords: list
    discarded_measurement_qubit_indices: Set[int]
    no_detector_mask: Optional[np.ndarray]


class CircuitBuilder:
    """
    Constructs the Stim circuit for QEC experiments.
    SyndromeTracker automatically generates detectors and logical observables.
    NoiseInjector automatically injects noise to appropriate places according to the given noise model.
    """

    def __init__(self,
                 tracker: SyndromeTracker,
                 system_config: Any,
                 if_detector: bool = True):
        """
        Args:
            tracker: Initialized SyndromeTracker instance.
            system_config: Object containing system specs:
                           - qubit_coords: Dict[int, List[float]] OR List[List[float]]
                           - data_indices: List[int]
                           - syndrome_indices: List[int]
                           - syndrome_coords: List[List[float]]
        """
        self.tracker = tracker
        self.system = system_config
        self.circuit = stim.Circuit()
        self.if_detector = if_detector
        # State set by apply_syndrome_extraction(z_only=True) for use by apply_data_readout
        self._z_only_syn_qubit_indices = None
        self._z_only_no_detector_mask  = None
        self._z_only_n_meas_per_round  = None

    # --------------------------------------------------------------------------
    # A. Setup & Initialization
    # --------------------------------------------------------------------------
    def write_coordinates(self, start_index: int = 0):
        """
        Generates QUBIT_COORDS instructions based on the system's layout.
        Essential for visualization.
        When start_index > 0, only writes coords for qubit indices >= start_index (for define-by-run).
        Tracks written indices to avoid duplicates from sequential coupler cycles.
        """
        if not hasattr(self, '_written_coord_indices'):
            self._written_coord_indices = set()

        coords_iterable = None
        if isinstance(self.system.qubit_coords, dict):
            coords_iterable = self.system.qubit_coords.items()
        elif isinstance(self.system.qubit_coords, list):
            coords_iterable = enumerate(self.system.qubit_coords)

        if coords_iterable:
            for q_index, coords in coords_iterable:
                if q_index >= start_index and q_index not in self._written_coord_indices:
                    self.circuit.append("QUBIT_COORDS", [q_index], list(coords))
                    self._written_coord_indices.add(q_index)

    def append_coordinates_for_new_qubits(self, start_index: int):
        """
        Insert QUBIT_COORDS for qubits with index >= start_index right after existing
        coords at the front of the circuit (instead of appending at the end).
        Used automatically by add_patch when builder is registered (define-by-run).
        """
        # Build circuit containing only the new QUBIT_COORDS
        new_coords_circuit = stim.Circuit()
        coords_iterable = None
        if isinstance(self.system.qubit_coords, dict):
            coords_iterable = self.system.qubit_coords.items()
        elif isinstance(self.system.qubit_coords, list):
            coords_iterable = enumerate(self.system.qubit_coords)
        if coords_iterable:
            for q_index, coords in coords_iterable:
                if q_index >= start_index:
                    new_coords_circuit.append("QUBIT_COORDS", [q_index], list(coords))
        # Insert at position start_index (first n_old instructions are existing coords)
        self.circuit = self.circuit[:start_index] + new_coords_circuit + self.circuit[start_index:]

    def initialize(self, init_dict: Dict[int, str], n: int, noiseless: bool = False):
        """
        Resets specific qubits in a given basis.

        Args:
            init_dict: Mapping from qubit index to basis ('X', 'Y', 'Z').
            n: Total number of qubits in the system.
            noiseless: If True, tag all reset instructions with 'noiseless' so that
                the noise injector skips them (useful for injection-phase init).
        """
        qubit_indices_x = [q for q, b in init_dict.items() if b == 'X']
        qubit_indices_z = [q for q, b in init_dict.items() if b == 'Z']
        qubit_indices_y = [q for q, b in init_dict.items() if b == 'Y']

        tag = "noiseless" if noiseless else ""
        # Apply Reset Gate
        if qubit_indices_x:
            self.circuit.append("RX", qubit_indices_x, tag=tag)
        if qubit_indices_z:
            self.circuit.append("R", qubit_indices_z, tag=tag)
        if qubit_indices_y:
            self.circuit.append("RY", qubit_indices_y, tag=tag)

        init_tableau = self._get_initialization_tableau(qubit_indices_x, qubit_indices_z, qubit_indices_y, n)

        self.tracker.process_initialization(init_tableau)

        # Track active qubits (logical lifetime)
        self.system.active_qubit_indices.update(init_dict.keys())

    def stabilizer_canonicalization(self, stabilizer_uids: Optional[Set[int]] = None) -> None:
        """
        Re-organize stabilizer tableau into stabilizers vs logicals using
        the code-defined canonical basis.

        This can be used after encoding or to finalize a composite SE round
        whose explicit measurement blocks were processed separately.
        """
        self.tracker.stabilizer_canonicalization(self.system, stabilizer_uids)

    def logical_canonicalization(self, canonical_logicals: Dict[int, "np.ndarray"]) -> None:
        """
        Replace logical operators with preferred canonical representatives.
        Call after stabilizer_canonicalization(), before SE.

        Args:
            canonical_logicals: {logical_index: pauli_vector (2n,)}
        """
        self.tracker.logical_canonicalization(canonical_logicals)

    # --------------------------------------------------------------------------
    # B. Syndrome Extraction
    # --------------------------------------------------------------------------
    def apply_syndrome_extraction(self,
                                  circuit_chunk: stim.Circuit,
                                  rounds: int = 1,
                                  noiseless: bool = False,
                                  z_only: bool = False,
                                  measurement_blocks: Optional[Tuple[stim.Circuit, ...]] = None):
        """
        Applies syndrome extraction with automated Tracker integration.

        Args:
            circuit_chunk: A Stim circuit representing ONE round of stabilizer measurement.
                           Only includes circuit operations. The last instruction has to be syndrome qubit measurement.
            rounds: Number of times to repeat.
            noiseless: If True, tag all gate instructions in circuit_chunk as 'noiseless'
                so the noise injector skips them. Useful for injection-stabilization rounds.
            z_only: If True, only Z-ancilla measurements emit DETECTOR instructions.
                X-ancilla are still measured (so the tableau update is correct) but
                suppressed from the DEM. State is stored for apply_data_readout(z_only=True).
        """
        # ONE engine: every round goes through the measurement-block path
        # (the engine on main).  A call without explicit blocks treats the
        # whole chunk as a single block; the legacy single-block engine is
        # gone (review: do not maintain two syndrome-extraction engines).
        return self._apply_syndrome_extraction_blocks(
            circuit_chunk, rounds=rounds, noiseless=noiseless,
            z_only=z_only,
            measurement_blocks=tuple(measurement_blocks or (circuit_chunk,)))

    def _apply_syndrome_extraction_blocks(self,
                                  circuit_chunk: stim.Circuit,
                                  rounds: int = 1,
                                  noiseless: bool = False,
                                  z_only: bool = False,
                                  measurement_blocks: Optional[Tuple[stim.Circuit, ...]] = None):
        """
        Applies syndrome extraction with automated Tracker integration.

        Args:
            circuit_chunk: The complete physical circuit for one SE round.
            measurement_blocks: Explicit tracker boundaries within the round.
                Each block starts from its input preparation and ends in one
                contiguous readout layer. When omitted, ``circuit_chunk`` is
                treated as one measurement block.
            rounds: Number of times to repeat.
            noiseless: If True, tag all gate instructions in circuit_chunk as 'noiseless'
                so the noise injector skips them. Useful for injection-stabilization rounds.
            z_only: If True, only Z-ancilla measurements emit DETECTOR instructions.
                X-ancilla are still measured (so the tableau update is correct) but
                suppressed from the DEM. State is stored for apply_data_readout(z_only=True).
        """
        if rounds < 1:
            return

        blocks = tuple(measurement_blocks or (circuit_chunk,))
        if not blocks:
            raise ValueError("measurement_blocks must not be empty.")
        if noiseless:
            blocks = tuple(_make_noiseless(block) for block in blocks)
        circuit_chunk = self._join_measurement_blocks(blocks)

        if not self.if_detector:
            self.circuit += circuit_chunk
            if rounds > 1:
                steady_round_body = stim.Circuit()
                steady_round_body.append("TICK")
                steady_round_body += circuit_chunk
                self.circuit += steady_round_body
                if rounds > 2:
                    self.circuit.append(stim.CircuitRepeatBlock(
                        rounds - 2,
                        steady_round_body,
                    ))
            return

        analyses = tuple(
            self._analyze_measurement_block(block, z_only=z_only)
            for block in blocks
        )
        if z_only:
            if len(analyses) != 1:
                raise ValueError(
                    "z_only readout currently supports one measurement block per SE round."
                )
            analysis = analyses[0]
            self._z_only_syn_qubit_indices = (
                analysis.measurement_qubit_indices
            )
            self._z_only_no_detector_mask = analysis.no_detector_mask
            self._z_only_n_meas_per_round = len(
                analysis.measurement_qubit_indices
            )

        _log.debug("Applying first round of syndrome extraction...")
        self._process_measurement_blocks(
            output_circuit=self.circuit,
            analyses=analyses,
            shift_round=True,
        )

        if rounds <= 1:
            return

        _log.debug("Applying second syndrome-extraction round...")
        first_round_logical_components = set(
            self.tracker.stabilizer_with_logical_components
        )
        steady_round_body = stim.Circuit()
        steady_round_body.append("TICK")
        self._process_measurement_blocks(
            output_circuit=steady_round_body,
            analyses=analyses,
            shift_round=True,
        )
        self.tracker.stabilizer_with_logical_components.update(
            first_round_logical_components
        )
        self.circuit += steady_round_body

        if rounds <= 2:
            return

        repeated_round_body = self._try_compress_steady_rounds(
            repetitions=rounds - 2,
            analyses=analyses,
        )
        if repeated_round_body is not None:
            self.circuit.append(stim.CircuitRepeatBlock(
                rounds - 2,
                repeated_round_body,
            ))
            return

        if not self._uses_disposable_syndrome_readout(analyses):
            persistent_logical_components = set(
                self.tracker.stabilizer_with_logical_components
            )
            for _ in range(rounds - 2):
                explicit_round_body = stim.Circuit()
                explicit_round_body.append("TICK")
                self._process_measurement_blocks(
                    output_circuit=explicit_round_body,
                    analyses=analyses,
                    shift_round=True,
                )
                self.circuit += explicit_round_body
                self.tracker.stabilizer_with_logical_components.update(
                    persistent_logical_components
                )
                persistent_logical_components.update(
                    self.tracker.stabilizer_with_logical_components
                )
            return

        self.circuit.append(stim.CircuitRepeatBlock(
            rounds - 2,
            steady_round_body,
        ))

        persistent_logical_components = set(
            self.tracker.stabilizer_with_logical_components
        )
        for _ in range(rounds - 2):
            promotable_stabilizer_rows = []
            for analysis in analyses:
                repeated_base_idx = self.tracker.total_measurements
                self.tracker.meas_rec_to_idx_map.update({
                    repeated_base_idx + i: qubit
                    for i, qubit in enumerate(
                        analysis.measurement_qubit_indices
                    )
                })
                promotable_stabilizer_rows.append(
                    self.tracker.process_mid_measurement(
                        circuit=stim.Circuit(),
                        forward_symplectic_matrix=(
                            analysis.forward_symplectic_matrix
                        ),
                        back_propagated_paulis=(
                            analysis.back_propagated_paulis
                        ),
                        reset_paulis=analysis.reset_paulis,
                        measurement_qubit_indices=(
                            analysis.measurement_qubit_indices
                        ),
                        measurement_bases=analysis.measurement_bases,
                        measurement_coords=analysis.measurement_coords,
                        discarded_measurement_qubit_indices=(
                            analysis.discarded_measurement_qubit_indices
                        ),
                        no_detector_mask=np.ones(
                            len(analysis.measurement_qubit_indices),
                            dtype=bool,
                        ),
                    )
                )
            self._finish_measurement_block_group(
                analyses,
                promotable_stabilizer_rows=tuple(
                    promotable_stabilizer_rows
                ),
            )
            self.tracker.stabilizer_with_logical_components.update(
                persistent_logical_components
            )
            persistent_logical_components.update(
                self.tracker.stabilizer_with_logical_components
            )

    @staticmethod
    def _join_measurement_blocks(
        measurement_blocks: Tuple[stim.Circuit, ...],
    ) -> stim.Circuit:
        circuit = stim.Circuit()
        for block in measurement_blocks:
            circuit += block
        return circuit

    @staticmethod
    def _uses_disposable_syndrome_readout(
        analyses: Tuple[_MeasurementBlockAnalysis, ...],
    ) -> bool:
        return all(
            analysis.discarded_measurement_qubit_indices
            == set(analysis.measurement_qubit_indices)
            for analysis in analyses
        )

    def _analyze_measurement_block(
        self,
        circuit: stim.Circuit,
        *,
        z_only: bool,
    ) -> _MeasurementBlockAnalysis:
        (
            back_propagated_paulis,
            measurement_qubit_indices,
            measurement_bases,
        ) = self._get_back_propagated_pauli(
            circuit,
            self.tracker.num_qubits,
            include_measurement_bases=True,
        )
        if circuit.num_measurements != len(measurement_qubit_indices):
            raise ValueError(
                "Each syndrome measurement block must contain exactly one "
                "terminal readout layer. Declare multiple measurement_blocks "
                "instead of placing an earlier measurement inside one block."
            )

        reset_paulis = self._get_terminal_reset_paulis(
            circuit,
            self.tracker.num_qubits,
            measurement_qubit_indices,
        )
        forward_symplectic_matrix = self.tracker.get_forward_symplectic_matrix(
            circuit,
            self.tracker.num_qubits,
        )
        measurement_coords = [
            self.system.qubit_coords[i]
            for i in measurement_qubit_indices
        ]
        patch_syndrome_qubits = set(self.system.active_syndrome_indices)
        discarded_measurement_qubit_indices = (
            set(measurement_qubit_indices) & patch_syndrome_qubits
        )
        if discarded_measurement_qubit_indices and (
            discarded_measurement_qubit_indices
            != set(measurement_qubit_indices)
        ):
            raise ValueError(
                "A measurement block cannot currently mix disposable syndrome "
                "ancillas with retained data-qubit measurements."
            )

        no_detector_mask = None
        if z_only:
            x_ancillas = set(self.system.active_syndrome_indices_x)
            no_detector_mask = np.array(
                [q in x_ancillas for q in measurement_qubit_indices],
                dtype=bool,
            )

        return _MeasurementBlockAnalysis(
            circuit=circuit,
            forward_symplectic_matrix=forward_symplectic_matrix,
            back_propagated_paulis=back_propagated_paulis,
            reset_paulis=reset_paulis,
            measurement_qubit_indices=measurement_qubit_indices,
            measurement_bases=measurement_bases,
            measurement_coords=measurement_coords,
            discarded_measurement_qubit_indices=(
                discarded_measurement_qubit_indices
            ),
            no_detector_mask=no_detector_mask,
        )

    def _try_canonicalize_stateful_code_frame(
        self,
        analysis: _MeasurementBlockAnalysis,
        *,
        tracker: Optional[SyndromeTracker] = None,
    ) -> bool:
        """Find and classify a code frame inside a retained-data block."""
        tracker = self.tracker if tracker is None else tracker
        if (
            tracker.expected_num_logicals == 0
            or tracker.logicals.count
            == tracker.expected_num_logicals
        ):
            return False

        prefix = stim.Circuit()
        saw_unitary = False
        unitary_instruction_count = 0
        checked_instruction_count = 0

        for instruction in analysis.circuit:
            if not isinstance(instruction, stim.CircuitInstruction):
                continue
            gate = stim.gate_data(instruction.name)
            if gate.is_unitary:
                prefix.append(
                    instruction.name,
                    instruction.targets_copy(),
                    instruction.gate_args_copy(),
                )
                saw_unitary = True
                unitary_instruction_count += 1
                continue
            if instruction.name != "TICK" or not saw_unitary:
                continue
            if unitary_instruction_count == checked_instruction_count:
                continue
            checked_instruction_count = unitary_instruction_count

            probe = copy.deepcopy(tracker)
            probe.process_unitary_block(prefix)
            try:
                probe.rebase_stabilizers_onto_code_basis(self.system)
            except RuntimeError:
                continue
            if probe.logicals.count != probe.expected_num_logicals:
                continue

            tracker.process_unitary_block(prefix)
            tracker.rebase_stabilizers_onto_code_basis(self.system)
            tracker.process_unitary_block(prefix.inverse())
            return True

        return False

    def _process_measurement_blocks(
        self,
        *,
        output_circuit: stim.Circuit,
        analyses: Tuple[_MeasurementBlockAnalysis, ...],
        shift_round: bool,
        tracker: Optional[SyndromeTracker] = None,
    ) -> None:
        tracker = self.tracker if tracker is None else tracker
        promotable_stabilizer_rows = []
        for block_index, analysis in enumerate(analyses):
            output_circuit += analysis.circuit
            measurement_base_idx = tracker.total_measurements
            tracker.meas_rec_to_idx_map.update({
                measurement_base_idx + i: qubit
                for i, qubit in enumerate(
                    analysis.measurement_qubit_indices
                )
            })
            if shift_round and block_index == 0:
                output_circuit.append("SHIFT_COORDS", [], [0, 0, 1])

            reset_paulis = analysis.reset_paulis
            if not analysis.discarded_measurement_qubit_indices:
                if reset_paulis is not None:
                    tracker.process_resets(reset_paulis)
                    reset_paulis = None
                self._try_canonicalize_stateful_code_frame(
                    analysis,
                    tracker=tracker,
                )

            promotable_stabilizer_rows.append(
                tracker.process_mid_measurement(
                    circuit=output_circuit,
                    forward_symplectic_matrix=(
                        analysis.forward_symplectic_matrix
                    ),
                    back_propagated_paulis=(
                        analysis.back_propagated_paulis
                    ),
                    reset_paulis=reset_paulis,
                    measurement_qubit_indices=(
                        analysis.measurement_qubit_indices
                    ),
                    measurement_bases=analysis.measurement_bases,
                    measurement_coords=analysis.measurement_coords,
                    discarded_measurement_qubit_indices=(
                        analysis.discarded_measurement_qubit_indices
                    ),
                    no_detector_mask=analysis.no_detector_mask,
                )
            )
        self._finish_measurement_block_group(
            analyses,
            promotable_stabilizer_rows=tuple(
                promotable_stabilizer_rows
            ),
            tracker=tracker,
        )

    def _finish_measurement_block_group(
        self,
        analyses: Tuple[_MeasurementBlockAnalysis, ...],
        *,
        promotable_stabilizer_rows: Tuple[Set[int], ...],
        tracker: Optional[SyndromeTracker] = None,
    ) -> None:
        """Apply Builder-owned state classification at an SE-round boundary."""
        tracker = self.tracker if tracker is None else tracker
        if self._uses_disposable_syndrome_readout(analyses):
            if len(analyses) == 1:
                tracker.promote_stabilizer_rows_to_logicals(
                    promotable_stabilizer_rows[0]
                )
            else:
                tracker.rebase_stabilizers_onto_code_basis(self.system)
        else:
            tracker.validate_logical_count(
                context="syndrome-extraction round"
            )

    def _try_compress_steady_rounds(
        self,
        *,
        repetitions: int,
        analyses: Tuple[_MeasurementBlockAnalysis, ...],
    ) -> Optional[stim.Circuit]:
        """Compress a verified repeated SE state transition into one Stim body.

        The second round defines a candidate canonical logical representative.
        Two additional rounds are evaluated on a tracker copy. If stabilizer
        records remain fixed to an old measurement, adjacent-round detector
        products remove that anchor before the verified body is repeated.
        """
        tracker = self.tracker
        if (
            repetitions < 1
            or tracker.logicals.count != 1
            or tracker.expected_num_logicals != 1
            or tracker.stabilizer_with_logical_components
            or tracker._gauge_logical_vectors
            or tracker.absorbed_ops.count
            or tracker.post_select_row_indices
            or self.circuit.num_observables > 0
            or tracker.total_observables > 0
        ):
            return None

        if any(
            record < 0
            for tableau in (tracker.stabilizers, tracker.logicals)
            for records in tableau.records
            for record in records
        ):
            return None

        canonical_logical = {0: tracker.logicals.matrix[0].copy()}
        baseline_logical_records = set(tracker.logicals.records[0])
        initial_total_measurements = tracker.total_measurements
        round_syn_qubits = [
            qubit
            for analysis in analyses
            for qubit in analysis.measurement_qubit_indices
        ]
        measurements_per_round = len(round_syn_qubits)
        probe = copy.deepcopy(tracker)
        probe_states = []
        probe_round_bodies = []
        previous_logical_records = baseline_logical_records

        try:
            for _ in range(2):
                probe_round_body = stim.Circuit()
                probe_round_body.append("TICK")
                self._process_measurement_blocks(
                    output_circuit=probe_round_body,
                    analyses=analyses,
                    shift_round=True,
                    tracker=probe,
                )
                probe.logical_canonicalization(canonical_logical)

                logical_records = set(probe.logicals.records[0])
                logical_delta = tuple(sorted(
                    record - probe.total_measurements
                    for record in previous_logical_records ^ logical_records
                ))
                stabilizer_record_offsets = tuple(
                    tuple(sorted(
                        record - probe.total_measurements
                        for record in records
                    ))
                    for records in probe.stabilizers.records
                )
                probe_states.append((
                    probe.stabilizers.matrix.copy(),
                    stabilizer_record_offsets,
                    logical_delta,
                ))
                probe_round_bodies.append(probe_round_body)
                previous_logical_records = logical_records
        except (RuntimeError, ValueError):
            return None

        first_matrix, first_stabilizer_offsets, first_logical_delta = probe_states[0]
        second_matrix, second_stabilizer_offsets, second_logical_delta = probe_states[1]
        if (
            not np.array_equal(first_matrix, second_matrix)
            or first_logical_delta != second_logical_delta
            or any(offset >= 0 for offset in first_logical_delta)
            or any(
                offset >= 0
                for records in first_stabilizer_offsets
                for offset in records
            )
        ):
            return None

        if (
            first_stabilizer_offsets == second_stabilizer_offsets
            and probe_round_bodies[0] == probe_round_bodies[1]
        ):
            repeated_round_body = probe_round_bodies[0].copy()
        else:
            anchored_stabilizers = all(
                tuple(
                    offset - measurements_per_round
                    for offset in first_offsets
                )
                == second_offsets
                for first_offsets, second_offsets in zip(
                    first_stabilizer_offsets,
                    second_stabilizer_offsets,
                )
            )
            if not anchored_stabilizers:
                return None
            repeated_round_body = self._make_periodic_detector_body(
                previous_body=probe_round_bodies[0],
                current_body=probe_round_bodies[1],
                measurements_per_round=measurements_per_round,
            )
            if repeated_round_body is None:
                return None

        final_total_measurements = (
            initial_total_measurements
            + repetitions * measurements_per_round
        )
        final_stabilizer_records = [
            [final_total_measurements + offset for offset in offsets]
            for offsets in first_stabilizer_offsets
        ]
        if any(
            record < 0
            for records in final_stabilizer_records
            for record in records
        ):
            return None

        if first_logical_delta:
            # Accumulation into ID 0 (the observable the terminal readout
            # allocates for this same logical) — the legal exception spelled
            # out in tracker.allocate_observable, not a second allocation.
            # Sound only because the guard above refused compression when
            # tracker.total_observables > 0: with no reservation ahead of
            # it, the sole logical's terminal readout is guaranteed to
            # allocate ID 0.
            repeated_round_body.append(
                "OBSERVABLE_INCLUDE",
                [stim.target_rec(offset) for offset in first_logical_delta],
                [0],
            )

        tracker.stabilizers.matrix = first_matrix
        tracker.stabilizers.records = final_stabilizer_records
        tracker.logicals.matrix = canonical_logical[0].reshape(1, -1)
        tracker.logicals.records = [sorted(baseline_logical_records)]
        tracker.total_measurements = final_total_measurements

        for records in final_stabilizer_records:
            for record in records:
                position = (
                    record - final_total_measurements
                ) % measurements_per_round
                tracker.meas_rec_to_idx_map[record] = round_syn_qubits[position]

        _log.debug(
            "Compressed %d steady syndrome-extraction rounds with logical delta %s.",
            repetitions,
            first_logical_delta,
        )
        return repeated_round_body

    @staticmethod
    def _make_periodic_detector_body(
        *,
        previous_body: stim.Circuit,
        current_body: stim.Circuit,
        measurements_per_round: int,
    ) -> Optional[stim.Circuit]:
        """Eliminate fixed record anchors using adjacent-round detectors."""
        previous_instructions = list(previous_body)
        current_instructions = list(current_body)
        if len(previous_instructions) != len(current_instructions):
            return None

        result = stim.Circuit()
        for previous, current in zip(
            previous_instructions,
            current_instructions,
        ):
            if (
                not isinstance(previous, stim.CircuitInstruction)
                or not isinstance(current, stim.CircuitInstruction)
            ):
                return None
            if current.name != "DETECTOR":
                if previous != current:
                    return None
                result.append(
                    current.name,
                    current.targets_copy(),
                    current.gate_args_copy(),
                    tag=current.tag,
                )
                continue
            if (
                previous.name != "DETECTOR"
                or previous.gate_args_copy() != current.gate_args_copy()
                or previous.tag != current.tag
            ):
                return None

            record_offsets = set()
            for target in current.targets_copy():
                if not target.is_measurement_record_target:
                    return None
                record_offsets.symmetric_difference_update([target.value])
            for target in previous.targets_copy():
                if not target.is_measurement_record_target:
                    return None
                record_offsets.symmetric_difference_update([
                    target.value - measurements_per_round
                ])
            result.append(
                "DETECTOR",
                [
                    stim.target_rec(offset)
                    for offset in sorted(record_offsets)
                ],
                current.gate_args_copy(),
                tag=current.tag,
            )

        return result

    # --------------------------------------------------------------------------
    # C. Unitary Block (Logical Gates, Unitary Encoding, etc.)
    # --------------------------------------------------------------------------

    @staticmethod
    def _get_terminal_reset_paulis(
        circuit_chunk: stim.Circuit,
        num_qubits: int,
        measurement_qubit_indices: List[int],
    ) -> Optional[np.ndarray]:
        """Return input-reset stabilizers for the terminal readout targets."""
        reset_basis_by_qubit = {}
        reset_bases = {
            "R": "Z",
            "RZ": "Z",
            "RX": "X",
            "RY": "Y",
        }
        measured_qubits = set(measurement_qubit_indices)
        instructions = list(circuit_chunk)
        measurement_names = {"M", "MZ", "MR", "MRZ", "MX", "MRX"}
        measurement_positions = [
            k
            for k, instruction in enumerate(instructions)
            if (
                isinstance(instruction, stim.CircuitInstruction)
                and instruction.name in measurement_names
            )
        ]
        if not measurement_positions:
            return None
        terminal_measurement_position = min(measurement_positions)
        selected_instructions = instructions[:terminal_measurement_position]

        for instruction in selected_instructions:
            if not isinstance(instruction, stim.CircuitInstruction):
                continue
            basis = reset_bases.get(instruction.name)
            if basis is None:
                continue
            for target in instruction.targets_copy():
                if target.is_qubit_target and target.value in measured_qubits:
                    reset_basis_by_qubit[target.value] = basis

        missing = sorted(measured_qubits - reset_basis_by_qubit.keys())
        if len(missing) == len(measured_qubits):
            return None
        if missing:
            raise ValueError(
                "A measurement block must reset either every terminal target before "
                "the readout or none of them; partial reset coverage is not yet supported. "
                f"Missing resets for qubits {missing}."
            )

        reset_paulis = np.zeros(
            (len(measurement_qubit_indices), 2 * num_qubits),
            dtype=np.uint8,
        )
        for row_idx, qubit in enumerate(measurement_qubit_indices):
            basis = reset_basis_by_qubit[qubit]
            if basis in ("X", "Y"):
                reset_paulis[row_idx, qubit] = 1
            if basis in ("Z", "Y"):
                reset_paulis[row_idx, num_qubits + qubit] = 1

        return reset_paulis

    @staticmethod
    def _get_back_propagated_pauli(
        circuit_chunk: stim.Circuit,
        num_qubits: int,
        *,
        include_measurement_bases: bool = False,
    ) -> Union[
        Tuple[np.ndarray, List[int]],
        Tuple[np.ndarray, List[int], List[str]],
    ]:
        """
        Back-propagate the terminal syndrome-readout layer through the circuit.

        The readout layer may contain several contiguous measurement
        instructions in X/Z bases. Rows are returned in Stim measurement
        record order, so a terminal ``M z0 z1; MX x0 x1`` produces the Z rows
        followed by the X rows.

        Terminal measurement targets, QEC-patch syndrome roles, and qubits
        explicitly reset by this block are distinct sets. Only Pauli factors
        fixed by an explicit reset in this physical block are removed from the
        returned input observables.

        The default two-item return preserves the existing helper contract.
        Builder internals request the optional measurement-basis list when
        constructing a physical measurement-block analysis.
        """
        measurement_basis_by_gate = {
            "M": "Z",
            "MZ": "Z",
            "MR": "Z",
            "MRZ": "Z",
            "MX": "X",
            "MRX": "X",
        }
        terminal_measurements = []
        for instruction in reversed(list(circuit_chunk)):
            if not isinstance(instruction, stim.CircuitInstruction):
                break
            basis = measurement_basis_by_gate.get(instruction.name)
            if basis is None:
                break
            terminal_measurements.append((instruction, basis))

        if not terminal_measurements:
            last_name = circuit_chunk[-1].name if len(circuit_chunk) else "<empty circuit>"
            raise ValueError(
                "The circuit chunk must end with an X/Z syndrome-qubit "
                f"measurement; got {last_name}."
            )
        terminal_measurements.reverse()

        # Step 1. Get the back-propagated Pauli string
        # Convert to tableau and get numpy representation (binary array in symplectic representation)
        # We ignore measurement/reset to treat the circuit as a unitary operation for analysis.
        se_tableau = stim.Tableau.from_circuit(circuit_chunk, ignore_noise=True, ignore_measurement=True, ignore_reset=True)
        se_tableau_inverse = se_tableau.inverse()

        # 6 outputs: x2x, x2z, z2x, z2z, x_signs, z_signs
        x2x, x2z, z2x, z2z, _, _ = se_tableau_inverse.to_numpy()

        # Convert to int
        x2x_int = x2x.astype(int)
        x2z_int = x2z.astype(int)
        z2x_int = z2x.astype(int)
        z2z_int = z2z.astype(int)

        measurement_qubit_indices = []
        measurement_bases = []
        back_pauli_x_rows = []
        back_pauli_z_rows = []
        for instruction, basis in terminal_measurements:
            for target in instruction.targets_copy():
                if not target.is_qubit_target:
                    continue
                qubit = target.value
                measurement_qubit_indices.append(qubit)
                measurement_bases.append(basis)

                if basis == "X":
                    row_x = x2x_int[qubit].copy()
                    row_z = x2z_int[qubit].copy()
                else:
                    row_x = z2x_int[qubit].copy()
                    row_z = z2z_int[qubit].copy()

                back_pauli_x_rows.append(row_x)
                back_pauli_z_rows.append(row_z)

        back_pauli_x = np.asarray(back_pauli_x_rows, dtype=np.uint8)
        back_pauli_z = np.asarray(back_pauli_z_rows, dtype=np.uint8)

        reset_bits = {
            "R": (0, 1),
            "RZ": (0, 1),
            "RX": (1, 0),
            "RY": (1, 1),
        }
        reset_by_qubit = {}
        for instruction in circuit_chunk:
            if not isinstance(instruction, stim.CircuitInstruction):
                continue
            if instruction.name in measurement_basis_by_gate:
                break
            bits = reset_bits.get(instruction.name)
            if bits is None:
                continue
            for target in instruction.targets_copy():
                if target.is_qubit_target:
                    reset_by_qubit[target.value] = bits

        # A factor can be projected out only when this block physically resets
        # that qubit into the same Pauli eigenbasis. Merely being measured at the
        # terminal boundary, or carrying a syndrome role in QECPatch, is not
        # sufficient.
        removable_reset_qubits = sorted(reset_by_qubit)
        if removable_reset_qubits:

            for row_idx in range(back_pauli_x.shape[0]):
                for qubit in removable_reset_qubits:
                    actual = (
                        int(back_pauli_x[row_idx, qubit]),
                        int(back_pauli_z[row_idx, qubit]),
                    )
                    if actual == (0, 0):
                        continue
                    expected = reset_by_qubit.get(qubit)
                    if actual != expected:
                        raise ValueError(
                            "Back-propagated syndrome factor "
                            f"{actual} on qubit {qubit} is not stabilized by "
                            f"its reset basis {expected}. The measurement is "
                            "not a deterministic syndrome or flag outcome."
                        )

            back_pauli_x[:, removable_reset_qubits] = 0
            back_pauli_z[:, removable_reset_qubits] = 0

        # Padding the back-propagated Pauli string to the full size of the system
        current_size = back_pauli_x.shape[1]

        if current_size < num_qubits:
            pad_width = num_qubits - current_size
            # Pad indices: ((top, bottom), (left, right))
            # We only pad columns on the right with 0s (Identity)
            back_pauli_x = np.pad(back_pauli_x, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            back_pauli_z = np.pad(back_pauli_z, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        elif current_size > num_qubits:
            raise ValueError(f"Circuit chunk has qubit index {current_size-1} which exceeds system size {num_qubits}.")

        # stack x_part and z_part to get the full 2n-bitstring
        back_pauli = np.hstack([back_pauli_x, back_pauli_z])

        if include_measurement_bases:
            return (
                back_pauli,
                measurement_qubit_indices,
                measurement_bases,
            )
        return back_pauli, measurement_qubit_indices


    def apply_unitary_block(self, unitary_block: stim.Circuit, noiseless: bool = False):
        """
        Applies a unitary circuit block and updates the tracker's tableau.

        This method is used for logical operations (e.g., transversal CNOT) that
        need to update the stabilizer tableau to reflect the unitary transformation.

        Args:
            unitary_block: A Stim circuit containing only unitary operations (no measurements/resets).
            noiseless: If True, tag all gate instructions with 'noiseless' so that
                       noise injection rules skip them.
        """
        # Append the unitary block to the circuit
        if self.circuit[-1].name != "TICK":
            self.circuit.append("TICK")

        if noiseless:
            # Re-emit each instruction with the noiseless tag
            for inst in unitary_block:
                if isinstance(inst, stim.CircuitInstruction):
                    self.circuit.append(
                        inst.name, inst.targets_copy(), inst.gate_args_copy(),
                        tag="noiseless",
                    )
        else:
            self.circuit += unitary_block

        # Update the tracker's tableau to reflect the unitary transformation
        self.tracker.process_unitary_block(unitary_block)

    # --------------------------------------------------------------------------
    # D. Logical Coupler Activity, Stabilizer Masking/Unmasking
    # --------------------------------------------------------------------------
    def activate_coupler(self, name: str):
        """
        Turn on the logical coupler. A wrapper for QECSystem.activate_coupler.
        This changes the active stabilizer set for the NEXT round of extraction.
        """
        # Call the system's state manager
        self.system.activate_coupler(name)

    def deactivate_coupler(self, name: str):
        """
        Turn off the logical coupler and restore original patch boundaries.
        A wrapper for QECSystem.deactivate_coupler.
        """
        self.system.deactivate_coupler(name)

    def mask_stabilizers(self, ids: Set[int]):
        """
        Mask (Deactivate) the stabilizers with the given ids.
        To be implemented.
        """
        pass

    def unmask_stabilizers(self, ids: Set[int]):
        """
        Unmask (Activate) the stabilizers with the given ids.
        To be implemented.
        """
        pass


    # --------------------------------------------------------------------------
    # E. Data Qubit Measurement
    # --------------------------------------------------------------------------
    def apply_mid_data_readout(
        self,
        measurements: Dict[int, str],
        *,
        noiseless: bool = False,
    ) -> None:
        """Measure retained data qubits without closing the experiment.

        Unlike :meth:`apply_data_readout`, this method updates the tracker and
        leaves all surviving state available for later syndrome-extraction
        rounds. The current measurement engine supports X- and Z-basis
        retained-data readout. Measured qubits leave the active lifetime, but
        stabilizer support is unchanged until the system geometry is updated.

        Args:
            measurements: Data-qubit index to measurement basis (``X`` or
                ``Z``).
            noiseless: Tag the physical measurements so noise injection skips
                them.
        """
        if not measurements:
            raise ValueError("measurements must not be empty.")

        normalized = {
            qubit: str(basis).upper()
            for qubit, basis in measurements.items()
        }
        bad_bases = sorted(set(normalized.values()) - {"X", "Z"})
        if bad_bases:
            raise ValueError(
                "Mid-circuit data readout supports only X and Z bases; got "
                f"{bad_bases}."
            )
        non_data = set(normalized) - set(self.system.data_indices)
        if non_data:
            raise ValueError(
                "Mid-circuit data readout requires registered data qubits; got "
                f"{sorted(non_data)}."
            )
        inactive = set(normalized) - set(self.system.active_qubit_indices)
        if inactive:
            raise ValueError(
                "Mid-circuit data readout requires active qubits; got inactive "
                f"indices {sorted(inactive)}."
            )

        if len(self.circuit) > 0 and self.circuit[-1].name != "TICK":
            self.circuit.append("TICK")

        block = stim.Circuit()
        tag = "noiseless" if noiseless else ""
        x_qubits = sorted(
            qubit for qubit, basis in normalized.items() if basis == "X"
        )
        z_qubits = sorted(
            qubit for qubit, basis in normalized.items() if basis == "Z"
        )
        if x_qubits:
            block.append("MX", x_qubits, tag=tag)
        if z_qubits:
            block.append("M", z_qubits, tag=tag)

        if self.if_detector:
            analysis = self._analyze_measurement_block(block, z_only=False)
            if analysis.discarded_measurement_qubit_indices:
                raise ValueError(
                    "Mid-circuit data readout cannot discard syndrome ancillas."
                )
            self._process_measurement_blocks(
                output_circuit=self.circuit,
                analyses=(analysis,),
                shift_round=True,
            )
        else:
            self.circuit += block
            self.circuit.append("SHIFT_COORDS", [], [0, 0, 1])

        self.system.active_qubit_indices.difference_update(normalized)

    def apply_data_readout(self, final_measurements: Dict[int, str] = None, noiseless: bool = False,
                           z_only: bool = False, resolve_absorbed: bool = True):
        """
        Applies destructive measurement on data qubits and calls Tracker to
        resolve remaining stabilizers into Detectors/Observables.

        Args:
            final_measurements: Dict mapping qubit index to measurement basis ('X', 'Y', 'Z').
            noiseless: If True, tag measurement instructions with 'noiseless' so that
                       noise injection rules skip them.
            z_only: If True, bypass tracker.process_data_measurement and construct
                DETECTOR / OBSERVABLE_INCLUDE manually from Z-stabilizers and Z-logicals.
                Requires apply_syndrome_extraction to have been called with z_only=True.
        """
        if final_measurements is None:
            final_measurements = {q: 'Z' for q in self.system.data_indices}

        # Final destructive data readout must start in a fresh moment. Some
        # extraction blocks, such as middle-out color-code circuits, end with
        # data-qubit gauge measurements instead of a trailing TICK.
        if len(self.circuit) > 0 and self.circuit[-1].name != "TICK":
            self.circuit.append("TICK")

        xs = [q for q, b in final_measurements.items() if b == 'X']
        ys = [q for q, b in final_measurements.items() if b == 'Y']
        zs = [q for q, b in final_measurements.items() if b == 'Z']

        tag = "noiseless" if noiseless else ""

        # Append gates (No manual noise here)
        if xs: self.circuit.append("MX", xs, tag=tag)
        if ys: self.circuit.append("MY", ys, tag=tag)
        if zs: self.circuit.append("M", zs, tag=tag)

        # Prepare Basis for Tracker (order matches circuit: X then Y then Z)
        sorted_indices = xs + ys + zs
        n = self.tracker.num_qubits
        final_paulis = np.zeros((len(sorted_indices), 2 * n), dtype=np.uint8)

        for i, q in enumerate(sorted_indices):
            basis = final_measurements[q]
            if basis == 'X':
                final_paulis[i, q] = 1
            elif basis == 'Y':
                final_paulis[i, q] = 1
                final_paulis[i, n + q] = 1
            else:
                final_paulis[i, n + q] = 1

        # Call Tracker — or, for z_only, build DETECTORs/OBSERVABLEs manually
        if self.if_detector:
            if z_only:
                syn_qubit_indices = self._z_only_syn_qubit_indices
                no_detector_mask  = self._z_only_no_detector_mask
                n_meas_per_round  = self._z_only_n_meas_per_round
                x_anc_set = {q for q, masked in zip(syn_qubit_indices, no_detector_mask) if masked}

                data_indices = sorted_indices  # already M'd above in Z basis
                data_pos     = {q: i for i, q in enumerate(data_indices)}
                n_data       = len(data_indices)
                z_anc_pos    = {q: i for i, q in enumerate(syn_qubit_indices) if q not in x_anc_set}

                for stab in self.system.active_stabilizers_z:
                    recs = [stim.target_rec(-n_data + data_pos[q]) for q in stab['data_indices']]
                    recs.append(stim.target_rec(-n_data - n_meas_per_round + z_anc_pos[stab['syn_idx']]))
                    self.circuit.append("DETECTOR", recs)

                # Delegate observable generation to the tracker so it handles
                # periodic-boundary codes (e.g. toric) correctly. We snapshot
                # the circuit length, let the tracker append both DETECTORs and
                # OBSERVABLE_INCLUDEs, then keep only the OBSERVABLE_INCLUDEs.
                n_before = len(self.circuit)
                self.tracker.process_data_measurement(
                    circuit=self.circuit,
                    final_paulis=final_paulis,
                    idx_to_coord_map=self.system.qubit_coords,
                    syndrome_qubit_indices=self.system.syndrome_indices,
                    resolve_absorbed=resolve_absorbed,
                )
                obs_insts = [
                    inst for inst in list(self.circuit)[n_before:]
                    if not isinstance(inst, stim.CircuitRepeatBlock)
                    and inst.name == "OBSERVABLE_INCLUDE"
                ]
                self.circuit = self.circuit[:n_before]
                for inst in obs_insts:
                    self.circuit.append(inst.name, inst.targets_copy(), inst.gate_args_copy())
            else:
                self.tracker.process_data_measurement(
                    circuit=self.circuit,
                    final_paulis=final_paulis,
                    idx_to_coord_map=self.system.qubit_coords,
                    syndrome_qubit_indices=self.system.syndrome_indices,
                    resolve_absorbed=resolve_absorbed,
                )

        # Remove measured qubits from active set (ready for reuse)
        self.system.active_qubit_indices.difference_update(final_measurements.keys())

    # --------------------------------------------------------------------------
    # F. Final Build & Noise Injection
    # --------------------------------------------------------------------------
    def build_noisy_circuit(
        self,
        noise_params: NoiseConfig,
        noise_model: str = 'circuit_level'
    ) -> stim.Circuit:
        """
        Consumes the clean circuit and applies noise using the specified model strategy.

        Args:
            noise_params: Noise parameters (NoiseConfig).
            noise_model: The name of the factory method in NoiseInjector to use.
                         e.g., 'circuit_level' -> calls NoiseInjector.from_circuit_level(...)
                         e.g., 'circuit_level_with_idling' -> adds idle noise every moment
                         e.g., 'custom_test'   -> calls NoiseInjector.from_custom_test(...)
        """
        # 1. Construct the expected factory method name
        method_name = f"from_{noise_model}"

        # 2. Dynamically retrieve the method from the NoiseInjector class
        if not hasattr(NoiseInjector, method_name):
            # Fallback or nice error message showing available options
            valid_methods = [m.replace("from_", "") for m in dir(NoiseInjector) if m.startswith("from_")]
            raise ValueError(f"Unknown noise model '{noise_model}'. "
                             f"Expected one of: {valid_methods}")

        factory_method = getattr(NoiseInjector, method_name)

        # 3. Inject noise
        # Most models target data qubits for idle noise. The per-moment model
        # instead needs the complete physical-qubit universe so that it can
        # take the complement of each moment's operated qubits.
        data_indices = [self.system.index_map[coord] for coord in self.system.data_coords]
        target_indices = (
            list(range(self.circuit.num_qubits))
            if noise_model == "circuit_level_with_idling"
            else data_indices
        )
        injector = factory_method(noise_params, target_indices)
        noisy_circuit = injector.inject_noise(self.circuit)

        return noisy_circuit

    # --------------------------------------------------------------------------
    # G. Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _get_initialization_tableau(qubit_indices_x: List[int], qubit_indices_z: List[int], qubit_indices_y: List[int], n: int):
        """
        Generates the tableau for the given qubit indices in X, Z, Y basis.
        Args:
            qubit_indices_x: List[int]
            qubit_indices_z: List[int]
            qubit_indices_y: List[int]
            n: int
        Returns:
            initialized_tableau: np.ndarray
        """
        # 1. X Basis: (X=1, Z=0)
        # Shape is (len, 2n). If len=0, it's safe.
        t_x = np.zeros((len(qubit_indices_x), 2 * n), dtype=int)
        if qubit_indices_x:
            t_x[np.arange(len(qubit_indices_x)), qubit_indices_x] = 1

        # 2. Z Basis: (X=0, Z=1)
        t_z = np.zeros((len(qubit_indices_z), 2 * n), dtype=int)
        if qubit_indices_z:
            # Use list comprehension or numpy add to shift index by n
            cols = [i + n for i in qubit_indices_z]
            t_z[np.arange(len(qubit_indices_z)), cols] = 1

        # 3. Y Basis: (X=1, Z=1) -> Critical Fix here
        t_y = np.zeros((len(qubit_indices_y), 2 * n), dtype=int)
        if qubit_indices_y:
            # Set X part
            t_y[np.arange(len(qubit_indices_y)), qubit_indices_y] = 1
            # Set Z part (same row!)
            cols_z = [i + n for i in qubit_indices_y]
            t_y[np.arange(len(qubit_indices_y)), cols_z] = 1

        # 4. Stack
        # Since all sub-matrices have 2*n columns, vstack works even if some have 0 rows.
        return np.vstack([t_x, t_z, t_y])


    def to_stim_circuit(self) -> stim.Circuit:
        return self.circuit
