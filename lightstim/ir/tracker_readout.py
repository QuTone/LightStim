"""Destructive readout for the bare/dressed S, L, L_absorbed tracker.

Active checks supply a role space, never eigenvalues. Observable extraction
uses the complete surviving S/L intersection with the measured Pauli span;
the redundant absorbed tableau participates only in projection.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import stim

from .tableau import PauliTableau
from .tracker_algebra import (
    Rows, ClosureRows, ReadoutClosures, RecordDependencies,
    readout_closures, record_dependencies,
)
from ..utils.linear_algebra import (
    check_commutativity, kernel_gf2, solve_linear_decomposition,
)
from ..utils.pauli_backend import multiply_binary
from ..utils.subsystem_algebra import (
    combine_records, independent_row_indices, intersect_row_spaces, reduce_modulo,
)


def _parity_mask(records):
    result = 0
    for record in records:
        result ^= 1 << int(record)
    return result


def _insert_parity(basis, value):
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            basis[pivot] = value
            return True
        value ^= basis[pivot]
    return False


def _contains_parity(basis, value):
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            return False
        value ^= basis[pivot]
    return True


def _coordinate_readout_closures(state, known_banks, unknown_logicals, measurements, postselection):
    """Close current native source coordinates without exporting physical state.

    Every closed physical row is reconstructed from its measurement lineage;
    this also preserves the exact row needed by detector coordinate selection.
    Full X/Z role coordinates are allowed. Source-lineage dependency ordering
    is invariant under the inverse tableau's invertible coordinate transform.
    """
    from ..utils.cpp._pauli_backend_cpp import (
        PackedRows, PackedBasis, packed_readout_closures, packed_dependencies,
        packed_multiply, packed_combine_records,
    )

    banks = [state.coordinate_rows(indices) for indices in known_banks]
    sources = banks[0].combined(banks[1]).combined(banks[2])
    indices = [index for bank in known_banks for index in bank]
    source_records = state.records_selected(indices)
    measurement_coordinates = state.coordinate_paulis(measurements.matrix)
    physical_measurements = PackedRows(measurements.matrix)
    empty = PackedRows(np.zeros((0, measurements.matrix.shape[1]), dtype=np.uint8))
    native = packed_readout_closures(*banks, measurement_coordinates, [])
    if unknown_logicals:
        known_closed = native["detectors"]["rows"].combined(native["observables"]["rows"])
        potential = packed_readout_closures(
            sources.combined(state.coordinate_rows(unknown_logicals)), empty, empty,
            measurement_coordinates, [])["detectors"]["rows"]
        if PackedBasis(known_closed).residues(potential).numpy().any():
            raise RuntimeError(
                "process_data_measurement: a logical observable closes with "
                "UNMEASURED sentinel records; its parity was never banked "
                "and cannot be emitted.")

    def dependencies(rows, records):
        lineage = packed_dependencies(rows)
        return RecordDependencies(lineage.numpy(), tuple(
            tuple(parity) for parity in packed_combine_records(lineage, records)))

    def translate(result, packed_sources, records):
        def closed(name):
            value = result[name]
            physical = packed_multiply(value["measurement_lineage"], physical_measurements)
            history = packed_combine_records(value["state_lineage"], records)
            measured = packed_combine_records(value["measurement_lineage"], measurements.records)
            return ClosureRows(
                Rows(physical.numpy(), history), value["state_lineage"].numpy(),
                value["measurement_lineage"].numpy(),
                tuple(tuple(sorted(set(old) ^ set(new))) for old, new in zip(history, measured)))
        return ReadoutClosures(closed("detectors"), closed("observables"),
                               closed("postselection"), dependencies(packed_sources, records))

    closure = translate(native, sources, source_records)
    selected_coordinates = state.coordinate_paulis(postselection.matrix)
    ps_native = packed_readout_closures(
        selected_coordinates, empty, empty, measurement_coordinates, [])
    ps_closure = translate(ps_native, selected_coordinates, postselection.records)
    bare_dependencies = dependencies(banks[0], state.records_selected(known_banks[0]))
    return closure, ps_closure, bare_dependencies


class V2ReadoutMixin:
    def _process_v2_data_measurement(
        self,
        circuit: stim.Circuit,
        final_paulis: np.ndarray,
        idx_to_coord_map: Dict[int, Tuple[float, float]],
        syndrome_qubit_indices: set = None,
        resolve_absorbed: bool = True,
        active_stabilizers: Optional[np.ndarray] = None,
        measurement_record_indices: Optional[Sequence[int]] = None,
    ):
        """Project once, close S/L relations, fold support, and reclassify S.

        Explicit record indices allow caller-declared tracking stages inside
        one already emitted physical measurement layer. In that case the
        caller has advanced ``total_measurements`` to the whole layer's end;
        this method neither increments it nor changes physical gate order.
        ``resolve_absorbed`` remains an API compatibility argument: v2 always
        rebuilds absorbed rows from the current dressed quotient.

        Measurement rows must be independent and commuting, as for Builder's
        single-qubit readout on distinct qubits. Repeated/dependent checks use
        the mid-measurement interface, which tracks their record relations.
        """
        del resolve_absorbed
        width = 2 * self.num_qubits
        measured = np.asarray(final_paulis, dtype=np.uint8)
        if measured.ndim != 2 or measured.shape[1] != width:
            raise ValueError("Readout Pauli rows must use the tracker's register")
        if check_commutativity(measured, measured).any():
            raise ValueError("One readout collection must contain commuting measurements")
        if len(independent_row_indices(measured)) != len(measured):
            raise ValueError("Readout requires independent measurement rows; "
                             "use mid-measurement for repeated or dependent checks")
        if measurement_record_indices is None:
            absolute = list(range(self.total_measurements,
                                  self.total_measurements + len(measured)))
            self.total_measurements += len(measured)
        else:
            absolute = [int(index) for index in measurement_record_indices]
            if len(absolute) != len(measured) or len(set(absolute)) != len(absolute):
                raise ValueError("Readout requires one distinct absolute record per measurement")
            if any(index < 0 or index >= self.total_measurements for index in absolute):
                raise ValueError("Explicit readout records must belong to the already emitted batch")
        measurements = Rows(measured, [[index] for index in absolute])
        empty = Rows.empty(width)

        # Explicit postselection is a physical constraint subspace. Preserve
        # that subspace across projection, not an OR-tag on multiplied rows.
        selected = sorted(self.post_select_row_indices)
        if any(index < 0 or index >= self.stabilizers.count for index in selected):
            raise IndexError("Postselected stabilizer row lies outside S")
        selected = [index for index in selected
                    if all(record >= 0 for record in self.stabilizers.records[index])]
        record_state = self._sync_record_state()
        if record_state is not None:
            # Reference readout projects a temporary full table. An unknown
            # logical closure can still reject the readout after projection;
            # the caller's S/L banks must retain their pre-call state then.
            # Measurement counters and any already emitted physical circuit
            # retain their existing reference error behavior.
            record_state = record_state.copy()
        selected_matrix = (record_state.rows_selected(selected) if record_state is not None
                           else self.stabilizers.matrix[selected])
        selected_records = [self.stabilizers.records[index] for index in selected]
        if len(selected_matrix):
            commutators = check_commutativity(selected_matrix, measured)
            ps_combinations = kernel_gf2(commutators.T)
            postselection = Rows(multiply_binary(ps_combinations, selected_matrix),
                                 combine_records(ps_combinations, selected_records))
        else:
            postselection = empty

        # B. The full physical state is S_bare, S_dressed, L. Absorbed rows
        # are redundant; their first anticommuting pivot would be an invariant
        # violation. Keep every tableau in the same projection update.
        dressed_indices = set(self._dressed_stabilizer_rows)
        bare_indices = [index for index in range(self.stabilizers.count)
                        if index not in dressed_indices]
        ordered_dressed = [index for index in range(self.stabilizers.count)
                           if index in dressed_indices]
        order = bare_indices + ordered_dressed
        n_bare = len(bare_indices)
        n_stabilizers = len(order)
        n_logicals = self.logicals.count
        physical_end = n_stabilizers + n_logicals
        full = PauliTableau(self.num_qubits)
        if record_state is not None:
            native_order = order + list(range(n_stabilizers,
                n_stabilizers + n_logicals + self.absorbed_ops.count))
            full.records = record_state.records_selected(native_order)
            record_state.adopt_coordinates(record_state.coordinate_rows(native_order), full.records)
        else:
            full.matrix = np.vstack([self.stabilizers.matrix[order],
                                     self.logicals.matrix, self.absorbed_ops.matrix]).copy()
            full.records = ([list(self.stabilizers.records[index]) for index in order]
                            + [list(row) for row in self.logicals.records]
                            + [list(row) for row in self.absorbed_ops.records])
        destroyed = set()
        for row, record in zip(measured, absolute):
            anti = (record_state.anticommuting(row) if record_state is not None else
                    np.flatnonzero(check_commutativity(row[None, :], full.matrix)[0]))
            anti = [int(index) for index in anti if int(index) not in destroyed]
            if not anti:
                continue
            pivot = anti[0]
            if pivot >= physical_end:
                raise RuntimeError("L_absorbed supplied the only anticommuting readout pivot")
            if record_state is not None:
                record_state.project(row, pivot, record, sorted(destroyed))
                destroyed.add(pivot)
                continue
            for other in anti[1:]:
                unknown = any(record < 0 for record in full.records[other] + full.records[pivot])
                full.update_row(other, pivot)
                if unknown and not any(record < 0 for record in full.records[other]):
                    full.records[other].append(-1)
            full.replace_row(pivot, row, [record])
            destroyed.add(pivot)

        if record_state is not None:
            full.records = record_state.records()

        def known_indices(start, stop):
            return [index for index in range(start, stop)
                    if index not in destroyed
                    and all(record >= 0 for record in full.records[index])]

        known_banks = [known_indices(0, n_bare), known_indices(n_bare, n_stabilizers),
                       known_indices(n_stabilizers, physical_end)]
        unknown_logicals = [index for index in range(n_stabilizers, physical_end)
                            if index not in destroyed
                            and any(record < 0 for record in full.records[index])]
        if record_state is not None:
            closure, ps_closure, bare_dependencies = _coordinate_readout_closures(
                record_state, known_banks, unknown_logicals, measurements, postselection)
        else:
            bare, dressed, logical = [Rows(full.matrix[indices],
                [full.records[index] for index in indices]) for indices in known_banks]
            if unknown_logicals:
                # Unknown L cannot supply a readout value, including a product
                # that becomes measurable only together with known rows.
                known_state = np.vstack([bare.matrix, dressed.matrix, logical.matrix])
                known_closed, _ = intersect_row_spaces(known_state, measured)
                potential_closed, _ = intersect_row_spaces(
                    np.vstack([known_state, full.matrix[unknown_logicals]]), measured)
                if reduce_modulo(potential_closed, known_closed).any():
                    raise RuntimeError(
                        "process_data_measurement: a logical observable closes with "
                        "UNMEASURED sentinel records; its parity was never banked "
                        "and cannot be emitted.")
            closure = readout_closures(bare, dressed, logical, measurements,
                                      include_dependencies=True)
            ps_closure = readout_closures(postselection, empty, empty, measurements,
                                         include_dependencies=True)
            bare_dependencies = record_dependencies(bare)

        # Coordinate selection follows the native final-readout convention:
        # historical syndrome position first, then support on measured data.
        used_coords = set()

        def coordinate(row, history):
            data_coord = None
            syndrome_coord = None
            for record in history:
                qubit = self.meas_rec_to_idx_map.get(record)
                if qubit is None or qubit not in idx_to_coord_map:
                    continue
                if syndrome_qubit_indices and qubit in syndrome_qubit_indices:
                    syndrome_coord = tuple(idx_to_coord_map[qubit])
                elif row[qubit] or row[self.num_qubits + qubit]:
                    data_coord = tuple(idx_to_coord_map[qubit])
            if data_coord is None:
                support = np.flatnonzero(row[:self.num_qubits] | row[self.num_qubits:])
                data_coord = next((tuple(idx_to_coord_map[int(qubit)]) for qubit in support
                                   if int(qubit) in idx_to_coord_map),
                                  tuple(next(iter(idx_to_coord_map.values()), (0, 0))))
            if syndrome_coord is not None and syndrome_coord not in used_coords:
                return syndrome_coord
            return data_coord

        # A selected closure can also be a logical observable. Its acceptance
        # parity must still be emitted as a tagged detector, without changing
        # the protected role or duplicating an already selected detector.
        ps_basis = {}
        for parity in ps_closure.detectors.parities:
            _insert_parity(ps_basis, _parity_mask(parity))
        for parity in ps_closure.source_dependencies.records:
            _insert_parity(ps_basis, _parity_mask(parity))
        emitted_ps = {}
        emitted_this_call = {}
        previous_relations = None

        def existing_relation_space():
            nonlocal previous_relations
            if previous_relations is None:
                previous_relations = {}
                record_cursor = 0
                observables = {}
                for instruction in circuit.flattened():
                    if instruction.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                        parity = _parity_mask(record_cursor + target.value
                                              for target in instruction.targets_copy()
                                              if target.is_measurement_record_target)
                        if instruction.name == "DETECTOR":
                            _insert_parity(previous_relations, parity)
                        else:
                            index = int(instruction.gate_args_copy()[0])
                            observables[index] = observables.get(index, 0) ^ parity
                    record_cursor += instruction.num_measurements
                for parity in observables.values():
                    _insert_parity(previous_relations, parity)
                for relation in self.logical_history:
                    _insert_parity(previous_relations, _parity_mask(relation.records))
            return previous_relations

        def emit(parity, row, history, *, observable=False, postselected=False,
                 record_only=False):
            from .tracker import _append_detector

            parity = tuple(sorted(parity))
            mask = _parity_mask(parity)
            if not mask:
                return
            if record_only and not postselected and (
                _contains_parity(emitted_this_call, mask)
                or _contains_parity(existing_relation_space(), mask)
            ):
                return
            args = [stim.target_rec(record - self.total_measurements) for record in parity]
            if observable:
                circuit.append("OBSERVABLE_INCLUDE", args, [self.allocate_observable()])
            else:
                coord = coordinate(row, history)
                coords = list(coord) + [1]
                postselected = (postselected or _contains_parity(ps_basis, mask)
                                or tuple(coords) in self.post_select_detector_coords)
                _append_detector(circuit, args, coords, post_select=postselected)
                used_coords.add(coord)
                if postselected:
                    _insert_parity(emitted_ps, mask)
            _insert_parity(emitted_this_call, mask)
            if previous_relations is not None:
                _insert_parity(previous_relations, mask)

        for closures, observable in ((closure.detectors, False), (closure.observables, True)):
            for row, history, parity in zip(closures.rows.matrix, closures.rows.records,
                                             closures.parities):
                emit(parity, row, history, observable=observable, record_only=not row.any())

        zero = np.zeros(width, dtype=np.uint8)
        # A dependent state representation can carry a genuine record-only
        # relation. Close it before basis reduction discards a Pauli-zero row.
        # Relations already emitted or archived retain their original role.
        for parity in bare_dependencies.records:
            emit(parity, zero, parity, record_only=True)
        for parity in closure.source_dependencies.records:
            emit(parity, zero, parity, observable=True, record_only=True)
        for row, history, parity in zip(ps_closure.detectors.rows.matrix,
                                        ps_closure.detectors.rows.records,
                                        ps_closure.detectors.parities):
            if not _contains_parity(emitted_ps, _parity_mask(parity)):
                emit(parity, row, history, postselected=True)
        for parity in ps_closure.source_dependencies.records:
            if not _contains_parity(emitted_ps, _parity_mask(parity)):
                emit(parity, zero, parity, postselected=True)

        # C. Fold the measured Pauli factors into their actual fresh records.
        # This supports arbitrary commuting readout spans as well as ordinary
        # single-qubit destructive measurement.
        def fold(matrix, records):
            folded = reduce_modulo(matrix, measured)
            coefficients, known, _ = solve_linear_decomposition(
                basis=measured, targets=matrix ^ folded, reduce_weight=False)
            if not np.all(known):
                raise RuntimeError("Readout support reduction escaped the measurement span")
            correction = combine_records(coefficients, measurements.records)
            folded_records = [sorted(set(old) ^ set(new))
                              for old, new in zip(records, correction)]
            return folded, folded_records

        stab_indices = [index for index in range(n_stabilizers) if index not in destroyed]
        log_indices = [index for index in range(n_stabilizers, physical_end)
                       if index not in destroyed]
        ps_matrix, ps_records = fold(postselection.matrix, postselection.records)
        ps_survivors = [index for index, row in enumerate(ps_matrix) if row.any()]
        ps_matrix = ps_matrix[ps_survivors]
        ps_records = [ps_records[index] for index in ps_survivors]

        # Preserve exact surviving PS constraints before the ordinary S
        # complement. Native folds retain physical-column canonical residues;
        # only closure and source dependency algebra is coordinate invariant.
        if record_state is not None:
            from ..utils.cpp._pauli_backend_cpp import PackedRows, PackedBasis

            stab_rows, stab_records = record_state.fold_rows(
                stab_indices, measured, measurements.records)
            log_rows, log_records = record_state.fold_rows(
                log_indices, measured, measurements.records)
            stab_order = ([i for i, records in enumerate(stab_records) if all(r >= 0 for r in records)]
                          + [i for i, records in enumerate(stab_records) if any(r < 0 for r in records)])
            ps_rows = PackedRows(ps_matrix)
            candidates = ps_rows.combined(stab_rows.select(stab_order))
            additional = [index - len(ps_rows) for index in PackedBasis(candidates).selected
                          if index >= len(ps_rows)]
            chosen_stabs = [stab_order[index] for index in additional]
            final_stabs = ps_rows.combined(stab_rows.select(chosen_stabs))
            chosen_logicals = PackedBasis(log_rows).selected
            final_logs = log_rows.select(chosen_logicals)
            final_records = (ps_records + [stab_records[index] for index in chosen_stabs]
                             + [log_records[index] for index in chosen_logicals])
            record_state.adopt_physical(final_stabs.combined(final_logs), final_records)
            self._record_state = record_state
            self._read_record_state((len(final_stabs), len(final_logs), 0))
        else:
            stab_matrix, stab_records = fold(full.matrix[stab_indices],
                                             [full.records[index] for index in stab_indices])
            log_matrix, log_records = fold(full.matrix[log_indices],
                                           [full.records[index] for index in log_indices])
            stab_order = ([i for i, records in enumerate(stab_records) if all(r >= 0 for r in records)]
                          + [i for i, records in enumerate(stab_records) if any(r < 0 for r in records)])
            candidates = np.vstack([ps_matrix, stab_matrix[stab_order]])
            additional = [index - len(ps_matrix) for index in independent_row_indices(candidates)
                          if index >= len(ps_matrix)]
            chosen_stabs = [stab_order[index] for index in additional]
            self.stabilizers.matrix = np.vstack([ps_matrix, stab_matrix[chosen_stabs]])
            self.stabilizers.records = ps_records + [stab_records[index] for index in chosen_stabs]
            chosen_logicals = independent_row_indices(log_matrix)
            self.logicals.matrix = log_matrix[chosen_logicals]
            self.logicals.records = [log_records[index] for index in chosen_logicals]
        self.post_select_row_indices = set(range(len(ps_matrix)))
        self._classified_stabilizer_rows = set(range(self.stabilizers.count))
        self.stabilizer_with_logical_components = set()
        self._gauge_logical_vectors = []
        active = (np.zeros((0, width), dtype=np.uint8) if active_stabilizers is None
                  else np.asarray(active_stabilizers, dtype=np.uint8))
        # Classification also rebuilds the current absorbed quotient and
        # synchronizes the rank exposed through the compatibility count API.
        self._v2_classify_stabilizers(reduce_modulo(active, measured))
