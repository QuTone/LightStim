"""Destructive readout for the bare/dressed S, L, L_absorbed tracker.

Active checks supply a role space, never eigenvalues. Observable extraction
uses the complete surviving S/L intersection with the measured Pauli span;
the redundant absorbed tableau participates only in projection.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import stim

from .tableau import PauliTableau
from .tracker_algebra import Rows, readout_closures, record_dependencies
from ..utils.linear_algebra import (
    check_commutativity, kernel_gf2, solve_linear_decomposition,
)
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
        selected_matrix = self.stabilizers.matrix[selected]
        selected_records = [self.stabilizers.records[index] for index in selected]
        if len(selected_matrix):
            commutators = check_commutativity(selected_matrix, measured)
            ps_combinations = kernel_gf2(commutators.T)
            postselection = Rows(ps_combinations @ selected_matrix % 2,
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
        full.matrix = np.vstack([self.stabilizers.matrix[order],
                                 self.logicals.matrix, self.absorbed_ops.matrix]).copy()
        full.records = ([list(self.stabilizers.records[index]) for index in order]
                        + [list(row) for row in self.logicals.records]
                        + [list(row) for row in self.absorbed_ops.records])
        destroyed = set()
        for row, record in zip(measured, absolute):
            anti = [int(index) for index in np.flatnonzero(
                check_commutativity(row[None, :], full.matrix)[0])
                if int(index) not in destroyed]
            if not anti:
                continue
            pivot = anti[0]
            if pivot >= physical_end:
                raise RuntimeError("L_absorbed supplied the only anticommuting readout pivot")
            for other in anti[1:]:
                unknown = any(record < 0 for record in full.records[other] + full.records[pivot])
                full.update_row(other, pivot)
                if unknown and not any(record < 0 for record in full.records[other]):
                    full.records[other].append(-1)
            full.replace_row(pivot, row, [record])
            destroyed.add(pivot)

        def known_part(start, stop):
            indices = [index for index in range(start, stop)
                       if index not in destroyed
                       and all(record >= 0 for record in full.records[index])]
            return Rows(full.matrix[indices], [full.records[index] for index in indices])

        bare = known_part(0, n_bare)
        dressed = known_part(n_bare, n_stabilizers)
        logical = known_part(n_stabilizers, physical_end)
        unknown_logicals = [index for index in range(n_stabilizers, physical_end)
                            if index not in destroyed
                            and any(record < 0 for record in full.records[index])]
        if unknown_logicals:
            # Geometry can show that a logical readout closes even though its
            # eigenvalue was never banked. Do not silently omit that requested
            # logical output. Compare full intersections so products of rows,
            # not just individually readable unknown L rows, are checked.
            known_state = np.vstack([bare.matrix, dressed.matrix, logical.matrix])
            known_closed, _ = intersect_row_spaces(known_state, measured)
            potential_closed, _ = intersect_row_spaces(
                np.vstack([known_state, full.matrix[unknown_logicals]]), measured)
            if reduce_modulo(potential_closed, known_closed).any():
                raise RuntimeError(
                    "process_data_measurement: a logical observable closes with "
                    "UNMEASURED sentinel records; its parity was never banked "
                    "and cannot be emitted.")
        # Unknown preparation placeholders are never allowed to manufacture
        # an eigenvalue by canceling two occurrences of the -1 sentinel.
        closure = readout_closures(bare, dressed, logical, measurements,
                                   include_dependencies=True)
        ps_closure = readout_closures(postselection, empty, empty, measurements,
                                      include_dependencies=True)

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
        bare_dependencies = record_dependencies(bare)
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
        stab_matrix, stab_records = fold(full.matrix[stab_indices],
                                         [full.records[index] for index in stab_indices])
        log_matrix, log_records = fold(full.matrix[log_indices],
                                       [full.records[index] for index in log_indices])
        ps_matrix, ps_records = fold(postselection.matrix, postselection.records)
        ps_survivors = [index for index, row in enumerate(ps_matrix) if row.any()]
        ps_matrix = ps_matrix[ps_survivors]
        ps_records = [ps_records[index] for index in ps_survivors]

        # Preserve exact surviving PS constraints before selecting an ordinary
        # S complement, so downstream selectors retain the same signed space.
        # Known rows precede unknown placeholders when choosing representatives.
        stab_order = ([i for i, records in enumerate(stab_records) if all(r >= 0 for r in records)]
                      + [i for i, records in enumerate(stab_records) if any(r < 0 for r in records)])
        candidates = np.vstack([ps_matrix, stab_matrix[stab_order]])
        additional = [index - len(ps_matrix) for index in independent_row_indices(candidates)
                      if index >= len(ps_matrix)]
        chosen_stabs = [stab_order[index] for index in additional]
        self.stabilizers.matrix = np.vstack([ps_matrix, stab_matrix[chosen_stabs]])
        self.stabilizers.records = ps_records + [stab_records[index] for index in chosen_stabs]
        self.post_select_row_indices = set(range(len(ps_matrix)))
        chosen_logicals = independent_row_indices(log_matrix)
        self.logicals.matrix = log_matrix[chosen_logicals]
        self.logicals.records = [log_records[index] for index in chosen_logicals]
        self._classified_stabilizer_rows = set(range(self.stabilizers.count))
        self.stabilizer_with_logical_components = set()
        self._gauge_logical_vectors = []
        active = (np.zeros((0, width), dtype=np.uint8) if active_stabilizers is None
                  else np.asarray(active_stabilizers, dtype=np.uint8))
        # Classification also rebuilds the current absorbed quotient and
        # synchronizes the rank exposed through the compatibility count API.
        self._v2_classify_stabilizers(reduce_modulo(active, measured))
