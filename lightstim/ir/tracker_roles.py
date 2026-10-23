"""Bare/dressed role algebra for the record-carrying Pauli tracker.

S and L describe the conditional quantum state.  The absorbed tableau is a
redundant representative of the current dressed quotient, never an additional
source of observable relations.  Active checks supply roles, not eigenvalues.
"""

import numpy as np

from ..utils.linear_algebra import (
    check_commutativity, kernel_gf2, solve_linear_decomposition,
)
from ..utils.subsystem_algebra import (
    independent_row_indices, combine_records, reduce_modulo,
)
from ..utils.pauli_backend import propagate_paulis


class BareDressedRoles:
    def _v2_capture_postselection(self):
        """Snapshot actual selected constraints for one atomic operation.

        This is transient algebra, not a second persistent PS tableau. Unknown
        placeholders cannot supply a signed acceptance constraint.
        """
        from .tracker_algebra import Rows

        selected = sorted(self.post_select_row_indices)
        if any(i < 0 or i >= self.stabilizers.count for i in selected):
            raise IndexError("Postselected stabilizer row lies outside S")
        selected = [i for i in selected if all(r >= 0 for r in self.stabilizers.records[i])]
        if not selected:
            return Rows.empty(2 * self.num_qubits)
        return Rows(self.stabilizers.matrix[selected],
                    [self.stabilizers.records[i] for i in selected])

    @staticmethod
    def _v2_postselection_kernel(selected, constraints):
        """Keep the full kernel, with original selected representatives first.

        The kernel is formed in coefficient space so a genuine Pauli-zero,
        record-only selected relation is retained until it can be emitted.
        """
        from .tracker_algebra import Rows

        count = len(selected.matrix)
        if not count:
            return Rows.empty(selected.matrix.shape[1])
        original = np.flatnonzero(~np.asarray(constraints).any(axis=1)).tolist()
        identity = np.eye(count, dtype=np.uint8)
        kernel = kernel_gf2(np.asarray(constraints, dtype=np.uint8).T)
        candidates = np.vstack([identity[original], kernel])
        coefficients = candidates[independent_row_indices(candidates)]
        return Rows(coefficients @ selected.matrix % 2,
                    combine_records(coefficients, selected.records))

    def _v2_project_postselection(self, selected, measurements):
        """Only selected products commuting with every measurement survive."""
        return self._v2_postselection_kernel(
            selected, check_commutativity(selected.matrix, measurements))

    def _v2_trace_postselection(self, selected, qubits):
        """A reset retains selected products with no X or Z on reset qubits."""
        qubits = sorted(set(int(q) for q in qubits))
        columns = qubits + [self.num_qubits + q for q in qubits]
        return self._v2_postselection_kernel(selected, selected.matrix[:, columns])

    def _v2_forward_postselection(
        self, selected, forward_symplectic_matrix, *,
        measurement_qubit_indices=None, measurement_bases=None,
        measurement_base_idx=0,
    ):
        """Transport selected constraints through F and disposable readout.

        Retained measurements pass no terminal ancilla arguments: their
        quantum factors remain in the state. Disposable measurements replace
        those factors by the actual terminal measurement records.
        """
        from .tracker_algebra import Rows

        matrix = propagate_paulis(selected.matrix, forward_symplectic_matrix)
        records = selected.records
        if measurement_qubit_indices is not None:
            qubits = list(measurement_qubit_indices)
            bases = list(measurement_bases or ())
            if len(qubits) != len(bases):
                raise ValueError("One terminal basis is required per disposable ancilla")
            measured = np.zeros((len(qubits), 2*self.num_qubits), dtype=np.uint8)
            for i, (qubit, basis) in enumerate(zip(qubits, bases)):
                if basis in ("X", "Y"):
                    measured[i, qubit] = 1
                if basis in ("Z", "Y"):
                    measured[i, self.num_qubits + qubit] = 1
                if basis not in ("X", "Y", "Z"):
                    raise ValueError(f"Unsupported terminal measurement basis {basis!r}")
            folded = reduce_modulo(matrix, measured)
            coefficients, known, _ = solve_linear_decomposition(
                basis=measured, targets=matrix ^ folded, reduce_weight=False)
            if not np.all(known):
                raise RuntimeError("Selected constraint escaped the terminal measurement span")
            corrections = combine_records(
                coefficients, [[measurement_base_idx+i] for i in range(len(qubits))])
            records = [sorted(set(old) ^ set(new)) for old, new in zip(records, corrections)]
            matrix = folded
        return Rows(matrix, records)

    def _v2_restore_postselection(self, selected, *, circuit=None,
                                  measurement_coords=None):
        """Retain exact selected operators, never independently tag factors.

        S permits redundant rows. Keeping an exact selected representative
        avoids enlarging span(P1*P2) into span(P1,P2) when refresh happens to
        choose P1 and P2 as the ordinary state basis. Record-only closures are
        emitted immediately when a circuit is available.
        """
        import stim
        from .tracker_readout import _contains_parity, _insert_parity, _parity_mask

        s = self.stabilizers
        selected_output = set()
        existing_selected_parities = None
        for row, records in zip(selected.matrix, selected.records):
            records = list(records)
            if not row.any():
                if not records:
                    continue
                if circuit is not None:
                    from .tracker import _append_detector

                    if existing_selected_parities is None:
                        existing_selected_parities = {}
                        # Builder may pass a buffered round instead of the
                        # full circuit. Its rec targets still refer to the
                        # global measurement stream, including earlier rounds.
                        cursor = self.total_measurements - circuit.num_measurements
                        for instruction in circuit.flattened():
                            if instruction.name == "DETECTOR" and instruction.tag == "post-select":
                                parity = _parity_mask(cursor + target.value
                                                      for target in instruction.targets_copy())
                                _insert_parity(existing_selected_parities, parity)
                            cursor += instruction.num_measurements
                    parity = _parity_mask(records)
                    if not _contains_parity(existing_selected_parities, parity):
                        coord = (list(measurement_coords[-1])
                                 if measurement_coords is not None and len(measurement_coords)
                                 else [0, 0])
                        _append_detector(
                            circuit,
                            [stim.target_rec(r-self.total_measurements) for r in records],
                            coord + [0], post_select=True)
                        _insert_parity(existing_selected_parities, parity)
                    continue
                # A direct reset API has no circuit argument. Keep this exact
                # record-only constraint for the next readout's PS closure.
            known_indices = [i for i, old in enumerate(s.records) if all(r >= 0 for r in old)]
            known_logicals = [i for i, old in enumerate(self.logicals.records)
                              if all(r >= 0 for r in old)]
            physical = np.vstack([s.matrix[known_indices], self.logicals.matrix[known_logicals]])
            _, represented, _ = solve_linear_decomposition(
                basis=physical, targets=row[None, :], reduce_weight=False)
            if not represented[0]:
                raise RuntimeError("Surviving selected constraint lies outside the physical S/L state")
            exact = next((i for i in range(s.count)
                          if np.array_equal(s.matrix[i], row) and s.records[i] == records), None)
            if exact is None:
                bare = [i for i in self._v2_bare_indices()
                        if all(r >= 0 for r in s.records[i])]
                _, is_bare, _ = solve_linear_decomposition(
                    basis=s.matrix[bare], targets=row[None, :], reduce_weight=False)
                exact = s.count
                s.add_stabilizers(row[None, :], [records])
                self._classified_stabilizer_rows.add(exact)
                if not is_bare[0]:
                    self._dressed_stabilizer_rows.add(exact)
            selected_output.add(exact)
        self.post_select_row_indices = selected_output

    def _v2_bare_indices(self):
        return [i for i in range(self.stabilizers.count)
                if i not in self._dressed_stabilizer_rows]

    def _v2_rebuild_absorbed(self):
        """Represent S / (S_bare + L) using actual dressed rows and records."""
        s, live = self.stabilizers, self.logicals
        bare = self._v2_bare_indices()
        dressed = [i for i in sorted(self._dressed_stabilizer_rows)
                   if i < s.count and all(r >= 0 for r in s.records[i])]
        known_bare = [i for i in bare if all(r >= 0 for r in s.records[i])]
        known_live = [i for i, row in enumerate(live.records) if all(r >= 0 for r in row)]
        state = self._sync_record_state()
        if state is not None:
            from ..utils.cpp._pauli_backend_cpp import PackedBasis

            ns, nl = s.count, live.count
            prefix = known_bare + [ns + i for i in known_live]
            selected = [dressed[i] for i in state.ordered_complement(prefix, dressed)]
            kept = list(range(ns + nl)) + selected
            state.adopt_coordinates(state.coordinate_rows(kept), state.records_selected(kept))
            self._read_record_state((ns, nl, len(selected)))
            self.stabilizer_with_logical_components = set(self._dressed_stabilizer_rows)
            self._gauge_logical_vectors = []
            return
        basis = np.vstack([s.matrix[known_bare], live.matrix[known_live]])
        if dressed:
            _, _, chosen = solve_linear_decomposition(
                basis=basis, targets=s.matrix[dressed], reduce_weight=False)
            selected = [dressed[int(i)] for i in chosen]
        else:
            selected = []
        self.absorbed_ops.matrix = s.matrix[selected].copy()
        self.absorbed_ops.records = [list(s.records[i]) for i in selected]
        # Compatibility consumers use this set to exclude protected S rows.
        # It is no longer a union of measurement indices across rounds.
        self.stabilizer_with_logical_components = set(self._dressed_stabilizer_rows)
        self._gauge_logical_vectors = []

    def _v2_sync_budget(self):
        """Expose the current protected rank through the historical count API.

        Active-code changes can change this rank.  It is not a conserved
        initial logical-qubit budget, and redundant absorbed rows add no
        further physical constraints.
        """
        s, live = self.stabilizers, self.logicals
        bare = [i for i in self._v2_bare_indices()
                if all(r >= 0 for r in s.records[i])]
        known_s = [i for i, row in enumerate(s.records) if all(r >= 0 for r in row)]
        known_l = [i for i, row in enumerate(live.records) if all(r >= 0 for r in row)]
        state = self._sync_record_state()
        if state is not None:
            from ..utils.cpp._pauli_backend_cpp import PackedBasis

            full = state.coordinate_rows(known_s + [s.count + i for i in known_l])
            self.expected_num_logicals = (
                PackedBasis(full).rank - PackedBasis(state.coordinate_rows(bare)).rank)
            return
        full = np.vstack([s.matrix[known_s], live.matrix[known_l]])
        self.expected_num_logicals = (
            len(independent_row_indices(full))
            - len(independent_row_indices(s.matrix[bare])))

    def _v2_classify_stabilizers(self, active_stabilizers):
        """Reconstruct known S_bare = span(S) intersect the declared role space.

        Unknown preparation placeholders are retained without supplying any
        value to the intersection.  They remain explicitly unknown until a
        physical measurement establishes their values.
        """
        from .tracker_algebra import Rows, classify_stabilizers

        s = self.stabilizers
        width = 2 * self.num_qubits
        active = (np.zeros((0, width), dtype=np.uint8)
                  if active_stabilizers is None
                  else np.asarray(active_stabilizers, dtype=np.uint8))
        if active.ndim != 2 or active.shape[1] != width:
            raise ValueError("Active role space must use the tracker register")
        self._v2_role_space = active.copy()
        known = [i for i, row in enumerate(s.records) if all(r >= 0 for r in row)]
        known_set = set(known)
        unknown = [i for i in range(s.count) if i not in known_set]
        old_classified = set(self._classified_stabilizer_rows)
        old_ps = set(self.post_select_row_indices)
        old_to_known = {old: new for new, old in enumerate(known)}
        state = self._sync_record_state()
        if state is not None:
            from ..utils.cpp._pauli_backend_cpp import packed_classify_stabilizers

            ns, nl = s.count, self.logicals.count
            result = packed_classify_stabilizers(
                state.coordinate_rows(known), state.coordinate_paulis(active),
                [old_to_known[i] for i in old_ps if i in old_to_known])
            nb, nu = len(result["bare"]), len(unknown)
            new_s = (result["bare"].combined(state.coordinate_rows(unknown))
                     .combined(result["dressed"]))
            new_s_records = (
                state.combine_records(result["bare_lineage"], known)
                + state.records_selected(unknown)
                + state.combine_records(result["dressed_lineage"], known))
            logical_indices = list(range(ns, ns + nl))
            output = new_s.combined(state.coordinate_rows(logical_indices))
            output_records = new_s_records + state.records_selected(logical_indices)
            lineage = result["bare_lineage"].combined(result["dressed_lineage"]).numpy()
            classified = set()
            for out, coefficients in enumerate(lineage):
                sources = {known[int(i)] for i in np.flatnonzero(coefficients)}
                new = out if out < nb else out + nu
                if sources and sources <= old_classified:
                    classified.add(new)
            classified.update(nb + j for j, old in enumerate(unknown) if old in old_classified)
            selected = {i if i < nb else i + nu
                        for i in result["safely_postselected_output_rows"]}
            selected.update(nb + j for j, old in enumerate(unknown) if old in old_ps)
            state.adopt_coordinates(output, output_records)
            self._read_record_state((len(new_s), nl, 0))
            self._classified_stabilizer_rows = classified
            self.post_select_row_indices = selected
            self._dressed_stabilizer_rows = set(range(nb + nu, len(new_s)))
            self._v2_rebuild_absorbed()
            self._v2_sync_budget()
            return
        result = classify_stabilizers(
            Rows(s.matrix[known], [s.records[i] for i in known]), active,
            postselected_rows=[old_to_known[i] for i in old_ps if i in old_to_known])
        nb, nu = len(result.bare.matrix), len(unknown)
        new_matrix = np.vstack([result.bare.matrix, s.matrix[unknown], result.dressed.matrix])
        new_records = ([list(r) for r in result.bare.records]
                       + [list(s.records[i]) for i in unknown]
                       + [list(r) for r in result.dressed.records])
        # The two helper lineages refer only to known input S rows.
        lineage = np.vstack([result.bare_lineage, result.dressed_lineage])
        classified = set()
        for out, coefficients in enumerate(lineage):
            sources = {known[int(i)] for i in np.flatnonzero(coefficients)}
            new = out if out < nb else out + nu
            if sources and sources <= old_classified:
                classified.add(new)
        classified.update(nb + j for j, old in enumerate(unknown) if old in old_classified)
        selected = {i if i < nb else i + nu
                    for i in result.safely_postselected_output_rows}
        selected.update(nb + j for j, old in enumerate(unknown) if old in old_ps)
        self.stabilizers.matrix = new_matrix
        self.stabilizers.records = new_records
        self._classified_stabilizer_rows = classified
        self.post_select_row_indices = selected
        self._dressed_stabilizer_rows = set(range(nb + nu, len(new_matrix)))
        self._v2_rebuild_absorbed()
        self._v2_sync_budget()

    def _v2_reclassify_after_readout(self, active_stabilizers):
        """Classify the known S/L intersection with the continuing code.

        Readout has already extracted relations using the input roles. Keep
        existing bare S representatives first: a redundant logical expression
        can differ from one of them by a previously extracted observable, and
        must not become a second bare representative. Add only missing bare
        directions, then retain S and L complements in that order.
        """
        self._v2_classify_stabilizers(active_stabilizers)
        active_stabilizers = self._v2_role_space
        s, logicals = self.stabilizers, self.logicals
        ns, nl = s.count, logicals.count
        if not nl:
            return

        known_s = [i for i, records in enumerate(s.records) if all(r >= 0 for r in records)]
        known_l = [ns + i for i, records in enumerate(logicals.records)
                   if all(r >= 0 for r in records)]
        if not known_l:
            return
        known_s_set, known_l_set = set(known_s), set(known_l)
        unknown_s = [i for i in range(ns) if i not in known_s_set]
        unknown_l = [i for i in range(ns, ns + nl) if i not in known_l_set]
        bare = [i for i in known_s if i not in self._dressed_stabilizer_rows]
        dressed = [i for i in known_s if i in self._dressed_stabilizer_rows]
        selected_dressed = [i for i in dressed if i in self.post_select_row_indices]
        other_dressed = [i for i in dressed if i not in self.post_select_row_indices]
        sources = known_s + known_l
        state = self._sync_record_state()

        if state is not None:
            from ..utils.cpp._pauli_backend_cpp import PackedBasis, packed_classify_stabilizers

            def rows(indices):
                return state.coordinate_rows(indices)

            def concatenate(left, right):
                return left.combined(right)

            def independent(matrix):
                return list(PackedBasis(matrix).selected)

            def select(matrix, indices):
                return matrix.select(indices)

            get_records = state.records_selected
            intersection = packed_classify_stabilizers(
                rows(sources), state.coordinate_paulis(active_stabilizers), [])
            all_bare = intersection["bare"]
            all_lineage = intersection["bare_lineage"]
        else:
            from .tracker_algebra import Rows, classify_stabilizers

            physical = np.vstack([s.matrix, logicals.matrix])
            records = [list(r) for r in s.records] + [list(r) for r in logicals.records]

            def rows(indices):
                return physical[indices]

            def concatenate(left, right):
                return np.vstack([left, right])

            def independent(matrix):
                return list(independent_row_indices(matrix))

            def select(matrix, indices):
                return matrix[indices]

            def get_records(indices):
                return [records[i] for i in indices]

            intersection = classify_stabilizers(
                Rows(rows(sources), get_records(sources)), active_stabilizers)
            all_bare, all_lineage = intersection.bare.matrix, intersection.bare_lineage

        old_bare = rows(bare)
        extra = [i - len(bare) for i in independent(concatenate(old_bare, all_bare))
                 if i >= len(bare)]
        new_bare = concatenate(old_bare, select(all_bare, extra))
        extra_lineage = select(all_lineage, extra)
        extra_records = (state.combine_records(extra_lineage, sources) if state is not None
                         else combine_records(extra_lineage, get_records(sources)))

        # Exact postselected S representatives survive even when redundant.
        prefix = concatenate(new_bare, rows(selected_dressed))
        chosen_dressed = [other_dressed[i - len(prefix)]
                          for i in independent(concatenate(prefix, rows(other_dressed)))
                          if i >= len(prefix)]
        kept_dressed = selected_dressed + chosen_dressed
        prefix = concatenate(new_bare, rows(kept_dressed))
        kept_l = [known_l[i - len(prefix)]
                  for i in independent(concatenate(prefix, rows(known_l)))
                  if i >= len(prefix)]
        kept_l = sorted(kept_l + unknown_l)

        # Unknown placeholders never supply eigenvalues to the intersection.
        new_s = concatenate(concatenate(new_bare, rows(unknown_s)), rows(kept_dressed))
        output = concatenate(new_s, rows(kept_l))
        output_records = (get_records(bare) + extra_records + get_records(unknown_s)
                          + get_records(kept_dressed) + get_records(kept_l))
        old_to_new = {old: i for i, old in enumerate(bare)}
        old_to_new.update({old: len(new_bare) + i for i, old in enumerate(unknown_s)})
        dressed_start = len(new_bare) + len(unknown_s)
        old_to_new.update({old: dressed_start + i for i, old in enumerate(kept_dressed)})
        selected = {old_to_new[i] for i in self.post_select_row_indices if i in old_to_new}
        classified = {old_to_new[i] for i in self._classified_stabilizer_rows if i in old_to_new}
        classified.update(range(len(bare), len(new_bare)))
        if state is not None:
            state.adopt_coordinates(output, output_records)
            self._read_record_state((len(new_s), len(kept_l), 0))
        else:
            self.stabilizers.matrix = new_s
            self.stabilizers.records = output_records[:len(new_s)]
            self.logicals.matrix = output[len(new_s):]
            self.logicals.records = output_records[len(new_s):]
        self.post_select_row_indices = selected
        self._classified_stabilizer_rows = classified
        self._dressed_stabilizer_rows = set(range(dressed_start, len(new_s)))
        self._v2_rebuild_absorbed()
        self._v2_sync_budget()

    def _v2_commuting_relation(self, full, num_stabs, num_logs, measurement,
                               previous_measurements, measurement_base_index, index,
                               *, context=None, record_state=None):
        """Classify a commuting measurement using bare and earlier measurements.

        A fresh measured constraint supplies repeat-check information even if
        its projection pivot occupied a logical row.  All temporary rows here
        have their actual record expressions; L_absorbed is never a source.
        Returns (kind, records, logical_indices). The last field records the
        participating current L rows for historical API diagnostics; dressed
        directions do not acquire fictitious L indices.
        """
        # The builder's projection loop invalidates this local context after
        # every row/record mutation. It never survives an atomic operation.
        if context is None:
            context = {}
        if not context:
            bare = [i for i in range(num_stabs)
                    if i not in self._dressed_stabilizer_rows
                    and all(r >= 0 for r in full.records[i])]
            exact_rows = {}
            for i in bare:
                exact_rows.setdefault(np.asarray(full.matrix[i], dtype=np.uint8).tobytes(), i)
            context.update(
                bare=bare, exact_rows=exact_rows,
                known=[i for i in range(num_stabs + num_logs)
                       if all(r >= 0 for r in full.records[i])])
        bare = context["bare"]
        prefix = previous_measurements[:index]
        # Preserve an existing check's record representative before choosing
        # another decomposition of a redundant S_bare basis.  The difference
        # is itself a valid state relation, but replacing a local repeat check
        # by a product of other checks can make elementary faults touch many
        # detectors.  Exact matching is also the common repeated-SE case and
        # needs no new elimination of the unchanged bare space.
        exact = context["exact_rows"].get(np.asarray(measurement, dtype=np.uint8).tobytes())
        if exact is not None:
            parity = set(full.records[exact])
            parity.symmetric_difference_update([measurement_base_index + index])
            return "detector", sorted(parity), ()
        repeated = np.flatnonzero(np.all(prefix == measurement, axis=1))
        if len(repeated):
            return "detector", [measurement_base_index + int(repeated[0]),
                                measurement_base_index + index], ()
        basis = np.vstack([full.matrix[bare], prefix])
        records = ([full.records[i] for i in bare]
                   + [[measurement_base_index + j] for j in range(index)])
        coeff, dependent, _ = solve_linear_decomposition(
            basis=basis, targets=measurement.reshape(1, -1), reduce_weight=False)
        if dependent[0]:
            parity = set(combine_records(coeff, records)[0])
            parity.symmetric_difference_update([measurement_base_index + index])
            return "detector", sorted(parity), ()
        known = context["known"]
        if record_state is None:
            coeff, dependent, _ = solve_linear_decomposition(
                basis=full.matrix[known], targets=measurement.reshape(1, -1),
                reduce_weight=False)
        else:
            native_coeff, native_dependent, native_parity = record_state.decompose(
                measurement, known)
            coeff = np.asarray(native_coeff, dtype=np.uint8)[known][None, :]
            dependent = [native_dependent]
        if not dependent[0]:
            if record_state is None:
                candidate, represented, _ = solve_linear_decomposition(
                    basis=full.matrix[:num_stabs + num_logs],
                    targets=measurement.reshape(1, -1), reduce_weight=False)
            else:
                candidate_row, represented_row, _ = record_state.decompose(
                    measurement, list(range(num_stabs + num_logs)))
                candidate = np.asarray(candidate_row, dtype=np.uint8)[None, :]
                represented = [represented_row]
            if represented[0] and any(
                i >= num_stabs and any(r < 0 for r in full.records[int(i)])
                for i in np.flatnonzero(candidate[0])
            ):
                raise ValueError("Cannot capture logical history from an unmeasured logical record")
            # An unknown placeholder cannot establish a deterministic parity.
            # The physical refresh below still records the new measurement.
            return "unknown", [], ()
        parity = set(native_parity if record_state is not None else
                     combine_records(coeff, [full.records[i] for i in known])[0])
        parity.symmetric_difference_update([measurement_base_index + index])
        logical_indices = tuple(known[int(i)] - num_stabs
                                for i in np.flatnonzero(coeff[0])
                                if known[int(i)] >= num_stabs)
        return "logical", sorted(parity), logical_indices

    def _v2_validate_roles(self, *, context="tracker state"):
        if any(i < 0 or i >= self.stabilizers.count for i in self._dressed_stabilizer_rows):
            raise RuntimeError(f"After {context}: dressed row index outside S")
        state = self._sync_record_state()
        if state is not None:
            from ..utils.cpp._pauli_backend_cpp import PackedBasis

            physical = self.stabilizers.count + self.logicals.count
            if self.absorbed_ops.count:
                basis = PackedBasis(state.coordinate_rows(range(physical)))
                _, contained, _ = basis.decompose_packed(state.coordinate_rows(
                    range(physical, physical + self.absorbed_ops.count)))
                if not np.all(contained):
                    raise RuntimeError(f"After {context}: absorbed representative outside span(S,L)")
            self._v2_sync_budget()
            return
        full = np.vstack([self.stabilizers.matrix, self.logicals.matrix])
        if self.absorbed_ops.count:
            _, contained, _ = solve_linear_decomposition(
                basis=full, targets=self.absorbed_ops.matrix, reduce_weight=False)
            if not np.all(contained):
                raise RuntimeError(f"After {context}: absorbed representative outside span(S,L)")
        self._v2_sync_budget()
