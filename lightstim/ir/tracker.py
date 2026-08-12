import stim
import warnings
import numpy as np
from ..utils.linear_algebra import check_commutativity, solve_linear_decomposition
from ..utils.tableau_utils import stabilizers_to_symplectic
from .tableau import PauliTableau
from typing import List, Dict, Tuple, Optional, Set, Any

# Tag for post-selection: detectors with this tag are used for post-selection filtering
POST_SELECT_TAG = "post-select"

# Sentinel for unmeasured stabilizer rows: treated as has_record (stays stabilizer), excluded from detector construction
UNMEASURED_STAB_RECORD = -1


def _gf2_rank(M):
    """GF(2) row rank of a 0/1 matrix."""
    M = (np.asarray(M, dtype=np.uint8) % 2).copy()
    if M.size == 0:
        return 0
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        mask = M[:, c].astype(bool).copy()
        mask[r] = False
        M[mask] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def _gf2_rref(M):
    """GF(2) reduced row-echelon form of a 0/1 matrix.

    Returns (R, pivots): R the non-zero reduced rows, pivots the pivot
    column of each row of R (each pivot column is zero in every other
    row of R).
    """
    M = (np.asarray(M, dtype=np.uint8) % 2).copy()
    if M.size == 0:
        return M.reshape(0, -1), []
    rows, cols = M.shape
    r = 0
    pivots = []
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        mask = M[:, c].astype(bool).copy()
        mask[r] = False
        M[mask] ^= M[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return M[:r], pivots


def _append_detector(
    circuit: stim.Circuit,
    args: list,
    coords: list,
    post_select: bool = False,
) -> None:
    """Append a DETECTOR instruction, optionally with post-select tag."""
    if post_select:
        try:
            circuit.append("DETECTOR", args, coords, tag=POST_SELECT_TAG)
        except TypeError:
            circuit.append("DETECTOR", args, coords)
    else:
        circuit.append("DETECTOR", args, coords)


class SyndromeTracker:
    def __init__(
        self,
        num_qubits: int,
        expected_num_logicals: int = 0,
        post_select_detector_coords: Optional[Set[Tuple[float, ...]]] = None,
    ):
        # Number of physical qubits (include data, syndrome, and potentially other ancilla qubits)
        self.num_qubits = num_qubits
        # Number of logical qubits (should be the total number of logical qubits in the QEC system)
        self.expected_num_logicals = expected_num_logicals
        # Total number of measurements
        self.total_measurements = 0
        self.total_observables = 0
        self.meas_rec_to_idx_map = {}

        # Track the current stabilizers and logicals of the system
        # Note 1: Technically, logicals are also stabilizers of the system, define the logical states
        # Note 2: Stabilizer tableau allows linear dependencies between rows (e.g. toric code, BB code), but logicals do not in general..
        self.stabilizers = PauliTableau(num_qubits)
        self.logicals = PauliTableau(num_qubits)
        # Absorbed logical operators (Fix C): the measured Pauli strings that were folded
        # into the stabilizer group by a merge (e.g. a joint ZZ over two |0>) and STILL
        # hold a trapped logical DOF. Persisted across rounds/PPMs, so an absorb that an
        # intervening round does not re-measure is not lost (this is what the old per-round
        # gauge tally missed — pitfall B). THE census ledger: every path that absorbs a
        # logical DOF records the operator here, and the count is always DERIVED via
        # num_absorbed_dof() (the ledger's own GF(2) rank; logical-equivalence dedup
        # happens at insertion — record_absorbed_op, the ledger's only entrance) —
        # there is deliberately no
        # separately maintained integer, because a bare counter demands perfect
        # increment/decrement pairing from every path and drifts silently when one
        # forgets. Self-correcting: once an absorbed relation is read out
        # (process_data_measurement), its row is dropped and stops being
        # counted.
        self.absorbed_ops = PauliTableau(num_qubits)
        self.stabilizer_with_logical_components = set()  # Row indices of stabilizers that contain logical components
        self._gauge_logical_vectors = []  # GF(2) vectors over logical indices for rank computation
        self.post_select_detector_coords = post_select_detector_coords or set()
        self.post_select_row_indices = set()  # Stabilizer row indices to post-select in process_data_measurement

    def set_expected_logicals(self, k: int):
        """
        Call this after the logical count is adjusted
        (e.g. some logicals are fixed into stabilizers)
        """
        self.expected_num_logicals = k

    def num_absorbed_dof(self) -> int:
        """Number of banked absorbed logical DOFs (cross-round persistent):
        the GF(2) rank of the ledger itself.

        Deliberately NOT quotiented by the current stabilizer bank: after a
        merge the joint's CLOSURE row (same relation, carrying the merge
        records) legitimately lives in the bank, and modding the ledger by
        the bank would cancel the banked DOF against its own reflection
        (found the hard way: the census tripped on a measurement-block
        round over post-PPM state).  Logical-equivalence deduplication
        happens at INSERTION instead — see record_absorbed_op — so two
        representatives differing by a stabilizer never both enter the
        ledger, while a banked relation keeps counting regardless of how
        the group later expresses it."""
        return _gf2_rank(self.absorbed_ops.matrix)

    def record_absorbed_op(self, op: np.ndarray, records=()) -> bool:
        """Bank one absorbed logical relation, deduplicated up to logical
        equivalence AT THIS MOMENT: the operator is reduced against the
        current stabilizer rows and the already-banked relations, and only
        an irreducible residue is recorded.  A second representative of an
        already-banked relation (differing by stabilizers) reduces to zero
        and is skipped.  Returns True iff a new row was banked."""
        op = np.asarray(op, dtype=np.uint8).reshape(1, -1)
        if not op.any():
            return False
        A = self.absorbed_ops
        basis_parts = [M for M in (self.stabilizers.matrix, A.matrix)
                       if M.shape[0] > 0]
        if basis_parts:
            basis = np.vstack(basis_parts)
            _, dep, _ = solve_linear_decomposition(basis=basis, targets=op,
                                                   reduce_weight=False)
            if dep[0]:
                return False
        A.matrix = (np.vstack([A.matrix, op]).astype(np.uint8)
                    if A.count else op.astype(np.uint8))
        A.records = list(A.records) + [list(records)]
        return True

    def allocate_observable(self) -> int:
        """Reserve and return the next OBSERVABLE_INCLUDE index.

        Every emitter of a NEW observable must allocate here — never from
        circuit.num_observables — so that two independent logical results
        can never share an ID (a shared ID XORs them into one observable,
        and the XOR of two deterministic bits stays deterministic, so p=0
        sampling cannot catch the collision).  Re-using a previously
        allocated index to ACCUMULATE into the same observable remains
        legal and does not go through this method.
        """
        idx = self.total_observables
        self.total_observables += 1
        return idx

    def expand(self, delta: int):
        """
        Expand the tracker to include delta new qubits (define-by-run).
        New qubits act as identity on existing stabilizers/logicals.
        """
        if delta <= 0:
            return
        self.stabilizers.expand(delta)
        self.logicals.expand(delta)
        self.absorbed_ops.expand(delta)
        self.num_qubits += delta


    def _remap_rows_after_removal(self, removed_sorted):
        """Shift stabilizer-row metadata down past removed stabilizer rows.

        Every deletion of stabilizer rows must come through here (or rebuild
        the sets itself, as the measurement-block paths do): both
        post_select_row_indices and stabilizer_with_logical_components hold
        ROW indices, and a stale index silently post-selects or reclassifies
        a different row.  stabilizer_with_logical_components pairs
        positionally with _gauge_logical_vectors via sorted order, so
        entries dropped here drop their paired vector too (the shift is
        monotone, so surviving pairs stay aligned).
        """
        removed = sorted(set(removed_sorted))
        if not removed:
            return
        rem = set(removed)

        def shift(idx):
            return idx - sum(1 for r in removed if r < idx)

        if self.post_select_row_indices:
            self.post_select_row_indices = {
                shift(i) for i in self.post_select_row_indices
                if i not in rem}
        if self.stabilizer_with_logical_components:
            swlc_sorted = sorted(self.stabilizer_with_logical_components)
            vectors = list(self._gauge_logical_vectors)
            paired = list(zip(swlc_sorted, vectors)) if vectors else \
                [(i, None) for i in swlc_sorted]
            kept = [(i, v) for i, v in paired if i not in rem]
            self.stabilizer_with_logical_components = {
                shift(i) for i, _ in kept}
            if vectors:
                self._gauge_logical_vectors = [v for _, v in kept]

    def reset_records_for_qubits(self, qubit_indices):
        """
        Clean up tracker state for qubits being re-initialized mid-circuit.

        Three cases for each stabilizer row:
        - Support ONLY on target qubits → remove (re-init adds fresh rows)
        - Support on BOTH target and non-target → keep but clear records
          (prevents detectors from comparing across state-change boundary)
        - No target support → keep as-is

        Same logic applied to logical rows.
        """
        n = self.num_qubits
        qubit_set = set(qubit_indices)

        def _clean_rows(tableau):
            new_indices = []
            new_records = []
            for i in range(tableau.count):
                row = tableau.matrix[i]
                has_target = any(row[q] != 0 or row[q + n] != 0 for q in qubit_set)
                if not has_target:
                    # No target support → keep as-is
                    new_indices.append(i)
                    new_records.append(tableau.records[i])
                elif has_target:
                    # Check if support extends beyond target qubits
                    has_non_target = False
                    for q in range(n):
                        if q not in qubit_set and (row[q] != 0 or row[q + n] != 0):
                            has_non_target = True
                            break
                    if has_non_target:
                        # Mixed support → keep but mark as unmeasured
                        # (prevents stale detector comparison, preserves Pauli operator)
                        new_indices.append(i)
                        new_records.append([UNMEASURED_STAB_RECORD])
                    # else: support ONLY on target → remove (skip)

            if new_indices:
                tableau.matrix = tableau.matrix[new_indices]
            else:
                tableau.matrix = np.zeros((0, 2 * n), dtype=np.uint8)
            tableau.records = new_records

        _clean_rows(self.stabilizers)
        _clean_rows(self.logicals)

    def _reject_pending_row_metadata(self, context: str) -> None:
        """Basis recombination replaces rows by linear combinations, so old
        row indices carry no meaning afterwards — a shift cannot fix them.
        Fail loud instead of silently post-selecting or reclassifying a
        recombined row.  (The measurement-block paths that CAN remap by
        decomposition do so themselves and never call this.)"""
        if self.post_select_row_indices:
            raise RuntimeError(
                f"{context}: post_select_row_indices is non-empty but the "
                f"stabilizer basis is about to be recombined; row indices "
                f"would silently point at different rows. Resolve or clear "
                f"the post-selection marks first.")
        if self.stabilizer_with_logical_components:
            raise RuntimeError(
                f"{context}: stabilizer_with_logical_components is non-empty "
                f"but the stabilizer basis is about to be recombined; the "
                f"pending gauge/logical classification would be lost.")

    def stabilizer_canonicalization(
        self,
        system: Any,
        stabilizer_uids: Optional[Set[int]] = None,
    ) -> None:
        """
        Re-organize stabilizer tableau into stabilizers vs logicals BEFORE any SE measurement.
        Basis = canonical stabilizer set (active_stabilizers or stabilizer_uids).
        Uses new_basis_indices (Logical Basis) to extract minimal logical dimension, not is_dependent.
        Aligned with process_mid_measurement Step 3.

        Call after encoding, before SE. Raises if logical count does not match expected.
        """
        self._reject_pending_row_metadata("stabilizer_canonicalization")
        n = self.num_qubits
        if stabilizer_uids is not None:
            stab_dicts = [system.stabilizers[i] for i in range(len(system.stabilizers)) if i in stabilizer_uids]
        else:
            stab_dicts = [system.stabilizers[i] for i in sorted(system.active_stabilizer_indices)]
        canonical_basis = stabilizers_to_symplectic(system, stab_dicts, n)

        if canonical_basis.shape[0] == 0:
            return

        num_stabs = self.stabilizers.count
        if num_stabs == 0:
            return

        # Full tableau = stabilizers + logicals (same structure as process_mid_measurement)
        existing_log_matrix = self.logicals.matrix
        existing_log_records = self.logicals.records
        if existing_log_matrix.shape[0] > 0:
            full_matrix = np.vstack([self.stabilizers.matrix, existing_log_matrix])
            full_records = self.stabilizers.records + existing_log_records
        else:
            full_matrix = self.stabilizers.matrix
            full_records = self.stabilizers.records

        coeffs, _, new_basis_indices = solve_linear_decomposition(
            basis=canonical_basis,
            targets=full_matrix,
            reduce_weight=True,
        )

        # new_basis_indices = Logical Basis (pivot columns). Rows not in it -> stabilizers.
        old_stab_indices = [i for i in range(num_stabs) if i not in new_basis_indices]
        new_log_basis_indices = list(new_basis_indices)

        # Split dependent rows into MEASURED (keep evolved form + records) and
        # UNMEASURED (replace with canonical basis rows to preserve raw structure).
        measured_indices = [i for i in old_stab_indices
                           if full_records[i] and full_records[i] != [UNMEASURED_STAB_RECORD]]
        unmeasured_indices = [i for i in old_stab_indices if i not in measured_indices]

        measured_rows = full_matrix[measured_indices] if measured_indices else np.zeros((0, 2*n), dtype=np.uint8)

        # For unmeasured slots: use canonical_basis rows that are independent of the measured rows.
        # This replaces evolved RREF representatives with raw canonical forms (e.g., original RM
        # generator matrix rows), preserving the structure for downstream gauge measurements.
        if measured_rows.shape[0] > 0 and canonical_basis.shape[0] > 0:
            _, _, indep_canonical = solve_linear_decomposition(
                basis=measured_rows, targets=canonical_basis, reduce_weight=False)
            unmeasured_rows = canonical_basis[indep_canonical]
        else:
            unmeasured_rows = canonical_basis

        new_stab_matrix = np.vstack([measured_rows, unmeasured_rows]) if measured_rows.shape[0] > 0 else unmeasured_rows
        measured_records = [full_records[i] for i in measured_indices]
        unmeasured_records = [[UNMEASURED_STAB_RECORD]] * unmeasured_rows.shape[0]
        new_stab_records = measured_records + unmeasured_records
        new_log_matrix = full_matrix[new_log_basis_indices]
        new_log_records = [full_records[i] for i in new_log_basis_indices]

        self.stabilizers.matrix = new_stab_matrix
        self.stabilizers.records = new_stab_records
        self.logicals.matrix = new_log_matrix
        self.logicals.records = new_log_records

        self.validate_logical_count(context="stabilizer canonicalization")

    def logical_canonicalization(
        self,
        canonical_logicals: Dict[int, np.ndarray],
    ) -> None:
        """
        Replace logical operators with preferred canonical representatives.

        For each (logical_index, canonical_pauli) pair:
          1. Verify that canonical_pauli is in span(stabilizers + logicals)
             but NOT in span(stabilizers) alone — i.e. it is a genuine logical.
          2. Replace the corresponding logical row and update remaining logicals
             to eliminate the canonical component (RREF on the logical subspace).

        This allows choosing minimal-weight logical representatives that lead
        to lower-weight observables in the detector error model.

        Args:
            canonical_logicals: {logical_index: pauli_vector} where
                logical_index is the row index in self.logicals (0-based),
                pauli_vector is a (2*num_qubits,) GF(2) array.
        """
        n = self.num_qubits
        num_stabs = self.stabilizers.count
        num_logs = self.logicals.count

        if num_logs == 0:
            raise ValueError("No logicals to canonicalize.")

        # Combined tableau: [stabilizers; logicals]
        full_matrix = np.vstack([self.stabilizers.matrix, self.logicals.matrix])
        full_records = self.stabilizers.records + self.logicals.records

        for log_idx, canonical_pauli in canonical_logicals.items():
            if log_idx < 0 or log_idx >= num_logs:
                raise ValueError(
                    f"Logical index {log_idx} out of range [0, {num_logs})."
                )

            canonical_pauli = np.asarray(canonical_pauli, dtype=np.uint8).reshape(1, -1)
            if canonical_pauli.shape[1] != 2 * n:
                raise ValueError(
                    f"Canonical pauli has {canonical_pauli.shape[1]} columns, "
                    f"expected {2 * n}."
                )

            # Step 1: Verify it's a valid logical (in full span but not in stab span)
            # Check against stabilizers only
            _, stab_dep, _ = solve_linear_decomposition(
                basis=self.stabilizers.matrix,
                targets=canonical_pauli,
                reduce_weight=False,
            )
            if stab_dep[0]:
                raise ValueError(
                    f"Canonical logical {log_idx} is in span(stabilizers) — "
                    "it's a stabilizer, not a logical."
                )

            # Check against full tableau
            coeffs_full, full_dep, _ = solve_linear_decomposition(
                basis=full_matrix,
                targets=canonical_pauli,
                reduce_weight=False,
            )
            if not full_dep[0]:
                raise ValueError(
                    f"Canonical logical {log_idx} is NOT in span(stabilizers + logicals) — "
                    "it does not belong to the current code space."
                )

            # Step 2: Decompose and compute records for the canonical logical
            # c = sum of contributing rows from full_matrix
            contributing_indices = np.where(coeffs_full[0])[0]
            canonical_records = []
            rec_set = set()
            for idx in contributing_indices:
                rec_set.symmetric_difference_update(full_records[idx])
            canonical_records = list(rec_set)

            # Replace the target logical row
            abs_idx = num_stabs + log_idx  # index in full_matrix
            old_pauli = full_matrix[abs_idx].copy()
            old_records = full_records[abs_idx]

            full_matrix[abs_idx] = canonical_pauli[0]
            full_records[abs_idx] = canonical_records

            # Step 3: Update other logical rows to eliminate the canonical component.
            # For each other logical row l_j, if l_j has a component along the
            # old canonical direction, XOR it out.
            # We do this by checking if l_j + canonical is simpler (in stab span
            # minus one dimension) — but a simpler approach: just XOR any logical
            # row that anti-commutes with the symplectic partner, or use RREF.
            #
            # Practical approach: for each other logical j != log_idx,
            # decompose l_j against the new full_matrix. If it depends on the
            # canonical row, XOR to remove that dependence.
            for j in range(num_logs):
                if j == log_idx:
                    continue
                j_abs = num_stabs + j
                row_j = full_matrix[j_abs]
                # Check if row_j has overlap with canonical in the logical subspace
                # by seeing if (row_j XOR canonical) is in span(stabs + other logicals)
                test = (row_j ^ canonical_pauli[0]).reshape(1, -1)
                _, test_dep, _ = solve_linear_decomposition(
                    basis=self.stabilizers.matrix,
                    targets=test,
                    reduce_weight=False,
                )
                # If row_j XOR canonical is a stabilizer, then row_j has the same
                # logical component as canonical — need to XOR them
                if test_dep[0]:
                    full_matrix[j_abs] ^= canonical_pauli[0]
                    rec_set_j = set(full_records[j_abs])
                    rec_set_j.symmetric_difference_update(canonical_records)
                    full_records[j_abs] = list(rec_set_j)

        # Write back
        self.logicals.matrix = full_matrix[num_stabs:]
        self.logicals.records = full_records[num_stabs:]

    def process_initialization(self, init_tableau: np.ndarray):
        """
        Registers new stabilizers from initialization into the tracker.

        For t=0 (System Start): Populates the empty tableau.
        For New Patch: Appends new independent stabilizers to the existing set.

        Args:
            init_tableau: Shape (k, 2n).
        """
        self._reject_reset_over_banked(
            {int(c) % self.num_qubits
             for c in np.flatnonzero(init_tableau.any(axis=0))},
            context="process_initialization")
        self.stabilizers.add_stabilizers(init_tableau)


    def process_unitary_block(self, circuit_chunk: stim.Circuit):
        """Forward-propagate the tracked state through a Clifford circuit."""
        self._apply_symplectic_matrix(
            self.get_forward_symplectic_matrix(circuit_chunk, self.num_qubits)
        )

    @staticmethod
    def get_forward_symplectic_matrix(
        circuit_chunk: stim.Circuit,
        num_qubits: int,
    ) -> np.ndarray:
        """Build the padded forward symplectic matrix of a Clifford circuit."""
        u_tableau = stim.Tableau.from_circuit(
            circuit_chunk,
            ignore_noise=True,
            ignore_measurement=True,
            ignore_reset=True,
        )
        x2x, x2z, z2x, z2z, _, _ = u_tableau.to_numpy()
        x2x = x2x.astype(np.uint8)
        x2z = x2z.astype(np.uint8)
        z2x = z2x.astype(np.uint8)
        z2z = z2z.astype(np.uint8)

        n_chunk = len(u_tableau)
        if n_chunk > num_qubits:
            raise ValueError(
                f"Circuit chunk involves qubit {n_chunk - 1}, exceeding "
                f"system size {num_qubits}."
            )

        if n_chunk == num_qubits:
            return np.vstack([
                np.hstack([x2x, x2z]),
                np.hstack([z2x, z2z]),
            ])

        full_M = np.eye(2 * num_qubits, dtype=np.uint8)
        if n_chunk:
            full_M[:n_chunk, :n_chunk] = x2x
            full_M[:n_chunk, num_qubits:num_qubits + n_chunk] = x2z
            full_M[num_qubits:num_qubits + n_chunk, :n_chunk] = z2x
            full_M[
                num_qubits:num_qubits + n_chunk,
                num_qubits:num_qubits + n_chunk,
            ] = z2z
        return full_M

    def _apply_symplectic_matrix(self, symplectic_matrix: np.ndarray) -> None:
        expected_shape = (2 * self.num_qubits, 2 * self.num_qubits)
        if symplectic_matrix.shape != expected_shape:
            raise ValueError(
                f"Expected a symplectic matrix with shape {expected_shape}; "
                f"got {symplectic_matrix.shape}."
            )

        if self.stabilizers.count > 0:
            self.stabilizers.matrix = (self.stabilizers.matrix @ symplectic_matrix) % 2
            self.stabilizers.matrix = self.stabilizers.matrix.astype(np.uint8)

        if self.logicals.count > 0:
            self.logicals.matrix = (self.logicals.matrix @ symplectic_matrix) % 2
            self.logicals.matrix = self.logicals.matrix.astype(np.uint8)
        # Fix C: the absorbed operators are Pauli strings too — apply the same
        # Clifford so they stay in the current frame (else the span overlap
        # with stabilizers drifts).
        if self.absorbed_ops.count > 0:
            self.absorbed_ops.matrix = (self.absorbed_ops.matrix @ symplectic_matrix) % 2
            self.absorbed_ops.matrix = self.absorbed_ops.matrix.astype(np.uint8)


    def _reject_reset_over_banked(self, touched_qubits, *, context):
        """A banked absorbed relation must never ride silently through a
        physical reset/re-initialisation: the reset destroys the relation's
        support while its recorded parity stays banked, so the census would
        go stale without any alarm.  Production flows never reset a qubit
        that still carries a banked relation (readouts resolve or fold the
        ledger first) — both real reset paths (process_resets, reached from
        every SE round's ancilla resets, and process_initialization, reached
        from CircuitBuilder.initialize) fail loud here instead of inventing
        discard semantics."""
        A = self.absorbed_ops
        if not A.count or not touched_qubits:
            return
        n = self.num_qubits
        cols = [q for q in touched_qubits] + [n + q for q in touched_qubits]
        if A.matrix[:, cols].any():
            raise RuntimeError(
                f"{context}: reset touches qubits that still carry banked "
                f"absorbed relations — resolve them via a corridor/patch "
                f"readout before re-initialising, or account for the "
                f"discarded DOF explicitly.")

    def process_resets(
        self,
        reset_paulis: np.ndarray,
    ) -> None:
        """Apply physical reset operations to the current tracked state."""
        if reset_paulis.shape[0] == 0:
            return


        self._reject_reset_over_banked(
            {int(c) % self.num_qubits
             for c in np.flatnonzero(reset_paulis.any(axis=0))},
            context="process_resets")

        num_stabs = self.stabilizers.count
        num_logs = self.logicals.count
        if num_logs:
            full_matrix = np.vstack([
                self.stabilizers.matrix,
                self.logicals.matrix,
            ])
            full_records = self.stabilizers.records + self.logicals.records
        else:
            full_matrix = self.stabilizers.matrix.copy()
            full_records = list(self.stabilizers.records)

        full = PauliTableau(self.num_qubits)
        full.matrix = full_matrix
        full.records = [list(records) for records in full_records]

        for reset_pauli in reset_paulis:
            reset_row = reset_pauli.reshape(1, -1)
            anti_commuting = np.flatnonzero(
                check_commutativity(reset_row, full.matrix)[0]
            )
            if len(anti_commuting):
                pivot = int(anti_commuting[0])
                for other in anti_commuting[1:]:
                    full.update_row(int(other), pivot)
                full.replace_row(pivot, reset_pauli, [])
            else:
                coeffs, is_dependent, _ = solve_linear_decomposition(
                    basis=full.matrix,
                    targets=reset_row,
                    reduce_weight=False,
                )
                if is_dependent[0]:
                    contributing = np.flatnonzero(coeffs[0])
                    pivot = int(contributing[0])
                    for other in contributing[1:]:
                        full.update_row(pivot, int(other))
                else:
                    pivot = num_stabs
                    full.matrix = np.insert(
                        full.matrix,
                        pivot,
                        reset_pauli,
                        axis=0,
                    )
                    full.records.insert(pivot, [])
                    num_stabs += 1

            reset_x = np.flatnonzero(reset_pauli[:self.num_qubits])
            reset_z = np.flatnonzero(reset_pauli[self.num_qubits:])
            reset_support = set(reset_x) | set(reset_z)
            for other in range(full.count):
                if other == pivot:
                    continue
                has_same_factor = all(
                    full.matrix[other, qubit]
                    == reset_pauli[qubit]
                    and full.matrix[other, self.num_qubits + qubit]
                    == reset_pauli[self.num_qubits + qubit]
                    for qubit in reset_support
                )
                if has_same_factor:
                    full.update_row(other, pivot)

            # The old pivot records were transferred while eliminating this
            # qubit from the other rows. Reset now prepares the +1 eigenstate.
            full.records[pivot] = []

        self.stabilizers.matrix = full.matrix[:num_stabs].copy()
        self.stabilizers.records = [
            list(records)
            for records in full.records[:num_stabs]
        ]
        self.logicals.matrix = full.matrix[num_stabs:].copy()
        self.logicals.records = [
            list(records)
            for records in full.records[num_stabs:]
        ]

    def _replace_measured_ancillas_with_records(
        self,
        *,
        measurement_qubit_indices: List[int],
        measurement_bases: List[str],
        measurement_base_idx: int,
    ) -> None:
        """Eliminate terminal measured-ancilla factors from the current state.

        After forward propagation, a data stabilizer may carry the measured
        Pauli of a syndrome ancilla. The ancilla Pauli's eigenvalue is exactly
        the corresponding measurement result, so removing that factor toggles
        the absolute measurement record into the row's sign history.
        """
        if len(measurement_qubit_indices) != len(measurement_bases):
            raise ValueError(
                "Syndrome measurement qubits and bases must have equal length."
            )

        n = self.num_qubits
        for tableau_name, tableau in (
            ("stabilizer", self.stabilizers),
            ("logical", self.logicals),
        ):
            for row_idx in range(tableau.count):
                records = set(tableau.records[row_idx])

                for meas_idx, (qubit, basis) in enumerate(
                    zip(measurement_qubit_indices, measurement_bases)
                ):
                    has_x = bool(tableau.matrix[row_idx, qubit])
                    has_z = bool(tableau.matrix[row_idx, n + qubit])
                    if not has_x and not has_z:
                        continue

                    expected_x = basis == "X"
                    expected_z = basis == "Z"
                    if (has_x, has_z) != (expected_x, expected_z):
                        actual_basis = "Y" if has_x and has_z else "X" if has_x else "Z"
                        raise RuntimeError(
                            f"Forward-propagated {tableau_name} row {row_idx} has "
                            f"{actual_basis} on syndrome qubit {qubit}, but that qubit "
                            f"is measured in the {basis} basis."
                        )

                    tableau.matrix[row_idx, qubit] = 0
                    tableau.matrix[row_idx, n + qubit] = 0

                    record = measurement_base_idx + meas_idx
                    if record in records:
                        records.remove(record)
                    else:
                        records.add(record)

                tableau.records[row_idx] = sorted(records)

    def _record_measurement_logical_effects(
        self,
        surviving_logical_indices: Set[int],
        old_logicals_current_frame: Optional[np.ndarray] = None,
        old_logicals_records: Optional[List[List[int]]] = None,
    ) -> None:
        """Account for logical DOFs consumed by the current measurement block.

        The consumed combinations are RECORDED as operator rows in
        absorbed_ops (the single census ledger); the census count is always
        derived from that ledger via num_absorbed_dof().  Callers pass the
        pre-block logical rows already pushed into the current frame
        (i.e. after the block's forward symplectic), so the recorded
        operators live in the same frame as every other tableau row —
        together with those rows' RECORDS: a consumed record-pinned
        logical's seed records are the banked parity of the absorbed
        relation, and a later readout of the relation (terminal or
        measured_absorbed) emits seed XOR measuring records.  Dropping the
        seeds here would corrupt that parity invisibly to the census.
        """
        if self._gauge_logical_vectors:
            gauge_matrix = np.array(
                self._gauge_logical_vectors,
                dtype=np.uint8,
            ).copy()
            for logical_idx in surviving_logical_indices:
                if logical_idx < gauge_matrix.shape[1]:
                    gauge_matrix[:, logical_idx] = 0
            if gauge_matrix.any():
                if old_logicals_current_frame is None:
                    raise ValueError(
                        "_record_measurement_logical_effects: gauge "
                        "measurements consumed logical DOFs but the caller "
                        "did not supply the pre-block logical rows — the "
                        "absorbed operators cannot be recorded.")
                n_logs = old_logicals_current_frame.shape[0]
                ops = (gauge_matrix[:, :n_logs]
                       @ old_logicals_current_frame) % 2
                for gv, op in zip(gauge_matrix, ops):
                    recs: Set[int] = set()
                    if old_logicals_records is not None:
                        for j in np.flatnonzero(gv[:n_logs]):
                            recs.symmetric_difference_update(
                                old_logicals_records[int(j)])
                    self.record_absorbed_op(op, records=sorted(recs))

        if surviving_logical_indices and self._gauge_logical_vectors:
            rows_to_remove = set()
            for row_idx, logical_vector in zip(
                sorted(self.stabilizer_with_logical_components),
                self._gauge_logical_vectors,
            ):
                logical_indices = set(np.flatnonzero(logical_vector))
                if logical_indices.issubset(surviving_logical_indices):
                    rows_to_remove.add(row_idx)
            self.stabilizer_with_logical_components -= rows_to_remove

    def process_mid_measurement(
        self,
        circuit: stim.Circuit,
        forward_symplectic_matrix: np.ndarray,
        back_propagated_paulis: np.ndarray,
        reset_paulis: Optional[np.ndarray],
        measurement_qubit_indices: List[int],
        measurement_bases: List[str],
        measurement_coords: list,
        discarded_measurement_qubit_indices: Set[int],
        no_detector_mask: Optional[np.ndarray] = None,
    ) -> Set[int]:
        """
        Update the tracked state across one physical measurement block.

        Args:
            forward_symplectic_matrix: Clifford forward evolution of the
                measurement block, padded to the full system size.
            reset_paulis: Initial syndrome-ancilla reset stabilizers, ordered
                like the terminal syndrome measurements. ``None`` means the
                block has no fresh resets; its projected current tableau is
                forward-propagated to form the output state.
            no_detector_mask: Optional boolean array of length num_meas. When
                no_detector_mask[i] is True the measurement still updates the
                stabilizer tableau (Step 3) but no DETECTOR instruction is
                emitted for it (Step 2). Useful for Z-only / X-only memory
                experiments where one ancilla type is measured without detectors.

        Returns:
            Stabilizer row indices that may be promoted to logical rows if the
            Builder declares this block to be a complete classification
            boundary.

        This operation has no knowledge of a surrounding syndrome-extraction
        round. Candidate state constraints remain stabilizer rows until the
        Builder explicitly classifies or rebases them.
        """
        num_meas = back_propagated_paulis.shape[0]
        if (
            reset_paulis is not None
            and reset_paulis.shape != back_propagated_paulis.shape
        ):
            raise ValueError(
                "Reset and back-propagated Pauli tableaus must have the same shape; "
                f"got {reset_paulis.shape} and {back_propagated_paulis.shape}."
            )
        measured_qubit_set = set(measurement_qubit_indices)
        if discarded_measurement_qubit_indices not in (
            set(),
            measured_qubit_set,
        ):
            raise ValueError(
                "A measurement block cannot currently mix discarded syndrome "
                "ancillas with retained measured data qubits."
            )
        retain_measured_qubits = not discarded_measurement_qubit_indices
        current_base_idx = self.total_measurements
        self.total_measurements += num_meas

        # Reset per-round tracking (these are only meaningful within a single PMM call)
        self.stabilizer_with_logical_components = set()
        self._gauge_logical_vectors = []

        # Reorder stabilizer rows: empty-record rows first.
        # Fresh init rows (rec=[]) represent newly introduced qubits whose DOF should
        # be consumed first by measurements, before established stabilizers with SE records.
        # This produces cleaner detectors and correct logical observable construction.
        num_stabs_pre = self.stabilizers.count
        if num_stabs_pre > 0:
            empty_indices = [i for i in range(num_stabs_pre) if self.stabilizers.records[i] == []]
            other_indices = [i for i in range(num_stabs_pre) if self.stabilizers.records[i] != []]
            if empty_indices and other_indices:  # only reorder if there's a mix
                reorder = empty_indices + other_indices
                self.stabilizers.matrix = self.stabilizers.matrix[reorder]
                self.stabilizers.records = [self.stabilizers.records[i] for i in reorder]
                # Remap post_select_row_indices
                if self.post_select_row_indices:
                    idx_map = {old: new for new, old in enumerate(reorder)}
                    self.post_select_row_indices = {
                        idx_map.get(k, k) for k in self.post_select_row_indices
                    }

        # ======================================================================
        # Step 1: Combine Stabilizers and Logicals into Full Tableau
        # ======================================================================
        num_stabs = self.stabilizers.count
        num_logs = self.logicals.count

        # If logicals is empty, full_tableau is just stabilizers
        if num_logs > 0:
            full_matrix = np.vstack([self.stabilizers.matrix, self.logicals.matrix])
            # Flatten records list
            full_records = self.stabilizers.records + self.logicals.records
        else:
            full_matrix = self.stabilizers.matrix.copy() # Copy to avoid reference issues during loop
            full_records = list(self.stabilizers.records) # Deep copy of list structure

        # ======================================================================
        # Step 2: Process Back-propagated Pauli measurements (Update / Detector)
        # ======================================================================
        # Use a temporary tableau view so we can call update_row / replace_row.
        temp_full = PauliTableau(self.num_qubits)
        temp_full.matrix = full_matrix
        temp_full.records = full_records

        for i in range(num_meas):
            meas_pauli = back_propagated_paulis[i]
            meas_row = meas_pauli.reshape(1, -1)
            meas_abs_idx = current_base_idx + i

            # Check commutativity against existing stabilizers and logicals
            comm_check = check_commutativity(meas_row, full_matrix)
            anti_comm_indices = np.where(comm_check[0])[0]

            if len(anti_comm_indices) > 0:
                # --- Case A: Anti-commutes (State Update) ---
                pivot = anti_comm_indices[0]
                # Update other anti-commuting rows
                for other in anti_comm_indices[1:]:
                    temp_full.update_row(other, pivot) # (target, source)

                # Replace the pivot with the back_propagated_paulis
                temp_full.replace_row(pivot, meas_pauli, [meas_abs_idx])

                if pivot >= num_stabs:
                    # If the pivot is a logical operator and is replaced by a measurement, decreases one degree of freedom
                    self.expected_num_logicals -= 1

            else:
                # --- Case B: Commutes (Detector) ---
                # Detector is formed by decomposing Back-propagated Pauli Measurements into existing STABILIZERS only (rows in the stabilizer tableau).
                # (Logicals do not contribute to the decomposition)

                # A measurement that reduces to identity after removing known
                # reset-ancilla factors is a flag. Its expected value is fixed,
                # so the measurement record is a detector by itself.
                if not np.any(meas_pauli):
                    if no_detector_mask is None or not no_detector_mask[i]:
                        coords = list(measurement_coords[i]) + [0]
                        _append_detector(
                            circuit,
                            [stim.target_rec(meas_abs_idx - self.total_measurements)],
                            coords,
                            post_select=tuple(coords) in self.post_select_detector_coords,
                        )
                    continue

                if num_stabs > 0:
                    # First check if meas_row is exactly one row in curr_stab_matrix
                    # Directly compare meas_row against current stabilizer rows
                    curr_stab_matrix = full_matrix[:num_stabs]
                    matching_rows = np.where(np.all(curr_stab_matrix == meas_row, axis=1))[0]
                    if len(matching_rows) > 0:
                        # Raise warning if there are multiple identical stabilizer rows matching this measurement
                        if len(matching_rows) > 1:
                            warnings.warn(
                                f"Found {len(matching_rows)} identical stabilizer rows matching this measurement. "
                                "Check that the stabilizer tableau has no duplicate rows.",
                                UserWarning,
                                stacklevel=2,
                            )
                        # Directly construct the detector
                        row_idx = matching_rows[0]  # Take the first matching row
                        args = [stim.target_rec(meas_abs_idx - self.total_measurements)]
                        for r in full_records[row_idx]:
                            if r >= 0:
                                args.append(stim.target_rec(r - self.total_measurements))
                        args.sort(key=lambda target: target.value)
                        if no_detector_mask is None or not no_detector_mask[i]:
                            coords = list(measurement_coords[i]) + [0]
                            _append_detector(
                                circuit, args, coords,
                                post_select=tuple(coords) in self.post_select_detector_coords,
                            )
                    else: # meas_row is not exactly one row in curr_stab_matrix, but a linear combination of rows in the full matrix
                        # decompose meas_row into existing stabilizers
                        coeffs, is_dependent, _ = solve_linear_decomposition(
                            basis=full_matrix,
                            targets=meas_row
                        )
                        # Note: Here we use the full matrix as the basis, not just the stabilizer tableau.
                        # The back-propagated Pauli may contain present logical operator components. e.g., Logical ZZ over two |0> states, then
                        # the last Z gauge operator consisting ZZ measurements can be written as the linear combination of previous Z gauges and two logical Z operators of two patches.
                        # If we don't use the full matrix as the basis, these measurements will be identified as independent basis and treated as logicals, which is incorrect.
                        # However, these measurements, although they can be decomposed, cannot be detectors, because their logical operator components cannot be measured in the middle of the circuit
                        # and cannot give syndrome information. This will be identified when we construct detectors.

                        # Construct a detector
                        if is_dependent[0]:
                            args = [stim.target_rec(meas_abs_idx - self.total_measurements)]
                            comp_indices = np.where(coeffs[0])[0]
                            if max(comp_indices) >= num_stabs:
                                # The measurement contains a logical component, and cannot be a detector
                                # Flag this row for further logical observable construction
                                self.stabilizer_with_logical_components.add(i)
                                # Track which logical indices this measurement involves (for rank computation)
                                log_vec = np.zeros(num_logs, dtype=np.uint8)
                                for c in comp_indices:
                                    if c >= num_stabs:
                                        log_vec[c - num_stabs] = 1
                                self._gauge_logical_vectors.append(log_vec)
                                continue
                            # Otherwise, purely depends on stabilizers, construct a detector
                            # Use set-based XOR for O(1) toggle instead of O(n) list scan
                            args_set = set(args)
                            for c_idx in comp_indices:
                                # Map back to full records (skip UNMEASURED_STAB_RECORD)
                                for r in full_records[c_idx]:
                                    if r < 0:
                                        continue
                                    rec_to_append = stim.target_rec(r - self.total_measurements)
                                    if rec_to_append in args_set:
                                        args_set.remove(rec_to_append)
                                    else:
                                        args_set.add(rec_to_append)  # addition modulo 2
                            args = sorted(
                                args_set,
                                key=lambda target: target.value,
                            )

                            if no_detector_mask is None or not no_detector_mask[i]:
                                coords = list(measurement_coords[i]) + [0]
                                _append_detector(
                                    circuit, args, coords,
                                    post_select=tuple(coords) in self.post_select_detector_coords,
                                )
                        else:
                            # Measurement row commute but is independent of the current full tableau,
                            # but this should never happen in a well-defined full tableau, unless there are degrees of freedom missing.
                            raise RuntimeError(
                                f"Measurement {i} commutes with all rows in the current full tableau (stabilizers + logicals) but is linearly independent.\n"
                                f"This implies the Full (Stabilizer + Logicals) Tableau is incomplete (Rank < num_qubits).\n"
                                f"Please ensure all qubits are initialized and added to the tracker before measurement."
                            )

        # ======================================================================
        # Step 3: Write Back with "Clean" Basis
        # ======================================================================
        if retain_measured_qubits:
            # The projected measurement rows are the freshest representatives
            # of the post-measurement state. Keep an independent subset with
            # their current records, then complete the state basis from the
            # projected old tableau. Preserving the old basis verbatim would
            # leave commuting measurements anchored to stale records.
            empty_basis = np.zeros(
                (0, 2 * self.num_qubits),
                dtype=np.uint8,
            )
            _, _, measured_basis_indices = solve_linear_decomposition(
                basis=empty_basis,
                targets=back_propagated_paulis,
                reduce_weight=False,
            )
            measured_basis = back_propagated_paulis[
                measured_basis_indices
            ]
            measured_records = [
                [current_base_idx + idx]
                for idx in measured_basis_indices
            ]

            if num_logs > 0:
                reordered_targets = np.vstack([
                    full_matrix[num_stabs:],
                    full_matrix[:num_stabs],
                ])
            else:
                reordered_targets = full_matrix

            _, _, complement_indices = solve_linear_decomposition(
                basis=measured_basis,
                targets=reordered_targets,
                reduce_weight=False,
            )

            old_stab_basis_indices = []
            new_log_basis_indices = []
            for reordered_idx in complement_indices:
                if reordered_idx < num_logs:
                    new_log_basis_indices.append(
                        num_stabs + reordered_idx
                    )
                else:
                    old_stab_basis_indices.append(
                        reordered_idx - num_logs
                    )

            old_stab_matrix = full_matrix[
                old_stab_basis_indices
            ]
            new_stab_matrix = np.vstack([
                measured_basis,
                old_stab_matrix,
            ])
            self.stabilizers.matrix = new_stab_matrix
            self.stabilizers.records = measured_records + [
                list(full_records[idx])
                for idx in old_stab_basis_indices
            ]
            self.logicals.matrix = full_matrix[
                new_log_basis_indices
            ].copy()
            self.logicals.records = [
                list(full_records[idx])
                for idx in new_log_basis_indices
            ]

            if self.post_select_row_indices:
                new_post_select_rows = set()
                for old_idx in self.post_select_row_indices:
                    if old_idx >= num_stabs:
                        continue
                    coeffs, is_dependent, _ = solve_linear_decomposition(
                        basis=new_stab_matrix,
                        targets=full_matrix[old_idx:old_idx + 1],
                        reduce_weight=False,
                    )
                    if is_dependent[0]:
                        new_post_select_rows.update(
                            int(idx)
                            for idx in np.flatnonzero(coeffs[0])
                        )
                self.post_select_row_indices = new_post_select_rows

            self._apply_symplectic_matrix(forward_symplectic_matrix)
            surviving_logical_indices = {
                idx - num_stabs
                for idx in new_log_basis_indices
                if idx >= num_stabs
            }
            self._record_measurement_logical_effects(
                surviving_logical_indices,
                old_logicals_current_frame=(
                    full_matrix[num_stabs:] @ forward_symplectic_matrix) % 2,
                old_logicals_records=[
                    list(r) for r in full_records[num_stabs:]],
            )
            return set()

        # After detector construction, we always decompose the system into the "Clean Basis" of the measurements we just performed.
        # - Dependent rows in Full Tableau -> Replaced by Clean Measurements (Stabilizers).
        # - Independent rows in Full Tableau -> Identified as Logicals.
        # This automatically determines the right logicals, e.g. after first round of syndrome extraction

        # Basis: The clean measurements (Back-propagated Paulis)
        # Targets: The updated Full Tableau (System State)
        # new_basis_indices: Indices in full_matrix that form the Logical Basis.
        #
        # IMPORTANT: Reorder targets so logical rows come BEFORE stabilizer rows.
        # This gives logicals priority for RREF pivots, preventing stabilizer rows
        # from "stealing" a logical's pivot when they share the same independent direction.
        # Without this, logicals can be lost (e.g., PQRM logical Z in CrossLS Z-state).
        if num_logs > 0:
            reordered_targets = np.vstack([full_matrix[num_stabs:], full_matrix[:num_stabs]])
            reordered_records = full_records[num_stabs:] + full_records[:num_stabs]
        else:
            reordered_targets = full_matrix
            reordered_records = full_records

        _, is_dependent, new_basis_indices = solve_linear_decomposition(
            basis=back_propagated_paulis,
            targets=reordered_targets
        )

        # Map reordered indices back to original full_matrix semantics:
        # reordered[0..num_logs-1]       = original logicals  (full_matrix[num_stabs..end])
        # reordered[num_logs..num_logs+num_stabs-1] = original stabs (full_matrix[0..num_stabs-1])
        def _remap(idx):
            if idx < num_logs:
                return num_stabs + idx          # logical row
            else:
                return idx - num_logs            # stabilizer row

        # 1. Update explicit logical rows and preserve unclassified constraints.
        # Independent rows fall into:
        # (a) Logical rows: stay as logicals
        # (b) Stabilizer rows: remain stabilizers until Builder classification
        if len(new_basis_indices) > 0:
            new_log_basis_indices = []
            old_stab_basis_indices = []

            for ri in new_basis_indices:
                orig_idx = _remap(ri)
                if orig_idx >= num_stabs:
                    # Logical row → stays as logical
                    new_log_basis_indices.append(orig_idx)
                else:
                    # A physical measurement update cannot decide whether an
                    # unmeasured state constraint is a code stabilizer or an
                    # encoded logical state. Preserve it until the Builder
                    # requests a code-basis rebase.
                    old_stab_basis_indices.append(orig_idx)

            self.logicals.matrix = full_matrix[new_log_basis_indices]
            self.logicals.records = [full_records[i] for i in new_log_basis_indices]
        else:
            new_log_basis_indices = []
            old_stab_basis_indices = []
            self.logicals = PauliTableau(self.num_qubits)  # empty logicals

        # 2. Update Stabilizers
        # The back-propagated measurement basis describes the input boundary
        # and was used above for detector construction and state projection.
        # The output stabilizers instead originate from the freshly reset
        # syndrome ancillas. Forward propagation turns these reset factors into
        # data stabilizers carrying the terminal measurement results.
        new_stab_records = [[] for _ in range(num_meas)]

        # Build old_stab part: rows with records that stayed as stabilizers
        old_stab_matrix = full_matrix[old_stab_basis_indices]
        old_stab_records = [full_records[i] for i in old_stab_basis_indices]
        promotable_old_positions = [
            position
            for position, old_idx in enumerate(old_stab_basis_indices)
            if full_records[old_idx] == []
        ]

        self.stabilizers.matrix = np.vstack([reset_paulis, old_stab_matrix])
        self.stabilizers.records = new_stab_records + old_stab_records

        # The measurement update above occurred at the block's input boundary.
        # Evolve the reset-seeded output basis through the extraction circuit,
        # then eliminate terminal syndrome-ancilla factors using their outcomes.
        self._apply_symplectic_matrix(forward_symplectic_matrix)
        self._replace_measured_ancillas_with_records(
            measurement_qubit_indices=measurement_qubit_indices,
            measurement_bases=measurement_bases,
            measurement_base_idx=current_base_idx,
        )

        # Flag reset rows forward-propagate into measured ancilla factors only.
        # Once those factors are replaced by their records, the Pauli row is
        # identity and carries no quantum-state information. The detector above
        # already records its deterministic parity, so do not retain the
        # identity row in the stabilizer tableau.
        output_rows = self.stabilizers.matrix[:num_meas]
        kept_output_indices = np.flatnonzero(np.any(output_rows, axis=1)).tolist()
        measurement_to_output_row = {
            measurement_idx: output_idx
            for output_idx, measurement_idx in enumerate(kept_output_indices)
        }
        num_output_stabilizers = len(kept_output_indices)
        if num_output_stabilizers != num_meas:
            old_output_indices = list(range(num_meas, self.stabilizers.count))
            kept_indices = kept_output_indices + old_output_indices
            self.stabilizers.matrix = self.stabilizers.matrix[kept_indices]
            self.stabilizers.records = [
                self.stabilizers.records[idx]
                for idx in kept_indices
            ]
            self.stabilizer_with_logical_components = {
                measurement_to_output_row[idx]
                for idx in self.stabilizer_with_logical_components
                if idx in measurement_to_output_row
            }

        # Update post_select_row_indices: map old full_matrix indices to new stabilizer indices.
        # - Old stab rows in old_stab_basis_indices follow the retained output rows.
        # - Dependent rows (absorbed into measurement basis) → find which measurement rows captured them
        if self.post_select_row_indices:
            new_ps = set()
            # Map old stab rows that survived
            for j, old_idx in enumerate(old_stab_basis_indices):
                if old_idx in self.post_select_row_indices:
                    new_ps.add(num_output_stabilizers + j)
            # Map dependent rows by finding which measurement rows captured them.
            for old_idx in self.post_select_row_indices:
                if old_idx < full_matrix.shape[0]:
                    # Decompose against original full_matrix (not reordered)
                    row = full_matrix[old_idx:old_idx+1]
                    c, dep, _ = solve_linear_decomposition(basis=back_propagated_paulis, targets=row, reduce_weight=False)
                    if dep[0]:
                        meas_indices = np.where(c[0])[0]
                        new_ps.update(
                            measurement_to_output_row[int(m)]
                            for m in meas_indices
                            if int(m) in measurement_to_output_row
                        )
            self.post_select_row_indices = new_ps

        # Account for logical components measured by this physical block. The
        # Builder validates the final logical count at its chosen boundary.
        surviving_log_indices = set(
            i - num_stabs for i in new_log_basis_indices
            if i >= num_stabs
        ) if new_log_basis_indices else set()
        self._record_measurement_logical_effects(
            surviving_log_indices,
            old_logicals_current_frame=(
                full_matrix[num_stabs:] @ forward_symplectic_matrix) % 2,
            old_logicals_records=[
                list(r) for r in full_records[num_stabs:]],
        )
        return {
            num_output_stabilizers + position
            for position in promotable_old_positions
        }

    def promote_stabilizer_rows_to_logicals(
        self,
        row_indices: Set[int],
    ) -> None:
        """Move selected tracked stabilizer rows into the logical table.

        This is a basis-classification operation, not a measurement update.
        ``process_mid_measurement`` identifies rows that are eligible for this
        classification, and the Builder chooses whether its physical protocol
        has reached a boundary where they should be promoted.
        """
        promoted_indices = sorted(row_indices)
        invalid_indices = [
            idx
            for idx in promoted_indices
            if idx < 0 or idx >= self.stabilizers.count
        ]
        if invalid_indices:
            raise IndexError(
                "Cannot promote stabilizer rows outside the tracked tableau: "
                f"{invalid_indices}."
            )
        if not promoted_indices:
            self.validate_logical_count(
                context="stabilizer-row classification"
            )
            return

        promoted_set = set(promoted_indices)
        kept_indices = [
            idx
            for idx in range(self.stabilizers.count)
            if idx not in promoted_set
        ]
        promoted_matrix = self.stabilizers.matrix[promoted_indices]
        promoted_records = [
            self.stabilizers.records[idx]
            for idx in promoted_indices
        ]

        if self.logicals.count:
            self.logicals.matrix = np.vstack([
                self.logicals.matrix,
                promoted_matrix,
            ])
            self.logicals.records.extend(promoted_records)
        else:
            self.logicals.matrix = promoted_matrix.copy()
            self.logicals.records = promoted_records

        old_to_new = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(kept_indices)
        }
        if kept_indices:
            self.stabilizers.matrix = self.stabilizers.matrix[kept_indices]
        else:
            self.stabilizers.matrix = np.zeros(
                (0, 2 * self.num_qubits),
                dtype=np.uint8,
            )
        self.stabilizers.records = [
            self.stabilizers.records[idx]
            for idx in kept_indices
        ]
        self.post_select_row_indices = {
            old_to_new[idx]
            for idx in self.post_select_row_indices
            if idx in old_to_new
        }
        self.stabilizer_with_logical_components = {
            old_to_new[idx]
            for idx in self.stabilizer_with_logical_components
            if idx in old_to_new
        }

        self.validate_logical_count(
            context="stabilizer-row classification"
        )

    def validate_logical_count(self, *, context: str = "tracker state") -> None:
        """Check that explicit and measurement-absorbed logical DOFs add up."""
        absorbed = self.num_absorbed_dof()
        actual = self.logicals.count + absorbed
        if actual != self.expected_num_logicals:
            raise RuntimeError(
                f"After {context}: logical count {self.logicals.count} plus "
                f"absorbed logical DOFs {absorbed} != "
                f"expected {self.expected_num_logicals}."
            )

    def rebase_stabilizers_onto_code_basis(
        self,
        system: Any,
        stabilizer_uids: Optional[Set[int]] = None,
    ) -> None:
        """Re-express tracked stabilizers in the active code's canonical basis.

        Every requested code stabilizer must already be in the current tracked
        stabilizer span. Reconstructing the canonical rows from that span also
        reconstructs their measurement-record parities. Independent remaining
        directions become the logical tableau.

        This differs from :meth:`stabilizer_canonicalization`, which may insert
        unmeasured canonical rows when preparing a code for its first SE round.
        """
        self._reject_pending_row_metadata("rebase_stabilizers_onto_code_basis")
        n = self.num_qubits
        if stabilizer_uids is None:
            stabilizer_uids = set(system.active_stabilizer_indices)
        stab_dicts = [
            system.stabilizers[uid]
            for uid in sorted(stabilizer_uids)
        ]
        canonical_basis = stabilizers_to_symplectic(system, stab_dicts, n)
        if canonical_basis.shape[0] == 0:
            return
        if self.stabilizers.count == 0:
            raise RuntimeError(
                "Cannot rebase an empty stabilizer tableau onto the code basis."
            )

        canonical_coeffs, is_dependent, _ = solve_linear_decomposition(
            basis=self.stabilizers.matrix,
            targets=canonical_basis,
            reduce_weight=False,
        )
        missing = np.flatnonzero(~is_dependent)
        if len(missing):
            raise RuntimeError(
                "Cannot rebase onto the active code basis; canonical rows "
                f"{missing.tolist()} are missing from the tracker stabilizer span."
            )

        canonical_records = []
        for coeffs in canonical_coeffs:
            records = set()
            for row_idx in np.flatnonzero(coeffs):
                records.symmetric_difference_update(
                    self.stabilizers.records[row_idx]
                )
            canonical_records.append(sorted(records))

        if self.logicals.count:
            full_matrix = np.vstack([
                self.stabilizers.matrix,
                self.logicals.matrix,
            ])
            full_records = self.stabilizers.records + self.logicals.records
        else:
            full_matrix = self.stabilizers.matrix.copy()
            full_records = list(self.stabilizers.records)

        _, _, logical_indices = solve_linear_decomposition(
            basis=canonical_basis,
            targets=full_matrix,
            reduce_weight=False,
        )
        self.stabilizers.matrix = canonical_basis
        self.stabilizers.records = canonical_records
        self.logicals.matrix = full_matrix[logical_indices]
        self.logicals.records = [
            full_records[idx]
            for idx in logical_indices
        ]

        self.validate_logical_count(context="code-basis rebase")


    def process_data_measurement(self,
                                  circuit: stim.Circuit,
                                  final_paulis: np.ndarray,
                                  idx_to_coord_map: Dict[int, Tuple[float, float]],
                                  syndrome_qubit_indices: set = None,
                                  resolve_absorbed: bool = True):
        """
        Handles Final Data Qubit Measurements using Gaussian Elimination.

        Args:
            circuit: Stim circuit to append to.
            final_paulis: (M, 2N) numpy array. The measurement basis.
                          Does NOT need to be single-qubit Paulis (can be general).
            idx_to_coord_map: Mapping from qubit index to coordinate. Determines
                the coordinate of the detector in the decoding graph.
            syndrome_qubit_indices: Optional set of syndrome (ancilla) qubit indices.
                When provided, historical measurement records that map to a syndrome
                qubit take coordinate priority over data qubit fallback — aligning
                final-round detectors with the syndrome grid rather than the data grid.
                Coordinates are deduplicated: if a syndrome coord is already used by
                a previous detector, the data qubit fallback is applied instead.
        """

        num_new_meas = final_paulis.shape[0]
        base_meas_idx = self.total_measurements
        self.total_measurements += num_new_meas

        # Budget delta (independent, matrix-driven): a PATCH readout that reads out a live
        # logical removes exactly that many DOFs from the live set. Snapshot the free
        # (standing) and absorbed counts BEFORE the readout mutates them; at the end we
        # decrement `expected_num_logicals` by however many were actually resolved. This is
        # a pure delta — it never re-syncs expected to standing+absorbed, so `expected`
        # stays an independent budget and a prior matrix discrepancy is preserved for the
        # guardrail to catch rather than silently erased.
        _std_before = self.logicals.count
        _abs_before = self.num_absorbed_dof()

        # Fix C: reconcile absorbed_ops with this readout.
        #   - PATCH readout (resolve_absorbed=True): the readout reads out a patch's logical
        #     value, so any absorbed relation supported on the measured qubits is RECORDED —
        #     drop those rows from absorbed_ops.
        #   - CORRIDOR/bus readout (resolve_absorbed=False): only the bus is measured out; the
        #     patch-patch relation persists. But an absorbed op stored on the bus qubits would
        #     otherwise dangle on measured-out columns, so FOLD the bus support out (zero the
        #     measured columns), leaving the patch-part of the relation. A later patch readout
        #     then resolves that patch-part.
        A = self.absorbed_ops
        if A.count and num_new_meas:
            n = self.num_qubits
            meas_q = set(int(c % n) for c in np.where(final_paulis.any(axis=0))[0])
            mcols = [q for q in meas_q] + [q + n for q in meas_q]
            if resolve_absorbed:
                # A resolve readout must cover every banked relation it
                # touches COMPLETELY.  A partial cover either loses the
                # unmeasured remainder (dropping the row wholesale) or needs
                # it re-banked against a standing logical — per-patch
                # readout semantics that belong to the future liveness/reuse
                # layer.  No production flow reads a strict subset of a
                # banked relation's support (main and the PPM path read out
                # full patch sets; corridor readouts take the
                # resolve_absorbed=False branch), so this fails loud instead
                # of inventing discard semantics here.
                keep = []
                _fold_basis = np.vstack([self.stabilizers.matrix,
                                         final_paulis])
                for r in range(A.count):
                    if not A.matrix[r, mcols].any():
                        keep.append(r)
                        continue
                    _cf, _depf, _ = solve_linear_decomposition(
                        basis=_fold_basis[:, mcols],
                        targets=A.matrix[r:r + 1][:, mcols],
                        reduce_weight=False)
                    _rem = None
                    if _depf[0]:
                        _comb = (_cf[0][None, :] @ _fold_basis) % 2
                        _rem = ((A.matrix[r] + _comb[0]) % 2).astype(np.uint8)
                        _rem[mcols] = 0
                    if not _depf[0] or _rem.any():
                        raise RuntimeError(
                            f"patch readout covers only part of banked "
                            f"absorbed relation {r}: the unmeasured "
                            f"remainder would be lost, and re-banking it "
                            f"is per-patch readout semantics that live "
                            f"with the future liveness layer.  Read out "
                            f"the relation's full support, or fold it off "
                            f"the measured qubits via a corridor readout "
                            f"(resolve_absorbed=False) first.")
                    # fully determined by this readout: resolved; the
                    # budget delta below retires the slot
            else:
                A.matrix = A.matrix.copy()
                A.records = [list(rr) for rr in A.records]
                # An absorbed relation is only defined MOD the stabilizer group; a
                # stored rep may sit entirely on bus qubits (e.g. the joint of a
                # colour-wall corridor reduces to a weight-2 bus operator).  Before
                # folding the bus columns out, re-express each touched row off the
                # measured qubits wherever the group allows — the patch-patch
                # relation survives the bus readout.  Every fold is a change of
                # REPRESENTATIVE: the folded row's records must ride along
                # (stabilizer rows' records, or the readout's measurement
                # records), same contract as reset_records_for_qubits — a
                # silent truncation corrupts the banked parity invisibly to
                # both the census (rank ignores records) and p=0 sampling.
                touched = [r for r in range(A.count) if A.matrix[r, mcols].any()]
                if touched and self.stabilizers.count:
                    coeffs, dep, _ = solve_linear_decomposition(
                        basis=self.stabilizers.matrix[:, mcols],
                        targets=A.matrix[np.ix_(touched, mcols)],
                        reduce_weight=False)
                    for k, r in enumerate(touched):
                        if dep[k]:
                            folded_recs = set(A.records[r])
                            for s in np.flatnonzero(coeffs[k]):
                                s_recs = self.stabilizers.records[int(s)]
                                if UNMEASURED_STAB_RECORD in s_recs:
                                    raise RuntimeError(
                                        f"corridor readout: re-expressing "
                                        f"absorbed relation {r} off the bus "
                                        f"needs stabilizer row {int(s)}, "
                                        f"whose records already carry the "
                                        f"UNMEASURED sentinel — the "
                                        f"relation's parity cannot be "
                                        f"reconstructed.")
                                folded_recs.symmetric_difference_update(
                                    s_recs)
                            comb = (coeffs[k][None, :]
                                    @ self.stabilizers.matrix) % 2
                            A.matrix[r] = (A.matrix[r] + comb[0]) % 2
                            A.records[r] = sorted(folded_recs)
                # Residual bus support the group could not cancel: fold it
                # against THIS readout's measured Paulis, XORing the matching
                # measurement records in.  A residue outside the readout's
                # span anticommutes with (or is undetermined by) the bus
                # readout — the banked parity is destroyed: fail loud.  A row
                # fully consumed by the fold lies entirely in the measured
                # bus: a corridor readout resolves no absorbed relation, so
                # silently dropping it would leak the DOF from the books
                # (the census then misfires at an unrelated later
                # checkpoint) — fail loud and let the caller route it as a
                # patch readout instead.
                still = [r for r in touched if A.matrix[r, mcols].any()]
                if still:
                    cf2, dep2, _ = solve_linear_decomposition(
                        basis=final_paulis[:, mcols],
                        targets=A.matrix[np.ix_(still, mcols)],
                        reduce_weight=False)
                    for k, r in enumerate(still):
                        if not dep2[k]:
                            raise RuntimeError(
                                f"corridor readout leaves absorbed relation "
                                f"{r} with bus support that neither the "
                                f"stabilizer group nor the readout basis "
                                f"determines — the banked parity would be "
                                f"corrupted.")
                        folded_recs = set(A.records[r])
                        for j in np.flatnonzero(cf2[k]):
                            folded_recs.symmetric_difference_update(
                                {base_meas_idx + int(j)})
                        comb = (cf2[k][None, :] @ final_paulis) % 2
                        newrow = ((A.matrix[r] + comb[0]) % 2).astype(
                            np.uint8)
                        if not newrow.any():
                            raise RuntimeError(
                                f"corridor readout (resolve_absorbed=False) "
                                f"would fully consume banked relation {r} "
                                f"(its support lies inside the measured "
                                f"bus); a bus readout resolves no absorbed "
                                f"relation — route this readout with "
                                f"resolve_absorbed=True instead.")
                        A.matrix[r] = newrow
                        A.records[r] = sorted(folded_recs)
                A.matrix = A.matrix.astype(np.uint8)
                keep = [r for r in range(A.count) if A.matrix[r].any()]
            if len(keep) != A.count:
                A.matrix = A.matrix[keep] if keep else np.zeros((0, 2 * n), dtype=np.uint8)
                A.records = [A.records[r] for r in keep] if A.records else []


        num_stabs = self.stabilizers.count
        num_logs = self.logicals.count

        # ======================================================================
        # Step 1: Combine Full Tableau
        # ======================================================================
        if num_logs > 0:
            full_matrix = np.vstack([self.stabilizers.matrix, self.logicals.matrix])
            full_records = self.stabilizers.records + self.logicals.records
        else:
            full_matrix = self.stabilizers.matrix.copy()
            full_records = list(self.stabilizers.records)

        # ======================================================================
        # Step 2: Update Tableau (Resolve Anti-commutation)
        # ======================================================================
        # Use a temporary tableau view so we can call update_row / replace_row.
        temp_full = PauliTableau(self.num_qubits)
        temp_full.matrix = full_matrix
        temp_full.records = full_records

        # We MUST track which rows represent stabilizers that are destroyed by the measurement.
        # A destroyed stabilizer cannot form a deterministic detector.
        destroyed_rows = set()

        for i in range(num_new_meas):
            meas_pauli = final_paulis[i]
            meas_row = meas_pauli.reshape(1, -1)
            meas_abs_idx = base_meas_idx + i

            comm_check = check_commutativity(meas_row, full_matrix)
            anti_comm_indices = [j for j in np.where(comm_check[0])[0] if j not in destroyed_rows]

            if len(anti_comm_indices) > 0:
                # Prefer rows that are: (1) not logicals, (2) not swlc rows
                # This preserves logicals and gauge-measurement rows for observable construction.
                safe = [j for j in anti_comm_indices
                        if j < num_stabs and j not in self.stabilizer_with_logical_components]
                if safe:
                    pivot = safe[0]
                else:
                    stab_candidates = [j for j in anti_comm_indices if j < num_stabs]
                    pivot = stab_candidates[0] if stab_candidates else anti_comm_indices[0]
                destroyed_rows.add(pivot) # Mark pivot as destroyed

                for other in anti_comm_indices[1:]:
                    temp_full.update_row(other, pivot)

                # Replace pivot to maintain valid tableau for subsequent loop steps
                temp_full.replace_row(pivot, meas_pauli, [meas_abs_idx])

        # ======================================================================
        # Step 3: Decomposition, Detectors/Logical Observables Construction
        # ======================================================================
        # Basis: The Final Measurements we just performed.
        # Targets: The Updated System State (Stabilizers + Logicals).
        # reduce_weight=False: detector/observable construction only needs correct
        #   linear combination, not minimal-weight; _greedy_reduce_weight is O(k^2)
        #   and can dominate runtime for large codes (e.g. BB [[144,12,12]]).
        coeffs, is_dependent, _ = solve_linear_decomposition(
            basis=final_paulis,
            targets=full_matrix,
            reduce_weight=False,
        )

        num_rows = full_matrix.shape[0]
        _used_final_coords: set = set()  # dedup: each final detector gets a unique (x,y) coord
        for k in range(num_rows):
            # Condition 1: Must NOT be destroyed (anti-commuted).
            if k in destroyed_rows:
                continue

            # Condition 2: Must be fully determined by the measurements (Linear Dependent).
            if not is_dependent[k]:
                continue

            # --- Construct Detector / Observable ---
            args = []

            # 1. Measurement Components (The decomposition result)
            basis_indices = np.where(coeffs[k])[0]
            for b_idx in basis_indices:
                stim_rec_target = b_idx - num_new_meas
                args.append(stim.target_rec(stim_rec_target))

            # 2. Historical Record Components — use set-based XOR for O(1) toggle
            det_coord = None       # best coord from data-qubit Pauli support
            syndrome_coord = None  # coord from a syndrome qubit historical record
            args_set = set(args)
            row_k = full_matrix[k]
            n_q = self.num_qubits
            for r in full_records[k]:
                if r < 0:
                    continue
                rec_to_append = stim.target_rec(r - self.total_measurements)
                if rec_to_append in args_set:
                    args_set.remove(rec_to_append)
                else:
                    args_set.add(rec_to_append)
                qubit_idx = self.meas_rec_to_idx_map.get(r)
                if qubit_idx is not None and qubit_idx in idx_to_coord_map:
                    if syndrome_qubit_indices and qubit_idx in syndrome_qubit_indices:
                        # Syndrome qubit: record directly which stabilizer was measured.
                        # Syndrome qubits are NOT in the stabilizer Pauli support, so we
                        # skip the support check and track this coord separately.
                        syndrome_coord = idx_to_coord_map[qubit_idx]
                    else:
                        # Data / other qubit: only use if in this row's Pauli support
                        # (avoids inheriting coords from unrelated qubits via row updates).
                        if row_k[qubit_idx] or row_k[n_q + qubit_idx]:
                            det_coord = idx_to_coord_map[qubit_idx]
            args = list(args_set)

            # Fallback when no data-qubit coord found via support check
            if det_coord is None:
                row = full_matrix[k]
                n = self.num_qubits
                first_support = next(
                    (i for i in range(n) if row[i] or row[n + i]),
                    None,
                )
                det_coord = (
                    idx_to_coord_map[first_support]
                    if first_support is not None and first_support in idx_to_coord_map
                    else next(iter(idx_to_coord_map.values()), (0, 0))
                )

            # Prefer syndrome coord (aligns final detectors with the syndrome grid).
            # Fall back to data coord if this syndrome position is already taken.
            if syndrome_coord is not None and syndrome_coord not in _used_final_coords:
                det_coord = syndrome_coord

            # 3. Output:
            if k < num_stabs and k not in self.stabilizer_with_logical_components:
                # Measurement-promoted joint closures (support > check weight)
                # are emitted like every other row, matching upstream main:
                # the long-range parity is real syndrome information (paired-
                # noise MWPM LER is ~30% better with it; review blocker #1).
                if -1 in full_records[k]:
                    # An UNWATCHED gauge direction's close-out (sentinel-
                    # tagged row = WriteBack's no-slot gauge branch).  Its
                    # only content is init-parity == readout-parity with
                    # nothing re-measured in between; per the gauge-branch
                    # rule such relations must not become detectors - each
                    # error along the string is already a 2-symptom event
                    # on neighbouring checks, and this global parity would
                    # attach an extra, irreducible symptom to every one of
                    # them.  Same user ruling as above.
                    continue
                _used_final_coords.add(det_coord)
                coords = list(det_coord) + [1]
                _append_detector(
                    circuit, args, coords,
                    post_select=(tuple(coords) in self.post_select_detector_coords
                                 or k in self.post_select_row_indices),
                )
            else:
                if any(r < 0 for r in full_records[k]):
                    # Same guard as the stab-row branch above: an
                    # path's measured-out refusal: a sentinel record means
                    # this row's parity was never banked, so the observable
                    # cannot be constructed — emitting it with the sentinel
                    # silently skipped would publish a wrong parity.
                    raise RuntimeError(
                        f"process_data_measurement: row {k} closes into a "
                        f"logical observable but its records carry the "
                        f"UNMEASURED sentinel — its parity was never "
                        f"banked and the observable cannot be emitted.")
                circuit.append("OBSERVABLE_INCLUDE", args,
                               [self.allocate_observable()])

        # ======================================================================
        # Step 4: Partial reduction of remaining rows (incremental support)
        # ======================================================================
        # Rows that commute with measurements but share support on measured
        # qubits (e.g. Z_L = Z1*Z2*Z3 when Z2,Z3 are measured in Z basis)
        # must be reduced by XOR-ing out the measurement Paulis. This ensures
        # remaining rows only depend on unmeasured qubits, enabling correct
        # decomposition in subsequent process_data_measurement calls.
        emitted_rows = {k for k in range(num_rows)
                        if k not in destroyed_rows and is_dependent[k]}
        rows_to_remove = destroyed_rows | emitted_rows

        # Build pivot map: for each single-qubit measurement, identify its
        # pivot column (the nonzero column in the 2n symplectic vector).
        # MZ(q) -> pivot at n+q; MX(q) -> pivot at q; MY(q) -> pivot at q.
        n = self.num_qubits
        meas_pivot_map = {}  # pivot_col -> (measurement index i, meas Pauli vector)
        for i in range(num_new_meas):
            meas = final_paulis[i]
            nonzero_cols = np.where(meas)[0]
            if len(nonzero_cols) == 0:
                continue
            # Use the first nonzero column as pivot
            pivot_col = int(nonzero_cols[0])
            meas_pivot_map[pivot_col] = (i, meas)

        for k in range(num_rows):
            if k in rows_to_remove:
                continue
            row = full_matrix[k]
            for pivot_col, (i, meas) in meas_pivot_map.items():
                if row[pivot_col]:
                    # XOR measurement Pauli out of this row
                    full_matrix[k] = row = row ^ meas
                    # Update records: symmetric difference with measurement index
                    meas_abs_idx = base_meas_idx + i
                    rec_set = set(full_records[k])
                    rec_set.symmetric_difference_update({meas_abs_idx})
                    full_records[k] = list(rec_set)

        # ======================================================================
        # Step 5: Persist updated tableau state
        # ======================================================================
        remaining_stab_rows = [k for k in range(num_stabs) if k not in rows_to_remove]
        remaining_log_rows = [k for k in range(num_stabs, num_rows) if k not in rows_to_remove]

        if remaining_stab_rows:
            self.stabilizers.matrix = full_matrix[remaining_stab_rows]
            self.stabilizers.records = [full_records[k] for k in remaining_stab_rows]
        else:
            self.stabilizers.matrix = np.zeros((0, 2 * self.num_qubits), dtype=np.uint8)
            self.stabilizers.records = []

        if remaining_log_rows:
            self.logicals.matrix = full_matrix[remaining_log_rows]
            self.logicals.records = [full_records[k] for k in remaining_log_rows]
        else:
            self.logicals.matrix = np.zeros((0, 2 * self.num_qubits), dtype=np.uint8)
            self.logicals.records = []

        # Rows consumed by this readout leave; surviving rows keep their
        # post-select marking for any later readout at their new positions.
        self._remap_rows_after_removal(
            [k for k in range(num_stabs) if k in rows_to_remove])
        # Reset per-call tracking that uses row indices (now invalidated)
        self.stabilizer_with_logical_components = set()
        self._gauge_logical_vectors = []

        # A PATCH readout retires however many live logical DOFs it actually read out
        # (free ones consumed + absorbed relations resolved). A CORRIDOR/bus readout
        # (resolve_absorbed=False) resolves no ABSORBED relation (their support is
        # reduced onto the patches, not read out) — but a LOGICAL row REMOVED by it
        # is a consumed DOF all the same: a restore-credit promotion can put a
        # corridor-ribbon direction into the logicals, and the split then destroys
        # it (anticommuting bus-basis readout) or emits it (row entirely on the
        # measured corridor).  Legacy corridor splits removed no logical rows, so
        # this term was always zero before.
        if resolve_absorbed:
            resolved = ((_std_before - self.logicals.count)
                        + (_abs_before - self.num_absorbed_dof()))
            self.expected_num_logicals -= resolved
        else:
            self.expected_num_logicals -= (_std_before - self.logicals.count)