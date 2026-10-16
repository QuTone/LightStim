"""Record-aware row-space operations for the S-bare/S-dressed tracker.

Every returned eigenvalue expression is transformed from actual input rows.
An active-code declaration K supplies a role space, never measurement values.
All lineages below act on original input rows over GF(2).
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

import numpy as np

from lightstim.utils.linear_algebra import kernel_gf2, solve_linear_decomposition
from lightstim.utils.subsystem_algebra import (
    combine_records,
    independent_row_indices,
    intersect_row_spaces,
    reduce_modulo,
)


@dataclass
class Rows:
    matrix: np.ndarray
    records: Sequence[Sequence[int]]

    def __post_init__(self):
        self.matrix = np.array(self.matrix, dtype=np.uint8, copy=True) % 2
        if self.matrix.ndim != 2 or self.matrix.shape[1] % 2:
            raise ValueError("Pauli rows require a two-dimensional even-width matrix")
        if len(self.records) != len(self.matrix):
            raise ValueError("One record expression is required for every Pauli row")
        # A negative record is an unknown placeholder, not an eigenvalue that
        # can be canceled by an ordinary GF(2) combination with another -1.
        if any(int(record) < 0 for row in self.records for record in row):
            raise ValueError("Only known rows are accepted; an unknown sentinel cannot supply an eigenvalue")
        clean = []
        for row in self.records:
            parity = set()
            for record in row:
                record = int(record)
                if record in parity:
                    parity.remove(record)
                else:
                    parity.add(record)
            clean.append(tuple(sorted(parity)))
        self.records = tuple(clean)

    @classmethod
    def empty(cls, width):
        return cls(np.zeros((0, width), dtype=np.uint8), ())

    @property
    def count(self):
        return len(self.matrix)


@dataclass
class KnownRows:
    known: Rows
    known_indices: Tuple[int, ...]
    unknown_indices: Tuple[int, ...]


def split_known_rows(tableau):
    """Separate unknown placeholders without using them in record algebra.

    Indices refer to the caller's unchanged tableau. The caller retains the
    unknown rows and decides how to handle their physical preparation role.
    """
    matrix = np.asarray(tableau.matrix, dtype=np.uint8)
    if matrix.ndim != 2 or len(matrix) != len(tableau.records):
        raise ValueError("One record expression is required for every Pauli row")
    known = tuple(i for i, records in enumerate(tableau.records)
                  if all(record >= 0 for record in records))
    unknown = tuple(i for i, records in enumerate(tableau.records)
                    if any(record < 0 for record in records))
    return KnownRows(Rows(matrix[list(known)], [tableau.records[i] for i in known]),
                     known, unknown)


@dataclass
class RecordDependencies:
    """Pauli-zero row combinations; nonempty records must not vanish silently.

    These are diagnostics, not newly emitted relations. A caller must know
    whether their record parity has already been archived or emitted.
    """
    lineage: np.ndarray
    records: Tuple[Tuple[int, ...], ...]


@dataclass
class ClassifiedStabilizers:
    bare: Rows
    dressed: Rows
    bare_lineage: np.ndarray
    dressed_lineage: np.ndarray
    # Concatenated output S row indices. Exact selected input rows are retained
    # even if several have the same Pauli or dependent quotient classes.
    exact_postselection_map: Dict[int, int]
    safely_postselected_output_rows: FrozenSet[int]
    original_row_map: Dict[int, int]
    dressed_quotient_rank: int
    source_dependencies: Optional[RecordDependencies]


@dataclass
class ClosureRows:
    """A Pauli closure with two independent, checkable reconstructions."""
    rows: Rows
    state_lineage: np.ndarray
    measurement_lineage: np.ndarray
    parities: Tuple[Tuple[int, ...], ...]


@dataclass
class ReadoutClosures:
    detectors: ClosureRows
    observables: ClosureRows
    postselection: ClosureRows
    source_dependencies: Optional[RecordDependencies]


def _rank(matrix):
    return len(independent_row_indices(matrix))


def _check_width(width, *matrices):
    if any(matrix.ndim != 2 or matrix.shape[1] != width for matrix in matrices):
        raise ValueError("All Pauli matrices must use the same register width")


def _decompose(basis, targets):
    if not len(targets):
        return np.zeros((0, len(basis)), dtype=np.uint8)
    coefficients, known, _ = solve_linear_decomposition(
        basis=basis, targets=targets, reduce_weight=False)
    if not np.all(known):
        raise ValueError(f"Rows {np.flatnonzero(~known).tolist()} lie outside the declared source span")
    return coefficients.astype(np.uint8)


def _annotated(matrix, lineage, records):
    return Rows(matrix, combine_records(lineage, records))


def _known(rows):
    return rows if isinstance(rows, Rows) else Rows(rows.matrix, rows.records)


def record_dependencies(rows):
    """Return all Pauli-zero input combinations and their record parities."""
    rows = _known(rows)
    if not len(rows.matrix):
        lineage = np.zeros((0, 0), dtype=np.uint8)
    else:
        lineage = kernel_gf2(rows.matrix.T)
    return RecordDependencies(
        lineage, tuple(tuple(row) for row in combine_records(lineage, rows.records)))


def _indices(indices, count):
    result = tuple(sorted(set(int(i) for i in indices)))
    if any(i < 0 or i >= count for i in result):
        raise IndexError("Postselected row index lies outside the supplied S table")
    return result


def classify_stabilizers(S, K, *, postselected_rows=(), include_dependencies=False):
    """Split S into its full intersection with K and a quotient complement.

    K is the active role space in this exact physical frame. L and absorbed
    history are deliberately absent: neither can supply S eigenvalues.
    Unknown code declarations must be separated with ``split_known_rows``
    rather than assigned an empty record expression.

    All original bare representatives are kept first, in their input order,
    including dependent rows, to preserve local checks and postselection.
    The full intersection supplies any missing bare directions. Dressed rows
    complete the physical span; exact selected dressed rows are also kept.
    Thus a group may be redundant: rank, not row count, is its dimension.
    Additional rows with lineage supported entirely on the selected input
    subspace can safely inherit postselection. Mixed ancestry is not OR-tagged.
    """
    S = _known(S)
    width = S.matrix.shape[1]
    K = np.asarray(K, dtype=np.uint8) % 2
    _check_width(width, K)
    selected = _indices(postselected_rows, len(S.matrix))
    identity = np.eye(len(S.matrix), dtype=np.uint8)

    residues = reduce_modulo(S.matrix, K)
    if not residues.any():
        # Every original row is already bare, so preserving those rows also
        # completes the entire intersection. Neither intersection elimination
        # nor a complementary basis can add a direction in this common case.
        # Copy the matrix just as the general path does; callers may mutate S.
        return ClassifiedStabilizers(
            bare=Rows(S.matrix, S.records),
            dressed=Rows.empty(width),
            bare_lineage=identity,
            dressed_lineage=np.zeros((0, len(S.matrix)), dtype=np.uint8),
            exact_postselection_map={i: i for i in selected},
            safely_postselected_output_rows=frozenset(selected),
            original_row_map={i: i for i in range(len(S.matrix))},
            dressed_quotient_rank=0,
            source_dependencies=record_dependencies(S) if include_dependencies else None,
        )

    bare_all, bare_all_lineage = intersect_row_spaces(S.matrix, K)
    original_bare = [int(i) for i in np.flatnonzero(~residues.any(axis=1))]
    selected_dressed = [i for i in selected if residues[i].any()]
    exact_bare = S.matrix[original_bare]
    bare_candidates = np.vstack([exact_bare, bare_all])
    additional_bare = [i - len(exact_bare) for i in independent_row_indices(bare_candidates)
                       if i >= len(exact_bare)]
    bare_matrix = np.vstack([exact_bare, bare_all[additional_bare]])
    bare_lineage = np.vstack([identity[original_bare], bare_all_lineage[additional_bare]])

    exact_dressed = S.matrix[selected_dressed]
    prefix = np.vstack([bare_matrix, exact_dressed])
    additional_dressed = [i - len(prefix) for i in independent_row_indices(
        np.vstack([prefix, S.matrix])) if i >= len(prefix)]
    dressed_matrix = np.vstack([exact_dressed, S.matrix[additional_dressed]])
    dressed_lineage = np.vstack([identity[selected_dressed], identity[additional_dressed]])
    bare = _annotated(bare_matrix, bare_lineage, S.records)
    dressed = _annotated(dressed_matrix, dressed_lineage, S.records)

    output_lineage = np.vstack([bare_lineage, dressed_lineage])
    original_map = {old: new for new, old in enumerate(original_bare)}
    original_map.update({old: len(bare.matrix) + new
                        for new, old in enumerate(selected_dressed + additional_dressed)})
    exact_map = {old: original_map[old] for old in selected}
    selected_set = set(selected)
    safely_selected = frozenset(i for i, coefficients in enumerate(output_lineage)
        if coefficients.any() and set(np.flatnonzero(coefficients)) <= selected_set)

    return ClassifiedStabilizers(
        bare, dressed, bare_lineage, dressed_lineage, exact_map,
        safely_selected, original_map, _rank(residues),
        record_dependencies(S) if include_dependencies else None,
    )


def _closure_rows(matrix, lineage, state, measurements):
    measurement_lineage = _decompose(measurements.matrix, matrix)
    rows = _annotated(matrix, lineage, state.records)
    measurement_records = combine_records(measurement_lineage, measurements.records)
    parities = tuple(tuple(sorted(set(old) ^ set(new)))
                     for old, new in zip(rows.records, measurement_records))
    return ClosureRows(rows, lineage, measurement_lineage, parities)


def _intersection_preserving_rows(matrix, basis):
    """Keep every original member first, then complete the intersection."""
    residues = reduce_modulo(matrix, basis)
    originals = np.flatnonzero(~residues.any(axis=1)).tolist()
    exact = matrix[originals]
    intersection, lineage = intersect_row_spaces(matrix, basis)
    extra = [i - len(exact) for i in independent_row_indices(
        np.vstack([exact, intersection])) if i >= len(exact)]
    return (np.vstack([exact, intersection[extra]]),
            np.vstack([np.eye(len(matrix), dtype=np.uint8)[originals], lineage[extra]]))


def readout_closures(S_bare, S_dressed, L, measurements, *, postselected_rows=(),
                    include_dependencies=False):
    """Close the complete physical span against M, with bare closure priority.

    Inputs must already be the actual surviving rows after projection; the
    replaced/destroyed rows are excluded. S_bare is the pre-readout role,
    not a reclassification against the post-readout surviving-qubit K.
    L_absorbed is intentionally absent from this API.

    The detector space is span(S_bare) intersect span(M). The observable
    space completes it to span(S_bare,S_dressed,L) intersect span(M).
    Hence closures that require products of supplied rows are included.
    Existing measurable bare rows are preserved first, including redundant
    local checks. Protected closures prefer original dressed/L rows, then
    products, and are independent modulo the full detector closure space.

    Postselection closures are returned separately with their exact source
    lineage; an emitter must preserve that physical parity space even when
    its preferred detector basis changes. They do not override DET/OBS roles.
    Pauli-zero source dependencies are reported, never silently claimed to
    have been emitted; the tracker owns record-only history bookkeeping.
    """
    S_bare, S_dressed, L, measurements = map(_known, (S_bare, S_dressed, L, measurements))
    width = S_bare.matrix.shape[1]
    _check_width(width, S_dressed.matrix, L.matrix, measurements.matrix)
    state = Rows(np.vstack([S_bare.matrix, S_dressed.matrix, L.matrix]),
                 S_bare.records + S_dressed.records + L.records)
    selected = _indices(postselected_rows, len(state.matrix))
    bare_closed, bare_lineage_local = _intersection_preserving_rows(
        S_bare.matrix, measurements.matrix)
    bare_lineage = np.zeros((len(bare_closed), len(state.matrix)), dtype=np.uint8)
    bare_lineage[:, :len(S_bare.matrix)] = bare_lineage_local
    full_closed, full_lineage = intersect_row_spaces(state.matrix, measurements.matrix)
    protected_source = np.vstack([S_dressed.matrix, L.matrix])
    protected_known = np.flatnonzero(~reduce_modulo(
        protected_source, measurements.matrix).any(axis=1)) + len(S_bare.matrix)
    candidates = np.vstack([state.matrix[protected_known], full_closed])
    candidate_lineage = np.vstack([
        np.eye(len(state.matrix), dtype=np.uint8)[protected_known], full_lineage])
    protected_indices = [i - len(bare_closed) for i in independent_row_indices(
        np.vstack([bare_closed, candidates])) if i >= len(bare_closed)]
    protected = candidates[protected_indices]
    protected_lineage = candidate_lineage[protected_indices]

    ps_closed, ps_local = _intersection_preserving_rows(
        state.matrix[list(selected)], measurements.matrix)
    ps_lineage = np.zeros((len(ps_closed), len(state.matrix)), dtype=np.uint8)
    ps_lineage[:, list(selected)] = ps_local
    return ReadoutClosures(
        _closure_rows(bare_closed, bare_lineage, state, measurements),
        _closure_rows(protected, protected_lineage, state, measurements),
        _closure_rows(ps_closed, ps_lineage, state, measurements),
        record_dependencies(state) if include_dependencies else None,
    )
