"""Record-only logical history, separate from the tracker's live state.

The tracker stores unsigned Paulis. A relation here therefore specifies a
deterministic *support*, not a promise that its ideal XOR is zero. Stim's
reference sample supplies that affine offset when observables are sampled.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, FrozenSet, Iterable, List, Tuple

import stim

if TYPE_CHECKING:
    from .tracker import SyndromeTracker


@dataclass(frozen=True)
class LogicalHistoryRelation:
    """A dependency captured before measurement writeback changes its basis.

    ``records`` are absolute, zero-based measurement indices with XOR
    multiplicity removed. ``logical_indices`` describe the logical tableau
    at capture time; they are provenance, not persistent logical-qubit IDs.
    These objects do not contribute to the live logical-DOF census.
    """

    records: Tuple[int, ...]
    measurement_index: int
    logical_indices: Tuple[int, ...]


LOGICAL_HISTORY_TAG = "logical-history"


def _record_support(records: Iterable[int], num_measurements: int) -> FrozenSet[int]:
    support = set()
    for record in records:
        if not 0 <= record < num_measurements:
            raise ValueError(
                f"Logical history requires known measurement indices; got {record} "
                f"for a circuit with {num_measurements} measurements."
            )
        record = int(record)
        if record in support:
            support.remove(record)
        else:
            support.add(record)
    return frozenset(support)


def _insert_row(pivots: Dict[int, FrozenSet[int]], support: FrozenSet[int]) -> bool:
    """Sparse GF(2) elimination; storage follows support, not absolute index."""
    row = set(support)
    while row:
        pivot = max(row)
        if pivot not in pivots:
            pivots[pivot] = frozenset(row)
            return True
        row.symmetric_difference_update(pivots[pivot])
    return False


def _annotations(circuit: stim.Circuit, *, include_detectors: bool):
    pivots: Dict[int, FrozenSet[int]] = {}
    observables: Dict[int, FrozenSet[int]] = {}
    history_ids = set()
    pauli_observable_ids = set()
    offset = 0
    for instruction in circuit.flattened():
        if instruction.name == "OBSERVABLE_INCLUDE" or (
            include_detectors and instruction.name == "DETECTOR"
        ):
            targets = instruction.targets_copy()
            record_targets = [t for t in targets if t.is_measurement_record_target]
            support = _record_support((offset + t.value for t in record_targets), offset)
            if instruction.name == "DETECTOR":
                _insert_row(pivots, support)
            else:
                key = int(instruction.gate_args_copy()[0])
                observables[key] = observables.get(key, frozenset()) ^ support
                if len(record_targets) != len(targets):
                    pauli_observable_ids.add(key)
                if instruction.tag == LOGICAL_HISTORY_TAG:
                    history_ids.add(key)
        offset += instruction.num_measurements
    if history_ids & pauli_observable_ids:
        raise ValueError("Tagged logical-history observables must use measurement records only.")
    if include_detectors and pauli_observable_ids:
        raise ValueError(
            "independent_only=True requires record-only observables; native Pauli "
            "targets are preserved by the default export but cannot be reduced "
            "in a measurement-record basis."
        )
    return pivots, observables, history_ids


def _annotation_basis(circuit: stim.Circuit) -> Dict[int, FrozenSet[int]]:
    pivots, observables, _ = _annotations(circuit, include_detectors=True)
    for support in observables.values():
        _insert_row(pivots, support)
    return pivots


def logical_history_observable_indices(circuit: stim.Circuit) -> Tuple[int, ...]:
    """IDs emitted as historical targets, available to task-selection policies."""
    return tuple(sorted({
        int(instruction.gate_args_copy()[0])
        for instruction in circuit.flattened()
        if instruction.name == "OBSERVABLE_INCLUDE"
        and instruction.tag == LOGICAL_HISTORY_TAG
    }))


def append_logical_history_observables(
    tracker: "SyndromeTracker", circuit: stim.Circuit, *, independent_only: bool = False
) -> List[int]:
    """Append all captured logical-history relations to the finished circuit.

    Native observable IDs/supports are preserved. Only identical previously
    emitted history supports are skipped by default; linear dependencies with
    native targets or detectors do not silently remove a task outcome.
    ``independent_only=True`` explicitly requests the former basis-extension
    policy for record-only annotations. It uses sparse supports rather than
    dense absolute-index bitsets. Default export preserves native Pauli targets.

    Validation precedes mutation. Fixed nonzero ideal parity is permitted;
    Stim evaluates errors against its reference sample. This covers the
    archived mid-measurement logical branch, not arbitrary-protocol completeness.
    Repeated calls on an unchanged circuit are idempotent.
    Complete circuit copies reuse previously allocated history IDs unless a
    supplied copy already uses that ID for another observable.
    """
    if not tracker.logical_history:
        return []
    if circuit.num_measurements != tracker.total_measurements:
        raise ValueError(
            "Logical-history export needs the complete circuit matching the tracker: "
            f"circuit has {circuit.num_measurements} measurements, "
            f"tracker has {tracker.total_measurements}."
        )
    num_measurements = circuit.num_measurements
    supports = [
        _record_support(relation.records, num_measurements)
        for relation in tracker.logical_history
    ]
    flows = [stim.Flow(measurements=relation.records) for relation in tracker.logical_history]
    if not circuit.has_all_flows(flows, unsigned=True):
        raise ValueError(
            "An archived logical-history parity is not deterministic in this circuit. "
            "Check circuit provenance and the tracker state at capture time."
        )

    pivots, observables, history_ids = _annotations(
        circuit, include_detectors=independent_only
    )
    already_emitted = {observables[index] for index in history_ids}
    if independent_only:
        for support in observables.values():
            _insert_row(pivots, support)
    selected = []
    for relation, support in zip(tracker.logical_history, supports):
        if support in already_emitted:
            continue
        if independent_only and not _insert_row(pivots, support):
            continue
        selected.append((relation, support))
        already_emitted.add(support)
    if not selected:
        return []

    # Respect earlier reservations and explicit native annotations. Reusing
    # an allocated history ID in another complete output is not a new target;
    # consuming the allocator again would leave phantom observable columns.
    tracker.total_observables = max(tracker.total_observables, circuit.num_observables)
    allocated_ids = dict(tracker._logical_history_observable_ids)
    occupied_ids = set(observables)
    new_ids = []
    for relation, support in selected:
        observable_id = allocated_ids.get(support)
        if observable_id is None or observable_id in occupied_ids:
            observable_id = tracker.allocate_observable()
        # A caller may add a native observable at our preferred ID in one
        # output. Keep the original binding for later ordinary copies.
        allocated_ids.setdefault(support, observable_id)
        tracker.total_observables = max(tracker.total_observables, observable_id + 1)
        occupied_ids.add(observable_id)
        circuit.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(record - num_measurements) for record in relation.records],
            observable_id,
            tag=LOGICAL_HISTORY_TAG,
        )
        new_ids.append(observable_id)
    # Do not mutate the dictionary shared with a shallow-copied tracker.
    tracker._logical_history_observable_ids = allocated_ids
    return new_ids
