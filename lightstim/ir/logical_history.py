"""Record-only logical history, separate from the tracker's live state.

The tracker stores unsigned Paulis. A relation here therefore specifies a
deterministic *support*, not a promise that its ideal XOR is zero. Stim's
reference sample supplies that affine offset when observables are sampled.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Tuple

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


def _record_word(records: Iterable[int], num_measurements: int) -> int:
    word = 0
    for record in records:
        if not 0 <= record < num_measurements:
            raise ValueError(
                f"Logical history requires known measurement indices; got {record} "
                f"for a circuit with {num_measurements} measurements."
            )
        word ^= 1 << int(record)
    return word


def _insert_row(pivots: Dict[int, int], word: int) -> bool:
    """Insert a sparse GF(2) row, returning whether it adds a direction."""
    while word:
        pivot = word.bit_length() - 1
        if pivot not in pivots:
            pivots[pivot] = word
            return True
        word ^= pivots[pivot]
    return False


def _annotation_basis(circuit: stim.Circuit) -> Dict[int, int]:
    """Span of detector rows and complete (possibly accumulated) OBS rows."""
    pivots: Dict[int, int] = {}
    observables: Dict[int, int] = {}
    offset = 0
    for instruction in circuit.flattened():
        if instruction.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            targets = instruction.targets_copy()
            if any(not target.is_measurement_record_target for target in targets):
                raise ValueError("Logical-history export requires record-only annotations.")
            word = _record_word(
                (offset + target.value for target in targets), offset
            )
            if instruction.name == "DETECTOR":
                _insert_row(pivots, word)
            else:
                key = int(instruction.gate_args_copy()[0])
                observables[key] = observables.get(key, 0) ^ word
        offset += instruction.num_measurements
    for word in observables.values():
        _insert_row(pivots, word)
    return pivots


def append_logical_history_observables(
    tracker: "SyndromeTracker", circuit: stim.Circuit
) -> List[int]:
    """Explicitly append independent historical targets to a finished circuit.

    Existing gates, detectors, and observable IDs/supports are preserved.
    Candidates already in their joint span are omitted. All archived
    dependencies are checked against the ideal circuit before either the
    circuit or observable allocator is modified. The check permits a fixed
    nonzero parity; Stim reports errors relative to its ideal reference.

    This changes the evaluation target when new IDs are returned. It is
    deliberately opt-in and does not claim to enumerate dependencies from
    tracker paths other than the archived mid-measurement logical branch.
    Calling it again on the same circuit adds nothing.
    """
    if circuit.num_measurements != tracker.total_measurements:
        raise ValueError(
            "Logical-history export needs the complete circuit matching the tracker: "
            f"circuit has {circuit.num_measurements} measurements, "
            f"tracker has {tracker.total_measurements}."
        )
    if not tracker.logical_history:
        return []

    num_measurements = circuit.num_measurements
    words = [
        _record_word(relation.records, num_measurements)
        for relation in tracker.logical_history
    ]
    flows = [stim.Flow(measurements=relation.records) for relation in tracker.logical_history]
    if not circuit.has_all_flows(flows, unsigned=True):
        raise ValueError(
            "An archived logical-history parity is not deterministic in this circuit. "
            "Check circuit provenance and the tracker state at capture time."
        )

    pivots = _annotation_basis(circuit)
    selected = [
        relation for relation, word in zip(tracker.logical_history, words)
        if _insert_row(pivots, word)
    ]
    if not selected:
        return []

    # Respect both earlier reservations and explicit annotations supplied by
    # the caller. Every NEW target still goes through the central allocator.
    tracker.total_observables = max(tracker.total_observables, circuit.num_observables)
    new_ids = []
    for relation in selected:
        observable_id = tracker.allocate_observable()
        circuit.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(record - num_measurements) for record in relation.records],
            observable_id,
        )
        new_ids.append(observable_id)
    return new_ids
