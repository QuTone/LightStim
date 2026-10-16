"""Persistent inverse-coordinate execution of the slabs measurement policy.

This is the same S_bare/S_dressed/L/L_absorbed transition as Tracker's reference
path. Only its storage and linear algebra change: current rows, record updates,
projection, refresh and quotient selection remain in one native engine.
"""

import numpy as np
import stim

from .logical_history import LogicalHistoryRelation
from ..utils.pauli_backend import propagate_paulis


def process_mid_measurement(
    tracker, circuit, forward, measurements, resets, qubits, bases, coords,
    retain, no_detector_mask, chunk, base, postselection, input_dressed,
):
    from .tracker import _append_detector
    from ..utils.cpp._pauli_backend_cpp import PackedBasis

    # The reference transition projects a temporary full tableau. A later
    # unknown-logical/record error must not expose those partial projections
    # through a public lazy bank. Clone the retained inverse frame directly;
    # this is a transaction snapshot, with no physical export or reconstruction.
    state = tracker._sync_record_state().copy()
    ns, nl, na = (tracker.stabilizers.count, tracker.logicals.count,
                  tracker.absorbed_ops.count)
    classified = set(tracker._classified_stabilizer_rows)
    state.begin_measurement_batch(measurements)
    for i, measurement in enumerate(measurements):
        anti = state.measurement_anticommuting(i)
        if anti:
            pivot = tracker._slabs_projection_pivot(None, ns, nl, anti)
            state.project(measurement, pivot, base + i)
            if pivot >= ns:
                classified.difference_update(int(j) for j in anti if j != pivot)
            else:
                classified.add(int(pivot))
            continue
        if not measurement.any():
            if no_detector_mask is None or not no_detector_mask[i]:
                position = list(coords[i]) + [0]
                _append_detector(
                    circuit, [stim.target_rec(base + i - tracker.total_measurements)],
                    position, post_select=tuple(position) in tracker.post_select_detector_coords)
            continue
        kind, parity, logical_indices = state.measurement_relation(
            i, ns, nl, sorted(tracker._dressed_stabilizer_rows), base)
        if kind == "logical":
            tracker.logical_history.append(LogicalHistoryRelation(
                records=tuple(parity), measurement_index=base + i,
                logical_indices=tuple(logical_indices)))
        elif kind == "detector" and (no_detector_mask is None or not no_detector_mask[i]):
            position = list(coords[i]) + [0]
            _append_detector(
                circuit, [stim.target_rec(r - tracker.total_measurements) for r in parity],
                position, post_select=tuple(position) in tracker.post_select_detector_coords)

    if retain:
        measured_indices = PackedBasis(state.coordinate_paulis(measurements)).selected
        measured = measurements[measured_indices]
    else:
        measured = measurements
    # Native physical-column canonicalization of L preserves the historical
    # representative, then one coordinate-space complement gives [L; S]/M.
    complement = state.refresh_complement(measured, ns, nl)
    logical = [ns + i for i in complement if i < nl]
    old_stabilizers = [i - nl for i in complement if i >= nl]
    old_records = state.records_selected(old_stabilizers)
    promotable = [j for j, (i, records) in enumerate(zip(old_stabilizers, old_records))
                  if not records and i not in classified]
    absorbed = list(range(ns + nl, ns + nl + na))
    prefix = measured if retain else resets
    prefix_records = ([[base + i] for i in measured_indices] if retain
                      else [[] for _ in range(len(measurements))])
    # Fresh reset rows can change the occupied inverse frame. Native refresh
    # imports only these new physical directions and reuses retained rows.
    state.refresh_sources(prefix, prefix_records, old_stabilizers + logical + absorbed)
    output_count = len(prefix_records)
    new_ns, new_nl = output_count + len(old_stabilizers), len(logical)
    tracker._dressed_stabilizer_rows = {
        output_count + j for j, i in enumerate(old_stabilizers) if i in input_dressed}
    tracker._classified_stabilizer_rows = set(range(output_count)) | {
        output_count + j for j, i in enumerate(old_stabilizers) if i in classified}
    state.unitary(str(chunk))
    if not retain:
        state.fold_measured(qubits, bases, base)
        kept_outputs = state.nonzero_rows(range(output_count))
        if len(kept_outputs) != output_count:
            kept = kept_outputs + list(range(output_count, new_ns + new_nl + na))
            state.adopt_coordinates(state.coordinate_rows(kept), state.records_selected(kept))
            new_ns -= output_count - len(kept_outputs)
        output_count = len(kept_outputs)
        tracker._dressed_stabilizer_rows = {
            output_count + j for j, i in enumerate(old_stabilizers) if i in input_dressed}
        tracker._classified_stabilizer_rows = set(range(output_count)) | {
            output_count + j for j, i in enumerate(old_stabilizers) if i in classified}
    tracker._record_state = state
    tracker._read_record_state((new_ns, new_nl, na))
    if tracker._v2_role_space is not None:
        tracker._v2_role_space = propagate_paulis(tracker._v2_role_space, forward)
    tracker._v2_restore_postselection(
        tracker._v2_forward_postselection(
            postselection, forward,
            measurement_qubit_indices=None if retain else qubits,
            measurement_bases=None if retain else bases,
            measurement_base_idx=base),
        circuit=circuit, measurement_coords=coords)
    tracker._record_measurement_logical_effects({i - ns for i in logical})
    return set() if retain else {output_count + position for position in promotable}
