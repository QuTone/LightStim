"""Readout completeness, explicit stage records, and postselection transport."""
import numpy as np
import pytest
import stim

from lightstim.ir.tracker import SyndromeTracker
from tests.tracker_test_utils import (
    _annotations, _basis, _bits, _full_record_parities,
)


def _rows(*paulis, n):
    return np.asarray([np.concatenate(stim.PauliString(pauli).to_numpy())
                       for pauli in paulis], dtype=np.uint8).reshape(len(paulis), 2*n)


def _tracker(n, *, stabilizers=(), logicals=(), dressed=(), selected=(),
             stab_records=None, logical_records=None):
    tracker = SyndromeTracker(n, logical_tracking="slabs")
    tracker.stabilizers.add_stabilizers(_rows(*stabilizers, n=n), stab_records)
    tracker.logicals.add_stabilizers(_rows(*logicals, n=n), logical_records)
    tracker._dressed_stabilizer_rows = set(dressed)
    tracker.post_select_row_indices = set(selected)
    tracker._v2_rebuild_absorbed()
    tracker._v2_sync_budget()
    return tracker


def _read(tracker, circuit, qubits, *, active=(), absolute=None):
    n = tracker.num_qubits
    if absolute is None:
        circuit.append("M", qubits)
    paulis = []
    for qubit in qubits:
        letters = ["I"] * n
        letters[qubit] = "Z"
        paulis.append("".join(letters))
    tracker.process_data_measurement(
        circuit, _rows(*paulis, n=n), {q: (q, 0) for q in range(n)},
        active_stabilizers=_rows(*active, n=n),
        measurement_record_indices=absolute)


def _assert_complete(tracker, circuit):
    physical, detectors, observables = _annotations(circuit)
    emitted = detectors + observables
    assert len(_basis(emitted)) == len(_basis(_full_record_parities(physical)))
    assert physical.has_all_flows([
        stim.Flow(measurements=list(_bits(row))) for row in emitted
    ], unsigned=False)
    n = tracker.num_qubits
    for tableau in (tracker.stabilizers, tracker.logicals, tracker.absorbed_ops):
        for row, records in zip(tableau.matrix, tableau.records):
            if any(record < 0 for record in records):
                continue
            pauli = stim.PauliString.from_numpy(xs=row[:n].astype(bool),
                                               zs=row[n:].astype(bool))
            assert physical.has_flow(stim.Flow(output=pauli, measurements=records),
                                     unsigned=False)
    circuit.detector_error_model()


@pytest.mark.parametrize("order", [("ZZ", "IZ"), ("IZ", "ZZ")])
def test_v2_readout_closes_product_of_live_logicals(order):
    tracker = _tracker(2, logicals=order)
    circuit = stim.Circuit("R 0 1")
    _read(tracker, circuit, [0])
    assert circuit.num_observables == 1
    assert tracker.expected_num_logicals == 1
    _assert_complete(tracker, circuit)
    _read(tracker, circuit, [1])
    assert circuit.num_observables == 2
    assert tracker.expected_num_logicals == 0
    _assert_complete(tracker, circuit)


def test_v2_readout_noncontiguous_absolute_records_share_one_physical_layer():
    tracker = _tracker(3, logicals=("ZII", "IZI", "IIZ"))
    circuit = stim.Circuit("R 0 1 2\nM 0 1 2")
    tracker.total_measurements = 3
    _read(tracker, circuit, [0, 2], absolute=[0, 2])
    assert tracker.total_measurements == 3
    assert tracker.expected_num_logicals == 1
    _read(tracker, circuit, [1], absolute=[1])
    assert tracker.total_measurements == 3
    assert tracker.expected_num_logicals == 0
    _, _, observables = _annotations(circuit)
    assert sorted(observables) == [1, 2, 4]
    _assert_complete(tracker, circuit)


def test_v2_readout_transports_selected_constraint_across_partial_stages():
    tracker = _tracker(3, stabilizers=("ZIZ",), logicals=("ZII", "IZI"),
                       dressed=(0,), selected=(0,))
    circuit = stim.Circuit("R 0 1 2")
    _read(tracker, circuit, [0], active=("IIZ",))
    assert tracker.post_select_row_indices
    selected = sorted(tracker.post_select_row_indices)
    assert any(np.array_equal(tracker.stabilizers.matrix[index], _rows("IIZ", n=3)[0])
               and tracker.stabilizers.records[index] == [0] for index in selected)
    _read(tracker, circuit, [2])
    selected_detectors = [instruction for instruction in circuit
                          if instruction.name == "DETECTOR" and instruction.tag == "post-select"]
    assert len(selected_detectors) == 1
    assert [target.value for target in selected_detectors[0].targets_copy()] == [-2, -1]
    assert not tracker.post_select_row_indices
    _read(tracker, circuit, [1])
    _assert_complete(tracker, circuit)
    faulty = stim.Circuit("R 0 1 2\nX_ERROR(1) 2") + circuit[1:]
    detectors, _ = faulty.compile_detector_sampler(seed=7).sample(8, separate_observables=True)
    assert detectors.all()


def test_v2_readout_closes_selected_products_before_individual_selected_rows():
    tracker = _tracker(3, stabilizers=("ZIZ", "IZZ"), logicals=("IIZ",),
                       selected=(0, 1))
    circuit = stim.Circuit("R 0 1 2")
    _read(tracker, circuit, [0, 1])
    # Neither selected input row is supported on the measured pair. Their
    # product is, and its postselection role must be visible at this boundary.
    selected = [instruction for instruction in circuit
                if instruction.name == "DETECTOR" and instruction.tag == "post-select"]
    assert len(selected) == 1
    assert [target.value for target in selected[0].targets_copy()] == [-2, -1]
    assert tracker.post_select_row_indices
    _assert_complete(tracker, circuit)
    _read(tracker, circuit, [2])
    _assert_complete(tracker, circuit)


def test_v2_projection_does_not_or_tag_an_unselected_pivot_into_postselection():
    tracker = _tracker(3, stabilizers=("ZIZ", "IZZ"), logicals=("XXX",),
                       selected=(1,))
    circuit = stim.Circuit("R 0 1 2\nH 0\nCX 0 1 0 2\nMX 2")
    tracker.process_data_measurement(
        circuit, _rows("IIX", n=3), {q: (q, 0) for q in range(3)},
        active_stabilizers=_rows("ZZI", n=3))
    # ZIZ is the unselected pivot; IZZ is selected but is destroyed as an
    # independently readable constraint. ZIZ*IZZ survives without selection.
    assert not tracker.post_select_row_indices
    _read(tracker, circuit, [0, 1])
    assert not any(instruction.name == "DETECTOR" and instruction.tag == "post-select"
                   for instruction in circuit)
    _assert_complete(tracker, circuit)


@pytest.mark.parametrize("spectator_first", [False, True])
def test_v2_postselection_and_logical_overlap_uses_physical_rank(spectator_first):
    tracker = _tracker(2, stabilizers=("ZI",), logicals=("ZI", "IZ"),
                       dressed=(0,), selected=(0,))
    assert tracker.expected_num_logicals == 2
    circuit = stim.Circuit("R 0 1")
    if spectator_first:
        _read(tracker, circuit, [1])
        assert tracker.expected_num_logicals == 1
        assert tracker.post_select_row_indices
        _read(tracker, circuit, [0])
    else:
        _read(tracker, circuit, [0, 1])
    assert circuit.num_observables == 2
    assert circuit.num_detectors == 1
    assert tracker.expected_num_logicals == 0
    assert all(instruction.tag == "post-select" for instruction in circuit
               if instruction.name == "DETECTOR")
    _assert_complete(tracker, circuit)


def test_v2_record_only_logical_dependency_is_closed_before_deduplication():
    # Both measurements prepare the same known Z value; their equality is a
    # genuine un-emitted record-only constraint represented by redundant rows.
    tracker = _tracker(1, stabilizers=("Z",), logicals=("Z",), dressed=(0,),
                       stab_records=[[0]], logical_records=[[1]])
    tracker.total_measurements = 2
    circuit = stim.Circuit("RX 0\nM 0 0")
    tracker.process_data_measurement(circuit, _rows(n=1), {0: (0, 0)},
                                     active_stabilizers=_rows(n=1))
    assert circuit.num_observables == 1
    assert circuit.has_flow(stim.Flow(measurements=[0, 1]), unsigned=False)
    assert tracker.expected_num_logicals == 1
    _assert_complete(tracker, circuit)


def test_v2_unknown_placeholders_do_not_cancel_into_a_known_detector():
    tracker = _tracker(2, stabilizers=("ZI", "IZ"), stab_records=[[-1], [-1]])
    circuit = stim.Circuit("RX 0 1")
    _read(tracker, circuit, [0, 1])
    assert circuit.num_detectors == circuit.num_observables == 0
    _assert_complete(tracker, circuit)


@pytest.mark.parametrize("stabilizers,logicals,measurements", [
    ((), ("ZZ",), ("ZI", "IZ")),
    (("IZ",), ("ZZ",), ("ZI",)),
    ((), ("ZI", "IZ"), ("ZZ",)),
])
def test_v2_unknown_logical_closure_refuses_unbanked_records(stabilizers, logicals, measurements):
    tracker = _tracker(2, stabilizers=stabilizers, logicals=logicals,
                       logical_records=[[-1] for _ in logicals])
    with pytest.raises(RuntimeError, match="UNMEASURED sentinel"):
        tracker.process_data_measurement(stim.Circuit(), _rows(*measurements, n=2),
                                         {0: (0, 0), 1: (1, 0)})


def test_v2_unknown_logical_destroyed_by_projection_does_not_raise():
    tracker = _tracker(1, logicals=("X",), logical_records=[[-1]])
    circuit = stim.Circuit("RX 0\nM 0")
    tracker.process_data_measurement(circuit, _rows("Z", n=1), {0: (0, 0)})
    assert circuit.num_observables == circuit.num_detectors == 0
    _assert_complete(tracker, circuit)


def test_v2_projection_keeps_unknown_when_two_placeholder_records_would_cancel():
    tracker = _tracker(2, stabilizers=("XX", "XI"), stab_records=[[-1], [-1]])
    circuit = stim.Circuit("RX 0\nR 1")
    _read(tracker, circuit, [0], active=("IX",))
    assert tracker.stabilizers.count == 1
    assert -1 in tracker.stabilizers.records[0]
    circuit.append("MX", [1])
    tracker.process_data_measurement(circuit, _rows("IX", n=2), {0: (0, 0), 1: (1, 0)})
    assert circuit.num_detectors == circuit.num_observables == 0
    _assert_complete(tracker, circuit)


def test_v2_native_unrotated_transversal_cnot_partial_readout():
    from lightstim.ir.builder import CircuitBuilder
    from lightstim.ir.qec_system import QECSystem
    from lightstim.qec_code.surface_code.unrotated import (
        UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock,
    )
    system = QECSystem()
    system.add_patch(UnrotatedSurfaceCode(distance=3), name="A")
    system.add_patch(UnrotatedSurfaceCode(distance=3), name="B", offset=(0, 6))
    tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=2)
    builder = CircuitBuilder(tracker, system)
    builder.write_coordinates()
    builder.initialize({q: "Z" for q in system.data_indices}, n=system.num_qubits)
    builder.apply_syndrome_extraction(UnrotatedSurfaceCodeExtractionBlock(system).circuit)
    unitary = stim.Circuit()
    for local, qubit in sorted(system.local_to_global_map["A"].items()):
        if qubit in system.data_indices:
            unitary.append("CX", [system.local_to_global_map["B"][local], qubit])
    builder.apply_unitary_block(unitary)
    A = [q for q in sorted(system.data_indices) if system.index_to_owner_map[q] == "A"]
    B = [q for q in sorted(system.data_indices) if system.index_to_owner_map[q] == "B"]
    builder.apply_data_readout({q: "Z" for q in A})
    assert builder.circuit.num_observables == 1
    _assert_complete(tracker, builder.circuit)
    builder.apply_data_readout({q: "Z" for q in B})
    assert builder.circuit.num_observables == 2
    _assert_complete(tracker, builder.circuit)
