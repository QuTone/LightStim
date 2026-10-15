"""Active code checks distinguish a repeated SE round from a new ZZ readout.

All physical protocols use the existing unrotated QECSystem/Builder blocks.
The oracle compares complete signed Stim flows and responses to independently
inserted physical logical Paulis; it does not infer roles from tracker rows.
"""

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock, UnrotatedTwoPatchCoupler,
)
from tests.test_slabs_acceptance import (
    _annotations, _basis, _bits, _contains, _full_record_parities, _mask,
    _validate_physical_relations,
)


def _rows(*paulis):
    return np.asarray([
        np.concatenate(stim.PauliString(pauli).to_numpy()) for pauli in paulis
    ], dtype=np.uint8)


def _tableau_snapshot(tracker):
    result = {}
    for name, tableau in (("S", tracker.stabilizers), ("L", tracker.logicals),
                          ("L_absorbed", tracker.absorbed_logicals)):
        result[name] = [
            dict(pauli=str(stim.PauliString.from_numpy(
                xs=row[:tracker.num_qubits].astype(bool),
                zs=row[tracker.num_qubits:].astype(bool))), records=list(records))
            for row, records in zip(tableau.matrix, tableau.records)
        ]
    result["logical_history"] = [list(relation.records) for relation in tracker.logical_history]
    return result


def _spectator_case(*, split_between, read_spectator_between):
    """ZZZ preparation, two AB SE rounds, optional intermediate split and C readout."""
    a = UnrotatedSurfaceCode(distance=3)
    a.rotate_coords(np.pi)
    a.reset_rotation_angle()
    system = QECSystem()
    system.add_patch(a, name="A")
    system.add_patch(UnrotatedSurfaceCode(distance=3), name="B", offset=(0, 6))
    system.add_patch(UnrotatedSurfaceCode(distance=3), name="C", offset=(20, 0))
    system.register_coupler(UnrotatedTwoPatchCoupler(), patch_names=["A", "B"],
                            name="AB", interaction_type="ZZ")
    tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=3,
                              logical_tracking="slabs")
    builder = CircuitBuilder(tracker, system)
    builder.write_coordinates()
    owner = system.index_to_owner_map
    patch_data = {q for q in system.data_indices if owner[q] in ("A", "B", "C")}
    c_data = {q for q in patch_data if owner[q] == "C"}
    corridor = {q for q in system.data_indices if owner[q] == "AB"}
    builder.initialize({q: "Z" for q in patch_data}, n=system.num_qubits)
    builder.apply_syndrome_extraction(UnrotatedSurfaceCodeExtractionBlock(system).circuit)

    operators = []
    for operator in system.logical_ops:
        if operator["patch_name"] not in ("A", "B", "C"):
            continue
        pauli = stim.PauliString(system.num_qubits)
        for qubit, factor in operator["pauli"].items():
            pauli[qubit] = factor
        operators.append((operator["patch_name"], pauli))
    cuts = [(_annotations(builder.circuit)[0].copy(), [pauli for _, pauli in operators])]

    def remaining_cut():
        cuts.append((_annotations(builder.circuit)[0].copy(),
                     [pauli for name, pauli in operators
                      if not read_spectator_between or name != "C"]))

    def merge_round():
        builder.activate_coupler("AB")
        builder.initialize({q: "X" for q in corridor}, n=system.num_qubits)
        builder.apply_syndrome_extraction(UnrotatedSurfaceCodeExtractionBlock(system).circuit)

    def split():
        builder.deactivate_coupler("AB")
        builder.apply_data_readout({q: "X" for q in corridor}, resolve_absorbed=False)

    merge_round()
    stages = {"after_first_merge": _tableau_snapshot(tracker)}
    if read_spectator_between:
        builder.apply_data_readout({q: "Z" for q in c_data})
        # The physical protocol retires already-read C checks, while declaring
        # precisely the same merged AB checks for the next extraction block.
        c_syndrome = {q for q in system.syndrome_indices if owner[q] == "C"}
        system.active_stabilizer_indices = {
            i for i in system.active_stabilizer_indices
            if system.stabilizers[i]["syn_idx"] not in c_syndrome}
        stages["after_C_readout"] = _tableau_snapshot(tracker)

    history_before = len(tracker.logical_history)
    if split_between:
        split()
        stages["after_intermediate_split"] = _tableau_snapshot(tracker)
        remaining_cut()
        merge_round()
    else:
        builder.apply_syndrome_extraction(UnrotatedSurfaceCodeExtractionBlock(system).circuit)
    stages["after_second_round"] = _tableau_snapshot(tracker)
    second_history = [list(relation.records) for relation in tracker.logical_history[history_before:]]

    split()
    stages["after_final_split"] = _tableau_snapshot(tracker)
    remaining_cut()
    builder.apply_data_readout({q: "Z" for q in patch_data
                               if not read_spectator_between or q not in c_data})
    circuit = builder.to_stim_circuit()
    return circuit, cuts, stages, second_history


@pytest.mark.parametrize("split_between", [False, True], ids=["same_merge", "split_and_merge"])
@pytest.mark.parametrize("read_spectator_between", [False, True], ids=["C_deferred", "C_between"])
def test_ab_second_se_is_detector_until_a_split(split_between, read_spectator_between):
    circuit, cuts, stages, second_history = _spectator_case(
        split_between=split_between, read_spectator_between=read_spectator_between)
    expected = 5 if split_between else 4
    _validate_physical_relations(circuit, cuts, expected)
    physical, detectors, observables = _annotations(circuit)
    assert physical.has_all_flows([
        stim.Flow(measurements=list(_bits(row))) for row in detectors + observables
    ], unsigned=False)
    assert len(stages["after_first_merge"]["logical_history"]) == 1
    assert len(second_history) == int(split_between)
    assert len(stages["after_second_round"]["logical_history"]) == 1 + int(split_between)
    if read_spectator_between:
        assert len(stages["after_C_readout"]["L_absorbed"]) == 1
    if split_between:
        assert stages["after_intermediate_split"]["L_absorbed"] == []
    assert stages["after_final_split"]["L_absorbed"] == []

    if read_spectator_between and not split_between:
        # Exact witness from the reported real-code counterexample:
        # h = m72^m73^m74 is the first ZZ readout; d = m74^m114 is
        # repeated extraction consistency. Their sum must not replace d
        # with an additional protected logical direction.
        first = _mask([72, 73, 74])
        repeat = _mask([74, 114])
        assert stages["after_first_merge"]["logical_history"] == [[72, 73, 74]]
        assert _contains(_basis(detectors), repeat)
        assert not _contains(_basis(detectors), first)
        assert _contains(_basis(observables + detectors), first)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["ZI_then_IZ", "IZ_then_ZI"])
def test_s_first_projection_updates_absorbed_copy_for_both_S_row_orders(order):
    """Physical Bell preparation reaches the user's exact two-row example."""
    tracker = SyndromeTracker(2, expected_num_logicals=1, logical_tracking="slabs")
    tracker.stabilizers.add_stabilizers(_rows("XX"), [[]])
    tracker.logicals.add_stabilizers(_rows("ZZ"), [[]])
    tracker.meas_rec_to_idx_map = {0: 0, 1: 1, 2: 0, 3: 1}
    circuit = stim.Circuit("RX 0\nR 1\nCX 0 1\nM 0 1")
    tracker.process_mid_measurement(
        circuit=circuit, forward_symplectic_matrix=np.eye(4, dtype=np.uint8),
        back_propagated_paulis=_rows("ZI", "IZ"), reset_paulis=None,
        measurement_qubit_indices=[0, 1], measurement_bases=["Z", "Z"],
        measurement_coords=[(0, 0), (1, 0)],
        discarded_measurement_qubit_indices=set())
    np.testing.assert_array_equal(tracker.stabilizers.matrix, _rows("ZI", "IZ"))
    np.testing.assert_array_equal(tracker.absorbed_logicals.matrix, _rows("ZZ"))
    assert tracker.stabilizers.records == [[0], [1]]
    assert tracker.absorbed_logicals.records == [[]]
    for relation in tracker.logical_history:
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(r - tracker.total_measurements)
                                              for r in relation.records],
                       tracker.allocate_observable())
    tracker.stabilizers.matrix = tracker.stabilizers.matrix[list(order)].copy()
    tracker.stabilizers.records = [tracker.stabilizers.records[i] for i in order]
    circuit.append("MX", [0])
    tracker.process_data_measurement(circuit, _rows("XI"), {0: (0, 0), 1: (1, 0)})
    np.testing.assert_array_equal(tracker.logicals.matrix, _rows("IZ"))
    assert tracker.logicals.records == [[0]]
    circuit.append("M", [1])
    tracker.process_data_measurement(circuit, _rows("IZ"), {0: (0, 0), 1: (1, 0)})
    physical, detectors, observables = _annotations(circuit)
    assert detectors == []
    assert set(observables) == {_mask([0, 1]), _mask([0, 3])}
    assert len(_basis(detectors + observables)) == len(_full_record_parities(physical)) == 2
    assert physical.has_all_flows([
        stim.Flow(measurements=list(_bits(row))) for row in detectors + observables
    ], unsigned=False)
    circuit.detector_error_model()


def _absorbed_product_after_spectator_readout():
    """Reach a two-dimensional absorbed bank using the original mid-measurement API."""
    tracker = SyndromeTracker(3, expected_num_logicals=3, logical_tracking="slabs")
    tracker.logicals.add_stabilizers(_rows("ZII", "IXI", "IIX"), [[], [], []])
    tracker.meas_rec_to_idx_map = {0: 1, 1: 2, 2: 0, 3: 1, 4: 2}
    circuit = stim.Circuit("R 0\nRX 1 2\nM 1 2")
    tracker.process_mid_measurement(
        circuit=circuit, forward_symplectic_matrix=np.eye(6, dtype=np.uint8),
        back_propagated_paulis=_rows("IZI", "IIZ"), reset_paulis=None,
        measurement_qubit_indices=[1, 2], measurement_bases=["Z", "Z"],
        measurement_coords=[(1, 0), (2, 0)],
        discarded_measurement_qubit_indices=set())
    np.testing.assert_array_equal(tracker.absorbed_logicals.matrix, _rows("IZI", "IIZ"))
    assert tracker.absorbed_logicals.records == [[0], [1]]
    coordinates = {q: (q, 0) for q in range(3)}
    circuit.append("M", [0])
    tracker.process_data_measurement(circuit, _rows("ZII"), coordinates,
                                     active_stabilizers=_rows("IZZ"))
    return tracker, circuit, coordinates


def test_active_intersection_keeps_an_absorbed_combination_with_its_records():
    """An active Z1Z2 check retains a product, though neither input row is active."""
    tracker, circuit, coordinates = _absorbed_product_after_spectator_readout()
    np.testing.assert_array_equal(tracker.absorbed_logicals.matrix, _rows("IZZ"))
    assert tracker.absorbed_logicals.records == [[0, 1]]
    assert tracker.logicals.count == 1
    tracker.validate_logical_count(context="active intersection retained product")
    physical, _, _ = _annotations(circuit)
    for tableau in (tracker.stabilizers, tracker.logicals, tracker.absorbed_logicals):
        for row, records in zip(tableau.matrix, tableau.records):
            pauli = stim.PauliString.from_numpy(xs=row[:3].astype(bool), zs=row[3:].astype(bool))
            assert physical.has_flow(stim.Flow(output=pauli, measurements=records), unsigned=False)

    circuit.append("M", [1, 2])
    tracker.process_data_measurement(circuit, _rows("IZI", "IIZ"), coordinates,
                                     active_stabilizers=np.zeros((0, 6), dtype=np.uint8))
    physical, detectors, observables = _annotations(circuit)
    assert tracker.absorbed_logicals.count == tracker.logicals.count == 0
    assert len(_basis(observables + detectors)) - len(_basis(detectors)) == 3
    assert len(_basis(observables + detectors)) == len(_full_record_parities(physical)) == 3
    assert physical.has_all_flows([
        stim.Flow(measurements=list(_bits(row))) for row in detectors + observables
    ], unsigned=False)
    circuit.detector_error_model()


def test_repeating_retained_active_combination_emits_detector_not_logical_history():
    """After release, S must represent the active check without using released L.

    Initial retained MZ1/MZ2 produces S=[Z1{m0},Z2{m1}], L=[Z0{}],
    L_absorbed=[Z1{m0},Z2{m1}]. After spectator Z0 readout, the continuing
    code declares Z1Z2 active. Its next MPP outcome m3 must close the detector
    m0^m1^m3 even though one independent absorbed direction was released.
    """
    tracker, circuit, coordinates = _absorbed_product_after_spectator_readout()
    history_before = len(tracker.logical_history)
    circuit.append("MPP", [stim.target_z(1), stim.target_combiner(), stim.target_z(2)])
    tracker.process_mid_measurement(
        circuit=circuit, forward_symplectic_matrix=np.eye(6, dtype=np.uint8),
        back_propagated_paulis=_rows("IZZ"), reset_paulis=None,
        measurement_qubit_indices=[1], measurement_bases=["Z"],
        measurement_coords=[(1, 0)], discarded_measurement_qubit_indices=set())
    assert len(tracker.logical_history) == history_before
    physical, detectors, observables = _annotations(circuit)
    assert _contains(_basis(detectors), _mask([0, 1, 3]))
    assert physical.has_all_flows([
        stim.Flow(measurements=list(_bits(row))) for row in detectors + observables
    ], unsigned=False)

    circuit.append("M", [1, 2])
    tracker.process_data_measurement(circuit, _rows("IZI", "IIZ"), coordinates,
                                     active_stabilizers=np.zeros((0, 6), dtype=np.uint8))
    physical, detectors, observables = _annotations(circuit)
    assert len(_basis(detectors + observables)) == len(_full_record_parities(physical)) == 4
    assert len(_basis(detectors + observables)) - len(_basis(detectors)) == 3
    assert physical.has_all_flows([
        stim.Flow(measurements=list(_bits(row))) for row in detectors + observables
    ], unsigned=False)
    circuit.detector_error_model()


@pytest.mark.parametrize("active_representative", ["ZI", "IZ"])
def test_joint_readout_uses_active_check_class_modulo_measurement(active_representative):
    """After measuring ZZ, active ZI and IZ are the same continuing check class."""
    tracker = SyndromeTracker(2, expected_num_logicals=1, logical_tracking="slabs")
    tracker.stabilizers.add_stabilizers(_rows("ZI"), [[]])
    tracker.absorbed_logicals.add_stabilizers(_rows("ZI"), [[]])
    tracker.meas_rec_to_idx_map = {0: 0, 1: 1, 2: 1}
    coordinates = {0: (0, 0), 1: (1, 0)}
    circuit = stim.Circuit("R 0\nRX 1\nMPP Z0*Z1")
    tracker.process_data_measurement(
        circuit, _rows("ZZ"), coordinates,
        active_stabilizers=_rows(active_representative))
    assert tracker.logicals.count == 0
    np.testing.assert_array_equal(tracker.absorbed_logicals.matrix, _rows("IZ"))
    assert tracker.absorbed_logicals.records == [[0]]
    np.testing.assert_array_equal(tracker.stabilizers.matrix, _rows("IZ"))
    assert tracker.stabilizers.records == [[0]]
    assert circuit.has_flow(stim.Flow(output=stim.PauliString("IZ"), measurements=[0]),
                            unsigned=False)

    # Measuring the continuing check again is a detector; subsequently
    # retiring that check and reading it out releases the logical direction.
    circuit.append("M", [1])
    tracker.process_mid_measurement(
        circuit=circuit, forward_symplectic_matrix=np.eye(4, dtype=np.uint8),
        back_propagated_paulis=_rows("IZ"), reset_paulis=None,
        measurement_qubit_indices=[1], measurement_bases=["Z"],
        measurement_coords=[(1, 0)], discarded_measurement_qubit_indices=set())
    assert tracker.logical_history == []
    circuit.append("M", [1])
    tracker.process_data_measurement(
        circuit, _rows("IZ"), coordinates,
        active_stabilizers=np.zeros((0, 4), dtype=np.uint8))
    physical, detectors, observables = _annotations(circuit)
    assert detectors == [_mask([0, 1])]
    assert observables == [_mask([0, 2])]
    assert len(_basis(detectors + observables)) == len(_full_record_parities(physical)) == 2
    assert physical.has_all_flows([
        stim.Flow(measurements=list(_bits(row))) for row in detectors + observables
    ], unsigned=False)
    circuit.detector_error_model()
