"""Absorb normalization preserves physical relations across equivalent bases.

Bell-pair logical consistency is tested with reversed commuting measurement
order and either final X readout. A known spectator also checks equivalent L
bases, including nonempty records that must follow every logical row XOR.
"""
import numpy as np
import pytest
import stim

from lightstim.ir.tracker import SyndromeTracker
from tests.test_slabs_acceptance import (
    _annotations, _basis, _bits, _contains, _full_record_parities,
)
from tests.test_slabs_partial_readout import _mid, _read


def _rows(*paulis):
    return np.asarray([
        np.concatenate(stim.PauliString(pauli).to_numpy()) for pauli in paulis
    ], dtype=np.uint8)


def _bell_readout(*, order, first_readout, spectator=False,
                  mixed_logical_basis=False, measured_spectator=False):
    n = 3 if spectator else 2
    tracker = SyndromeTracker(n, expected_num_logicals=2 if spectator else 1,
                              logical_tracking="slabs")
    circuit = stim.Circuit()
    spectator_records = []
    if spectator:
        if measured_spectator:
            circuit.append("RX", [2])
            circuit.append("M", [2])
            spectator_records = [0]
            tracker.total_measurements = 1
            tracker.meas_rec_to_idx_map[0] = 2
        else:
            circuit.append("R", [2])
    circuit += stim.Circuit("R 0 1\nH 0\nCX 0 1")
    tracker.stabilizers.add_stabilizers(_rows("XXI" if spectator else "XX"), [[]])
    if spectator:
        tracker.logicals.add_stabilizers(
            _rows("ZZZ" if mixed_logical_basis else "ZZI", "IIZ"),
            [list(spectator_records) if mixed_logical_basis else [], list(spectator_records)])
    else:
        tracker.logicals.add_stabilizers(_rows("ZZ"), [[]])

    z_rows = ["ZII", "IZI"] if spectator else ["ZI", "IZ"]
    x_rows = ["XII", "IXI"] if spectator else ["XI", "IX"]
    start = tracker.total_measurements
    for i, qubit in enumerate(order):
        tracker.meas_rec_to_idx_map[start+i] = qubit
    circuit.append("M", list(order))
    tracker.process_mid_measurement(
        circuit=circuit, forward_symplectic_matrix=np.eye(2*n, dtype=np.uint8),
        back_propagated_paulis=_rows(*(z_rows[q] for q in order)), reset_paulis=None,
        measurement_qubit_indices=list(order), measurement_bases=["Z", "Z"],
        measurement_coords=[(q, 0) for q in order],
        discarded_measurement_qubit_indices=set())
    # Match the Builder's history export. No expected parity is obtained from
    # tracker tableaus: signed Stim flows below check every emitted relation.
    for relation in tracker.logical_history:
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(r - tracker.total_measurements)
                                              for r in relation.records],
                       tracker.allocate_observable())

    coordinates = {q: (q, 0) for q in range(n)}
    tracker.meas_rec_to_idx_map[tracker.total_measurements] = first_readout
    circuit.append("MX", [first_readout])
    tracker.process_data_measurement(circuit, _rows(x_rows[first_readout]), coordinates)
    remaining = [1-first_readout] + ([2] if spectator else [])
    for offset, qubit in enumerate(remaining):
        tracker.meas_rec_to_idx_map[tracker.total_measurements+offset] = qubit
    circuit.append("M", remaining)
    tracker.process_data_measurement(
        circuit, _rows(z_rows[1-first_readout], *(["IIZ"] if spectator else [])), coordinates)
    return tracker, circuit


def _validate_all_relations_protected(tracker, circuit, expected):
    physical, detectors, observables = _annotations(circuit)
    complete = _full_record_parities(physical)
    complete_basis = _basis(complete)
    emitted = detectors + observables
    assert all(_contains(complete_basis, row) for row in emitted)
    assert len(_basis(emitted)) == len(complete_basis) == expected
    assert len(_basis(emitted)) - len(_basis(detectors)) == expected
    assert physical.has_all_flows([
        stim.Flow(measurements=list(_bits(row))) for row in emitted], unsigned=False)
    assert len(complete_basis) == physical.count_determined_measurements()
    circuit.detector_error_model()
    tracker.validate_logical_count(context="Bell consistency after absorb normalization")
    assert tracker.expected_num_logicals == 0


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["M01", "M10"])
@pytest.mark.parametrize("first_readout", [0, 1], ids=["X_left", "X_right"])
def test_absorbed_bell_consistency_is_independent_of_measurement_order(order, first_readout):
    tracker, circuit = _bell_readout(order=order, first_readout=first_readout)
    # One Bell ZZ relation in the mid block and one final Z consistency.
    _validate_all_relations_protected(tracker, circuit, 2)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["M01", "M10"])
@pytest.mark.parametrize("first_readout", [0, 1], ids=["X_left", "X_right"])
@pytest.mark.parametrize("mixed_logical_basis", [False, True], ids=["ZZI_IIZ", "ZZZ_IIZ"])
@pytest.mark.parametrize("measured_spectator", [False, True], ids=["prepared_spectator", "recorded_spectator"])
def test_absorb_normalization_preserves_equivalent_L_bases_and_records(
        order, first_readout, mixed_logical_basis, measured_spectator):
    tracker, circuit = _bell_readout(
        order=order, first_readout=first_readout, spectator=True,
        mixed_logical_basis=mixed_logical_basis, measured_spectator=measured_spectator)
    # The independent spectator adds one Z consistency relation. In the
    # recorded case its initial measurement is random and must stay in parity.
    _validate_all_relations_protected(tracker, circuit, 3)


@pytest.mark.parametrize("prior_mid", [False, True], ids=["empty_record", "recorded_S"])
def test_projected_L_rref_keeps_zero_rows_from_postselected_S_overlap(prior_mid):
    """A retained selected S can annihilate a released L during the next B.

    This checks parity completeness and postselection preservation. The existing
    logical census does not account for this S/L overlap after its destruction;
    that separate baseline limitation is not an absorb-normalization assertion.
    """
    tracker = SyndromeTracker(2, expected_num_logicals=2, logical_tracking="slabs")
    tracker.stabilizers.add_stabilizers(_rows("ZI"), [[]])
    tracker.post_select_row_indices = {0}
    tracker.logicals.add_stabilizers(_rows("IZ"), [[]])
    tracker.absorbed_logicals.add_stabilizers(_rows("ZI"), [[]])
    circuit = stim.Circuit("R 0 1")
    if prior_mid:
        _mid(tracker, circuit, "ZI")
        tracker.post_select_row_indices = {0}

    _read(tracker, circuit, 1)
    assert tracker.post_select_row_indices
    circuit.append("MX", [0])
    tracker.process_mid_measurement(
        circuit=circuit, forward_symplectic_matrix=np.eye(4, dtype=np.uint8),
        back_propagated_paulis=_rows("XI"), reset_paulis=None,
        measurement_qubit_indices=[0], measurement_bases=["X"],
        measurement_coords=[(0, 0)], discarded_measurement_qubit_indices=set())
    circuit.append("MX", [0])
    tracker.process_data_measurement(circuit, _rows("XI"), {0: (0, 0), 1: (1, 0)})

    physical, detectors, observables = _annotations(circuit)
    complete = _basis(_full_record_parities(physical))
    emitted = detectors + observables
    assert len(observables) == 1
    assert len(detectors) == 1 + int(prior_mid)
    assert len(_basis(emitted)) == len(complete) == 2 + int(prior_mid)
    assert all(_contains(complete, row) for row in emitted)
    assert physical.has_all_flows([
        stim.Flow(measurements=list(_bits(row))) for row in emitted], unsigned=False)
    selected = [instruction for instruction in circuit
                if instruction.name == "DETECTOR" and instruction.tag == "post-select"]
    assert len(selected) == 1
    assert {target.value for target in selected[0].targets_copy()} == {-2, -1}
    assert tracker.post_select_row_indices == set()
    circuit.detector_error_model()
