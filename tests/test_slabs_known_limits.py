"""Constructed-state check of low-level logical-count bookkeeping.

This directly populates tracker tableaus; it is not a native benchmark failure.
Fixing the count should produce XPASS and require removing its marker.
Only assertion failures are expected; crashes remain ordinary test failures.
"""
import numpy as np
import pytest
import stim

from lightstim.ir.tracker import SyndromeTracker
from tests.test_slabs_acceptance import (
    _annotations, _basis, _bits, _contains, _full_record_parities,
)
from tests.test_slabs_absorb_normalization import _rows
from tests.test_slabs_partial_readout import _read


def _require_valid_complete_parities(circuit):
    physical, detectors, observables = _annotations(circuit)
    complete = _basis(_full_record_parities(physical))
    emitted = detectors + observables
    if not all(_contains(complete, row) for row in emitted):
        raise RuntimeError("Known-limit reproduction emitted an invalid physical relation")
    if len(_basis(emitted)) != len(complete):
        raise RuntimeError("Known-limit reproduction lost physical parity completeness")
    if not physical.has_all_flows([
            stim.Flow(measurements=list(_bits(row))) for row in emitted], unsigned=False):
        raise RuntimeError("Known-limit reproduction emitted an invalid signed relation")
    circuit.detector_error_model()
    return detectors, observables


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "Existing PS-overlap census limitation: an S pivot annihilates overlapping L "
    "but expected_num_logicals remains 1 after complete destructive readout"))
def test_postselected_S_L_overlap_is_removed_from_final_logical_census():
    tracker = SyndromeTracker(2, expected_num_logicals=2, logical_tracking="slabs")
    tracker.stabilizers.add_stabilizers(_rows("ZI"), [[]])
    tracker.post_select_row_indices = {0}
    tracker.logicals.add_stabilizers(_rows("IZ"), [[]])
    tracker.absorbed_logicals.add_stabilizers(_rows("ZI"), [[]])
    circuit = stim.Circuit("R 0 1")
    _read(tracker, circuit, 1)
    circuit.append("MX", [0])
    tracker.process_mid_measurement(
        circuit=circuit, forward_symplectic_matrix=np.eye(4, dtype=np.uint8),
        back_propagated_paulis=_rows("XI"), reset_paulis=None,
        measurement_qubit_indices=[0], measurement_bases=["X"],
        measurement_coords=[(0, 0)], discarded_measurement_qubit_indices=set())
    circuit.append("MX", [0])
    tracker.process_data_measurement(circuit, _rows("XI"), {0: (0, 0), 1: (1, 0)})
    _require_valid_complete_parities(circuit)
    if tracker.logicals.count or tracker.absorbed_logicals.count:
        raise RuntimeError("Known-limit reproduction unexpectedly retained a logical Pauli")
    assert tracker.expected_num_logicals == 0, (
        f"All logical Paulis were destroyed or read, but census is {tracker.expected_num_logicals}")
