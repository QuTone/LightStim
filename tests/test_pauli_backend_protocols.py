"""The acceleration engine must retain the merged tracker's surgery semantics."""

import numpy as np
import pytest
import stim

from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
from lightstim.utils import pauli_backend
from tests.tracker_test_utils import _annotations, _basis, _bits, _full_record_parities


pytestmark = pytest.mark.integration


def build_surgery(initial, final):
    experiment = TwoPatchLSExperiment(
        patch1_config={"distance": 3},
        patch2_config={"distance": 3},
        offset=(0, 6),
        interaction_type="ZZ",
        rounds=1,
        initial_state_patch1=initial[0],
        initial_state_patch2=initial[1],
        measure_state_patch1=final[0],
        measure_state_patch2=final[1],
        noise_params=None,
        rotate_patch1=True,
    )
    return experiment.build(), experiment.tracker


@pytest.mark.parametrize("initial,final,detectors,observables,parity_rank", [
    ("ZZ", "ZZ", 48, 3, 51),
    ("ZZ", "XZ", 48, 2, 50),
    ("XX", "ZZ", 48, 1, 49),
    ("ZX", "XZ", 48, 1, 49),
])
def test_surgery_acceleration_preserves_roles_history_and_complete_parities(
    monkeypatch, initial, final, detectors, observables, parity_rank,
):
    if not pauli_backend._NATIVE_AVAILABLE:
        pytest.skip("optional native Pauli backend has not been built")

    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "reference")
    reference_circuit, reference_tracker = build_surgery(initial, final)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    native_circuit, native_tracker = build_surgery(initial, final)

    # Preserve physical gates, record supports, annotation roles, IDs and tags.
    assert str(native_circuit) == str(reference_circuit)
    assert (native_circuit.num_detectors, native_circuit.num_observables) == (
        detectors, observables,
    )
    assert native_tracker.logical_history == reference_tracker.logical_history
    assert native_tracker.expected_num_logicals == reference_tracker.expected_num_logicals
    assert native_tracker._dressed_stabilizer_rows == reference_tracker._dressed_stabilizer_rows
    np.testing.assert_array_equal(native_tracker._v2_role_space, reference_tracker._v2_role_space)
    for name in ("stabilizers", "logicals", "absorbed_ops"):
        native_rows = getattr(native_tracker, name)
        reference_rows = getattr(reference_tracker, name)
        np.testing.assert_array_equal(native_rows.matrix, reference_rows.matrix)
        assert native_rows.records == reference_rows.records

    # A valid DEM alone cannot reveal a lost historical logical relation or a
    # relation reclassified as a detector. Check all physical parity dimensions.
    physical, detector_rows, observable_rows = _annotations(native_circuit)
    emitted = detector_rows + observable_rows
    assert len(_basis(emitted)) == parity_rank
    assert len(_basis(_full_record_parities(physical))) == parity_rank
    assert physical.count_determined_measurements() == parity_rank
    assert physical.has_all_flows([
        stim.Flow(measurements=list(_bits(row))) for row in emitted
    ], unsigned=False)
    native_circuit.detector_error_model()
