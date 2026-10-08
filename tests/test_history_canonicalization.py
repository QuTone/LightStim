"""Canonical basis changes preserve known record expressions for history.

An initialized eigenvalue is known even without a preceding measurement.
Conversely, the shared unknown sentinel is not an ordinary GF(2) variable.
"""

from types import SimpleNamespace
import contextlib
import io

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker, UNMEASURED_STAB_RECORD
from lightstim.qec_code.repetition.repetition import RepetitionCode


def _z(n, *qubits):
    row = np.zeros(2 * n, dtype=np.uint8)
    row[[n + q for q in qubits]] = 1
    return row


def _declared_z_checks(*supports):
    """Only the code declaration needed by stabilizer_canonicalization."""
    checks = [{"pauli": {q: "Z" for q in support}} for support in supports]
    return SimpleNamespace(
        active_stabilizer_indices=set(range(len(checks))),
        stabilizers=checks,
        effective_stabilizer=checks.__getitem__,
    )


def test_prepared_repetition_canonicalization_preserves_known_eigenvalues():
    system = QECSystem()
    system.add_patch(RepetitionCode(distance=3))
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    builder.initialize({q: "Z" for q in system.data_indices}, system.num_qubits)

    builder.stabilizer_canonicalization()

    assert tracker.stabilizers.records == [[], []]
    assert tracker.logicals.records == [[]]
    builder.apply_mid_data_readout({2: "Z"})
    assert len(tracker.logical_history) == 1
    assert tracker.logical_history[0].records == (0,)
    assert builder.circuit.has_flow(stim.Flow(measurements=[0]), unsigned=True)


def test_canonical_replacement_reconstructs_records_from_the_old_state():
    # The old conditional state has Z0 = +1 and Z1 = (-1)^m0. Its
    # canonical code stabilizer Z0 Z1 therefore has the record [0].
    tracker = SyndromeTracker(2, 1)
    tracker.stabilizers.add_stabilizers(_z(2, 0).reshape(1, -1), [[]])
    tracker.logicals.add_stabilizers(_z(2, 1).reshape(1, -1), [[0]])
    circuit = stim.Circuit("R 0\nH 1\nM 1")

    tracker.stabilizer_canonicalization(_declared_z_checks((0, 1)))

    np.testing.assert_array_equal(tracker.stabilizers.matrix, [_z(2, 0, 1)])
    assert tracker.stabilizers.records == [[0]]
    assert circuit.has_flow(stim.Flow(
        output=stim.PauliString("ZZ"), measurements=[0]), unsigned=True)


def test_unrelated_unknown_rows_cannot_cancel_their_sentinels():
    tracker = SyndromeTracker(3, 1)
    tracker.stabilizers.add_stabilizers(
        np.vstack([_z(3, 0), _z(3, 1), _z(3, 2)]),
        [[UNMEASURED_STAB_RECORD], [UNMEASURED_STAB_RECORD], []],
    )

    tracker.stabilizer_canonicalization(_declared_z_checks((0, 1), (1, 2)))

    # The product of two rows with unknown signs is still unknown. Merely
    # XORing [-1] with [-1] would incorrectly turn the first check into [].
    assert tracker.stabilizers.records == [[UNMEASURED_STAB_RECORD]] * 2


def test_known_empty_record_alternative_is_used_instead_of_unknown_rows():
    tracker = SyndromeTracker(3, 1)
    tracker.stabilizers.add_stabilizers(
        np.vstack([_z(3, 0), _z(3, 1), _z(3, 0, 1), _z(3, 2)]),
        [[UNMEASURED_STAB_RECORD], [UNMEASURED_STAB_RECORD], [], []],
    )

    tracker.stabilizer_canonicalization(_declared_z_checks((0, 1), (1, 2)))

    assert tracker.stabilizers.records == [[], [UNMEASURED_STAB_RECORD]]


def test_measured_code_row_keeps_its_historical_records():
    tracker = SyndromeTracker(3, 1)
    tracker.stabilizers.add_stabilizers(
        np.vstack([_z(3, 0, 1), _z(3, 1, 2), _z(3, 0)]),
        [[0, 2], [], []],
    )

    tracker.stabilizer_canonicalization(_declared_z_checks((0, 1), (1, 2)))

    assert tracker.stabilizers.records == [[0, 2], []]
    tracker.stabilizer_canonicalization(_declared_z_checks((0, 1), (1, 2)))
    assert tracker.stabilizers.records == [[0, 2], []]


def test_classified_roles_follow_partial_readout_and_clifford_updates():
    tracker = SyndromeTracker(3, 0)
    tracker.process_initialization(np.vstack([_z(3, q) for q in range(3)]))
    tracker.stabilizer_canonicalization(_declared_z_checks((0,), (1,), (2,)))
    tracker.process_unitary_block(stim.Circuit("H 2"))
    assert tracker._classified_stabilizer_rows == {0, 1, 2}
    circuit = stim.Circuit("R 0 1 2\nH 2\nM 0")
    tracker.process_data_measurement(
        circuit, _z(3, 0).reshape(1, -1), {q: (q, 0) for q in range(3)},
    )
    assert tracker.stabilizers.count == 2
    assert tracker._classified_stabilizer_rows == {0, 1}
    assert tracker.stabilizers.matrix[1, 2] == 1  # H moved Z2 to X2.


def test_classified_roles_reset_and_auxiliary_initialization_are_distinct():
    tracker = SyndromeTracker(2, 0)
    tracker.process_initialization(np.vstack([_z(2, 0), _z(2, 1)]))
    tracker.stabilizer_canonicalization(_declared_z_checks((0,), (1,)))
    tracker.process_resets(_z(2, 0).reshape(1, -1))
    # Reset's new preparation has not been classified by a protocol yet.
    assert tracker._classified_stabilizer_rows == {1}
    tracker.reset_records_for_qubits({0})
    assert tracker._classified_stabilizer_rows == {0}
    tracker.process_initialization(
        _z(2, 0).reshape(1, -1), stabilizer_qubits={0},
    )
    assert tracker._classified_stabilizer_rows == {0, 1}


@pytest.mark.parametrize("pqrm_state", ["X", "Z"])
def test_cross_ls_known_unmeasured_checks_keep_stabilizer_roles(pqrm_state):
    from lightstim.protocols.cross_ls import CrossLSExperiment

    experiment = CrossLSExperiment(
        PQRM_para=[1, 2, 4], d_surf=3, rounds=2,
        PQRM_state=pqrm_state, noise_params=None,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        circuit = experiment.build()
    # Canonical PQRM checks and fresh bridge preparation can have empty
    # records; neither may spuriously become a protected logical direction.
    experiment.tracker.validate_logical_count(context="CrossLS complete")
    for relation in experiment.tracker.logical_history:
        assert circuit.has_flow(stim.Flow(measurements=relation.records), unsigned=True)
    circuit.detector_error_model()
    detectors, observables = circuit.compile_detector_sampler(seed=58).sample(
        64, separate_observables=True,
    )
    assert not detectors.any() and not observables.any()
