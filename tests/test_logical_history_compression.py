"""Repeated-round probing must not discard record-only logical history."""

import copy
from types import SimpleNamespace

import numpy as np
import stim

from lightstim.ir.builder import CircuitBuilder, _MeasurementBlockAnalysis
from lightstim.ir.tracker import SyndromeTracker


def _builder_and_analysis(*, measure_joint):
    # The two qubits start in |00>, with Z0 classified as a stabilizer and
    # Z1 as the encoded logical state. Measuring Z0*Z1 has a deterministic
    # logical-dependent parity while leaving Z1 explicitly represented.
    tracker = SyndromeTracker(2, expected_num_logicals=1)
    tracker.stabilizers.matrix = np.array([[0, 0, 1, 0]], dtype=np.uint8)
    tracker.stabilizers.records = [[]]
    tracker.logicals.matrix = np.array([[0, 0, 0, 1]], dtype=np.uint8)
    tracker.logicals.records = [[]]
    builder = CircuitBuilder(tracker, SimpleNamespace(active_gauges=()))
    builder.circuit = stim.Circuit("R 0 1")
    if measure_joint:
        circuit = stim.Circuit("CX 1 0\nM 0")
        # Forward conjugation by CX(1, 0), in [X0, X1, Z0, Z1] order.
        forward = np.array([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ], dtype=np.uint8)
        backward = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    else:
        circuit = stim.Circuit("M 0")
        forward = np.eye(4, dtype=np.uint8)
        backward = np.array([[0, 0, 1, 0]], dtype=np.uint8)
    analysis = _MeasurementBlockAnalysis(
        circuit=circuit,
        forward_symplectic_matrix=forward,
        back_propagated_paulis=backward,
        reset_paulis=None,
        measurement_qubit_indices=[0],
        measurement_bases=["Z"],
        measurement_coords=[(0, 0)],
        discarded_measurement_qubit_indices=set(),
        no_detector_mask=None,
    )
    return builder, analysis


def _assert_tracker_unchanged(actual, before):
    for attribute in (
        "total_measurements", "total_observables", "expected_num_logicals",
        "meas_rec_to_idx_map", "logical_history",
        "stabilizer_with_logical_components", "post_select_row_indices",
    ):
        assert getattr(actual, attribute) == getattr(before, attribute)
    for attribute in ("stabilizers", "logicals", "absorbed_ops"):
        np.testing.assert_array_equal(
            getattr(actual, attribute).matrix, getattr(before, attribute).matrix
        )
        assert getattr(actual, attribute).records == getattr(before, attribute).records
    assert len(actual._gauge_logical_vectors) == len(before._gauge_logical_vectors)
    for actual_vector, before_vector in zip(
        actual._gauge_logical_vectors, before._gauge_logical_vectors
    ):
        np.testing.assert_array_equal(actual_vector, before_vector)


def test_existing_history_declines_shortcut_even_when_next_round_adds_none():
    builder, joint_analysis = _builder_and_analysis(measure_joint=True)
    builder._process_measurement_blocks(
        output_circuit=builder.circuit,
        analyses=(joint_analysis,),
        shift_round=True,
    )
    assert [relation.records for relation in builder.tracker.logical_history] == [(0,)]
    assert builder.tracker.logicals.count == 1
    assert not builder.tracker.stabilizer_with_logical_components
    assert builder.tracker.absorbed_ops.count == 0

    _, stabilizer_analysis = _builder_and_analysis(measure_joint=False)
    # A later stabilizer-only block clears the transient logical-vector
    # metadata. The persistent history must itself prevent the shortcut.
    builder._process_measurement_blocks(
        output_circuit=builder.circuit,
        analyses=(stabilizer_analysis,),
        shift_round=True,
    )
    assert not builder.tracker._gauge_logical_vectors
    assert [relation.records for relation in builder.tracker.logical_history] == [(0,)]
    before = copy.deepcopy(builder.tracker)
    circuit_before = builder.circuit.copy()
    assert builder._try_compress_steady_rounds(
        repetitions=4, analyses=(stabilizer_analysis,)
    ) is None
    _assert_tracker_unchanged(builder.tracker, before)
    assert builder.circuit == circuit_before


def test_new_history_discovered_by_real_probe_requires_explicit_updates(monkeypatch):
    builder, analysis = _builder_and_analysis(measure_joint=True)
    before = copy.deepcopy(builder.tracker)
    circuit_before = builder.circuit.copy()
    observed_probes = []
    process_blocks = builder._process_measurement_blocks

    def observe_real_processing(**kwargs):
        process_blocks(**kwargs)
        probe = kwargs.get("tracker")
        if probe is not None:
            observed_probes.append(probe)

    monkeypatch.setattr(builder, "_process_measurement_blocks", observe_real_processing)
    assert builder._try_compress_steady_rounds(
        repetitions=4, analyses=(analysis,)
    ) is None

    # The actual measurement/decomposition path found this relation on the
    # copied tracker; neither merely entering the probe nor throwing a state
    # validation exception would establish this behavior.
    assert len(observed_probes) == 1
    probe = observed_probes[0]
    assert probe is not builder.tracker
    assert [relation.records for relation in probe.logical_history] == [(0,)]
    probe.validate_logical_count()
    _assert_tracker_unchanged(builder.tracker, before)
    assert builder.circuit == circuit_before

    # Ordinary updates retain each record index instead of advancing only the
    # state frontier. Two repeated rounds create two independent known parities.
    for _ in range(2):
        builder._process_measurement_blocks(
            output_circuit=builder.circuit,
            analyses=(analysis,),
            shift_round=True,
        )
    relations = builder.tracker.logical_history
    assert [relation.records for relation in relations] == [(0,), (0, 1)]
    assert builder.tracker.total_measurements == 2
    assert builder.circuit.has_all_flows(
        [stim.Flow(measurements=relation.records) for relation in relations],
        unsigned=True,
    )
    assert builder.circuit.count_determined_measurements() == 2
    assert builder.append_logical_history_observables() == [0, 1]
