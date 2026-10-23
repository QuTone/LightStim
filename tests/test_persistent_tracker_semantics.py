"""A persistent physical engine must preserve every current tracker boundary.

These comparisons keep the Pauli kernels enabled in both runs, so they isolate
the tracking/record engine instead of conflating it with linear-algebra speedups.
Intermediate snapshots also cover the copied trackers used by repeat probes.
"""

import copy
from functools import wraps

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.bacon_shor import BaconShorCode, BaconShorCodeExtractionBlock
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)
from tests.test_pauli_backend_protocols import build_surgery
from tests.test_tracker_v2_builder_state import _postselected_transport
from tests.test_tracker_v2_native import _two_patch_circuit
from tests.test_tracker_v2_readout import _assert_complete, _read, _rows, _tracker


pytestmark = pytest.mark.integration


@pytest.fixture
def persistent_extension():
    extension = pytest.importorskip("lightstim.utils.cpp._pauli_backend_cpp")
    if not hasattr(extension, "PersistentRecordState"):
        pytest.skip("optional persistent tracking extension has not been built")
    return extension


def _freeze(value):
    if isinstance(value, np.ndarray):
        # Existing row-space operations may return int64 while the native
        # state exports uint8. The public binary entries and shape are the
        # contract; the incidental integer storage width is not.
        return (value.shape, np.asarray(value, dtype=np.int64).tobytes())
    if isinstance(value, dict):
        return tuple(sorted((repr(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze(item) for item in value))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _snapshot(tracker):
    names = (
        "num_qubits", "expected_num_logicals", "_declared_num_logicals",
        "total_measurements", "total_observables", "meas_rec_to_idx_map",
        "logical_history", "_logical_history_observable_ids",
        "_classified_stabilizer_rows", "_dressed_stabilizer_rows",
        "_v2_role_space", "stabilizer_with_logical_components",
        "_gauge_logical_vectors", "post_select_row_indices",
    )
    result = {name: _freeze(getattr(tracker, name)) for name in names}
    for name in ("stabilizers", "logicals", "absorbed_ops"):
        tableau = getattr(tracker, name)
        result[name] = (_freeze(tableau.matrix), _freeze(tableau.records))
    return result


_BOUNDARIES = (
    "expand", "process_initialization", "process_resets",
    "reset_records_for_qubits", "process_unitary_block",
    "process_mid_measurement", "process_data_measurement",
    "stabilizer_canonicalization", "logical_canonicalization",
    "classify_subsystem_state", "rebase_stabilizers_onto_code_basis",
    "promote_stabilizer_rows_to_logicals",
)


def _traced_build(monkeypatch, factory, mode):
    trace, trackers = [], {}
    with monkeypatch.context() as scoped:
        scoped.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
        scoped.setenv("LIGHTSTIM_TRACKER_BACKEND", mode)
        for name in _BOUNDARIES:
            original = getattr(SyndromeTracker, name)

            @wraps(original)
            def record_boundary(tracker, *args, _name=name, _original=original, **kwargs):
                result = _original(tracker, *args, **kwargs)
                trackers[id(tracker)] = tracker
                trace.append((_name, _snapshot(tracker)))
                return result

            scoped.setattr(SyndromeTracker, name, record_boundary)
        result = factory()
        circuit = result[0] if isinstance(result, tuple) else result
    return circuit, trace, tuple(trackers.values())


def _assert_same_build(monkeypatch, factory):
    reference, expected_trace, _ = _traced_build(monkeypatch, factory, "reference")
    persistent, actual_trace, trackers = _traced_build(monkeypatch, factory, "persistent")
    assert str(persistent) == str(reference)
    assert len(actual_trace) == len(expected_trace)
    for index, (actual, expected) in enumerate(zip(actual_trace, expected_trace)):
        assert actual == expected, f"tracker boundary {index}: {expected[0]}"
    engines = [tracker._record_state for tracker in trackers
               if tracker._record_state is not None]
    assert engines, "persistent mode must execute the native tracking engine"
    assert all(engine.stats["constructions"] == 1 for engine in engines)
    persistent.detector_error_model()
    assert not persistent.compile_detector_sampler(seed=71).sample(
        32, append_observables=True).any()
    return persistent, trackers


@pytest.mark.parametrize("initial,final", [
    ("ZZ", "ZZ"), ("ZZ", "XZ"), ("XX", "ZZ"), ("ZX", "XZ"),
])
def test_persistent_surgery_preserves_intermediate_roles_and_history(
    persistent_extension, monkeypatch, initial, final,
):
    _, trackers = _assert_same_build(monkeypatch, lambda: build_surgery(initial, final))
    stats = [tracker._record_state.stats for tracker in trackers
             if tracker._record_state is not None]
    # Explicit instrumentation prevents a differential pass caused by silently
    # running both circuit builds through the original tracking implementation.
    for operation in ("queries", "unitaries", "folds"):
        assert sum(item[operation] for item in stats) > 0, operation


def _dynamic_subsystem():
    system = QECSystem()
    system.add_patch(BaconShorCode(distance=3), name="first")
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    system.register_tracker(tracker)
    system.register_builder(builder)

    def extract():
        block = BaconShorCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(
            block.circuit, measurement_blocks=block.measurement_blocks)

    builder.initialize({q: "X" for q in system.data_indices}, system.num_qubits)
    extract()
    previous_data = set(system.data_indices)
    system.add_patch(BaconShorCode(distance=3), name="second", offset=(20, 0))
    builder.initialize({q: "X" for q in set(system.data_indices) - previous_data},
                       system.num_qubits)
    extract()
    builder.apply_data_readout({q: "X" for q in system.data_indices})
    assert tracker.expected_num_logicals == 0
    return builder.to_stim_circuit(), tracker


def _steady_rounds():
    system = QECSystem()
    system.add_patch(UnrotatedSurfaceCode(distance=3), name="memory")
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    builder.initialize({q: "Z" for q in system.data_indices}, system.num_qubits)
    builder.apply_syndrome_extraction(
        UnrotatedSurfaceCodeExtractionBlock(system).circuit, rounds=7)
    builder.apply_data_readout({q: "Z" for q in system.data_indices})
    circuit = builder.to_stim_circuit()
    assert any(isinstance(instruction, stim.CircuitRepeatBlock) for instruction in circuit)
    return circuit, tracker


@pytest.mark.parametrize("factory", [
    _dynamic_subsystem,
    _steady_rounds,
    lambda: _two_patch_circuit("ZZ", "ZZ", 1, 2, True),
    lambda: _postselected_transport(4, buffered=True),
], ids=["dynamic_noncommuting_gauges", "steady_repeat_probe",
        "staged_surgery_readout", "buffered_postselection"])
def test_persistent_builder_boundaries(persistent_extension, monkeypatch, factory):
    _assert_same_build(monkeypatch, factory)


def _partial_selected_readout():
    tracker = _tracker(3, stabilizers=("ZIZ",), logicals=("ZII", "IZI"),
                       dressed=(0,), selected=(0,))
    circuit = stim.Circuit("R 0 1 2")
    _read(tracker, circuit, [0], active=("IIZ",))
    assert tracker.post_select_row_indices
    _read(tracker, circuit, [2])
    assert not tracker.post_select_row_indices
    _read(tracker, circuit, [1])
    _assert_complete(tracker, circuit)
    return circuit, tracker


def _record_only_relation():
    tracker = _tracker(1, stabilizers=("Z",), logicals=("Z",), dressed=(0,),
                       stab_records=[[0]], logical_records=[[1]])
    tracker.total_measurements = 2
    circuit = stim.Circuit("RX 0\nM 0 0")
    tracker.process_data_measurement(circuit, _rows(n=1), {0: (0, 0)},
                                     active_stabilizers=_rows(n=1))
    assert circuit.num_observables == 1
    assert circuit.has_flow(stim.Flow(measurements=[0, 1]), unsigned=False)
    _assert_complete(tracker, circuit)
    return circuit, tracker


def _copied_and_mutated_state():
    tracker = SyndromeTracker(2)
    tracker.process_initialization(_rows("ZI", "IZ", n=2))
    tracker.process_unitary_block(stim.Circuit("CX 0 1"))
    before = _snapshot(tracker)
    probe = copy.deepcopy(tracker)
    probe.process_unitary_block(stim.Circuit("H 0"))
    probe.process_resets(_rows("ZI", n=2))
    assert _snapshot(tracker) == before

    # Direct public ndarray mutation must invalidate/reconcile native state.
    # Replacing Z0 by Z0*Z1 preserves this all-zero state's known eigenvalue.
    tracker.stabilizers.matrix[0] ^= tracker.stabilizers.matrix[1]
    tracker.process_unitary_block(stim.Circuit("H 0\nH 0"))
    circuit = stim.Circuit("R 0 1\nCX 0 1")
    _read(tracker, circuit, [0, 1])
    _assert_complete(tracker, circuit)
    return circuit, tracker


@pytest.mark.parametrize("factory", [
    _partial_selected_readout, _record_only_relation, _copied_and_mutated_state,
], ids=["selected_subspace_transport", "record_only_relation", "deepcopy_and_public_mutation"])
def test_persistent_records_and_public_state_contract(
    persistent_extension, monkeypatch, factory,
):
    _assert_same_build(monkeypatch, factory)
