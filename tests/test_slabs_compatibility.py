"""The S/L/L_absorbed path retains existing preparation and postselection.

These tests reuse the existing physical CrossLS and seven-input distillation
oracles.  They explicitly select the new tracker path even when the protocol
constructor does not expose the experimental option.
"""
from contextlib import contextmanager

import numpy as np
import pytest
import stim

from lightstim.ir.tracker import SyndromeTracker
from lightstim.protocols.state_injection import StateInjectionExperiment
from tests import test_cross_ls_postselection as cross_oracle
from tests import test_distillation_logical_history as distillation_oracle
from tests.conftest import assert_dem_valid, assert_noiseless, build_quiet


@contextmanager
def _slabs_tracker(monkeypatch):
    original_init = SyndromeTracker.__init__

    def initialize(self, *args, **kwargs):
        kwargs["logical_tracking"] = "slabs"
        original_init(self, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(SyndromeTracker, "__init__", initialize)
        # The existing geometry oracle caches complete protocol constructions.
        # Never reuse a legacy build in a test of the new path, or vice versa.
        cross_oracle._build.cache_clear()
        try:
            yield
        finally:
            cross_oracle._build.cache_clear()


@pytest.mark.parametrize("state", ["Z", "X", "Y"])
@pytest.mark.parametrize("mode", ["full_postselection", "hybrid", "full_qec"])
def test_slabs_injection_preserves_selected_parity_space(monkeypatch, state, mode):
    def build():
        return StateInjectionExperiment(
            distance=3, rounds=2, inject_state=state,
            post_select_mode=mode, noise_params=None,
        ).build()

    legacy = build_quiet(build)
    with _slabs_tracker(monkeypatch):
        candidate = build_quiet(build)
    # Identical physical operations make absolute-record parity comparison
    # meaningful even if annotations use another independent basis.
    def physical(circuit):
        return [instruction for instruction in circuit.flattened()
                if instruction.name not in {"DETECTOR", "OBSERVABLE_INCLUDE"}]

    assert physical(candidate) == physical(legacy)
    cross_oracle._assert_same_space(
        cross_oracle._selected_rows(candidate),
        cross_oracle._selected_rows(legacy),
    )
    assert_noiseless(candidate)
    assert_dem_valid(candidate)


@pytest.mark.parametrize("state", ["Z", "X", "Y"])
@pytest.mark.parametrize("mode", ["pqrm_only", "hybrid", "surface_se"])
def test_slabs_cross_ls_preserves_physical_postselection(monkeypatch, state, mode):
    with _slabs_tracker(monkeypatch):
        build_quiet(lambda: cross_oracle.
                    test_cross_ls_selects_entire_pqrm_x_constraint_space(
                        state, 2, mode))


@pytest.mark.parametrize("mode", ["pqrm_only", "hybrid", "surface_se"])
def test_slabs_cross_ls_rejects_each_terminal_pqrm_fault(monkeypatch, mode):
    with _slabs_tracker(monkeypatch):
        build_quiet(lambda: cross_oracle.
                    test_cross_ls_postselection_accepts_clean_and_rejects_pqrm_fault(mode))


@pytest.mark.parametrize("canonical", [False, True])
def test_slabs_cross_ls_keeps_optional_canonicalization(monkeypatch, canonical):
    with _slabs_tracker(monkeypatch):
        build_quiet(lambda: cross_oracle.
                    test_cross_ls_optional_canonicalization_preserves_selection(canonical))


def test_slabs_ls_distillation_repreparation_and_all_input_fault_patterns(monkeypatch):
    # Includes three W5 readout/repreparation cycles and all 128 combinations
    # of seven input logical faults, with independent exact acceptance counts.
    with _slabs_tracker(monkeypatch):
        distillation_oracle.test_distillation_all_seven_preparation_fault_patterns("ls")


def test_slabs_projection_uses_first_S_even_with_an_absorbed_copy():
    """Redundancy against L_absorbed must not disqualify an S pivot.

    S=[X0,X1], L_absorbed=[X0]. Measuring Z0Z1 pivots on X0: S retains
    X0X1 and the absorbed copy becomes identity. The continuing code declares
    X0X1 active, so its final MX closure is a detector. This replaces the
    experimental test's former independent-row prefilter expectation.
    """
    def rows(*paulis):
        return np.asarray([
            np.concatenate(stim.PauliString(p).to_numpy()) for p in paulis
        ], dtype=np.uint8)

    tracker = SyndromeTracker(2, expected_num_logicals=1, logical_tracking="slabs")
    tracker.stabilizers.add_stabilizers(rows("X_", "_X"))
    tracker.absorbed_logicals.add_stabilizers(rows("X_"))
    circuit = stim.Circuit("RX 0 1\nMPP Z0*Z1")
    coordinates = {0: (0.0, 0.0), 1: (1.0, 0.0)}
    tracker.process_data_measurement(circuit, rows("ZZ"), coordinates,
                                     active_stabilizers=rows("XX"))
    np.testing.assert_array_equal(tracker.stabilizers.matrix, rows("XX"))
    assert tracker.stabilizers.records == [[]]
    assert tracker.logicals.count == tracker.absorbed_logicals.count == 0
    tracker.validate_logical_count(context="projected joint readout")

    circuit.append("MX", [0, 1])
    tracker.process_data_measurement(circuit, rows("X_", "_X"), coordinates)
    assert circuit.num_observables == 0
    assert circuit.num_detectors == 1
    assert_noiseless(circuit)
    assert_dem_valid(circuit)
    assert circuit.has_flow(stim.Flow(measurements=[1, 2]))
    faulty = stim.Circuit("RX 0 1\nMPP Z0*Z1\nZ_ERROR(1) 0")
    faulty += circuit[2:]
    detectors, observables = faulty.compile_detector_sampler(seed=8).sample(
        8, separate_observables=True)
    assert detectors.all()
    assert observables.shape == (8, 0)


@pytest.mark.parametrize("entry", ["process_resets", "process_initialization", "reset_records_for_qubits"])
def test_slabs_reset_traces_bell_partner_and_keeps_spectator_records(entry):
    """Resetting half a Bell pair must leave the other half undetermined.

    The unrelated third qubit retains its measured logical value, its absolute
    record, and its one protected degree of freedom through both reset APIs.
    """
    def rows(*paulis):
        return np.asarray([
            np.concatenate(stim.PauliString(p).to_numpy()) for p in paulis
        ], dtype=np.uint8)

    tracker = SyndromeTracker(3, expected_num_logicals=2, logical_tracking="slabs")
    tracker.stabilizers.add_stabilizers(rows("XX_", "ZZ_"))
    tracker.logicals.add_stabilizers(rows("__Z"), [[0]])
    tracker.absorbed_logicals.add_stabilizers(rows("XX_"))
    tracker.total_measurements = 1
    if entry == "reset_records_for_qubits":
        tracker.reset_records_for_qubits([0])
        tracker.process_initialization(rows("Z__"))
    else:
        getattr(tracker, entry)(rows("Z__"))
    np.testing.assert_array_equal(tracker.stabilizers.matrix, rows("Z__"))
    np.testing.assert_array_equal(tracker.logicals.matrix, rows("__Z"))
    assert tracker.stabilizers.records == [[]]
    assert tracker.logicals.records == [[0]]
    assert tracker.absorbed_logicals.count == 0
    assert tracker.absorbed_logicals.records == []
    assert tracker.expected_num_logicals == 1
    tracker.validate_logical_count(context="reset after absorbed Bell preparation")

    circuit = stim.Circuit("RX 2\nM 2\nR 0 1\nH 0\nCX 0 1\nR 0\nM 0 1 2")
    tracker.process_data_measurement(
        circuit, rows("Z__", "_Z_", "__Z"),
        {q: (float(q), 0.0) for q in range(3)},
    )
    assert circuit.num_detectors == circuit.num_observables == 1
    assert_noiseless(circuit)
    assert_dem_valid(circuit)
    assert circuit.has_flow(stim.Flow(measurements=[0, 3]))
    assert not circuit.has_flow(stim.Flow(measurements=[2]))


def test_slabs_legacy_absorbed_insertion_keeps_record_expression():
    tracker = SyndromeTracker(2, logical_tracking="slabs")
    tracker.stabilizers.add_stabilizers(np.asarray([[0, 0, 1, 0]], dtype=np.uint8), [[1]])
    tracker.logicals.add_stabilizers(np.asarray([[0, 0, 1, 1]], dtype=np.uint8), [[0]])
    assert tracker.record_absorbed_op(np.asarray([0, 0, 1, 1], dtype=np.uint8))
    np.testing.assert_array_equal(tracker.absorbed_logicals.matrix, [[0, 0, 0, 1]])
    assert tracker.absorbed_logicals.records == [[0, 1]]
    physical = stim.Circuit("RX 0 1\nMPP Z0*Z1\nM 0")
    assert physical.has_flow(stim.Flow(output=stim.PauliString("_Z"), measurements=[0, 1]))


def test_slabs_random_measurement_banks_new_value_without_claiming_old_value():
    """Destroying input X does not make the new Z outcome deterministic.

    Its measured value is nevertheless known conditionally and must support a
    later Z-readout consistency observable. The old input constraint is gone.
    """
    tracker = SyndromeTracker(1, expected_num_logicals=1, logical_tracking="slabs")
    tracker.logicals.add_stabilizers(np.asarray([[1, 0]], dtype=np.uint8))
    circuit = stim.Circuit("RX 0\nM 0")
    tracker.process_mid_measurement(
        circuit=circuit, forward_symplectic_matrix=np.eye(2, dtype=np.uint8),
        back_propagated_paulis=np.asarray([[0, 1]], dtype=np.uint8),
        reset_paulis=None, measurement_qubit_indices=[0],
        measurement_bases=["Z"], measurement_coords=[(0, 0)],
        discarded_measurement_qubit_indices=set(),
    )
    assert tracker.logical_history == []
    assert tracker.logicals.count == 0
    np.testing.assert_array_equal(tracker.absorbed_logicals.matrix, [[0, 1]])
    assert tracker.absorbed_logicals.records == [[0]]
    assert tracker.expected_num_logicals == 1
    tracker.validate_logical_count(context="new conditional Z constraint")
    circuit.append("M", [0])
    tracker.process_data_measurement(
        circuit, np.asarray([[0, 1]], dtype=np.uint8), {0: (0.0, 0.0)})
    assert circuit.num_detectors == 0
    assert circuit.num_observables == 1
    assert circuit.has_flow(stim.Flow(measurements=[0, 1]))
    assert not circuit.has_flow(stim.Flow(measurements=[0]))
    assert_noiseless(circuit)
