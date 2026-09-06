"""Physical state inference, including products, partial rounds and preparation."""

import copy

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.bacon_shor import BaconShorCode
from lightstim.qec_code.generic_css import GenericCSSGaugeExtractionBlock
from lightstim.utils.linear_algebra import check_commutativity
from lightstim.utils.subsystem_algebra import (
    combine_records, intersect_row_spaces, reduce_modulo,
)
from lightstim.utils.tableau_utils import stabilizers_to_symplectic


def make_builder(*, bells=False):
    system = QECSystem()
    system.add_patch(BaconShorCode(distance=2), name="bs")
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    builder.initialize({q: "Z" for q in system.data_indices}, system.num_qubits)
    if bells:
        builder.apply_unitary_block(stim.Circuit("H 0 2\nCX 0 1 2 3"))
    return builder


def measure_gauge(builder, basis, index=0):
    gauge = [g for g in builder.system.active_gauges if g["type"] == basis][index]
    ancilla = gauge["syn_idx"]
    block = stim.Circuit()
    block.append("RX" if basis == "X" else "R", [ancilla])
    block.append("TICK")
    for data in gauge["data_indices"]:
        block.append("CX", [ancilla, data] if basis == "X" else [data, ancilla])
        block.append("TICK")
    block.append("MX" if basis == "X" else "M", [ancilla])
    builder.apply_syndrome_extraction(block)


def assert_same_span(left, right):
    assert not reduce_modulo(left, right).any()
    assert not reduce_modulo(right, left).any()


def assert_physical_flows(builder, tableau):
    for row, records in zip(tableau.matrix, tableau.records):
        n = builder.tracker.num_qubits
        pauli = stim.PauliString.from_numpy(xs=row[:n].astype(bool), zs=row[n:].astype(bool))
        assert builder.circuit.has_flow(stim.Flow(output=pauli, measurements=records), unsigned=False)


def assert_phase(builder, basis):
    fixed = builder.tracker.infer_gauge_fixed_stabilizers(builder.system)
    expected = stabilizers_to_symplectic(
        builder.system,
        [g for g in builder.system.active_gauges if g["type"] == basis]
        + [s for s in builder.system.active_stabilizers if s["type"] != basis],
        builder.system.num_qubits,
    )
    assert fixed.count == 3
    assert_same_span(fixed.matrix, expected)
    assert_physical_flows(builder, fixed)
    assert_physical_flows(builder, builder.tracker.logicals)
    assert builder.tracker.logicals.count == builder.tracker.expected_num_logicals == 1
    gauges = stabilizers_to_symplectic(builder.system, builder.system.active_gauges, builder.system.num_qubits)
    assert not check_commutativity(builder.tracker.logicals.matrix, gauges).any()


def test_intersection_finds_products_and_xors_record_lineage():
    # Neither original state row is in the right span; their product is.
    left = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.uint8)
    right = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    intersection, coefficients = intersect_row_spaces(left, right)
    assert_same_span(intersection, right)
    np.testing.assert_array_equal(intersection, (coefficients @ left) % 2)
    assert combine_records(coefficients, [[1, 3], [2, 3]]) == [[1, 2]]


@pytest.mark.parametrize("left_count,right_count", [(0, 0), (0, 2), (2, 0)])
def test_empty_intersections_have_consistent_shapes(left_count, right_count):
    rows, coefficients = intersect_row_spaces(
        np.zeros((left_count, 8), dtype=np.uint8), np.zeros((right_count, 8), dtype=np.uint8),
    )
    assert rows.shape == (0, 8)
    assert coefficients.shape == (0, left_count)


def test_bell_preparation_requires_span_intersection_and_both_tableaus():
    builder = make_builder(bells=True)
    tracker = builder.tracker
    original = tracker.stabilizers.matrix.copy()
    gauges = stabilizers_to_symplectic(builder.system, builder.system.active_gauges, tracker.num_qubits)
    individually_in_g = [row for row in original if not reduce_modulo(row[None, :], gauges).any()]
    assert len(individually_in_g) == 2
    assert tracker.infer_gauge_fixed_stabilizers(builder.system).count == 3
    # A previous classification can leave a gauge direction in the logical bank.
    tracker.logicals.add_stabilizers(tracker.stabilizers.matrix[:1].copy(), [tracker.stabilizers.records[0]])
    tracker.stabilizers.remove_rows([0])
    before = copy.deepcopy(tracker)
    assert tracker.infer_gauge_fixed_stabilizers(builder.system).count == 3
    np.testing.assert_array_equal(tracker.stabilizers.matrix, before.stabilizers.matrix)
    assert tracker.logicals.records == before.logicals.records
    builder.stabilizer_canonicalization()
    assert_phase(builder, "X")
    assert_same_span(original, np.vstack([tracker.stabilizers.matrix, tracker.logicals.matrix]))


def test_partial_omitted_and_repeated_gauges_infer_unmeasured_products():
    builder = make_builder(bells=True)
    declaration = copy.deepcopy((builder.system.stabilizers, builder.system.gauges,
                                 builder.system.active_stabilizer_indices, builder.system.active_gauge_indices))
    builder.stabilizer_canonicalization()
    assert_phase(builder, "X")
    for step, (basis, index) in enumerate([("Z", 0), ("Z", 0), ("X", 0), ("X", 1), ("Z", 1)]):
        old_detectors = builder.circuit.num_detectors
        measure_gauge(builder, basis, index)
        assert_phase(builder, basis)
        if step == 1:
            assert builder.circuit.num_detectors == old_detectors + 1
    assert declaration == (builder.system.stabilizers, builder.system.gauges,
                           builder.system.active_stabilizer_indices, builder.system.active_gauge_indices)
    builder.apply_data_readout({q: "Z" for q in builder.system.data_indices})
    assert builder.circuit.num_observables == 1
    assert not builder.circuit.compile_detector_sampler(seed=7).sample(256, append_observables=True).any()
    builder.circuit.detector_error_model()


def test_preparation_preserves_physical_constraints_until_center_established():
    builder = make_builder()
    original = builder.tracker.stabilizers.matrix.copy()
    builder.stabilizer_canonicalization()
    assert builder.tracker.infer_gauge_fixed_stabilizers(builder.system).count == 2
    assert builder.tracker.logicals.count == 1
    assert_same_span(original, np.vstack([builder.tracker.stabilizers.matrix, builder.tracker.logicals.matrix]))
    assert all(r >= 0 for records in builder.tracker.stabilizers.records for r in records)
    measure_gauge(builder, "X", 0)
    before = builder.circuit.copy()
    with pytest.raises(RuntimeError, match="centre is not yet established"):
        builder.apply_data_readout()
    assert builder.circuit == before  # Failure precedes destructive readout.
    measure_gauge(builder, "X", 1)
    assert_phase(builder, "X")
    measure_gauge(builder, "Z", 0)
    assert_phase(builder, "Z")
    builder.apply_data_readout()
    builder.circuit.detector_error_model()


class LogicalAndGauge(QECPatch):
    def _process_params(self):
        pass

    def build(self):
        self.add_qubit(0, 0, "data")
        self.add_qubit(1, 0, "data")
        self.create_stim_gauge({(1, 0): "X"}, type="X")
        self.create_stim_gauge({(1, 0): "Z"}, type="Z")
        self.num_logicals = 1


def test_logical_gauge_entanglement_is_not_counted_as_two_known_logicals():
    system = QECSystem()
    system.add_patch(LogicalAndGauge(), name="mixed")
    tracker = SyndromeTracker(2, 1)
    builder = CircuitBuilder(tracker, system)
    builder.initialize({0: "Z", 1: "Z"}, 2)
    builder.apply_unitary_block(stim.Circuit("H 0\nCX 0 1"))
    before = tracker.stabilizers.matrix.copy()
    assert tracker.infer_gauge_fixed_stabilizers(system).count == 0
    with pytest.raises(RuntimeError, match="0 known bare logical constraints"):
        builder.stabilizer_canonicalization()
    np.testing.assert_array_equal(tracker.stabilizers.matrix, before)


@pytest.mark.parametrize("metadata", ["post_select_row_indices", "stabilizer_with_logical_components"])
def test_pending_row_metadata_is_rejected_before_recombination(metadata):
    builder = make_builder(bells=True)
    setattr(builder.tracker, metadata, {0})
    before = builder.circuit.copy()
    block = GenericCSSGaugeExtractionBlock(builder.system)
    with pytest.raises(RuntimeError, match=metadata):
        builder.apply_syndrome_extraction(block.circuit, measurement_blocks=block.measurement_blocks)
    assert builder.circuit == before


def test_unmeasured_placeholder_cannot_supply_gauge_knowledge():
    builder = make_builder(bells=True)
    builder.tracker.stabilizers.records[0] = [-1]
    with pytest.raises(RuntimeError, match="unmeasured placeholders"):
        builder.tracker.infer_gauge_fixed_stabilizers(builder.system)
    with pytest.raises(RuntimeError, match="unmeasured placeholders"):
        builder.stabilizer_canonicalization()


def test_subsystem_rebase_uses_gauge_fixed_span_after_preparation():
    builder = make_builder(bells=True)
    builder.tracker.rebase_stabilizers_onto_code_basis(builder.system)
    assert_phase(builder, "X")
    measure_gauge(builder, "Z", 0)
    builder.tracker.rebase_stabilizers_onto_code_basis(builder.system)
    assert_phase(builder, "Z")


def test_absorbed_logical_metadata_is_not_silently_dropped():
    builder = make_builder(bells=True)
    builder.tracker.absorbed_ops.matrix = builder.tracker.stabilizers.matrix[:1].copy()
    before = builder.tracker.stabilizers.matrix.copy()
    with pytest.raises(RuntimeError, match="pending absorbed logical relations"):
        builder.stabilizer_canonicalization()
    np.testing.assert_array_equal(builder.tracker.stabilizers.matrix, before)


def test_unverified_compression_falls_back_to_full_subsystem_updates(monkeypatch):
    builder = make_builder()
    block = GenericCSSGaugeExtractionBlock(builder.system)
    monkeypatch.setattr(builder, "_try_compress_steady_rounds", lambda **kwargs: None)
    builder.apply_syndrome_extraction(block.circuit, rounds=5, measurement_blocks=block.measurement_blocks)
    assert not any(isinstance(instruction, stim.CircuitRepeatBlock) for instruction in builder.circuit)
    assert_phase(builder, "Z")
    builder.apply_data_readout()
    builder.circuit.detector_error_model()


def test_z_only_legacy_shortcut_is_rejected_before_mutation():
    builder = make_builder()
    block = GenericCSSGaugeExtractionBlock(builder.system, basis_order=("Z",))
    before = builder.circuit.copy()
    with pytest.raises(ValueError, match="full detector pipeline"):
        builder.apply_syndrome_extraction(block.circuit, z_only=True)
    assert builder.circuit == before
