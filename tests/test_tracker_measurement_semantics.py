import inspect

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.color_code import (
    ColorCode,
    ColorCodeSpaceMultiplexingBlock,
    ColorCodeTimeMultiplexingBlock,
)


def _build_one_round(*, layout, block_class):
    code = ColorCode(distance=3, layout=layout)
    system = QECSystem()
    system.add_patch(code, name="color")
    tracker = SyndromeTracker(code.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system, if_detector=True)
    builder.initialize(
        {qubit: "Z" for qubit in system.data_indices},
        n=code.num_qubits,
    )
    block = block_class(system)
    builder.apply_syndrome_extraction(
        block.circuit,
        rounds=1,
        measurement_blocks=block.measurement_blocks,
    )


def test_tracker_measurement_update_has_no_round_finalization_policy():
    parameters = inspect.signature(
        SyndromeTracker.process_mid_measurement
    ).parameters
    assert "finalize_logicals" not in parameters
    assert not hasattr(
        SyndromeTracker,
        "finalize_composite_syndrome_measurement",
    )


def test_builder_owns_measurement_group_classification(monkeypatch):
    events = []
    original_promote = SyndromeTracker.promote_stabilizer_rows_to_logicals
    original_rebase = SyndromeTracker.rebase_stabilizers_onto_code_basis

    def record_promote(self, row_indices):
        events.append("promote")
        return original_promote(self, row_indices)

    def record_rebase(self, system, stabilizer_uids=None):
        events.append("rebase")
        return original_rebase(self, system, stabilizer_uids)

    monkeypatch.setattr(
        SyndromeTracker,
        "promote_stabilizer_rows_to_logicals",
        record_promote,
    )
    monkeypatch.setattr(
        SyndromeTracker,
        "rebase_stabilizers_onto_code_basis",
        record_rebase,
    )

    _build_one_round(
        layout="superdense",
        block_class=ColorCodeSpaceMultiplexingBlock,
    )
    assert events == ["promote"]

    events.clear()
    _build_one_round(
        layout="triangular",
        block_class=ColorCodeTimeMultiplexingBlock,
    )
    assert events == ["rebase"]
