"""add_patch on dormant (retired) qubit indices must clear retirement state.

The retire bookkeeping (retired-qubit masking, Fix C absorbed-relation
resolution) treats every index in tracker.retired_qubits as measured-out.
When add_patch re-births a patch on those indices, the qubits are live
again: leaving them marked retired makes later retirements resolve
absorbed relations against a live qubit.  The coupler-activation path
already discards reused indices; this pins the ordinary add_patch path.
"""
import numpy as np
import pytest

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
from lightstim.qec_code.surface_code.rotated.SE_block import (
    RotatedSurfaceCodeExtractionBlock,
)

D = 3


def _retired_system():
    """P born, stabilized, read out, retired; every P qubit dormant."""
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=D), name="P", offset=(0, 0))
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.initialize(init_dict={q: 'Z' for q in system.data_indices},
                       n=system.num_qubits)
    builder.apply_syndrome_extraction(
        circuit_chunk=RotatedSurfaceCodeExtractionBlock(system).circuit,
        rounds=1)
    p_data = sorted(system.data_indices)
    builder.apply_data_readout(
        final_measurements={q: 'Z' for q in p_data})
    system.retire_measured_patch("P")
    # retire_measured_patch's documented scope: only DATA qubits go dormant;
    # ancilla release is the corridor-reuse layer's responsibility.  Play
    # that layer's role here so a same-footprint rebirth exercises pure
    # dormant reuse.
    system.active_qubit_indices.difference_update(system.syndrome_indices)
    return system, tracker, set(p_data)


def test_add_patch_reuse_clears_retired_qubits():
    system, tracker, p_data = _retired_system()
    assert p_data <= tracker.retired_qubits  # retirement really happened

    system.add_patch(RotatedSurfaceCode(distance=D), name="Q", offset=(0, 0))

    reused = set(system.local_to_global_map["Q"].values())
    assert reused & p_data, "rebirth did not reuse the dormant indices"
    assert tracker.retired_qubits.isdisjoint(reused), (
        "reused qubits are still marked retired")


def test_add_patch_reuse_rejects_stale_tracker_support():
    system, tracker, p_data = _retired_system()
    q0 = min(p_data)
    # Corrupt: a stabilizer row still referencing the retired qubit.  The
    # retirement contract eliminated these columns; rebirth on top of a
    # stale tableau must fail loud, not silently build.
    n = tracker.num_qubits
    row = np.zeros(2 * n, dtype=np.uint8)
    row[n + q0] = 1
    tracker.stabilizers.matrix = row.reshape(1, -1)
    tracker.stabilizers.records = [[0]]

    with pytest.raises(RuntimeError, match="stale retirement state"):
        system.add_patch(RotatedSurfaceCode(distance=D), name="Q",
                         offset=(0, 0))
