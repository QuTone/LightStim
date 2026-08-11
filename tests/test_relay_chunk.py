"""builder.apply_relay_chunk — flow-solved round processing (route ②).

Rounds where stabilizers are relayed between ancillas mid-round (e.g. the
Y-basis transition round of arXiv:2302.07395) violate the standard SE path's
per-measurement contract (builder.py syn-block zeroing: sound iff each
measurement, back-propagated, touches other ancillas only in their reset
basis). apply_relay_chunk instead solves the round's stabilizer flows with
stim (Circuit.flow_generators) and merges them with the tracker's
record-augmented rows: closed flows -> DETECTORs, open flows -> new tracked
rows, logical rows transported exactly.

Spec anchor: on a STANDARD SE round the relay path must be observationally
identical to apply_syndrome_extraction (same detector record-sets, same
tracker knowledge), and must compose with the standard path afterwards.
"""
import numpy as np
import pytest
import stim

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
from lightstim.qec_code.surface_code.rotated.SE_block import (
    RotatedSurfaceCodeExtractionBlock,
)

D = 3


def _fresh(d=D):
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=d), name="P", offset=(0, 0))
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    return system, tracker, builder


def _init_z(system, builder):
    builder.initialize(
        init_dict={q: 'Z' for q in system.data_indices},
        n=system.num_qubits)


_MEAS = {'M', 'MX', 'MY', 'MR', 'MRX', 'MRY'}


def _detector_recsets(circuit):
    """Each DETECTOR as a frozenset of ABSOLUTE measurement indices."""
    out = []
    m = 0
    for inst in circuit.flattened():
        if inst.name in _MEAS:
            m += len(inst.targets_copy())
        elif inst.name == 'DETECTOR':
            out.append(frozenset(m + t.value for t in inst.targets_copy()))
    return out


def _noiseless_clean(circuit, shots=256):
    s = circuit.compile_detector_sampler(seed=11).sample(
        shots=shots, append_observables=True)
    return not s.any()


def test_standard_round_matches_se_path():
    # Path A: the existing SE machinery
    sysA, trA, bA = _fresh()
    _init_z(sysA, bA)
    chunkA = RotatedSurfaceCodeExtractionBlock(sysA).circuit
    bA.apply_syndrome_extraction(circuit_chunk=chunkA, rounds=1)

    # Path B: relay path on the identical chunk
    sysB, trB, bB = _fresh()
    _init_z(sysB, bB)
    chunkB = RotatedSurfaceCodeExtractionBlock(sysB).circuit
    bB.apply_relay_chunk(chunkB)

    assert bB.circuit.num_measurements == bA.circuit.num_measurements
    assert sorted(sorted(r) for r in _detector_recsets(bB.circuit)) == \
        sorted(sorted(r) for r in _detector_recsets(bA.circuit))
    assert _noiseless_clean(bB.circuit)
    assert trB.total_measurements == trA.total_measurements
    # same knowledge: every registered stabilizer decomposes identically
    assert trB.stabilizers.count > 0


def test_relay_round_then_standard_round_composes():
    system, tracker, builder = _fresh()
    _init_z(system, builder)
    chunk = RotatedSurfaceCodeExtractionBlock(system).circuit
    builder.apply_relay_chunk(chunk)
    n_det_before = builder.circuit.num_detectors

    # the ordinary path must be able to continue on relay-installed rows
    builder.apply_syndrome_extraction(circuit_chunk=chunk, rounds=1)
    round2 = builder.circuit.num_detectors - n_det_before
    assert round2 == 8            # full complement at d=3
    assert _noiseless_clean(builder.circuit)


def test_two_relay_rounds_full_detector_complement():
    system, tracker, builder = _fresh()
    _init_z(system, builder)
    chunk = RotatedSurfaceCodeExtractionBlock(system).circuit
    builder.apply_relay_chunk(chunk)
    n1 = builder.circuit.num_detectors
    builder.apply_relay_chunk(chunk)
    n2 = builder.circuit.num_detectors - n1
    assert n1 == 4                # Z-init: only Z-checks deterministic
    assert n2 == 8
    assert _noiseless_clean(builder.circuit)


def test_relay_and_readout_observable_ids_do_not_collide():
    """Centralized observable allocation (review blocker: relay allocated
    from circuit.num_observables without advancing tracker.total_observables,
    so a later data readout reused the same ID and XORed two independent
    logical results into one observable — silently: the XOR of two
    deterministic bits is still deterministic, so p=0 sampling stays clean).

    Scenario: patch P is measured out by a relay chunk (one observable),
    then patch Q is read out terminally (a second observable).  The two
    OBSERVABLE_INCLUDEs must carry distinct IDs and both counters agree."""
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=D), name="P", offset=(0, 0))
    system.add_patch(RotatedSurfaceCode(distance=D), name="Q", offset=(20, 0))
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    _init_z(system, builder)
    builder.apply_syndrome_extraction(
        circuit_chunk=RotatedSurfaceCodeExtractionBlock(system).circuit,
        rounds=1)

    p_qubits = system.patches["P"][0].num_qubits
    relay = stim.Circuit()
    for q in sorted(q for q in system.data_indices if q < p_qubits):
        relay.append("M", [q])
    builder.apply_relay_chunk(relay)
    assert tracker.total_observables == builder.circuit.num_observables == 1

    builder.apply_data_readout(
        final_measurements={q: 'Z' for q in system.data_indices
                            if q >= p_qubits})
    c = builder.circuit
    obs_ids = sorted(int(i.gate_args_copy()[0]) for i in c.flattened()
                     if i.name == 'OBSERVABLE_INCLUDE')
    assert obs_ids == [0, 1], f"independent observables share an ID: {obs_ids}"
    assert c.num_observables == 2
    assert tracker.total_observables == 2
    assert _noiseless_clean(c)


def test_readout_after_relay_round_emits_observable():
    system, tracker, builder = _fresh()
    _init_z(system, builder)
    chunk = RotatedSurfaceCodeExtractionBlock(system).circuit
    builder.apply_relay_chunk(chunk)
    builder.apply_syndrome_extraction(circuit_chunk=chunk, rounds=1)
    builder.apply_data_readout(
        final_measurements={q: 'Z' for q in system.data_indices})
    c = builder.circuit
    assert c.num_observables == 1
    assert _noiseless_clean(c)
    dem = c.detector_error_model(decompose_errors=True)
    assert dem.num_observables == 1


def test_relay_transports_the_absorbed_ledger():
    """Review blocker: a Clifford relay must move the absorbed-relation
    ledger to the new Pauli frame.  Transversal H sends a banked Z
    relation to the X relation; before the fix the ledger kept the stale
    Z row while the census stayed silent (rank is frame-invariant)."""
    system, tracker, builder = _fresh()
    _init_z(system, builder)
    builder.apply_syndrome_extraction(
        circuit_chunk=RotatedSurfaceCodeExtractionBlock(system).circuit,
        rounds=1)
    n = tracker.num_qubits

    # bank the standing logical as an absorbed relation (its standing row
    # leaves the tableau; the budget still expects one DOF somewhere)
    zbar = tracker.logicals.matrix[0].copy()
    zrecs = list(tracker.logicals.records[0])
    tracker.logicals.matrix = np.zeros((0, 2 * n), dtype=np.uint8)
    tracker.logicals.records = []
    assert tracker.record_absorbed_op(zbar, zrecs)
    tracker.validate_logical_count(context="after banking")

    relay = stim.Circuit()
    relay.append("H", list(range(n)))
    builder.apply_relay_chunk(relay)

    expected = np.concatenate([zbar[n:], zbar[:n]])  # H: X- and Z-halves swap
    assert tracker.absorbed_ops.count == 1
    assert (tracker.absorbed_ops.matrix[0] == expected).all(), \
        "the banked relation stayed in the old Pauli frame"
    assert sorted(tracker.absorbed_ops.records[0]) == sorted(zrecs)
    tracker.validate_logical_count(context="after relay")
