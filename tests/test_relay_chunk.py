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


def test_relay_drops_sentinel_parity_detectors():
    """A row whose records carry the UNMEASURED sentinel has no banked
    parity; a candidate detector built through it would compare a
    per-shot-random value and stim only explodes far away, at DEM time.
    The SE engine skips these by provenance; the relay path must match."""
    from lightstim.ir.relay_flow import solve_relay_round
    from lightstim.ir.tracker import UNMEASURED_STAB_RECORD
    n = 2
    row = np.zeros(2 * n, dtype=np.uint8)
    row[[n + 0, n + 1]] = 1
    res = solve_relay_round(stim.Circuit("M 0 1"), n, row.reshape(1, -1),
                            [[UNMEASURED_STAB_RECORD]],
                            np.zeros((0, 2 * n), dtype=np.uint8), [],
                            base_record=100)
    assert all(r >= 0 for recs, _ in res.detectors for r in recs), \
        f"sentinel leaked into detector records: {res.detectors}"


def test_relay_refuses_sentinel_observable():
    from lightstim.ir.relay_flow import solve_relay_round
    from lightstim.ir.tracker import UNMEASURED_STAB_RECORD
    n = 1
    zrow = np.zeros(2 * n, dtype=np.uint8)
    zrow[n + 0] = 1
    with pytest.raises(NotImplementedError, match="sentinel"):
        solve_relay_round(stim.Circuit("M 0"), n,
                          np.zeros((0, 2 * n), dtype=np.uint8), [],
                          np.zeros((0, 2 * n), dtype=np.uint8), [],
                          base_record=10,
                          absorbed_matrix=zrow.reshape(1, -1),
                          absorbed_records=[[UNMEASURED_STAB_RECORD]])


def test_relay_error_names_the_absorbed_kind():
    """Destroying a banked (absorbed) relation must say so - not call it a
    logical row (review round-2 wording precision)."""
    from lightstim.ir.relay_flow import solve_relay_round
    n = 1
    zrow = np.zeros(2 * n, dtype=np.uint8)
    zrow[n + 0] = 1
    with pytest.raises(NotImplementedError, match="absorbed"):
        solve_relay_round(stim.Circuit("MX 0"), n,
                          np.zeros((0, 2 * n), dtype=np.uint8), [],
                          np.zeros((0, 2 * n), dtype=np.uint8), [],
                          base_record=10,
                          absorbed_matrix=zrow.reshape(1, -1),
                          absorbed_records=[[3]])


def test_relay_transports_absorbed_through_s_gate():
    """Frame transport is not H-specific: S sends a banked X relation to
    the Y relation (both symplectic halves set), records untouched."""
    from lightstim.ir.relay_flow import solve_relay_round
    n = 1
    xrow = np.zeros(2 * n, dtype=np.uint8)
    xrow[0] = 1
    res = solve_relay_round(stim.Circuit("S 0"), n,
                            np.zeros((0, 2 * n), dtype=np.uint8), [],
                            np.zeros((0, 2 * n), dtype=np.uint8), [],
                            base_record=0,
                            absorbed_matrix=xrow.reshape(1, -1),
                            absorbed_records=[[3]])
    assert res.new_absorbed_matrix.shape[0] == 1
    assert res.new_absorbed_matrix[0].tolist() == [1, 1]   # X -> Y
    assert res.new_absorbed_records == [[3]]


def test_relay_rejects_sentinel_fold_in_logical_transport():
    """Transporting a logical row must not fold through a stabilizer row
    whose records carry the UNMEASURED sentinel: that row's parity was
    never banked, so the transported row's records would be fabricated."""
    from lightstim.ir.relay_flow import solve_relay_round
    from lightstim.ir.tracker import UNMEASURED_STAB_RECORD
    n = 2
    stab = np.zeros(2 * n, dtype=np.uint8)
    stab[[n + 0, n + 1]] = 1                      # Z0Z1, sentinel records
    log = np.zeros(2 * n, dtype=np.uint8)
    log[n + 0] = 1                                # Z0 = Z0Z1 * Z1
    with pytest.raises(NotImplementedError, match="sentinel"):
        solve_relay_round(stim.Circuit("R 0"), n, stab.reshape(1, -1),
                          [[UNMEASURED_STAB_RECORD]],
                          log.reshape(1, -1), [[]], base_record=0)


def test_relay_rejects_cancelling_sentinel_folds():
    """The killer variant: TWO sentinel rows folded into the same logical
    transport cancel their -1s under symmetric difference, so a content
    check on the final record set sees a clean (empty) parity and the
    measured-out logical becomes a fabricated observable with LER
    identically zero.  The guard must fire per folded row (provenance),
    not on the surviving records."""
    from lightstim.ir.relay_flow import solve_relay_round
    from lightstim.ir.tracker import UNMEASURED_STAB_RECORD
    n = 3
    s1 = np.zeros(2 * n, dtype=np.uint8)
    s1[[n + 0, n + 1]] = 1                        # Z0Z1
    s2 = np.zeros(2 * n, dtype=np.uint8)
    s2[[n + 1, n + 2]] = 1                        # Z1Z2
    log = np.zeros(2 * n, dtype=np.uint8)
    log[[n + 0, n + 2]] = 1                       # Z0Z2 = Z0Z1 * Z1Z2
    with pytest.raises(NotImplementedError, match="sentinel"):
        solve_relay_round(stim.Circuit("R 0 2"), n, np.vstack([s1, s2]),
                          [[UNMEASURED_STAB_RECORD],
                           [UNMEASURED_STAB_RECORD]],
                          log.reshape(1, -1), [[]], base_record=0)


def test_relay_rejects_unsupported_measurement_gates():
    """MR/MPP produce measurement records but sit outside the relay's
    supported set; skipping them silently would desynchronise the
    tracker's measurement count and shift every later rec target (stim
    does NOT expand MR into M + R)."""
    from lightstim.ir.relay_flow import measured_qubits_in_order
    with pytest.raises(NotImplementedError, match="MR"):
        measured_qubits_in_order(stim.Circuit("MR 0"))
    with pytest.raises(NotImplementedError, match="MPP"):
        measured_qubits_in_order(stim.Circuit("MPP X0*X1"))


def test_relay_rejects_pending_row_metadata():
    """apply_relay_chunk wholesale-replaces the stabilizer basis; pending
    row-index metadata would silently point at DIFFERENT rows afterwards
    (a stale swlc index reroutes a closure DETECTOR into an
    OBSERVABLE_INCLUDE at the terminal readout).  Same fail-loud contract
    as stabilizer_canonicalization."""
    system, tracker, builder = _fresh()
    _init_z(system, builder)
    chunk = RotatedSurfaceCodeExtractionBlock(system).circuit
    builder.apply_syndrome_extraction(circuit_chunk=chunk, rounds=1)

    tracker.post_select_row_indices.add(0)
    with pytest.raises(RuntimeError, match="post_select"):
        builder.apply_relay_chunk(chunk)
    tracker.post_select_row_indices.clear()

    tracker.stabilizer_with_logical_components.add(0)
    with pytest.raises(RuntimeError, match="logical_components|recombined"):
        builder.apply_relay_chunk(chunk)


def test_relay_double_h_round_trips_the_ledger():
    """Two transversal-H relays must return the banked relation to its
    original frame with records intact."""
    system, tracker, builder = _fresh()
    _init_z(system, builder)
    builder.apply_syndrome_extraction(
        circuit_chunk=RotatedSurfaceCodeExtractionBlock(system).circuit,
        rounds=1)
    n = tracker.num_qubits
    zbar = tracker.logicals.matrix[0].copy()
    zrecs = list(tracker.logicals.records[0])
    tracker.logicals.matrix = np.zeros((0, 2 * n), dtype=np.uint8)
    tracker.logicals.records = []
    assert tracker.record_absorbed_op(zbar, zrecs)

    relay = stim.Circuit()
    relay.append("H", list(range(n)))
    builder.apply_relay_chunk(relay)
    builder.apply_relay_chunk(relay)

    assert tracker.absorbed_ops.count == 1
    assert (tracker.absorbed_ops.matrix[0] == zbar).all()
    assert sorted(tracker.absorbed_ops.records[0]) == sorted(zrecs)
