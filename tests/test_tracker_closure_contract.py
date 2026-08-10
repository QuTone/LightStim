"""Terminal-readout closure contract, independent of any patch/PPM geometry.

The tracker's stabilizer bank is not a canonical basis: row updates during
measurement processing multiply rows together, so a bank row can be a
product of local checks whose Pauli support exceeds any single physical
check.  Its terminal closure is legitimate syndrome information and must be
emitted like every other determined row's — suppressing it by a support
weight threshold silently costs decoder information (paired-noise MWPM LER
is ~30% worse without the closure; PR #68 review, blocker #1).

This test drives SyndromeTracker directly with a deliberately non-canonical
bank so the contract is pinned at the tracker layer, not via any lattice
surgery construction.
"""
import numpy as np
import stim

from lightstim.ir.tracker import SyndromeTracker


def test_high_weight_product_row_still_emits_closure_detector():
    n = 6
    # Physical history on |0..0>: MPP C2 = Z3Z4Z5 (rec 0), then
    # MPP C1 = Z0Z1Z2Z3 (rec 1), then transversal Z readout (recs 2..7).
    # Every parity below is deterministic.
    circuit = stim.Circuit("""
        R 0 1 2 3 4 5
        MPP Z3*Z4*Z5
        MPP Z0*Z1*Z2*Z3
        M 0 1 2 3 4 5
    """)

    tracker = SyndromeTracker(n, 0)
    tracker.total_measurements = 2  # the two MPPs above

    # Non-canonical bank: row A = C2 (support 3, record [0]); row B = C1*C2
    # = Z0 Z1 Z2 Z4 Z5 (support 5 — above any single check), records [0, 1].
    row_a = np.zeros(2 * n, dtype=np.uint8)
    row_a[[n + 3, n + 4, n + 5]] = 1
    row_b = np.zeros(2 * n, dtype=np.uint8)
    row_b[[n + 0, n + 1, n + 2, n + 4, n + 5]] = 1
    tracker.stabilizers.matrix = np.vstack([row_a, row_b])
    tracker.stabilizers.records = [[0], [0, 1]]

    final_paulis = np.zeros((n, 2 * n), dtype=np.uint8)
    for q in range(n):
        final_paulis[q, n + q] = 1  # Z readout, order matches the M above

    tracker.process_data_measurement(
        circuit,
        final_paulis,
        idx_to_coord_map={q: (float(q), 0.0) for q in range(n)},
    )

    detectors = [i for i in circuit.flattened() if i.name == "DETECTOR"]
    assert len(detectors) == 2, (
        "every determined stabilizer bank row must emit its closure "
        f"detector; got {len(detectors)}"
    )
    # Row A closes over 1 historical + 3 final records; row B (the
    # high-support product) over 2 historical + 5 final records.
    assert sorted(len(i.targets_copy()) for i in detectors) == [4, 7]

    # The assembled history is physically consistent: silent at p=0.
    assert not circuit.compile_detector_sampler(seed=0).sample(1024).any()


def test_sentinel_tagged_gauge_row_emits_no_detector():
    """The sentinel-record branch (kept by design, review follow-up: 'keep
    its semantics covered separately'): a stabilizer row whose records
    carry UNMEASURED_STAB_RECORD is an unwatched gauge direction's
    close-out — its only content is init-parity == readout-parity with no
    banked init record, so emitting it would attach an unbanked,
    per-shot-random parity.  The terminal readout must skip exactly these
    rows, by PROVENANCE (the sentinel), never by support weight."""
    import numpy as np
    import stim
    from lightstim.ir.tracker import SyndromeTracker, UNMEASURED_STAB_RECORD

    n = 4
    circuit = stim.Circuit("""
        R 0 1 2 3
        MPP Z0*Z1
        M 0 1 2 3
    """)
    tracker = SyndromeTracker(n, 0)
    tracker.total_measurements = 1

    watched = np.zeros(2 * n, dtype=np.uint8)
    watched[[n + 0, n + 1]] = 1
    unwatched = np.zeros(2 * n, dtype=np.uint8)
    unwatched[[n + 2, n + 3]] = 1
    tracker.stabilizers.matrix = np.vstack([watched, unwatched])
    tracker.stabilizers.records = [[0], [UNMEASURED_STAB_RECORD]]

    final_paulis = np.zeros((n, 2 * n), dtype=np.uint8)
    for q in range(n):
        final_paulis[q, n + q] = 1
    tracker.process_data_measurement(
        circuit, final_paulis,
        idx_to_coord_map={q: (float(q), 0.0) for q in range(n)})

    detectors = [i for i in circuit.flattened() if i.name == "DETECTOR"]
    assert len(detectors) == 1, (
        f"only the record-carrying row may close out; got {len(detectors)}")
    recs = {t.value + circuit.num_measurements
            for t in detectors[0].targets_copy()}
    assert recs == {0, 1, 2}, (
        f"the emitted closure must be the WATCHED row's (MPP rec 0 + finals "
        f"of qubits 0/1); got absolute records {sorted(recs)}")
