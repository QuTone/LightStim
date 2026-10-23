"""Readout reclassifies the full current state without rewriting past relations."""

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)
from lightstim.utils.linear_algebra import check_commutativity, solve_linear_decomposition
from lightstim.utils.subsystem_algebra import independent_row_indices, reduce_modulo
from tests.test_tracker_v2_readout import _assert_complete, _read, _rows, _tracker
from tests.tracker_test_utils import _annotations, _basis, _contains, _mask


@pytest.fixture(params=["reference", "persistent"])
def tracker_backend(request, monkeypatch):
    if request.param == "persistent":
        extension = pytest.importorskip("lightstim.utils.cpp._pauli_backend_cpp")
        if not hasattr(extension, "PersistentRecordState"):
            pytest.skip("optional persistent tracking extension has not been built")
        monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    else:
        monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "reference")
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", request.param)
    return request.param


def _pauli(row, n):
    return stim.PauliString.from_numpy(xs=row[:n].astype(bool), zs=row[n:].astype(bool))


def _bare_rows(tracker):
    indices = [i for i in range(tracker.stabilizers.count)
               if i not in tracker._dressed_stabilizer_rows]
    return (tracker.stabilizers.matrix[indices],
            [tracker.stabilizers.records[i] for i in indices])


def test_surface_code_readout_moves_bare_logical_row_without_leaking_history(tracker_backend):
    """A code-preserving Clifford changes L to Lg; reading L leaves bare g."""
    system = QECSystem()
    system.add_patch(UnrotatedSurfaceCode(distance=3), name="A")
    tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=1)
    builder = CircuitBuilder(tracker, system)
    builder.initialize({q: "Z" for q in system.data_indices}, n=system.num_qubits)
    builder.apply_syndrome_extraction(UnrotatedSurfaceCodeExtractionBlock(system).circuit)
    n = tracker.num_qubits
    logical = tracker.logicals.matrix[0].copy()
    support = np.flatnonzero(logical[:n] | logical[n:])
    assert not logical[:n].any()
    selected = independent_row_indices(tracker.stabilizers.matrix)
    stabilizers = tracker.stabilizers.matrix[selected].copy()
    g_index = next(i for i, row in enumerate(stabilizers)
                   if row.any() and not (row[:n][support] | row[n:][support]).any())
    g = stabilizers[g_index].copy()
    g_records = list(tracker.stabilizers.records[int(selected[g_index])])
    encoder = stim.Tableau.from_stabilizers(
        [_pauli(row, n) for row in stabilizers] + [_pauli(logical, n)],
        allow_underconstrained=True,
    )
    unitary = (encoder.inverse().to_circuit()
               + stim.Circuit(f"CX {g_index} {len(stabilizers)}")
               + encoder.to_circuit())
    action = unitary.to_tableau()
    assert all(action(_pauli(row, n)) == _pauli(row, n) for row in stabilizers)
    assert action(_pauli(logical, n)) == _pauli(logical ^ g, n)
    builder.apply_unitary_block(unitary)

    readout_start = builder.circuit.num_measurements
    builder.apply_data_readout({int(q): "Z" for q in support})
    logical_readout_records = set(range(readout_start, builder.circuit.num_measurements))
    active = tracker._v2_role_space
    assert not check_commutativity(active, active).any()
    state = np.vstack([tracker.stabilizers.matrix, tracker.logicals.matrix])
    _, known, _ = solve_linear_decomposition(basis=state, targets=active, reduce_weight=False)
    assert np.all(known)
    assert tracker.logicals.count == 0
    bare, records = _bare_rows(tracker)
    assert any(np.array_equal(row, g) and rec == g_records
               for row, rec in zip(bare, records))
    assert not reduce_modulo(bare, active).any()

    _, detectors, observables = _annotations(builder.circuit)
    assert len(detectors) == 6
    assert observables == [_mask(set(g_records) ^ logical_readout_records)]
    _assert_complete(tracker, builder.circuit)

    # Continuing the protocol must retain g's syndrome history, rather than
    # importing the already extracted logical parity into the next detector.
    g_pauli = _pauli(g, n)
    g_readout = {q: "IXYZ"[g_pauli[q]] for q in range(n) if g_pauli[q]}
    g_readout_start = builder.circuit.num_measurements
    builder.apply_data_readout(g_readout)
    _, final_detectors, final_observables = _annotations(builder.circuit)
    assert final_observables == observables
    assert final_detectors[:len(detectors)] == detectors
    expected = _mask(set(g_records) ^ set(range(g_readout_start, builder.circuit.num_measurements)))
    assert final_detectors[len(detectors):] == [expected]
    assert not expected & _mask(logical_readout_records)
    _assert_complete(tracker, builder.circuit)


def test_readout_finds_bare_product_across_stabilizer_and_logical_rows(tracker_backend):
    """Neither Z0 nor Z1 lies in K=<Z0 Z1>, but their product does."""
    tracker = _tracker(
        3, stabilizers=("ZII", "IIZ"), logicals=("IZI",), dressed=(0,),
        stab_records=[[0], []], logical_records=[[1]],
    )
    tracker.total_measurements = 2
    circuit = stim.Circuit("RX 0 1\nM 0 1\nR 2")
    active = _rows("ZZI", n=3)
    assert reduce_modulo(_rows("ZII", "IZI", n=3), active).any(axis=1).all()

    _read(tracker, circuit, [2], active=("ZZI",))
    bare, records = _bare_rows(tracker)
    assert any(np.array_equal(row, active[0]) and rec == [0, 1]
               for row, rec in zip(bare, records))
    assert not reduce_modulo(bare, active).any()
    state = np.vstack([tracker.stabilizers.matrix, tracker.logicals.matrix])
    assert len(independent_row_indices(state)) == 2
    protected = np.vstack([
        tracker.stabilizers.matrix[sorted(tracker._dressed_stabilizer_rows)],
        tracker.logicals.matrix,
    ])
    assert reduce_modulo(protected, active).any(axis=1).all()
    assert tracker.expected_num_logicals == 1
    _, detectors, observables = _annotations(circuit)
    assert detectors == [_mask([2])]
    assert observables == []
    _assert_complete(tracker, circuit)

    _read(tracker, circuit, [0, 1])
    _, detectors, observables = _annotations(circuit)
    assert detectors == [_mask([2]), _mask([0, 1, 3, 4])]
    assert len(observables) == 1
    assert not _contains(_basis(detectors), observables[0])
    # All three deterministic relations are retained; only the code parity
    # belongs in the detector space, while one independent logical value remains.
    assert len(_basis(detectors + observables)) == 3
    _assert_complete(tracker, circuit)
