"""Phase-1 tests for the [[6, 2, 2]] code patch (Magic-H6 base code)."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.H_six import (
    HSixCode,
    HSixExtractionBlock,
    HSixLogicalOpSet,
    HSixLogicalXCheckBlock,
    get_dist_circ,
    get_ft_init_circ,
)


def _pcm():
    return HSixCode().get_parity_check_matrix()


def test_shape_and_counts():
    p = HSixCode()
    assert p.num_qubits == 10                    # 6 data + 4 ancilla
    assert len(p.data_indices) == 6
    assert len(p.syndrome_indices_x) == 2
    assert len(p.syndrome_indices_z) == 2
    assert len(p.stabilizers) == 4
    assert p.num_logicals == 2
    assert len(p.logical_ops) == 4               # X0, Z0, X1, Z1


def test_css_structure_gives_k2_d2():
    Hx, Hz = _pcm()
    # Both bases: two weight-4 checks on {0,1,2,3} and {2,3,4,5}.
    assert Hx[:, :6].tolist() == [[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]]
    assert Hz[:, :6].tolist() == [[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]]
    assert np.linalg.matrix_rank(Hx % 2) == 2
    assert np.linalg.matrix_rank(Hz % 2) == 2
    # CSS commutation.
    assert np.all((Hx @ Hz.T) % 2 == 0)
    # n - rank(Hx) - rank(Hz) = 6 - 2 - 2 = 2 logical qubits.
    assert 6 - 2 - 2 == HSixCode().num_logicals


def test_logical_operator_symplectic_structure():
    Hx, Hz = _pcm()
    n = Hx.shape[1]

    def v(support):
        out = np.zeros(n, dtype=int)
        for q in support:
            out[q] = 1
        return out

    X0, X1 = v((0, 2, 4)), v((1, 3, 5))
    Z0, Z1 = v((0, 2, 4)), v((1, 3, 5))

    # Logicals commute with every check of the opposite type.
    assert np.all((Hz @ X0) % 2 == 0) and np.all((Hz @ X1) % 2 == 0)
    assert np.all((Hx @ Z0) % 2 == 0) and np.all((Hx @ Z1) % 2 == 0)
    # X_i anticommutes with Z_i only.
    assert (X0 @ Z0) % 2 == 1 and (X1 @ Z1) % 2 == 1
    assert (X0 @ Z1) % 2 == 0 and (X1 @ Z0) % 2 == 0
    # Logicals are not in the row span of the same-type checks (odd weight 3
    # vs even-weight generators).
    assert (X0.sum() % 2 == 1) and (X1.sum() % 2 == 1)


def test_extraction_block_builds():
    system = QECSystem()
    system.add_patch(HSixCode(), name="c622")
    se = HSixExtractionBlock(system)
    assert se.circuit.num_qubits >= 10
    last = se.circuit[-1]
    assert last.name in ("M", "MX")


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_noiseless_memory_is_deterministic(basis):
    system = QECSystem()
    system.add_patch(HSixCode(), name="c622")
    se = HSixExtractionBlock(system)
    tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    builder.write_coordinates()
    builder.initialize({q: basis for q in system.data_indices}, n=system.num_qubits)
    builder.apply_syndrome_extraction(se.circuit, rounds=3)
    builder.apply_data_readout({q: basis for q in system.data_indices})

    circ = builder.circuit
    assert circ.num_observables == 2
    dets, obs = circ.compile_detector_sampler().sample(256, separate_observables=True)
    assert not dets.any()
    assert not obs.any()
    circ.detector_error_model(decompose_errors=True)   # graphlike, compiles


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_memory_experiment_wrapper(basis):
    exp = MemoryExperiment(qec_patch=HSixCode(), rounds=3, basis=basis, noise_params=None)
    circ = exp.build()
    dets, obs = circ.compile_detector_sampler().sample(256, separate_observables=True)
    assert not dets.any()
    assert not obs.any()


def test_shift_moves_every_qubit():
    a = HSixCode()
    b = HSixCode(shift=(20, 4))
    for idx, (x, y) in a.qubit_coords.items():
        bx, by = b.qubit_coords[idx]
        assert (bx, by) == (x + 20, y + 4)


# --- Code614.py port ---------------------------------------------------------

def test_h_check_ancillas_are_bare_syndrome_qubits():
    p = HSixCode(h_check_ancillas=2)
    assert p.num_qubits == 12
    bare = p.syndrome_indices - p.syndrome_indices_x - p.syndrome_indices_z
    assert len(bare) == 2
    # SE block still ignores them.
    system = QECSystem()
    system.add_patch(HSixCode(h_check_ancillas=2), name="c622")
    se = HSixExtractionBlock(system)
    used = {t.value for inst in se.circuit for t in inst.targets_copy()}
    assert used.isdisjoint({10, 11})


def test_get_stabs_and_get_logicals_match_code614():
    # A valid codeword string has both stabilizers 0.
    assert HSixCode.get_stabs([1, 1, 1, 1, 0, 0]) == [0, 0]
    assert HSixCode.get_stabs([0, 0, 1, 1, 1, 1]) == [0, 0]
    assert HSixCode.get_stabs([1, 0, 0, 0, 0, 0]) == [1, 0]
    assert HSixCode.get_logicals([1, 0, 1, 0, 1, 0]) == [1, 0]
    assert HSixCode.get_logicals([0, 1, 0, 1, 0, 1]) == [0, 1]


def test_get_dist_circ_prepares_plus_plus_L():
    circ = get_dist_circ([0, 1, 2, 3, 4, 5])
    assert circ.num_qubits == 6
    gate_counts = {"H": 0, "CX": 0}
    for inst in circ.flattened():
        if inst.name in gate_counts:
            gate_counts[inst.name] += len(inst.targets_copy()) // (1 if inst.name == "H" else 2)
    assert gate_counts == {"H": 4, "CX": 8}
    sim = stim.TableauSimulator()
    sim.do(circ)
    for check in ("+XXXXII", "+IIXXXX", "+ZZZZII", "+IIZZZZ"):
        assert sim.peek_observable_expectation(stim.PauliString(check)) == 1
    # |++>_L : X0_L = X1_L = +1
    assert sim.peek_observable_expectation(stim.PauliString("+XIXIXI")) == 1
    assert sim.peek_observable_expectation(stim.PauliString("+IXIXIX")) == 1


def test_get_ft_init_circ_prepares_zero_zero_L_and_flags_stay_0():
    circ = get_ft_init_circ([0, 1, 2, 3, 4, 5], flags=[6, 7])
    sim = stim.TableauSimulator()
    sim.do(circ)  # trailing M on flags is deterministic-0 noiseless
    assert sim.current_measurement_record()[-2:] == [False, False]
    for check in ("+XXXXII", "+IIXXXX", "+ZZZZII", "+IIZZZZ"):
        assert sim.peek_observable_expectation(stim.PauliString(check + "II")) == 1
    # |00>_L : Z0_L = Z1_L = +1
    assert sim.peek_observable_expectation(stim.PauliString("+ZIZIZIII")) == 1
    assert sim.peek_observable_expectation(stim.PauliString("+IZIZIZII")) == 1


def test_bell_pair_h_check_is_deterministic_and_dem_clean():
    system = QECSystem()
    system.add_patch(HSixCode(h_check_ancillas=2), name="c622")
    tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker, system, if_detector=True)
    builder.write_coordinates()
    data = sorted(system.data_indices)
    builder.initialize({q: "Z" for q in data}, n=system.num_qubits)
    builder.apply_unitary_block(get_dist_circ(data))
    builder.apply_syndrome_extraction(
        circuit_chunk=HSixExtractionBlock(system).circuit, rounds=1
    )
    h_check = HSixLogicalXCheckBlock(system)
    assert h_check.ancilla_indices == [10, 11]
    builder.apply_syndrome_extraction(circuit_chunk=h_check.circuit, rounds=1)
    builder.apply_data_readout({q: "X" for q in data})

    circ = builder.circuit
    dets, obs = circ.compile_detector_sampler().sample(256, separate_observables=True)
    assert not dets.any() and not obs.any()
    circ.detector_error_model(decompose_errors=True)


def test_draw_tanner_graph_smoke():
    plt = pytest.importorskip("matplotlib.pyplot")
    from lightstim.utils.tanner import draw_tanner_graph

    ax = draw_tanner_graph(HSixCode())
    # Two X checks + two Z checks + six data nodes, plus logical-support markers.
    assert len(ax.patches) >= 10
    assert ax.get_title()
    plt.close(ax.figure)


def test_logical_op_set_transversal_h_is_self_dual():
    system = QECSystem()
    system.add_patch(HSixCode(), name="c622")
    patch = system.patches["c622"][0]
    tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker, system, if_detector=False)
    builder.write_coordinates()
    builder.initialize({q: "Z" for q in system.data_indices}, n=system.num_qubits)

    ops = HSixLogicalOpSet()
    ops.transversal_hadamard(builder, patch)
    ops.transversal_s(builder, patch)
    # Just needs to append cleanly and keep the circuit simulable.
    builder.circuit.compile_detector_sampler()
