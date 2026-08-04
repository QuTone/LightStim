"""Full Y-basis memory experiment: birth + memory + DEATH (measure-direction
transition), reproducing Gidney's b=Y reference circuits end to end with NO
noiseless idealisation anywhere.

The death direction requires measuring a tracked logical row through
apply_relay_chunk: the logical's transport flow ends at identity, so instead
of a new row the relay emits an OBSERVABLE_INCLUDE whose records are the
logical's banked seed records XOR the chunk's measuring records — the same
parity golden splits across its two OBSERVABLE_INCLUDE statements.

Golden reference (zenodo.7487893, r=3,d=3,b=Y,rb=1): 74 detectors,
1 observable, per-layer counts [5,8,8,8,8,8,8,8,13], fault distance 3.
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
from lightstim.qec_code.surface_code.rotated.y_boundary_patch import (
    make_degenerate_y_boundary_patch,
)
from lightstim.qec_code.surface_code.rotated.y_transition_round import (
    make_y_transition_chunk,
)
from lightstim.noise.config import NoiseConfig

NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def build_y_memory(d, memory_rounds=None, pad_rounds=None):
    """Gidney b=Y memory: [birth | memory x r | death], fully honest."""
    r = memory_rounds if memory_rounds is not None else d
    rb = pad_rounds if pad_rounds is not None else d // 2

    system = QECSystem()
    degen = make_degenerate_y_boundary_patch(d)
    gp_degen = system.add_patch(degen, name="P", offset=(0, 0))
    degen_uids = set(gp_degen._registered_stabilizer_uids)
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=0)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)

    # birth: anti-diagonal reset, reversed-order degenerate rounds, transition
    init = {}
    for q in system.data_indices:
        x, y = system.qubit_coords[q]
        init[q] = 'Z' if (x - 1) // 2 + (y - 1) // 2 < d else 'X'
    builder.initialize(init_dict=init, n=system.num_qubits)
    ch_rev = RotatedSurfaceCodeExtractionBlock(
        system, scheduling='gidney_reversed').circuit
    builder.apply_syndrome_extraction(circuit_chunk=ch_rev, rounds=1)
    builder.apply_syndrome_extraction(circuit_chunk=ch_rev, rounds=rb)
    system.grow_patch("P", RotatedSurfaceCode(distance=d), offset=(0, 0))
    qubit_uids = set(system.active_stabilizer_indices)
    t_init = make_y_transition_chunk(system, "P", gp_degen, direction='init')
    builder.apply_relay_chunk(t_init)
    n = tracker.num_qubits
    yl = np.zeros(2 * n, dtype=np.uint8)
    for rec in [x for x in system.logical_ops if x.get('patch_name') == 'P']:
        for q, p in rec['pauli'].items():
            if p in ('X', 'Y'):
                yl[q] ^= 1
            if p in ('Z', 'Y'):
                yl[n + q] ^= 1
    tracker.declare_logical(yl)

    # memory
    ch_fwd = RotatedSurfaceCodeExtractionBlock(
        system, scheduling='gidney').circuit
    builder.apply_syndrome_extraction(circuit_chunk=ch_fwd, rounds=r)

    # death: measure-direction transition, then degenerate rounds + readout
    system.active_stabilizer_indices.clear()
    system.active_stabilizer_indices.update(degen_uids)
    t_meas = make_y_transition_chunk(system, "P", gp_degen,
                                     direction='measure')
    builder.apply_relay_chunk(t_meas)
    ch_deg = RotatedSurfaceCodeExtractionBlock(
        system, scheduling='gidney').circuit
    builder.apply_syndrome_extraction(circuit_chunk=ch_deg, rounds=rb)
    # golden's final_round = one more SE round merged with the data readout;
    # our builder emits them as two layers (same detectors, split 13 -> 8+5)
    builder.apply_syndrome_extraction(circuit_chunk=ch_deg, rounds=1)
    final = {}
    for q in system.data_indices:
        if q not in gp_degen.data_indices:
            continue                     # corner was measured in the chunk
        x, y = system.qubit_coords[q]
        final[q] = 'Z' if (x - 1) // 2 + (y - 1) // 2 < d else 'X'
    builder.apply_data_readout(final_measurements=final)
    return system, tracker, builder


def _per_layer_counts(circuit):
    out, in_layer = [], False
    for inst in circuit.flattened():
        if inst.name in ('M', 'MX', 'MY'):
            if not in_layer:
                out.append(0)
                in_layer = True
        elif inst.name == 'DETECTOR':
            if out:
                out[-1] += 1
            in_layer = False
        elif inst.name not in ('SHIFT_COORDS', 'QUBIT_COORDS',
                               'OBSERVABLE_INCLUDE'):
            in_layer = False
    return out


def test_full_y_memory_d3_matches_golden_structure():
    system, tracker, builder = build_y_memory(3)
    c = builder.circuit
    assert c.num_observables == 1
    s = c.compile_detector_sampler(seed=7).sample(
        shots=512, append_observables=True)
    assert not s.any(), "noiseless detectors/observable must be silent"
    # golden: [5,8,8,8,8,8,8,8,13] — its final layer merges the last SE round
    # with the data readout; our builder splits them (8 + 5), same 74 total
    assert _per_layer_counts(c) == [5, 8, 8, 8, 8, 8, 8, 8, 8, 5]
    assert c.num_detectors == 74
    # lifecycle closed: the logical was measured out
    assert tracker.logicals.count == 0
    assert tracker.expected_num_logicals == 0


def test_full_y_memory_d3_fault_distance():
    _, _, builder = build_y_memory(3)
    noisy = builder.build_noisy_circuit(noise_params=NP,
                                        noise_model='circuit_level')
    assert len(noisy.shortest_graphlike_error()) == 3


@pytest.mark.slow
def test_full_y_memory_d5_fault_distance():
    _, tracker, builder = build_y_memory(5)
    c = builder.circuit
    assert c.num_observables == 1
    s = c.compile_detector_sampler(seed=7).sample(
        shots=256, append_observables=True)
    assert not s.any()
    noisy = builder.build_noisy_circuit(noise_params=NP,
                                        noise_model='circuit_level')
    assert len(noisy.shortest_graphlike_error()) == 5
