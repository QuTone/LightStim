"""Fault distance of the Gidney Y-basis birth (arXiv:2302.07395, Fig. 5/7).

Regression guard for a distance loss that was NOT in the detector bookkeeping.
``make_y_transition_chunk`` is a gate-for-gate port of Gidney's transition round,
and his Y-memory experiment builds the whole first half out of *inverted* chunks
(``final_round.inverted()``, ``boundary_round.inverted()``,
``qubit_to_boundary_round.inverted()`` in ``midout/circuits/_y_memory_circuit.py``).
The rounds on either side of the transition therefore have to use HIS interaction
order (``order_S`` for X tiles, ``order_N`` for Z tiles), time-reversed on the
degenerate-patch side.  Running LightStim's default 'perpendicular' zig-zag next
to the ported chunk mismatches the hook errors across the round boundary and
costs exactly one unit of fault distance (d -> d-1) while leaving every detector
deterministic, so only a distance check catches it.

The experiment ends with one NOISELESS syndrome-extraction round: a transversal
Y readout of the rotated patch admits no data-readout detectors (the code has no
Y-type stabilizer -- that is the point of the paper), so without a perfect final
stabilizer measurement the last noisy round is unchecked and the distance is 1.
"""
import numpy as np
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.noise.config import NoiseConfig
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

LOW_NOISE = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3,
                        p_idle=1e-3)
_MEAS = {'M', 'MX', 'MY', 'MR', 'MRX', 'MRY'}


def _build_y_birth(d, pre_sched='gidney_reversed', post_sched='gidney'):
    """Degenerate patch -> Y-birth transition -> d memory rounds -> Y readout."""
    system = QECSystem()
    degen = make_degenerate_y_boundary_patch(d)
    gp = system.add_patch(degen, name="P", offset=(0, 0))
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=0)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)

    # Gidney's init basis: Z on the q.real + q.imag < d triangle, X elsewhere.
    init = {}
    for q in system.data_indices:
        x, y = system.qubit_coords[q]
        init[q] = 'Z' if (x - 1) // 2 + (y - 1) // 2 < d else 'X'
    builder.initialize(init_dict=init, n=system.num_qubits)

    boundary = RotatedSurfaceCodeExtractionBlock(
        system, scheduling=pre_sched).circuit
    builder.apply_syndrome_extraction(circuit_chunk=boundary, rounds=1)
    builder.apply_syndrome_extraction(circuit_chunk=boundary,
                                      rounds=max(1, d // 2))

    system.grow_patch("P", RotatedSurfaceCode(distance=d), offset=(0, 0))
    builder.apply_relay_chunk(
        make_y_transition_chunk(system, "P", gp, direction='init'))

    n = tracker.num_qubits
    yl = np.zeros(2 * n, dtype=np.uint8)
    for rec in system.logical_ops:
        if rec.get('patch_name') != 'P':
            continue
        for q, p in rec['pauli'].items():
            if p in ('X', 'Y'):
                yl[q] ^= 1
            if p in ('Z', 'Y'):
                yl[n + q] ^= 1
    tracker.declare_logical(yl)

    memory = RotatedSurfaceCodeExtractionBlock(
        system, scheduling=post_sched).circuit
    builder.apply_syndrome_extraction(circuit_chunk=memory, rounds=d)
    builder.apply_syndrome_extraction(circuit_chunk=memory, rounds=1,
                                      noiseless=True)
    builder.apply_data_readout(
        final_measurements={q: 'Y' for q in system.data_indices},
        noiseless=True)
    return system, tracker, builder


def _graphlike_distance(builder):
    noisy = builder.build_noisy_circuit(noise_params=LOW_NOISE,
                                        noise_model='circuit_level')
    return len(noisy.shortest_graphlike_error())


def _per_layer_detector_counts(circuit):
    out, in_layer = [], False
    for inst in circuit.flattened():
        if inst.name in _MEAS:
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


def test_gidney_schedule_is_the_reverse_of_its_reverse():
    s = RotatedSurfaceCodeExtractionBlock.SCHEDULES
    assert s['gidney_reversed'] == s['gidney'][::-1]
    # arXiv:2302.07395 order_S = [UR, UL, DR, DL] (X), order_N = [UR, DR, UL, DL]
    # (Z), with UR=(+1,-1) UL=(-1,-1) DR=(+1,+1) DL=(-1,+1) in LightStim units.
    assert [x for x, _ in s['gidney']] == [(+1, -1), (-1, -1), (+1, +1), (-1, +1)]
    assert [z for _, z in s['gidney']] == [(+1, -1), (+1, +1), (-1, -1), (-1, +1)]


@pytest.mark.smoke
def test_y_birth_is_noiseless_deterministic_d3():
    _, _, builder = _build_y_birth(3)
    circuit = builder.circuit
    shots = circuit.compile_detector_sampler(seed=5).sample(
        shots=256, append_observables=True)
    assert not shots.any(), "Y-birth circuit has a non-deterministic detector"


@pytest.mark.smoke
def test_y_birth_detector_counts_match_golden_d3():
    """Golden r=3,d=3,b=Y,rb=1 emits 5/8/8/8/8 over its first five layers."""
    _, _, builder = _build_y_birth(3)
    counts = _per_layer_detector_counts(builder.circuit)
    assert counts[:5] == [5, 8, 8, 8, 8], counts


@pytest.mark.smoke
def test_y_birth_has_full_fault_distance_d3():
    _, _, builder = _build_y_birth(3)
    assert _graphlike_distance(builder) == 3


@pytest.mark.slow
def test_y_birth_has_full_fault_distance_d5():
    _, _, builder = _build_y_birth(5)
    circuit = builder.circuit
    shots = circuit.compile_detector_sampler(seed=5).sample(
        shots=64, append_observables=True)
    assert not shots.any()
    assert _graphlike_distance(builder) == 5


@pytest.mark.parametrize("pre,post", [
    ('perpendicular', 'perpendicular'),   # LightStim default on both sides
    ('gidney_reversed', 'perpendicular'),  # only the boundary side fixed
    ('gidney', 'gidney'),                  # boundary side not time-reversed
])
def test_mismatched_interaction_order_loses_distance(pre, post):
    """Documents the failure mode: still clean, but one distance short."""
    _, _, builder = _build_y_birth(3, pre_sched=pre, post_sched=post)
    shots = builder.circuit.compile_detector_sampler(seed=5).sample(
        shots=64, append_observables=True)
    assert not shots.any(), "mismatch must be invisible to the detectors"
    assert _graphlike_distance(builder) < 3
