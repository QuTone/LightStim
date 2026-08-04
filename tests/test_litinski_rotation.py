"""End-to-end verification of the full Litinski patch rotation (Fig 45 steps 4→9).

Layout ground truth (800/1600-dpi crops, bar-local y-up coords):

* panel 8 (the bar regrown upward from the rotated patch) — the ROTATED convention
  on the full (2d−1)×d bar: left Z(0,8),(0,4); right Z(6,6),(6,2); top X(4,10);
  bottom X(2,0); bulk unchanged;
* panel 9 (home tile, rotated) — tile-local layout IDENTICAL to panel 7:
  the two shrinks produce the same rotated-convention square.

End-to-end gates for the ``litinski_rotation`` logical op (5 steps × d diagonal rounds):
p=0 determinism, graphlike distance d for both memory bases at d = 3 and 5, the
final data footprint EQUAL to the starting one (the patch is back home), and the
final logical representatives orientation-swapped (X̄ vertical, Z̄ horizontal).
"""
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock)
from lightstim.qec_code.surface_code.rotated.rule_based_patch import RuleBasedRectPatch
from lightstim.qec_code.surface_code.rotated.litinski_layouts import (
    start_w2, shrunk_w2, bar2_w2)
from lightstim.qec_code.surface_code.rotated.operation import (
    RotatedSurfaceCodeLogicalOpSet,
)

_OPS = RotatedSurfaceCodeLogicalOpSet()


def _rotate(system, builder, name, **kwargs):
    """Rotate via the LogicalOpSet adapter (the op under test)."""
    return _OPS.litinski_rotation(builder, system.patches[name][0], **kwargs)
from lightstim.noise.config import NoiseConfig

pytestmark = pytest.mark.smoke

NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)

PANEL8_W2 = {   # Fig 45 panel 8, hand-read (d=3, bar-local coords)
    (0, 8):  ('Z', [(1, 7), (1, 9)]),
    (0, 4):  ('Z', [(1, 3), (1, 5)]),
    (6, 6):  ('Z', [(5, 5), (5, 7)]),
    (6, 2):  ('Z', [(5, 1), (5, 3)]),
    (4, 10): ('X', [(3, 9), (5, 9)]),
    (2, 0):  ('X', [(1, 1), (3, 1)]),
}
PANEL9_W2 = {   # Fig 45 panel 9's q2, hand-read, tile-local coords
    (0, 4): ('Z', [(1, 3), (1, 5)]),
    (6, 2): ('Z', [(5, 1), (5, 3)]),
    (4, 6): ('X', [(3, 5), (5, 5)]),
    (2, 0): ('X', [(1, 1), (3, 1)]),
}


def _norm(w2):
    return {sc: (t, sorted(map(tuple, sup))) for sc, (t, sup) in w2.items()}


def test_bar2_matches_panel8():
    assert _norm(bar2_w2(3)) == _norm(PANEL8_W2)


def test_final_matches_panel9_and_equals_panel7():
    assert _norm(shrunk_w2(3)) == _norm(PANEL9_W2)


def _run(d, basis):
    dy = 2 * (d - 1)
    system = QECSystem()
    system.add_patch(RuleBasedRectPatch(distance_x=d, distance_z=d,
                                        forced=start_w2(d), phase=0),
                     offset=(0, dy), name="q2")
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()

    start_coords = {system.qubit_coords[q] for q in sorted(system.data_indices)}
    builder.initialize({q: basis for q in sorted(system.data_indices)},
                       n=system.num_qubits)
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=d)

    summary = _rotate(system, builder, "q2")

    live = [q for q in sorted(system.data_indices)
            if system.index_to_owner_map.get(q) == "q2"
            and q not in tracker.retired_qubits]
    final_coords = {system.qubit_coords[q] for q in live}
    builder.apply_data_readout({q: basis for q in live})
    return builder, system, summary, start_coords, final_coords, live


def _check(d, basis):
    builder, system, summary, start_coords, final_coords, live = _run(d, basis)
    # the patch is back on its home tile
    assert final_coords == start_coords
    assert summary['grow_new'] == summary['regrow_new'] == d * (d - 1)
    # the final logicals are orientation-swapped: X̄ vertical, Z̄ horizontal
    ops = {op['type']: op for op in system.logical_ops
           if op.get('patch_name') == "q2"}
    xsup = {system.qubit_coords[i] for i in ops['X']['data_indices']}
    zsup = {system.qubit_coords[i] for i in ops['Z']['data_indices']}
    assert len({x for x, y in xsup}) == 1     # X̄ on one column (vertical)
    assert len({y for x, y in zsup}) == 1     # Z̄ on one row (horizontal)

    det, obs = builder.circuit.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any(), f"d={d} {basis}: detector fired at p=0"
    assert not obs.any(), f"d={d} {basis}: observable not deterministic at p=0"
    noisy = builder.build_noisy_circuit(noise_params=NP, noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == d


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_full_rotation_d3(basis):
    _check(3, basis)


def test_full_rotation_plus_y_direction():
    """The '+y' growth direction (stim-rendering frames): same gates at d=3."""
    from lightstim.qec_code.surface_code.rotated.litinski_layouts import flip_w2
    d = 3
    system = QECSystem()
    system.add_patch(RuleBasedRectPatch(distance_x=d, distance_z=d,
                                        forced=flip_w2(start_w2(d), 2 * d), phase=1),
                     offset=(0, 0), name="q2")
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    start_coords = {system.qubit_coords[q] for q in sorted(system.data_indices)}
    builder.initialize({q: 'X' for q in sorted(system.data_indices)},
                       n=system.num_qubits)
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=d)
    _rotate(system, builder, "q2", direction='+y')
    live = [q for q in sorted(system.data_indices)
            if system.index_to_owner_map.get(q) == "q2"
            and q not in tracker.retired_qubits]
    assert {system.qubit_coords[q] for q in live} == start_coords
    builder.apply_data_readout({q: 'X' for q in live})
    det, obs = builder.circuit.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = builder.build_noisy_circuit(noise_params=NP, noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == d


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_full_rotation_d5(basis):
    _check(5, basis)


@pytest.mark.parametrize("lr,tb,phase,direction,msg", [
    # X̄-vertical starts: used to SILENTLY no-op ('-y'@phase0, '+y'@phase1,
    # burning 5d rounds) or die deep inside shrink_patch — now a clear raise
    ('Z', 'X', 0, '-y', "does not match the '-y' start convention"),
    ('Z', 'X', 0, '+y', "does not match the '\\+y' start convention"),
    ('Z', 'X', 1, '-y', "does not match the '-y' start convention"),
    ('Z', 'X', 1, '+y', "does not match the '\\+y' start convention"),
    # legal layout, wrong direction (the direction is fixed by the phase;
    # the two directions' start layouts are position-identical)
    ('X', 'Z', 0, '+y', "requires direction '-y'"),
    ('X', 'Z', 1, '-y', r"requires direction '\+y'"),
])
def test_illegal_start_raises(lr, tb, phase, direction, msg):
    from lightstim.qec_code.surface_code.rotated.litinski_layouts import (
        rect_boundary)
    d, dy = 3, 4
    system = QECSystem()
    system.add_patch(RuleBasedRectPatch(distance_x=d, distance_z=d,
                                        forced=rect_boundary(d, d, lr, tb, phase),
                                        phase=phase),
                     offset=(0, dy), name="q2")
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    builder.initialize({q: 'Z' for q in sorted(system.data_indices)},
                       n=system.num_qubits)
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=1)
    with pytest.raises(ValueError, match=msg):
        _rotate(system, builder, "q2", rounds=1,
                direction=direction)


def test_concurrent_litinski_two_patches_lockstep():
    # Two disjoint patches rotated in parallel (rotate_patches_litinski):
    # the five stages advance in lockstep sharing system-wide SE rounds —
    # total duration ≈ a single rotation (5·d rounds), not 2×; both patches
    # pass every gate (clean at p=0, graphlike=d, both home with flipped
    # orientation)
    d = 3
    dy = 2 * (d - 1)

    def build(names_offsets):
        system = QECSystem()
        for nm, off in names_offsets:
            system.add_patch(RuleBasedRectPatch(distance_x=d, distance_z=d,
                                                forced=start_w2(d), phase=0),
                             offset=off, name=nm)
        tracker = SyndromeTracker(num_qubits=system.num_qubits,
                                  expected_num_logicals=system.num_logicals)
        builder = CircuitBuilder(tracker=tracker, system_config=system,
                                 if_detector=True)
        system.register_tracker(tracker)
        system.register_builder(builder)
        builder.write_coordinates()
        builder.initialize({q: 'Z' for q in sorted(system.data_indices)},
                           n=system.num_qubits)
        builder.apply_syndrome_extraction(
            DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=d)
        return system, builder

    # single-patch baseline duration
    s1, b1 = build([("qA", (0, dy))])
    _rotate(s1, b1, "qA")
    b1.apply_data_readout(
        {q: 'Z' for q in sorted(s1.data_indices)
         if s1.index_to_owner_map.get(q) == "qA"
         and q not in s1._tracker.retired_qubits})
    t1 = b1.circuit.num_ticks

    # two patches in parallel
    s2, b2 = build([("qA", (0, dy)), ("qB", (20, dy))])
    summaries = _OPS.litinski_rotation(
        b2, s2.patches["qA"][0], s2.patches["qB"][0])
    assert set(summaries) == {"qA", "qB"}
    live = [q for q in sorted(s2.data_indices)
            if q not in s2._tracker.retired_qubits]
    b2.apply_data_readout({q: 'Z' for q in live})
    t2 = b2.circuit.num_ticks
    assert abs(t2 - t1) <= 6, (t1, t2)    # lockstep shared SE: total duration ≈ one rotation
    det, obs = b2.circuit.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = b2.build_noisy_circuit(noise_params=NP,
                                   noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == d


# ---- ±x variants: the transpose-conjugated protocol (X̄-vertical start) ----

def _run_x(d, basis, sign):
    """X̄-vertical start (transpose of paper-q2), bar grows along ±x."""
    from lightstim.qec_code.surface_code.rotated.litinski_layouts import (
        rect_boundary)
    dy = 2 * (d - 1)
    system = QECSystem()
    off = (dy, 0) if sign == '-' else (0, 0)
    system.add_patch(RuleBasedRectPatch(distance_x=d, distance_z=d,
                                        forced=rect_boundary(d, d, 'Z', 'X'),
                                        phase=0),
                     offset=off, name="q2")
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    data = sorted(system.data_indices)
    start_coords = {system.qubit_coords[q] for q in data}
    builder.initialize({q: basis for q in data}, n=system.num_qubits)
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=d)
    _rotate(system, builder, "q2", direction=sign + 'x')
    live = [q for q in sorted(system.data_indices)
            if system.index_to_owner_map.get(q) == "q2"
            and q not in tracker.retired_qubits]
    final_coords = {system.qubit_coords[q] for q in live}
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=d)
    builder.apply_data_readout({q: basis for q in live})
    return builder, system, start_coords, final_coords


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_x_variant_full_distance(d, basis):
    builder, system, start_coords, final_coords = _run_x(d, basis, '-')
    assert final_coords == start_coords          # back on the home tile
    # final logicals orientation-swapped: X̄ horizontal after an X̄-vertical start
    ops = {op['type']: op for op in system.logical_ops
           if op.get('patch_name') == "q2"}
    xsup = {system.qubit_coords[i] for i in ops['X']['data_indices']}
    assert len({y for x, y in xsup}) == 1        # X̄ on one row (horizontal)
    c = builder.circuit
    det, obs = c.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any(), f"d={d} {basis} -x: detector fired at p=0"
    assert not obs.any(), f"d={d} {basis} -x: observable not deterministic"
    noisy = builder.build_noisy_circuit(noise_params=NP,
                                        noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == d


def test_x_variant_phase_direction_lock():
    # phase 0 fixes '-x'; asking for '+x' raises with the legal direction
    with pytest.raises(ValueError, match=r"requires direction '-x'"):
        _run_x(3, "Z", '+')
