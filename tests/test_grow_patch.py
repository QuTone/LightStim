"""Circuit-level grow_patch verification with the rule-based (paper) geometry.

Litinski Fig 45 step 4->5: the paper-q2-orientation d×d patch grows DOWN through its
Z boundary onto the (2d-1)×d bar (new data qubits initialised |+>), geometry from
``RuleBasedRectPatch(..., pair_type='X')`` — the layout verified check-by-check against
panel 5 in ``test_rule_based_patch.py``.

Gates (spec §5): p=0 detectors quiet + observable deterministic (both memory bases,
i.e. both logicals' eigenvalues are carried through the growth); anchoring — 7/8 (d=3)
resp. 22/24 (d=5) old checks keep their uid across the grow; noisy graphlike
distance == d.

Scheduling note: boundary-deformation steps NEED the K&F diagonal schedule.  The
grown bar's Z̄ representative bends (vertical left leg + horizontal bottom leg), and
an exhaustive sweep of every (phase-1, phase-2) combination of the N/Z variants
('perpendicular'/'swapped'/'parallel') caps the Z-memory graphlike distance at 2 —
hook pairs around the handoff seam are unavoidable for any single N/Z orientation.
The diagonal schedule (hooks never align with any logical direction) reaches full
distance for both bases at d = 3 and 5.  For PLAIN memory on the mirrored (paper-q2)
patch, the N/Z block still works if and only if ``scheduling='swapped'`` — kept as a
regression witness below.
"""
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated.SE_block import (
    RotatedSurfaceCodeExtractionBlock)
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock)
from lightstim.qec_code.surface_code.rotated.rule_based_patch import RuleBasedRectPatch
from lightstim.noise.config import NoiseConfig

pytestmark = pytest.mark.smoke

NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


# closed-form stage layouts live in the library now (litinski_layouts); the names are
# re-exported here because the demo notebook and sibling test modules import them
from lightstim.qec_code.surface_code.rotated.litinski_layouts import (   # noqa: F401
    ptype, start_w2 as paper_start_w2, grown_forced)


def build_grow_memory(d, basis):
    """Init memory in ``basis`` -> d SE rounds -> grow -> d SE rounds -> readout."""
    dy = 2 * (d - 1)
    start = RuleBasedRectPatch(distance_x=d, distance_z=d,
                               forced=paper_start_w2(d), phase=0)
    system = QECSystem()
    system.add_patch(start, offset=(0, dy), name="q2")
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    builder.initialize({q: basis for q in sorted(system.data_indices)},
                       n=system.num_qubits)
    se = DiagonalSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(se.circuit, rounds=d)

    grown_geom = RuleBasedRectPatch(distance_x=2 * d - 1, distance_z=d,
                                    forced=grown_forced(d), phase=0)
    old_active = set(system.active_stabilizer_indices)
    new_data, _ = system.grow_patch("q2", grown_geom, offset=(0, 0))
    kept = len(old_active & set(system.active_stabilizer_indices))

    # growth through the Z boundary extends Z̄, so the new region is initialised in
    # the Z basis — |0⟩, NOT the spec's original |+⟩ (Fig 40d transposed: "wider
    # patch" = X̄-extension = X-basis init; confirmed by the K&F reference
    # implementation, patch_rotation_manual.py, z=1 block reset="z").  Every new
    # Z check is then first-round deterministic (Fig 41 trivial outcome) — the
    # retiring lobes' successors continue their banked values and every
    # handoff-window error is detector-spanned.
    builder.initialize({q: 'Z' for q in new_data}, n=system.num_qubits)
    se2 = DiagonalSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(se2.circuit, rounds=d)
    builder.apply_data_readout({q: basis for q in sorted(system.data_indices)})
    return builder, kept, new_data


def _check(d, basis, kept_expect):
    builder, kept, new_data = build_grow_memory(d, basis)
    assert len(new_data) == (d - 1) * d
    assert kept == kept_expect
    det, obs = builder.circuit.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any(), f"d={d} {basis}: detector fired at p=0"
    assert not obs.any(), f"d={d} {basis}: observable not deterministic at p=0"
    noisy = builder.build_noisy_circuit(noise_params=NP, noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == d


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_grow_d3(basis):
    _check(3, basis, kept_expect=7)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_grow_d5(basis):
    _check(5, basis, kept_expect=22)


def test_diagonal_fixed_table_is_seven_slots():
    """The schedule is the FIXED compact-7 table of K&F Fig 4 (Z at slots 1-4,
    X at 4-7) — a lookup, not a packing search, so T is always exactly 7."""
    d = 3
    dy = 2 * (d - 1)
    system = QECSystem()
    system.add_patch(RuleBasedRectPatch(distance_x=d, distance_z=d,
                                        forced=paper_start_w2(d), phase=0),
                     offset=(0, dy), name="q2")
    assert DiagonalSurfaceCodeExtractionBlock(system).coupling_slots == 7


def test_mirror_patch_needs_swapped_scheduling():
    """Regression witness for the scheduling discovery: with the default
    'perpendicular' zigzag the mirrored start patch's memory distance halves."""
    d = 3
    start = RuleBasedRectPatch(distance_x=d, distance_z=d,
                               forced=paper_start_w2(d), phase=0)
    system = QECSystem()
    system.add_patch(start, offset=(0, 0), name="q")
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    builder.initialize({q: 'X' for q in sorted(system.data_indices)},
                       n=system.num_qubits)
    se = RotatedSurfaceCodeExtractionBlock(system, scheduling='perpendicular')
    builder.apply_syndrome_extraction(se.circuit, rounds=d)
    builder.apply_data_readout({q: 'X' for q in sorted(system.data_indices)})
    noisy = builder.build_noisy_circuit(noise_params=NP, noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == 2
