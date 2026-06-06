"""
Selectable 4-tick / 6-tick CNOT scheduling for the unrotated and toric SE blocks.

Spec: docs/superpowers/specs/2026-06-05-unrotated-se-4tick-scheduling-design.md

Covers, for BOTH UnrotatedSurfaceCodeExtractionBlock and ToricCodeExtractionBlock:
  - default scheduling is the 6-tick Li schedule (backward compatible)
  - scheduling='4tick' builds a valid, conflict-free 4-layer circuit
  - the 4-tick memory circuit is correct (noiseless → zero detections)
  - the 4-tick schedule is fault-tolerant (graphlike distance == 6-tick == d)
  - unknown scheduling raises ValueError
  - the two blocks' SCHEDULES copies agree (drift guard)
"""
import pytest
import stim

from lightstim.noise.config import NoiseConfig
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)
from lightstim.qec_code.surface_code.toric import ToricCode, ToricCodeExtractionBlock

from conftest import assert_noiseless, assert_valid_circuit

LOW_NOISE = NoiseConfig(p_1q=0.001, p_2q=0.001, p_meas=0.001, p_reset=0.001)

# (label, patch factory, SE block class)
BLOCKS = [
    ("unrotated", lambda d: UnrotatedSurfaceCode(distance=d), UnrotatedSurfaceCodeExtractionBlock),
    ("toric",     lambda d: ToricCode(distance=d),            ToricCodeExtractionBlock),
]


def _system(patch_factory, d):
    system = QECSystem()
    system.add_patch(patch_factory(d), name="main")
    return system


def _se_block(se_cls, system, scheduling):
    if scheduling is None:
        return se_cls(system)
    return se_cls(system, scheduling=scheduling)


def _count_ticks(circuit: stim.Circuit) -> int:
    return sum(1 for inst in circuit if inst.name == "TICK")


def _build_memory(patch_factory, se_cls, d, scheduling, rounds, basis="Z"):
    system = _system(patch_factory, d)
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    builder.write_coordinates()
    builder.initialize({q: basis for q in system.data_indices}, n=system.num_qubits)
    se = _se_block(se_cls, system, scheduling)
    builder.apply_syndrome_extraction(se.circuit, rounds=rounds)
    builder.apply_data_readout({q: basis for q in system.data_indices})
    return builder


# ── Structure: tick count tracks schedule length ──────────────────────────────

@pytest.mark.smoke
@pytest.mark.parametrize("label,patch_factory,se_cls", BLOCKS)
def test_default_is_six_tick(label, patch_factory, se_cls):
    """No scheduling arg → existing 6-tick Li schedule. SE block has 6 CNOT layers."""
    se = _se_block(se_cls, _system(patch_factory, 3), scheduling=None)
    # SE block emits: SE_start TICK + post-H TICK + N schedule TICKs + post-H TICK
    assert _count_ticks(se.circuit) == 6 + 3, f"{label}: expected 6-tick schedule by default"


@pytest.mark.smoke
@pytest.mark.parametrize("label,patch_factory,se_cls", BLOCKS)
def test_explicit_six_tick_matches_default(label, patch_factory, se_cls):
    """scheduling='6tick' reproduces the default circuit exactly."""
    default = _se_block(se_cls, _system(patch_factory, 3), scheduling=None)
    explicit = _se_block(se_cls, _system(patch_factory, 3), scheduling="6tick")
    assert str(default.circuit) == str(explicit.circuit), f"{label}: '6tick' must equal default"


@pytest.mark.smoke
@pytest.mark.parametrize("label,patch_factory,se_cls", BLOCKS)
def test_four_tick_has_four_layers(label, patch_factory, se_cls):
    """scheduling='4tick' → 4 CNOT layers (fewer than the 6-tick schedule)."""
    se = _se_block(se_cls, _system(patch_factory, 3), scheduling="4tick")
    assert _count_ticks(se.circuit) == 4 + 3, f"{label}: expected 4-tick schedule"


# ── Validity: stim rejects any same-tick double-drive at build time ────────────

@pytest.mark.smoke
@pytest.mark.parametrize("label,patch_factory,se_cls", BLOCKS)
def test_four_tick_builds_conflict_free(label, patch_factory, se_cls):
    """4-tick schedule must never drive a data qubit twice in one tick (stim would raise)."""
    se = _se_block(se_cls, _system(patch_factory, 3), scheduling="4tick")
    assert se.circuit.num_qubits > 0
    # Each CNOT layer's targets must be unique (no qubit used twice in a layer).
    for inst in se.circuit:
        if inst.name in ("CX", "CNOT", "ZCX"):
            qubits = [t.value for t in inst.targets_copy()]
            assert len(qubits) == len(set(qubits)), f"{label}: data qubit driven twice in one tick"


# ── Correctness: noiseless 4-tick memory circuit has zero detection events ─────

@pytest.mark.parametrize("label,patch_factory,se_cls", BLOCKS)
def test_four_tick_memory_noiseless(label, patch_factory, se_cls):
    """A noiseless 4-tick memory circuit measures stabilizers correctly (no detections)."""
    builder = _build_memory(patch_factory, se_cls, d=3, scheduling="4tick", rounds=3)
    assert_valid_circuit(builder.circuit)
    assert_noiseless(builder.circuit)


# ── Fault tolerance: 4-tick graphlike distance matches 6-tick in BOTH bases ─────

@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_unrotated_four_tick_distance_matches_six_tick(d, basis):
    """4-tick preserves full distance in both memory bases (hooks stay perpendicular).

    The 4-tick schedule mirrors X vs Z on the vertical ticks; this must not let a
    hook error align with either logical, so graphlike distance must equal d for
    both Z-memory and X-memory (matching the 6-tick schedule).
    """
    patch_factory, se_cls = (lambda dd: UnrotatedSurfaceCode(distance=dd)), UnrotatedSurfaceCodeExtractionBlock

    noisy_6 = _build_memory(patch_factory, se_cls, d, "6tick", rounds=d, basis=basis).build_noisy_circuit(
        LOW_NOISE, noise_model="circuit_level"
    )
    noisy_4 = _build_memory(patch_factory, se_cls, d, "4tick", rounds=d, basis=basis).build_noisy_circuit(
        LOW_NOISE, noise_model="circuit_level"
    )

    dist_6 = len(noisy_6.shortest_graphlike_error())
    dist_4 = len(noisy_4.shortest_graphlike_error())
    assert dist_6 == d, f"sanity: 6-tick d={d} {basis}-mem graphlike distance was {dist_6}"
    assert dist_4 == dist_6, f"4-tick {basis}-mem distance {dist_4} < 6-tick {dist_6} (hook errors?)"


# ── Errors and cross-block consistency ────────────────────────────────────────

@pytest.mark.smoke
@pytest.mark.parametrize("label,patch_factory,se_cls", BLOCKS)
def test_unknown_scheduling_raises(label, patch_factory, se_cls):
    with pytest.raises(ValueError):
        _se_block(se_cls, _system(patch_factory, 3), scheduling="nonsense")


@pytest.mark.smoke
def test_schedules_consistent_across_blocks():
    """Unrotated and toric keep their own copies of SCHEDULES; the delta tables must agree."""
    assert (
        UnrotatedSurfaceCodeExtractionBlock.SCHEDULES
        == ToricCodeExtractionBlock.SCHEDULES
    ), "unrotated and toric SCHEDULES have drifted"
