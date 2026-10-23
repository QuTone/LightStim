"""Native lattice-surgery output and explicit readout-boundary regressions."""
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pytest

from lightstim.ir.builder import CircuitBuilder, DataReadoutStage
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedTwoPatchCoupler,
)


def _two_patch_circuit(initial, final, episodes, rounds, combined, *, held=False):
    system = QECSystem()
    first = UnrotatedSurfaceCode(distance=3)
    first.rotate_coords(np.pi)
    first.reset_rotation_angle()
    system.add_patch(first, name="A")
    system.add_patch(UnrotatedSurfaceCode(distance=3), name="B", offset=(0, 6))
    system.register_coupler(
        UnrotatedTwoPatchCoupler(), patch_names=["A", "B"],
        name="AB", interaction_type="ZZ",
    )
    tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=2)
    builder = CircuitBuilder(tracker, system)
    builder.write_coordinates()
    owner = system.index_to_owner_map
    patch_data = sorted(q for q in system.data_indices if owner[q] in ("A", "B"))
    corridor = sorted(q for q in system.data_indices if owner[q] == "AB")
    bases = dict(zip(("A", "B"), initial))
    builder.initialize({q: bases[owner[q]] for q in patch_data}, n=system.num_qubits)
    builder.apply_syndrome_extraction(UnrotatedSurfaceCodeExtractionBlock(system).circuit)

    for episode in range(episodes):
        builder.activate_coupler("AB")
        builder.initialize({q: "X" for q in corridor}, n=system.num_qubits)
        builder.apply_syndrome_extraction(
            UnrotatedSurfaceCodeExtractionBlock(system).circuit, rounds=rounds,
        )
        if episode == episodes - 1 and (combined or held):
            break
        builder.deactivate_coupler("AB")
        builder.apply_data_readout({q: "X" for q in corridor})

    bases = dict(zip(("A", "B"), final))
    readout = {q: bases[owner[q]] for q in patch_data}
    if held:
        # A terminal Z measurement of the continuing merged code is one
        # readout operation, with no preceding split declaration.
        readout.update({q: "Z" for q in corridor})
        builder.apply_data_readout(readout)
        assert "AB" in system.paused_stabilizer_indices
    elif combined:
        readout.update({q: "X" for q in corridor})
        builder.apply_staged_data_readout(readout, stages=(
            DataReadoutStage(tuple(corridor), ("AB",)),
            DataReadoutStage(tuple(patch_data)),
        ))
    else:
        builder.apply_data_readout(readout)
    assert tracker.expected_num_logicals == 0
    return builder.to_stim_circuit()


def _assert_outputs(circuit, expected):
    assert circuit.num_observables == expected
    circuit.detector_error_model()
    detectors, observables = circuit.compile_detector_sampler(seed=17).sample(
        32, separate_observables=True,
    )
    assert not detectors.any()
    assert not observables.any()


CASES = [
    ("split", "ZZ", "ZZ", 1, 1, False, 3),
    ("staged_final", "ZZ", "ZZ", 1, 1, True, 3),
    ("two_merges", "ZZ", "ZZ", 2, 1, False, 4),
    ("repeated_extraction", "ZZ", "ZZ", 1, 2, False, 3),
    ("teleportation_readout", "ZX", "XZ", 1, 1, False, 1),
    ("mixed_initialization", "ZX", "ZZ", 1, 1, False, 2),
    ("x_initialization", "XX", "ZZ", 1, 1, False, 1),
]


@pytest.mark.parametrize(
    "name,initial,final,episodes,rounds,combined,expected", CASES,
    ids=[case[0] for case in CASES],
)
def test_native_two_patch_measurement_outputs(
        name, initial, final, episodes, rounds, combined, expected):
    circuit = _two_patch_circuit(initial, final, episodes, rounds, combined)
    _assert_outputs(circuit, expected)


@pytest.mark.parametrize("rounds", [1, 2])
def test_held_merge_terminal_Z_readout_has_two_observables(rounds):
    circuit = _two_patch_circuit("ZZ", "ZZ", 1, rounds, False, held=True)
    _assert_outputs(circuit, 2)


@pytest.mark.parametrize("initial,expected", [("ZZ", 3), ("ZX", 2), ("XX", 1)])
@pytest.mark.parametrize("rounds", [1, 2])
def test_default_two_patch_experiment_outputs(initial, expected, rounds):
    experiment = TwoPatchLSExperiment(
        patch1_config={"distance": 3}, patch2_config={"distance": 3},
        offset=(0, 6), interaction_type="ZZ", rounds=rounds,
        initial_state_patch1=initial[0], initial_state_patch2=initial[1],
        measure_state_patch1="Z", measure_state_patch2="Z", noise_params=None,
    )
    with redirect_stdout(StringIO()):
        circuit = experiment.build()
    _assert_outputs(circuit, expected)
    assert experiment.tracker.expected_num_logicals == 0
