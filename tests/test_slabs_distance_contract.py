"""Joint-consistency observables intentionally change the one-round distance.

An independent nine-qubit protocol and an encoded fault at a physical split
boundary distinguish logical information from a pure detector constraint.
"""
import numpy as np
import stim

from examples.integrations.logical_compiler_rotated_surface_ppm import (
    DEFAULT_PROGRAM, compile_program, load_program,
)
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.rotated_surface_ppm import RotatedSurfacePPMExperiment
from tests.test_slabs_acceptance import (
    _annotations, _basis, _bits, _full_record_parities, _mask,
)


def _compile(mode, monkeypatch):
    original = RotatedSurfacePPMExperiment.__init__
    original_step = RotatedSurfacePPMExperiment._apply_ppm_step
    boundary = {}

    def initialize(self, *args, **kwargs):
        kwargs["logical_tracking"] = mode
        original(self, *args, **kwargs)

    def step(self, index, specification):
        result = original_step(self, index, specification)
        if index == 0:
            boundary["prefix"] = _annotations(self.builder.circuit)[0]
            logical_x = next(op for op in self.system.logical_ops
                             if op["patch_name"] == "q00" and op["type"] == "X")
            fault = stim.PauliString(self.system.num_qubits)
            for qubit, letter in logical_x["pauli"].items():
                fault[qubit] = letter
            boundary["fault"] = fault
            boundary["checks"] = []
            for check in self.system.active_stabilizers:
                pauli = stim.PauliString(self.system.num_qubits)
                for qubit, letter in check["pauli"].items():
                    pauli[qubit] = letter
                boundary["checks"].append(pauli)
        return result

    program = load_program(DEFAULT_PROGRAM.with_name("checkerboard_3x3_ppm.json"))
    with monkeypatch.context() as patch:
        patch.setattr(RotatedSurfacePPMExperiment, "__init__", initialize)
        patch.setattr(RotatedSurfacePPMExperiment, "_apply_ppm_step", step)
        result = compile_program(program, rounds=1)
        assert result.experiment.logical_tracking == mode
        assert result.experiment.tracker.logical_tracking == mode
        return result, boundary


def _retarget(circuit, observable_rows):
    result = stim.Circuit()
    for instruction in circuit.flattened():
        if instruction.name != "OBSERVABLE_INCLUDE":
            result.append(instruction)
    for index, row in enumerate(observable_rows):
        result.append("OBSERVABLE_INCLUDE", [
            stim.target_rec(record - result.num_measurements) for record in _bits(row)
        ], index)
    return result


def test_checkerboard_one_round_protects_joint_consistency_as_logical_output(monkeypatch):
    # Three deterministic logical PPM outputs plus nine matching final readouts.
    # TICKs preserve the independent oracle's first-PPM insertion boundary.
    ideal = stim.Circuit("R 0 2 3 6 7\nRX 1 4 5 8\nMPP Z0*Z2\nTICK\n"
                         "MPP X1*X5\nTICK\nMPP X4*Z6*X8")
    ideal_prefix = ideal[:3]
    for qubit, axis in enumerate("ZXZZXXZZX"):
        ideal.append("M" if axis == "Z" else "MX", [qubit])
    ideal_fault = ideal_prefix + stim.Circuit("X 0") + ideal[len(ideal_prefix):]
    assert ideal.num_measurements == ideal.count_determined_measurements() == 12
    assert len(_basis(_full_record_parities(ideal))) == 12
    # Logical X after the first PPM flips the final q00 result, not an SE check.
    assert np.flatnonzero(ideal.reference_sample() ^ ideal_fault.reference_sample()).tolist() == [3]

    legacy, legacy_boundary = _compile("legacy", monkeypatch)
    slabs, slabs_boundary = _compile("slabs", monkeypatch)
    legacy_physical, legacy_detectors, legacy_observables = _annotations(legacy.circuit)
    slabs_physical, slabs_detectors, slabs_observables = _annotations(slabs.circuit)

    if legacy_physical != slabs_physical:
        raise RuntimeError("Tracker modes no longer compile the same physical circuit")
    complete_rank = len(_basis(_full_record_parities(legacy_physical)))
    for mode, result, detectors, observables in (
        ("legacy", legacy, legacy_detectors, legacy_observables),
        ("slabs", slabs, slabs_detectors, slabs_observables),
    ):
        relations = detectors + observables
        if len(_basis(relations)) != complete_rank:
            raise RuntimeError(f"{mode}: emitted record relations are incomplete")
        flows = [stim.Flow(measurements=list(_bits(row))) for row in relations]
        if not legacy_physical.has_all_flows(flows, unsigned=False):
            raise RuntimeError(f"{mode}: an emitted signed relation is invalid")
        if result.circuit.detector_error_model().num_errors:
            raise RuntimeError(f"{mode}: ideal circuit has detector/observable errors")
    assert len(_basis(legacy_detectors + legacy_observables)) - len(_basis(legacy_detectors)) == 9
    assert len(_basis(slabs_detectors + slabs_observables)) - len(_basis(slabs_detectors)) == 12

    # Capture the real protocol boundary, without a hard-coded instruction
    # offset: PPM0 has split and PPM1 has not started. The encoded X is X1 X7 X13.
    prefix, fault = legacy_boundary["prefix"], legacy_boundary["fault"]
    assert prefix == slabs_boundary["prefix"] == legacy_physical[:len(prefix)]
    assert fault == slabs_boundary["fault"]
    assert legacy_boundary["checks"] == slabs_boundary["checks"]
    assert len(legacy_boundary["checks"]) == 72
    assert all(fault.commutes(check) for check in legacy_boundary["checks"])
    fault_gates = stim.Circuit()
    for qubit in fault.pauli_indices():
        fault_gates.append("_XYZ"[fault[qubit]], [qubit])
    changed = prefix + fault_gates + legacy_physical[len(prefix):]
    delta = _mask(np.flatnonzero(legacy_physical.reference_sample() ^ changed.reference_sample()))
    events = lambda rows: [i for i, row in enumerate(rows) if (row & delta).bit_count() % 2]
    assert events(legacy_detectors) == [395]
    assert events(slabs_detectors) == []
    assert events(legacy_observables) == [0]
    assert events(slabs_observables) == [0, 1]

    # Each removed detector is precisely a PPM-to-final-joint consistency.
    for ppm, old_detector, new_observable in ((0, 395, 1), (1, 396, 3), (2, 394, 8)):
        joint_output = _mask(legacy.experiment.ppm_outcomes[ppm].records_post_split)
        assert legacy_detectors[old_detector] == joint_output ^ slabs_observables[new_observable]

    noise = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3,
                        p_reset=1e-3, p_idle=1e-3)
    legacy_noisy = legacy.experiment.builder.build_noisy_circuit(
        noise_params=noise, noise_model="circuit_level")
    legacy_distance = len(legacy_noisy.detector_error_model(
        decompose_errors=True).shortest_graphlike_error())
    if legacy_distance != 3:
        raise RuntimeError(f"Legacy control distance changed to {legacy_distance}")

    slabs_noisy = slabs.experiment.builder.build_noisy_circuit(
        noise_params=noise, noise_model="circuit_level")
    assert _annotations(legacy_noisy)[0] == _annotations(slabs_noisy)[0]
    assert len(slabs_noisy.detector_error_model(
        decompose_errors=True).shortest_graphlike_error()) == 1
    old_targets = _retarget(slabs_noisy, legacy_observables)
    distance = len(old_targets.detector_error_model(
        decompose_errors=True).shortest_graphlike_error())
    # Keeping the old target list cannot restore checks whose logical response
    # disqualifies them as pure detectors. One SE round exposes this time-like
    # measurement fault; the separately recorded three-round control has d=3.
    assert distance == 1
