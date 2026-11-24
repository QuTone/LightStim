import hashlib
import json
from pathlib import Path

import pytest

from lightstim.config import ExperimentConfig, compile, load_config
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.cnot_ls import CNOTLSExperiment
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock


EXAMPLES = Path(__file__).resolve().parents[1] / "examples/config"


def _hash(circuit):
    return hashlib.sha256(str(circuit).encode()).hexdigest()


@pytest.mark.parametrize(("distance", "basis", "p"), [(3, "Z", 0.001), (5, "X", 0.0)])
def test_memory_hash(distance, basis, p):
    point = load_config(EXAMPLES / "rotated-surface-memory.json").resolve(
        {"distance": distance, "basis": basis, "p": p}
    )
    circuit, layout = compile(point)
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=distance), name="memory")
    original = MemoryExperiment(
        qec_system=system, extraction_block_class=RotatedSurfaceCodeExtractionBlock,
        rounds=distance, basis=basis,
        noise_params=NoiseConfig(p_1q=p, p_2q=p, p_idle=p, p_meas=p, p_reset=p),
    )
    assert _hash(circuit) == _hash(original.build())
    assert layout.qubit_coords == system.qubit_coords


@pytest.mark.parametrize(("ctl", "tgt", "mc", "mt"), [("X", "X", "X", "X"), ("X", "Z", "Z", "Z")])
def test_lattice_surgery_hash(ctl, tgt, mc, mt, capsys):
    p = 0.001
    config = load_config(EXAMPLES / "lattice-surgery-cnot.json")
    point = config.resolve({"control_init": ctl, "target_init": tgt,
                            "control_meas": mc, "target_meas": mt})
    circuit, layout = compile(point)
    original = CNOTLSExperiment(
        patch_configs={role: {"distance": 3} for role in ("a", "c", "t")},
        offset_ta=(6, 0), offset_ca=(0, 6), rounds=2,
        initial_state_dict={"a": "X", "c": ctl, "t": tgt},
        measure_state_dict={"a": "Z", "c": mc, "t": mt},
        noise_params=NoiseConfig(p_1q=p, p_2q=p, p_idle=p, p_meas=p, p_reset=p),
    )
    assert _hash(circuit) == _hash(original.build())
    assert layout.qubit_coords == original.system.qubit_coords
    assert capsys.readouterr().out == ""


def test_custom_lattice_surgery_geometry():
    config = json.loads((EXAMPLES / "lattice-surgery-cnot.json").read_text())
    translation = (2, 4)
    spec = load_config(EXAMPLES / "lattice-surgery-cnot.json").resolve()
    transforms = {}
    for decl, role, patch in zip(config["system"]["patches"], ("a", "c", "t"), spec["system"]["patches"]):
        decl["offset"] = [a + b for a, b in zip(patch["offset"], translation)]
        transforms[role] = decl.get("transform", [])
    config["noise"]["p"] = 0
    circuit, system = compile(config)
    protocol = CNOTLSExperiment(
        patch_configs={role: {"distance": 3} for role in transforms},
        offset_ancilla=translation, patch_transforms=transforms, rotate_patches=False,
        noise_params=NoiseConfig(),
    )
    assert _hash(circuit) == _hash(protocol.build())
    assert system.patches["control"][1] == (2, 10)
    assert system.patches["target"][1] == (8, 4)


def test_ancilla_measure_basis_is_private_to_each_experiment():
    kwargs = {"patch_configs": {role: {"distance": 3} for role in ("a", "c", "t")}}
    initial = {"a": "Z", "c": "Z", "t": "Z"}
    measure = {"a": "Z", "c": "Z", "t": "Z"}
    first = CNOTLSExperiment(**kwargs, initial_state_dict=initial, measure_state_dict=measure)
    assert first.measure_state_dict["a"] == "X"
    assert measure["a"] == "Z"
    assert CNOTLSExperiment(**kwargs, initial_state_dict=initial).measure_state_dict["a"] == "X"
    assert CNOTLSExperiment(**kwargs).measure_state_dict["a"] == "Z"


def test_compile_only_section_and_grid_transforms():
    config = load_config(EXAMPLES / "rotated-surface-memory.json")
    section = {name: config[name] for name in ("parameters", "system", "operations", "noise")}
    assert compile(section)[0] == compile(config)[0]
    section["system"]["patches"][0]["transform"] = [{"rotate_deg": 45}]
    with pytest.raises(ValueError, match="90 degrees"):
        compile(section)


@pytest.mark.parametrize(("field", "value", "match"), [
    ("op", "mask_stabilizers", "not supported"),
    ("params", {"basis": {"wrong_patch": "Z"}}, "exactly the targets"),
    ("params", {"basis": "Z", "misspelled": 1}, "basis parameter"),
])
def test_unsupported_operations_fail_at_stage(field, value, match):
    config = json.loads((EXAMPLES / "rotated-surface-memory.json").read_text())
    config["operations"][0][field] = value
    with pytest.raises(ValueError, match=match):
        compile(ExperimentConfig.from_dict(config))
