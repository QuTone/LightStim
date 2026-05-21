"""
Extract real LightStim circuit data for frontend demo.
Outputs JSON files into qubeats-lightstim-ide/src/data/
"""
import sys, os, json
sys.path.insert(0, '/home/xiang/workspace/LightStim')

import stim
import numpy as np

# ── LightStim imports ──────────────────────────────────────────────────────────
from src.noise.config import NoiseConfig
from experiments.memory import MemoryExperiment
from experiments.two_patch_LS_unrotated import TwoPatchLSExperiment
from experiments.CNOT_trans import CNOTTransExperiment
from src.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock
from src.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock, UnrotatedTwoPatchCoupler
)
from src.ir.qec_system import QECSystem

OUT_DIR = '/home/xiang/workspace/LightStim/qubeats-lightstim-ide/src/data'
os.makedirs(OUT_DIR, exist_ok=True)

NOISE = NoiseConfig(p_1q=0.001, p_2q=0.001, p_meas=0.001, p_reset=0.001, p_idle=0.001)

# ══════════════════════════════════════════════════════════════════════════════
# DEM extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_dem(circuit: stim.Circuit, metadata: dict, decompose: bool = False) -> dict:
    dem = circuit.detector_error_model(decompose_errors=decompose)
    det_coords = circuit.get_detector_coordinates()

    detectors = []
    for det_id in sorted(det_coords.keys()):
        coords = det_coords[det_id]
        if len(coords) >= 3:
            detectors.append({"id": det_id, "coords": {"x": coords[0], "y": coords[1], "t": coords[2]}})
        elif len(coords) == 2:
            detectors.append({"id": det_id, "coords": {"x": coords[0], "y": coords[1], "t": 0.0}})

    error_mechanisms = []
    num_obs = 0
    for inst in dem.flattened():
        if inst.type == "error":
            prob = float(inst.args_copy()[0])
            detector_ids, observable_ids = [], []
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    detector_ids.append(t.val)
                elif t.is_logical_observable_id():
                    observable_ids.append(t.val)
                    num_obs = max(num_obs, t.val + 1)
            error_mechanisms.append({
                "probability": round(prob, 10),
                "detector_ids": detector_ids,
                "observable_ids": observable_ids,
            })

    return {
        "metadata": {**metadata, "decompose_errors": decompose},
        "detectors": detectors,
        "observables": [{"id": i} for i in range(num_obs)],
        "error_mechanisms": error_mechanisms,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Timeline extraction
# ══════════════════════════════════════════════════════════════════════════════

GATE_MAP = {
    "H": "H", "H_XZ": "H",
    "R": "R", "RZ": "R", "RX": "RX",
    "M": "M", "MR": "MR", "MZ": "M", "MRZ": "MR", "MX": "MX", "MRX": "MX",
    "S": "S", "SQRT_X": "S",
    "CX": "CX", "CNOT": "CX", "CZ": "CZ",
}
TWO_Q_GATES = {"CX", "CNOT", "CZ"}
NOISE_GATES = {
    "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2",
}

def extract_timeline(circuit: stim.Circuit, metadata: dict) -> dict:
    qubit_coords: dict = {}
    for inst in circuit:
        if inst.name == "QUBIT_COORDS":
            c = inst.gate_args_copy()
            for t in inst.targets_copy():
                qubit_coords[t.value] = c

    flat = circuit.flattened()
    ticks, current_ops, current_noise = [], [], []
    num_meas = 0

    for inst in flat:
        name = inst.name
        if name == "TICK":
            if current_ops or current_noise:
                ticks.append({"operations": current_ops, "noise": current_noise})
            current_ops, current_noise = [], []
        elif name in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"):
            pass
        elif name in NOISE_GATES:
            prob = float(inst.gate_args_copy()[0])
            targets = [t.value for t in inst.targets_copy()]
            if name == "DEPOLARIZE2" or name == "PAULI_CHANNEL_2":
                pairs = [(targets[i], targets[i+1]) for i in range(0, len(targets), 2)]
                current_noise.append({"gate": name, "qubit_pairs": pairs, "probability": prob})
            else:
                current_noise.append({"gate": name, "qubits": targets, "probability": prob})
        elif name in GATE_MAP:
            mapped = GATE_MAP[name]
            targets = [t.value for t in inst.targets_copy()]
            if name in TWO_Q_GATES:
                pairs = [(targets[i], targets[i+1]) for i in range(0, len(targets), 2)]
                current_ops.append({"gate": mapped, "qubit_pairs": pairs})
            else:
                if "M" in name:
                    num_meas += len(targets)
                current_ops.append({"gate": mapped, "qubits": targets})

    if current_ops or current_noise:
        ticks.append({"operations": current_ops, "noise": current_noise})

    qubits = []
    for qid in sorted(qubit_coords.keys()):
        c = qubit_coords[qid]
        if len(c) >= 2:
            qubits.append({"id": qid, "coords": {"x": c[0], "y": c[1]}})
        else:
            qubits.append({"id": qid})

    return {
        "metadata": {
            **metadata,
            "num_qubits": len(qubit_coords),
            "num_ticks": len(ticks),
            "num_measurements": num_meas,
        },
        "qubits": qubits,
        "ticks": ticks,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DetSlice extraction — built from QECPatch stabilizer records
# ══════════════════════════════════════════════════════════════════════════════

def _pauli_type_of(stab: dict) -> str:
    t = stab.get("type", "Z")
    return "X" if t == "X" else "Z"

def extract_detslice_from_patch(patch, system, circuit: stim.Circuit, metadata: dict,
                                 rounds: int, patch_name: str = None) -> dict:
    """
    Build detslice data from patch stabilizer records.
    Works for single-patch memory and is adapted for multi-patch by iterating
    patches in the QECSystem.
    """
    det_coords = circuit.get_detector_coordinates()

    # Build qubit list from system qubit_coords (only those with 2D coords)
    src = system if system is not None else patch
    qubits = []
    coord_to_id = {}
    for qid, coords in sorted(src.qubit_coords.items()):
        x, y = coords[0], coords[1]
        qubits.append({"id": qid, "coords": {"x": x, "y": y}})
        coord_to_id[(x, y)] = qid

    # Collect stabilizers from all code patches (skip coupler patches)
    coupler_names = set(system.coupler_patches.keys()) if system is not None else set()
    all_stabs = []
    if system is not None:
        for pname, (p, offset) in system.patches.items():
            if pname in coupler_names:
                continue
            for stab in p.stabilizers:
                global_data_indices = []
                for local_idx in stab["data_indices"]:
                    local_coord = p.qubit_coords.get(local_idx)
                    if local_coord:
                        global_coord = (local_coord[0] + offset[0], local_coord[1] + offset[1])
                        global_idx = coord_to_id.get(global_coord)
                        if global_idx is not None:
                            global_data_indices.append(global_idx)
                all_stabs.append({
                    "data_indices": global_data_indices,
                    "pauli": _pauli_type_of(stab),
                    "syn_coord": stab.get("syn_coord"),
                })
    else:
        for stab in patch.stabilizers:
            all_stabs.append({
                "data_indices": stab["data_indices"],
                "pauli": _pauli_type_of(stab),
                "syn_coord": stab.get("syn_coord"),
            })

    num_stabs = len(all_stabs)

    # detector_coordinates dict
    det_coord_dict = {}
    for det_id, coords in det_coords.items():
        if len(coords) >= 3:
            det_coord_dict[str(det_id)] = {"x": coords[0], "y": coords[1], "t": coords[2]}
        elif len(coords) == 2:
            det_coord_dict[str(det_id)] = {"x": coords[0], "y": coords[1], "t": 0.0}

    # Build slices: one per round (rounds total from SE + 1 final)
    slices = []
    num_rounds_shown = min(rounds + 1, 5)  # cap for display

    for r in range(num_rounds_shown):
        det_list = []
        for s_idx, stab in enumerate(all_stabs):
            det_id = r * num_stabs + s_idx
            pauli_support = [
                {"qubit_id": qid, "pauli": stab["pauli"]}
                for qid in stab["data_indices"]
            ]
            if pauli_support:
                det_list.append({
                    "detector_id": det_id,
                    "pauli_support": pauli_support,
                })
        slices.append({"tick": r + 1, "detectors": det_list})

    return {
        "metadata": {
            **metadata,
            "num_ticks": num_rounds_shown,
            "slices_with_data": num_rounds_shown,
        },
        "qubits": qubits,
        "detector_coordinates": det_coord_dict,
        "slices": slices,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Code snippets (real LightStim API)
# ══════════════════════════════════════════════════════════════════════════════

SNIPPET_MEMORY = """\
from src.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
)
from src.noise.config import NoiseConfig
from experiments.memory import MemoryExperiment

code = RotatedSurfaceCode(distance=3)
noise = NoiseConfig(p=0.001)

exp = MemoryExperiment(
    qec_system=code,
    extraction_block_class=RotatedSurfaceCodeExtractionBlock,
    rounds=3,
    noise_params=noise,
    noise_model="circuit_level",
    basis="Z",
)
circuit = exp.build()
dem = circuit.detector_error_model()
"""

SNIPPET_LS = """\
from src.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedTwoPatchCoupler,
)
from src.noise.config import NoiseConfig
from experiments.two_patch_LS_unrotated import TwoPatchLSExperiment

exp = TwoPatchLSExperiment(
    patch1_config={"distance": 3},
    patch2_config={"distance": 3},
    offset=(6.0, 0.0),          # side-by-side XX merge
    interaction_type="XX",
    coupler_protocol=UnrotatedTwoPatchCoupler(),
    initial_state_patch1="X",
    initial_state_patch2="Z",
    measure_state_patch1="Z",
    measure_state_patch2="X",
    extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
    rounds=3,
    noise_params=NoiseConfig(p_1q=0.001, p_2q=0.001,
                             p_meas=0.001, p_reset=0.001),
    noise_model="circuit_level",
    rotate_patch1=False,
)
circuit = exp.build()
dem = circuit.detector_error_model()
"""

SNIPPET_CNOT = """\
from src.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)
from src.noise.config import NoiseConfig
from experiments.CNOT_trans import CNOTTransExperiment

exp = CNOTTransExperiment(
    code_patch_class=UnrotatedSurfaceCode,
    extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
    code_params_control={"distance": 3},
    code_params_target={"distance": 3},
    offset_target=(8.0, 0.0),
    initial_basis_control="Z",
    initial_basis_target="Z",
    measure_basis_control="Z",
    measure_basis_target="Z",
    rounds_before=3,
    rounds_after=3,
    noise_params=NoiseConfig(p=0.001),
    noise_model="circuit_level",
)
circuit = exp.build()
dem = circuit.detector_error_model()
"""


# ══════════════════════════════════════════════════════════════════════════════
# Build each experiment
# ══════════════════════════════════════════════════════════════════════════════

def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path} ({os.path.getsize(path)//1024}KB)")


def build_exp1():
    print("\n=== Exp 1: Rotated SC Z Memory (d=3, r=3) ===")
    code = RotatedSurfaceCode(distance=3)
    system = QECSystem()
    system.add_patch(code, name="surface_code")
    exp = MemoryExperiment(
        qec_system=system,
        extraction_block_class=RotatedSurfaceCodeExtractionBlock,
        rounds=3,
        noise_params=NOISE,
        noise_model="circuit_level",
        basis="Z",
    )
    circuit = exp.build()
    meta = {"source": "rotated_surface_code:memory_z", "distance": 3, "rounds": 3,
            "noise_model": "circuit_level", "physical_error_rate": 0.001}

    d = f"{OUT_DIR}/exp1_rotated_memory"
    os.makedirs(d, exist_ok=True)
    save(f"{d}/dem_raw.json", extract_dem(circuit, dict(meta), decompose=False))
    save(f"{d}/dem_decomposed.json", extract_dem(circuit, dict(meta), decompose=True))
    save(f"{d}/timeline.json", extract_timeline(circuit, dict(meta)))
    save(f"{d}/detslice.json", extract_detslice_from_patch(code, system, circuit, dict(meta), rounds=3))
    save(f"{d}/snippet.json", {"code": SNIPPET_MEMORY})
    print("  Done.")


def build_exp2():
    print("\n=== Exp 2: Unrotated SC Two-Patch Lattice Surgery (d=3, XX) ===")
    exp = TwoPatchLSExperiment(
        patch1_config={"distance": 3},
        patch2_config={"distance": 3},
        offset=(6.0, 0.0),
        interaction_type="XX",
        coupler_protocol=UnrotatedTwoPatchCoupler(),
        initial_state_patch1="X",
        initial_state_patch2="Z",
        measure_state_patch1="Z",
        measure_state_patch2="X",
        extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
        rounds=3,
        noise_params=NOISE,
        noise_model="circuit_level",
        rotate_patch1=False,
    )
    circuit = exp.build()
    meta = {"source": "unrotated_surface_code:lattice_surgery_xx", "distance": 3, "rounds": 3,
            "noise_model": "circuit_level", "physical_error_rate": 0.001}

    d = f"{OUT_DIR}/exp2_unrotated_ls"
    os.makedirs(d, exist_ok=True)
    save(f"{d}/dem_raw.json", extract_dem(circuit, dict(meta), decompose=False))
    save(f"{d}/dem_decomposed.json", extract_dem(circuit, dict(meta), decompose=True))
    save(f"{d}/timeline.json", extract_timeline(circuit, dict(meta)))
    save(f"{d}/detslice.json", extract_detslice_from_patch(None, exp.system, circuit, dict(meta), rounds=3))
    save(f"{d}/snippet.json", {"code": SNIPPET_LS})
    print("  Done.")


def build_exp3():
    print("\n=== Exp 3: Unrotated SC Transversal CNOT (d=3) ===")
    exp = CNOTTransExperiment(
        code_patch_class=UnrotatedSurfaceCode,
        extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
        code_params_control={"distance": 3},
        code_params_target={"distance": 3},
        offset_target=(8.0, 0.0),
        initial_basis_control="Z",
        initial_basis_target="Z",
        measure_basis_control="Z",
        measure_basis_target="Z",
        rounds_before=3,
        rounds_after=3,
        noise_params=NOISE,
        noise_model="circuit_level",
    )
    circuit = exp.build()
    meta = {"source": "unrotated_surface_code:transversal_cnot", "distance": 3, "rounds": 3,
            "noise_model": "circuit_level", "physical_error_rate": 0.001}

    d = f"{OUT_DIR}/exp3_unrotated_cnot"
    os.makedirs(d, exist_ok=True)
    save(f"{d}/dem_raw.json", extract_dem(circuit, dict(meta), decompose=False))
    save(f"{d}/dem_decomposed.json", extract_dem(circuit, dict(meta), decompose=True))
    save(f"{d}/timeline.json", extract_timeline(circuit, dict(meta)))
    save(f"{d}/detslice.json", extract_detslice_from_patch(None, exp.system, circuit, dict(meta), rounds=3))
    save(f"{d}/snippet.json", {"code": SNIPPET_CNOT})
    print("  Done.")


if __name__ == "__main__":
    build_exp1()
    build_exp2()
    build_exp3()
    print("\n✓ All done. Data written to", OUT_DIR)
