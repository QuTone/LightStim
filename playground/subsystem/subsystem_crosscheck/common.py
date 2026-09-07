"""Reproducible, review-only subsystem benchmarks. All new detectors use LightStim."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import stim
from scipy.stats import beta, binom

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.bacon_shor import BaconShorCode
from lightstim.qec_code.shyps import SHYPSCode
from lightstim.qec_code.generic_css import GenericCSSGaugeExtractionBlock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results/2026-09-06"
REFERENCE = ROOT.parents[1] / "External/ComputingEfficientlyInQLDPCCodes"
ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(errors, shots, alpha=.05):
    """Two-sided exact Clopper-Pearson interval (zero counts get an upper bound)."""
    if not shots:
        return [0., 1.]
    return [float(beta.ppf(alpha / 2, errors, shots - errors + 1)) if errors else 0.,
            float(beta.ppf(1 - alpha / 2, errors + 1, shots - errors)) if errors < shots else 1.]


def bs_theory(d, p):
    # Napp-Preskill equations (10), (12), (19)-(20), one logical sector.
    q = -np.expm1(d * np.log1p(-2 * p)) / 2
    return float(binom.sf(d // 2, d, q))


def physical(circuit):
    out = stim.Circuit()
    for op in circuit.flattened():
        if op.name not in ANNOTATIONS:
            out.append(op.name, op.targets_copy(), op.gate_args_copy())
    return out


def annotation_rows(circuit):
    """Absolute measurement-record parities, affine-augmented by ideal reference."""
    reference = circuit.reference_sample()
    reference_int = sum(int(b) << i for i, b in enumerate(reference))
    ds = []
    obs = [0] * circuit.num_observables
    count = 0
    for op in circuit.flattened():
        if op.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            row = 0
            for t in op.targets_copy():
                assert t.is_measurement_record_target
                row ^= 1 << (count + t.value)
            if op.name == "DETECTOR":
                ds.append(row)
            else:
                obs[int(op.gate_args_copy()[0])] ^= row
        else:
            c = stim.Circuit()
            c.append(op)
            count += c.num_measurements
    augment = lambda row: row | (((row & reference_int).bit_count() % 2) << count)
    return list(map(augment, ds)), list(map(augment, obs))


def basis(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p not in pivots:
                pivots[p] = row
                break
            row ^= pivots[p]
    return pivots


def compare_spaces(a, b):
    ra, rb, union = len(basis(a)), len(basis(b)), len(basis(a + b))
    return dict(rank_a=ra, rank_b=rb, rank_union=union,
                a_subset_b=union == rb, b_subset_a=union == ra, equal=ra == rb == union)


def z_detectors(circuit):
    """Select pure Z-record detector rows AFTER the full LightStim tracker ran."""
    out = stim.Circuit()
    types = []
    for op in circuit.flattened():
        if op.name == "DETECTOR":
            if all(types[len(types) + t.value] == "Z" for t in op.targets_copy()):
                out.append(op)
        else:
            out.append(op)
            c = stim.Circuit()
            c.append(op)
            if c.num_measurements:
                assert op.name in {"M", "MX", "MR", "MRX"}, op
                types.extend(["X" if "X" in op.name else "Z"] * c.num_measurements)
    return out


def initialize(patch):
    system = QECSystem()
    system.add_patch(patch, name="memory")
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    builder.write_coordinates()
    builder.initialize({q: "Z" for q in sorted(system.data_indices)}, system.num_qubits)
    system.active_qubit_indices.update(system.data_indices)
    return system, builder


def bacon_capacity(d, p):
    """Perfect preparation/extraction/readout, ONE independent X/Z data channel.

    The first XZ pair prepares the protected memory. Following the data channel,
    another XZ pair erases the previous Z gauge fixing before measuring Z again.
    This leaves the centre syndrome, as in Napp-Preskill's perfect-syndrome model.
    """
    system, builder = initialize(BaconShorCode(distance=d))
    se = GenericCSSGaugeExtractionBlock(system)
    builder.apply_syndrome_extraction(se.circuit, rounds=1, measurement_blocks=se.measurement_blocks)
    builder.circuit.append("X_ERROR", sorted(system.data_indices), p)
    builder.circuit.append("Z_ERROR", sorted(system.data_indices), p)
    builder.apply_syndrome_extraction(se.circuit, rounds=1, measurement_blocks=se.measurement_blocks)
    builder.apply_data_readout({q: "Z" for q in sorted(system.data_indices)})
    return builder.circuit


def reference_noise(clean, p, n_data):
    """Match the PUBLIC reference .stim assets, including noiseless final M.

    Reset flip p; CNOT DEPOLARIZE2(p); ancilla measurement flip p; NO idle
    noise. This is an explicit benchmark noise adapter, not a claim that the
    paper's prose noise model and published assets are identical.
    """
    out = stim.Circuit()
    for op in clean.flattened():
        targets = op.targets_copy()
        if op.name in {"M", "MX"}:
            assert all(t.value >= n_data for t in targets) or all(t.value < n_data for t in targets)
            if all(t.value >= n_data for t in targets):
                out.append(op.name, targets, p)
            else:
                out.append(op.name, targets)
        else:
            out.append(op)
            if op.name in {"R", "RX"}:
                out.append("X_ERROR" if op.name == "R" else "Z_ERROR", targets, p)
            elif op.name == "CX":
                out.append("DEPOLARIZE2", targets, p)
    return out


def generic_memory(code, size, p, rounds=None, order=("X", "Z"), z_only=False):
    patch = BaconShorCode(distance=size) if code == "bacon_shor" else SHYPSCode(r=size)
    d = size if code == "bacon_shor" else patch.code_distance
    # Pin the extraction used by the archived cross-check: Bacon-Shor's public
    # default may evolve, but regenerating these assets must keep their schedule.
    clean = MemoryExperiment(qec_patch=patch, extraction_block_class=GenericCSSGaugeExtractionBlock,
                             basis="Z", rounds=rounds or d,
                             se_block_kwargs={"basis_order": order}).build()
    if z_only:
        clean = z_detectors(clean)
    return reference_noise(clean, p, len(patch.data_indices))


def reference_path(r, p, detectors="Z"):
    folder = REFERENCE / f"src/stim_circuits/shyps_r{r}_memory_circuits"
    if not folder.is_dir():
        folder = RESULTS / "reference_snapshot" / f"shyps_r{r}_memory_circuits"
    matches = list(folder.glob(f"*_simulation_{detectors}_detectors_circuit_p_{p}.stim"))
    if len(matches) != 1:
        raise ValueError((r, p, detectors, matches))
    return matches[0]


def reference_circuit(r, p, detectors="Z"):
    try:
        return stim.Circuit.from_file(reference_path(r, p, detectors))
    except ValueError:
        # The archive has a sparse p grid. Its fault locations all use the same
        # scalar p, so intermediate points can be obtained without a new circuit.
        source = stim.Circuit.from_file(reference_path(r, .001, detectors))
        out = stim.Circuit()
        for op in source.flattened():
            args = op.gate_args_copy()
            if op.name in {"X_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2", "M", "MX"} and args:
                assert args == [.001], (op.name, args)
                args = [p]
            out.append(op.name, op.targets_copy(), args)
        return out


def gauge_blocks(circuit, n_data):
    """Read the reset/CX/measure physical blocks; no annotations are imported."""
    blocks = []
    active = None
    # Iterate the noisy original: without_noise() can fuse successive CX
    # instructions, which would lose the original noise-location boundaries.
    for op in circuit.flattened():
        if op.name in ANNOTATIONS or op.name.endswith("ERROR") or op.name.startswith("DEPOLARIZE"):
            continue
        if op.name in {"R", "RX"} and op.targets_copy()[0].value >= n_data:
            assert active is None
            active = stim.Circuit()
        if active is not None:
            active.append(op.name, op.targets_copy())
            if op.name == "CX":
                active.append("TICK")
            if op.name in {"M", "MX"}:
                blocks.append(active)
                active = None
    assert active is None
    return blocks


def gauge_supports(block, n_data):
    supports = {}
    for op in block:
        if op.name == "CX":
            ts = [t.value for t in op.targets_copy()]
            for a, b in zip(ts[::2], ts[1::2]):
                anc, data = (a, b) if a >= n_data else (b, a)
                supports.setdefault(anc, set()).add(data)
    return supports


def shyps_reference_mapping(r, circuit):
    """Find a gauge-graph isomorphism; explicitly verify every mapped gauge."""
    patch = SHYPSCode(r=r)
    m, n = patch.simplex_length, patch.num_data_qubits
    blocks = gauge_blocks(circuit, n)
    z_support = gauge_supports(blocks[0], n)
    x_support = gauge_supports(blocks[1], n)
    h_ref = np.zeros((m, m), dtype=np.uint8)
    for i in range(m):
        h_ref[i, list(z_support[n + i])] = 1
    def graph(h):
        g = nx.Graph()
        g.add_nodes_from(range(m), kind="data")
        g.add_nodes_from(range(m, 2*m), kind="check")
        for i, j in zip(*np.nonzero(h)):
            g.add_edge(m + int(i), int(j))
        return g
    match = nx.algorithms.isomorphism.GraphMatcher(
        graph(h_ref), graph(patch.simplex_parity_check),
        node_match=lambda a, b: a["kind"] == b["kind"])
    if not match.is_isomorphic():
        raise ValueError(f"r={r}: reference and default gauge graphs are not isomorphic")
    perm = [match.mapping[i] for i in range(m)]
    mapping = {row*m+col: perm[row]*m+perm[col] for row in range(m) for col in range(m)}
    declared = {(g["type"], frozenset(g["pauli"])): g["syn_idx"] for g in patch.gauges}
    for kind, supports in [("Z", z_support), ("X", x_support)]:
        for anc, support in supports.items():
            mapped = frozenset(mapping[q] for q in support)
            mapping[anc] = declared[(kind, mapped)]
    assert sorted(mapping.values()) == list(range(3*n))
    return patch, mapping, blocks, h_ref


def remap(circuit, mapping):
    out = stim.Circuit()
    for op in circuit.flattened():
        ts = [mapping[t.value] if t.is_qubit_target else t for t in op.targets_copy()]
        out.append(op.name, ts, op.gate_args_copy())
    return out


def regenerate_reference(r, p):
    """Run the authors' physical schedule through native LightStim Builder.

    Reference DETECTOR and OBSERVABLE_INCLUDE instructions never enter Tracker.
    Qubit remapping preserves the public SHYPSCode declaration. Initialization
    and final readout order match the reference measurement record exactly.
    """
    source = reference_circuit(r, p)
    patch, mapping, blocks, h_ref = shyps_reference_mapping(r, source)
    system, builder = initialize(patch)
    builder.circuit.append("TICK")
    n = patch.num_data_qubits
    mapped_blocks = [remap(block, mapping) for block in blocks]
    assert all(block == mapped_blocks[i % 2] for i, block in enumerate(mapped_blocks))
    pair = mapped_blocks[:2]
    builder.apply_syndrome_extraction(pair[0] + pair[1], rounds=len(mapped_blocks)//2,
                                      measurement_blocks=pair)
    builder.apply_data_readout({mapping[q]: "Z" for q in range(n)})
    # The initialization R order differs only by commuting resets. Match it for
    # physical-stream equality; it changes neither measurement order nor state.
    clean = stim.Circuit()
    initialized = False
    for op in builder.circuit.flattened():
        if op.name == "R" and not initialized:
            clean.append("R", [mapping[q] for q in range(n)])
            initialized = True
        else:
            clean.append(op)
    full = reference_noise(clean, p, n)
    mapped_source = remap(source, mapping)
    if physical(mapped_source) != physical(full):
        physical(mapped_source).to_file(RESULTS / "assets/debug_reference.stim")
        physical(full).to_file(RESULTS / "assets/debug_regenerated.stim")
        raise AssertionError("Physical/noise sequence differs; see debug assets")
    return full, mapped_source, {"r": r, "data_qubits": n,
         "physical_round_pairs": len(blocks)//2, "reference_to_lightstim_qubit_map": mapping,
         "reference_H": h_ref.tolist(), "gauge_graph_isomorphic": True,
         "physical_and_noise_stream_equal": True}
