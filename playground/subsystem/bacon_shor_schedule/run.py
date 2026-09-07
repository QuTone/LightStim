"""Compare dedicated and generic Bacon-Shor extraction via LightStim.

Saves circuits, signed-flow/detector checks, circuit-distance bounds and an
optional fixed-shot Z-memory MWPM comparison. No detector is handwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lightstim-dedicated-mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import stim

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.bacon_shor import BaconShorCode, BaconShorCodeExtractionBlock
from lightstim.qec_code.generic_css import GenericCSSGaugeExtractionBlock
from lightstim.simulation.decoder_backend import get_decoder
from playground.subsystem.subsystem_crosscheck.common import (
    annotation_rows, compare_spaces, interval, reference_noise, save_json, sha256,
)

BLOCKS = {"dedicated": BaconShorCodeExtractionBlock, "generic": GenericCSSGaugeExtractionBlock}


def select_basis_detectors(circuit, basis):
    out = stim.Circuit()
    record_bases = []
    for op in circuit.flattened():
        if op.name == "DETECTOR":
            if all(record_bases[len(record_bases) + t.value] == basis for t in op.targets_copy()):
                out.append(op)
        else:
            out.append(op)
            single = stim.Circuit()
            single.append(op)
            if single.num_measurements:
                assert op.name in {"M", "MX"}
                record_bases.extend(["X" if op.name == "MX" else "Z"] * single.num_measurements)
    return out


def add_readout_noise(circuit, n_data, p):
    out = stim.Circuit()
    for op in circuit.flattened():
        if op.name in {"M", "MX"} and all(t.value < n_data for t in op.targets_copy()):
            out.append(op.name, op.targets_copy(), p)
        else:
            out.append(op)
    return out


def distance_bounds(circuit, basis, d, stem, out):
    full = circuit.detector_error_model(decompose_errors=False).flattened()
    projection = select_basis_detectors(circuit, basis).detector_error_model(decompose_errors=False).flattened()
    # The complete projection must be graphlike; never split or discard faults
    # when calculating the lower bound. The full model may have hyperedges.
    for op in projection:
        if op.type == "error":
            assert not any(t.is_separator() for t in op.targets_copy())
            assert sum(t.is_relative_detector_id() for t in op.targets_copy()) <= 2
    lower = len(projection.shortest_graphlike_error(ignore_ungraphlike_errors=False))
    witness = full.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    upper = len(witness)
    detector_xor, logical_xor = set(), set()
    for op in witness:
        for t in op.targets_copy():
            assert t.is_relative_detector_id() or t.is_logical_observable_id()
            (detector_xor if t.is_relative_detector_id() else logical_xor).symmetric_difference_update({t.val})
    assert not detector_xor and logical_xor == {0}
    assert lower == upper == d
    explanations = circuit.explain_detector_error_model_errors(
        dem_filter=witness, reduce_to_one_representative_error=True)
    assert len(explanations) == d and all(e.circuit_error_locations for e in explanations)
    files = {}
    for extension, contents in [("stim", circuit), ("dem", full), ("projected.dem", projection),
                                ("witness.dem", witness), ("witness.txt", "\n\n".join(map(str, explanations)))]:
        path = out / f"{stem}.{extension}"
        path.write_text(str(contents) + "\n")
        files[path.name] = sha256(path)
    return dict(lower_bound=lower, upper_bound=upper, artifact_sha256=files)


def sample(circuit, d, label, shots, out):
    projected = select_basis_detectors(circuit, "Z")
    dem = projected.detector_error_model(decompose_errors=False)
    decoder = get_decoder("pymatching").compile_decoder_for_dem(dem=dem)
    batches = []
    for batch, offset in enumerate(range(0, shots, 10_000)):
        n = min(10_000, shots-offset)
        # Fresh independent runs, not paired physical-fault samples.
        seed = 2026090610000 + (0 if label == "dedicated" else 100000) + 1000*d + batch
        dets, obs = projected.compile_detector_sampler(seed=seed).sample(
            n, separate_observables=True, bit_packed=True)
        predictions = decoder.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
        errors = int(np.any((predictions ^ obs) & 1, axis=1).sum())
        batches.append(dict(seed=seed, shots=n, errors=errors))
        if batch == 0:
            np.savez_compressed(out / f"d{d}_{label}_first_batch.npz", detectors=dets,
                                observables=obs, predictions=predictions, seed=seed)
    errors = sum(b["errors"] for b in batches)
    ci = interval(errors, shots)
    result = dict(d=d, rounds=d, schedule=label, shots=shots, errors=errors, ler=errors/shots,
                  ci95=ci, batches=batches)
    save_json(out / f"d{d}_{label}_mwpm.json", result)
    return result


def run(out, shots):
    out.mkdir(parents=True, exist_ok=True)
    checks, distances, samples = [], [], []
    for d in [3, 5, 7, 9]:
        system = QECSystem()
        system.add_patch(BaconShorCode(distance=d), name="bs")
        blocks = {label: cls(system) for label, cls in BLOCKS.items()}
        left, right = [blocks[label].circuit for label in ["dedicated", "generic"]]
        same_instrument = (left.num_measurements == right.num_measurements
                           and all(right.has_flow(f, unsigned=False) for f in left.flow_generators())
                           and all(left.has_flow(f, unsigned=False) for f in right.flow_generators()))
        assert same_instrument
        for basis in ["Z", "X"]:
            clean = {label: MemoryExperiment(qec_patch=BaconShorCode(distance=d),
                                             extraction_block_class=cls, basis=basis, rounds=d).build()
                     for label, cls in BLOCKS.items()}
            ds1, ls1 = annotation_rows(clean["dedicated"])
            ds2, ls2 = annotation_rows(clean["generic"])
            comparison = compare_spaces(ds1, ds2)
            assert comparison["equal"]
            assert compare_spaces(ds1, ds1 + [ls1[0] ^ ls2[0]])["equal"]
            checks.append(dict(d=d, basis=basis, cnot_depths={k:v.cnot_depth for k,v in blocks.items()},
                               signed_instrument_equal=same_instrument,
                               literal_se_equal=left == right, detector_spaces=comparison,
                               logical_equivalent_mod_detectors=True))
            for label, ideal in clean.items():
                assert not ideal.compile_detector_sampler(seed=52).sample(256, append_observables=True).any()
                circuit = reference_noise(ideal, .001, d*d)
                for model, noisy in [("reference", circuit),
                                     ("readout_flips", add_readout_noise(circuit, d*d, .001))]:
                    stem = f"d{d}_{basis}_{label}_{model}"
                    bounds = distance_bounds(noisy, basis, d, stem, out)
                    distances.append(dict(d=d, basis=basis, schedule=label, model=model, **bounds))
                if basis == "Z" and shots:
                    result = sample(circuit, d, label, shots, out)
                    samples.append(result)
                    print(label, d, "MWPM", result["errors"], "/", shots, flush=True)
            print("d", d, basis, "instruments/detectors/logicals agree; both circuit distances =", d, flush=True)
            save_json(out / "checks.json", dict(checks=checks, distances=distances))

    sources = ["lightstim/qec_code/bacon_shor/SE_block.py", "lightstim/qec_code/bacon_shor/code_patch.py",
               "lightstim/qec_code/generic_css/gauge_SE_block.py"]
    summary = dict(date="2026-09-06", stim_version=stim.__version__,
                   source_sha256={s:sha256(ROOT/s) for s in sources},
                   noise="reset and ancilla measurement flips p=0.001; CX DEPOLARIZE2(p); no idle; final data readout ideal except readout_flips audit cases",
                   decoder="LightStim pymatching registry, only same-basis Z detectors; one logical observable per complete memory shot",
                   distance_method="Complete undecomposed graphlike detector projection gives lower bound; physical witness in full DEM gives upper bound",
                   checks=checks, distances=distances, mwpm=samples)
    save_json(out / "summary.json", summary)
    if samples:
        with (out / "mwpm.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=["schedule", "d", "rounds", "shots", "errors", "ler", "ci_low", "ci_high"])
            writer.writeheader()
            for result in samples:
                writer.writerow({**{k:result[k] for k in writer.fieldnames if k in result},
                                 "ci_low":result["ci95"][0], "ci_high":result["ci95"][1]})
    print("Saved", out / "summary.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/2026-09-06")
    parser.add_argument("--shots", type=int, default=1_000_000, help="Z-memory shots per size/schedule; 0 skips decoding")
    args = parser.parse_args()
    if args.shots < 0:
        parser.error("--shots must be nonnegative")
    run(args.output, args.shots)
