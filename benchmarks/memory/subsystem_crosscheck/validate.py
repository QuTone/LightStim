"""Independent algebraic and exact-probability checks behind the benchmark."""
import argparse
from collections import Counter
import json
import time

import numpy as np
import stim

from common import *
from lightstim.simulation.decoder_backend import get_decoder, SimulationPipeline, DecoderConfig
from lightstim.simulation.decoder_backend.dem_matrices import dem_to_matrices


def exact_capacity(d, p):
    c = bacon_capacity(d, p)
    h, l, probabilities = dem_to_matrices(c.detector_error_model())
    assert l.shape[0] == 1
    assert h.shape[1] <= 20, "Enumeration is restricted to these small circuits"
    # Select independent original detector rows; no syndrome information is lost.
    selected, pivots = [], {}
    for i, row in enumerate(h):
        value = sum(int(b) << j for j, b in enumerate(row))
        while value:
            pivot = value.bit_length()-1
            if pivot not in pivots:
                selected.append(i)
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
    hr = h[selected]
    joint = np.vstack([hr, l])
    patterns = np.array([0], dtype=np.int64)
    weights = np.array([1.])
    for j, p_fault in enumerate(probabilities):
        flip = sum(int(b) << i for i, b in enumerate(joint[:, j]))
        patterns = np.concatenate([patterns, patterns ^ flip])
        weights = np.concatenate([weights*(1-p_fault), weights*p_fault])
    distribution = np.bincount(patterns, weights=weights, minlength=2**len(joint))
    half = 2**len(selected)
    p0, p1 = distribution[:half], distribution[half:]
    optimal = float(np.minimum(p0, p1).sum())  # DEGNERATE maximum-likelihood class
    theory = bs_theory(d, p)
    assert np.isclose(optimal, theory, rtol=1e-10, atol=1e-16), (d,p,optimal,theory)
    return dict(distance=d, p=p, enumerated_fault_patterns=len(patterns),
                detector_rank=len(selected), dem_error_mechanisms=len(probabilities),
                exact_degenerate_ml_ler=optimal, napp_preskill_ler=theory,
                absolute_difference=abs(optimal-theory))


def detector_details(c):
    records, round_idx, measurement_rounds = [], -1, []
    details = []
    for op in c.flattened():
        if op.name == "R" and op.targets_copy()[0].value >= c.num_qubits//3:
            round_idx += 1
        if op.name in {"M", "MX"}:
            for t in op.targets_copy():
                kind = "data" if t.value < c.num_qubits//3 else ("Z" if op.name=="M" else "X")
                records.append(kind)
                measurement_rounds.append(round_idx if kind != "data" else round_idx+1)
        elif op.name == "DETECTOR":
            indices = [len(records)+t.value for t in op.targets_copy()]
            kinds = sorted(set(records[i] for i in indices))
            rounds = sorted(set(measurement_rounds[i] for i in indices))
            details.append(dict(kinds=kinds, rounds=rounds, weight=len(indices)))
    return details


def verify_flows(c):
    clean = c.without_noise()
    ds, obs = annotation_rows(c)
    count = c.num_measurements
    flows = []
    for row in ds+obs:
        sign = "-" if row >> count else "+"
        ms = [i for i in range(count) if row >> i & 1]
        flows.append(stim.Flow(output=stim.PauliString(sign), measurements=ms))
    assert clean.has_all_flows(flows), "An annotation is not a signed deterministic flow"
    return len(flows)


def validate_reference(r):
    t = time.perf_counter()
    full, reference, meta = regenerate_reference(r, .001)
    auto_z = z_detectors(full)
    a, oa = annotation_rows(reference)
    b, ob = annotation_rows(auto_z)
    meta.update(reference_z_detectors=reference.num_detectors,
                auto_z_detectors=auto_z.num_detectors, auto_xz_detectors=full.num_detectors,
                z_detector_space=compare_spaces(a,b),
                logical_space_modulo_auto_detectors=compare_spaces(b+oa,b+ob))
    assert meta["z_detector_space"]["a_subset_b"]
    assert meta["logical_space_modulo_auto_detectors"]["equal"]
    for name, circuit in [("auto_z", auto_z), ("auto_xz", full), ("reference_mapped", reference)]:
        meta[name] = dict(signed_flows_verified=verify_flows(circuit),
                          num_error_mechanisms=circuit.detector_error_model().num_errors,
                          detector_structure=detector_details(circuit))
        circuit.to_file(RESULTS / "assets" / f"shyps_r{r}_{name}.stim")
    # Compare against the authors' full X+Z annotation set as well.
    try:
        source_xz = reference_circuit(r,.001,"X_and_Z")
        mapped_xz = remap(source_xz, {int(k):v for k,v in meta["reference_to_lightstim_qubit_map"].items()})
        meta["reference_xz_physical_equal_to_z_file"] = physical(mapped_xz) == physical(reference)
        if meta["reference_xz_physical_equal_to_z_file"]:
            dx, ox = annotation_rows(mapped_xz)
            df, of = annotation_rows(full)
            meta["xz_detector_space"] = compare_spaces(dx,df)
            meta["xz_logical_space_modulo_auto_detectors"] = compare_spaces(df+ox,df+of)
    except ValueError as exc:
        meta["reference_xz_unavailable"] = str(exc)
    meta["elapsed_seconds"] = time.perf_counter()-t
    save_json(RESULTS / f"shyps_r{r}_equivalence.json", meta)
    print('reference',r,'Z space',meta['z_detector_space'],'seconds',meta['elapsed_seconds'],flush=True)
    return meta


def single_fault_check(circuit, decoder="bposd", params=None):
    h, l, ps = dem_to_matrices(circuit.detector_error_model())
    compiled = get_decoder(decoder, **(params or dict(max_iterations=100))).compile_decoder_for_dem(dem=circuit.detector_error_model())
    packed = np.packbits(h.T,axis=1,bitorder="little")
    pred = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
    actual = np.packbits(l.T,axis=1,bitorder="little")
    failures = np.flatnonzero(np.any(pred ^ actual,axis=1))
    return dict(decoder=decoder, params=params, num_mechanisms=len(ps),
                failed_single_mechanisms=failures.tolist(),
                sum_failed_mechanism_priors=float(ps[failures].sum()))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--part",choices=["capacity","shyps3","shyps4","single_faults"],required=True)
    args=parser.parse_args()
    if args.part == "capacity":
        results=[exact_capacity(d,p) for d in (3,5,7,9) for p in (.005,.01,.02,.04)]
        published=2.638e-28
        anchor=dict(p=.001,distance=173,published_ler=published,
                    recomputed_ler=bs_theory(173,.001),
                    note="Analytic evaluation, NOT Monte Carlo or a d=173 LightStim build")
        save_json(RESULTS/"bacon_capacity_exact_validation.json",dict(points=results,published_anchor=anchor))
        # Check the standard public orchestration path on a generated circuit.
        stats=SimulationPipeline(DecoderConfig("bposd",params={"max_iterations":100}),
            max_errors=100,max_shots=20000,batch_size=1000,num_workers=1,
            print_progress=False).run(bacon_capacity(3,.04))
        save_json(RESULTS/"public_pipeline_smoke.json",dict(shots=stats.shots,
            ler=stats.logical_error_rate,theory=bs_theory(3,.04),
            note="Independent pipeline smoke, not added to sweep counts"))
        print('Exact validation passed',len(results),'anchor',anchor,flush=True)
    elif args.part.startswith("shyps"):
        validate_reference(int(args.part[-1]))
    else:
        result=[]
        for p in (.0001,.0005,.001,.002):
            c=generic_memory("bacon_shor",3,p)
            for decoder,params in [("bposd",dict(max_iterations=100)),
                                   ("bposd",dict(max_iterations=1000)),
                                   ("mwpf",dict(cluster_node_limit=50,timeout=1.))]:
                result.append(dict(p=p,**single_fault_check(c,decoder,params)))
        save_json(RESULTS/"bacon_circuit_single_fault_decoder_audit.json",result)
        print(result,flush=True)
