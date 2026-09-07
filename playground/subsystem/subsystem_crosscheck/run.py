"""Checkpointed per-configuration sampling using LightStim's decoder registry.

    PYTHONPATH=. python playground/subsystem/subsystem_crosscheck/run.py --suite initial --workers 8

No circuit/detector is handwritten here. Stim supplies samples; LightStim supplies
decoders and all annotations on generated circuits. Individual batches have fixed
seeds, making interruption/resumption reproducible. Batches are shared across
decoder configurations on a given circuit (seed excludes decoder name).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lightstim-crosscheck-mpl")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import stim
from common import (RESULTS, ROOT, bacon_capacity, bs_theory, generic_memory,
                    interval, reference_circuit, save_json, sha256)
from lightstim.simulation.decoder_backend import get_decoder


def circuit_id(c):
    return f"{c['experiment']}_s{c['size']}_p{c['p']:.7g}"


def job_id(c):
    return f"{circuit_id(c)}_{c['decoder']}_{c.get('variant', 'default')}"


def configuration(experiment, size, p, decoder="bposd", **kwargs):
    params = dict(max_iterations=100, osd_order=10, osd_method="osd_cs",
                  bp_method="min_sum", ms_scaling_factor=0)
    if decoder == "mwpf":
        params = dict(cluster_node_limit=50, timeout=1.)
    elif decoder == "mle-ilp":
        params = dict(time_limit=.2, max_cut_rounds=25, max_rpc_rounds=0)
    return dict(experiment=experiment, size=size, p=p, decoder=decoder,
                decoder_params=params, target_errors=200, max_shots=1_000_000,
                batch_size=1000, max_seconds=600, **kwargs)


def make_circuit(c):
    label, size, p = c["experiment"], c["size"], c["p"]
    if label == "bacon_capacity":
        return bacon_capacity(size, p)
    if label == "bacon_circuit":
        return generic_memory("bacon_shor", size, p)
    if label == "shyps_reference_z":
        return reference_circuit(size, p, "Z")
    if label == "shyps_reference_xz":
        return reference_circuit(size, p, "X_and_Z")
    asset = RESULTS / "assets/circuits" / (circuit_id(c) + ".stim")
    if asset.exists():
        return stim.Circuit.from_file(asset)
    raise ValueError(label)


def run_job(c):
    jid = job_id(c)
    result_path = RESULTS / "jobs" / (jid + ".json")
    assets = RESULTS / "assets"
    (assets / "circuits").mkdir(parents=True, exist_ok=True)
    (assets / "samples").mkdir(parents=True, exist_ok=True)
    state = json.loads(result_path.read_text()) if result_path.exists() else dict(
        config=c, job_id=jid, shots=0, errors=0, decode_failures=0, next_batch=0,
        elapsed_seconds=0., batches=[])
    # Explicit reruns may increase budgets; completed data and seeds are retained.
    state["config"] = c
    start = time.perf_counter()
    try:
        circuit = make_circuit(c)
        dem = circuit.detector_error_model()
        circuit_path = assets / "circuits" / (circuit_id(c) + ".stim")
        circuit.to_file(circuit_path)
        dem.to_file(circuit_path.with_suffix(".dem"))
        ideal = circuit.without_noise().compile_detector_sampler(seed=20260906).sample(
            256, append_observables=True)
        assert not ideal.any(), "Nonzero ideal detector/observable sample"
        decoder_params = dict(c["decoder_params"])
        if c["decoder"] == "mwpf":
            from mwpf.sinter_decoders import PanicAction
            decoder_params["panic_action"] = PanicAction.RAISE
        compiled = get_decoder(c["decoder"], **decoder_params).compile_decoder_for_dem(dem=dem)
        state.update(num_qubits=circuit.num_qubits, num_detectors=circuit.num_detectors,
                     num_observables=circuit.num_observables, num_error_mechanisms=dem.num_errors,
                     circuit_sha256=sha256(circuit_path), status="running")
        if c["experiment"] == "bacon_capacity":
            state["theory_one_sector"] = bs_theory(c["size"], c["p"])
        base_seed = (int.from_bytes(hashlib.sha256(circuit_id(c).encode()).digest()[:6], "little")
                     + int(c.get("seed_offset", 0)) * 10_000_000)
        save_json(result_path, state)
        while (state["errors"] < c["target_errors"] and state["shots"] < c["max_shots"]
               and state["elapsed_seconds"] + time.perf_counter() - start < c["max_seconds"]):
            n = min(c["batch_size"], c["max_shots"] - state["shots"])
            seed = base_seed + state["next_batch"]
            sampler = circuit.compile_detector_sampler(seed=seed)
            dets, obs = sampler.sample(n, separate_observables=True, bit_packed=True)
            t = time.perf_counter()
            pred = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
            residual = np.unpackbits(pred ^ obs, axis=1, bitorder="little")[:, :circuit.num_observables]
            failed = residual.any(axis=1)
            flags = getattr(compiled, "last_flags", None)
            timeout = np.zeros(n, dtype=bool) if flags is None else ~np.asarray(flags, dtype=bool)
            failed |= timeout  # conservative fail-closed; no postselection
            count = int(failed.sum())
            if state["next_batch"] == 0:
                np.savez_compressed(assets / "samples" / (jid + ".npz"),
                                    detectors=dets, observables=obs, predictions=pred,
                                    decode_failures=timeout, seed=seed,
                                    num_detectors=circuit.num_detectors,
                                    num_observables=circuit.num_observables)
            state["shots"] += n
            state["errors"] += count
            state["decode_failures"] += int(timeout.sum())
            per_obs = residual.sum(axis=0).astype(int).tolist()
            state["batches"].append(dict(seed=seed, shots=n, errors=count,
                                        per_observable_errors=per_obs,
                                        decode_failures=int(timeout.sum()),
                                        decode_seconds=time.perf_counter()-t))
            state["next_batch"] += 1
            state["ler"] = state["errors"] / state["shots"]
            state["ci95"] = interval(state["errors"], state["shots"])
            snapshot = {**state, "elapsed_seconds": state["elapsed_seconds"] + time.perf_counter()-start}
            save_json(result_path, snapshot)
        state["status"] = ("error_target" if state["errors"] >= c["target_errors"] else
                           "shot_cap" if state["shots"] >= c["max_shots"] else "time_cap")
    except Exception:
        state.update(status="exception", exception=traceback.format_exc())
    state["elapsed_seconds"] += time.perf_counter()-start
    save_json(result_path, state)
    print(jid, state["status"], state["errors"], "/", state["shots"], flush=True)
    if state["status"] == "exception":
        print(state["exception"], flush=True)


def suite(name):
    jobs = []
    if name == "initial":
        for d in (3, 5, 7, 9):
            for p in (.005, .01, .02, .04):
                jobs.append(configuration("bacon_capacity", d, p))
        for d in (3, 5, 7):
            for p in (.0005, .001, .002, .004):
                jobs.append(configuration("bacon_circuit", d, p))
        for r, ps in [(3, (.0005, .001, .003, .005)), (4, (.0005, .001, .002, .003))]:
            for p in ps:
                jobs.append(configuration("shyps_reference_z", r, p))
    elif name == "alternatives":
        for experiment, size, ps in [("bacon_capacity", 3, (.01, .04)),
                                      ("bacon_circuit", 3, (.001, .004)),
                                      ("shyps_reference_z", 3, (.001, .005))]:
            for p in ps:
                for decoder in ("mwpf", "mle-ilp"):
                    c = configuration(experiment, size, p, decoder)
                    c.update(max_seconds=300, max_shots=100_000 if decoder == "mwpf" else 5000,
                             batch_size=100 if decoder == "mle-ilp" else 1000)
                    jobs.append(c)
    else:
        jobs = json.loads(Path(name).read_text())
    return jobs


def launch(c):
    jid = job_id(c)
    path = RESULTS / "configs" / (jid + ".json")
    save_json(path, c)
    log = RESULTS / "logs" / (jid + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        try:
            result = subprocess.run([sys.executable, __file__, "--job", str(path)],
                                    cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=c["max_seconds"] + 180)
            print(jid, "exit", result.returncode, flush=True)
        except subprocess.TimeoutExpired:
            # A native decoder may overrun its soft timeout; preserve last batch.
            f.write("Supervisor hard timeout; last checkpoint remains available.\n")
            p = RESULTS / "jobs" / (jid + ".json")
            state = json.loads(p.read_text()) if p.exists() else dict(config=c, job_id=jid)
            state["status"] = "hard_timeout"
            save_json(p, state)
            print(jid, "HARD TIMEOUT", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="initial")
    parser.add_argument("--job")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.job:
        run_job(json.loads(Path(args.job).read_text()))
    else:
        assert 1 <= args.workers <= 48
        configs = suite(args.suite)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch, configs))
