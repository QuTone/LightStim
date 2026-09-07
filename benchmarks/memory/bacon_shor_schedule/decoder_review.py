"""Checkpointed CPU decoder comparison on the dedicated LightStim SE circuit.

The default memory profile is MWPM on matching-basis detectors. For comparison,
BP+OSD consumes exactly the same samples. Full-XZ BP+OSD is a separate profile.
No circuit or detector is handwritten; the projection selects original rows.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lightstim-decoder-review")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import stim
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.bacon_shor import BaconShorCode
from lightstim.simulation.decoder_backend import get_decoder
from benchmarks.memory.bacon_shor_schedule.run import select_basis_detectors
from benchmarks.memory.subsystem_crosscheck.common import interval, reference_noise, save_json, sha256

OUT = Path(__file__).resolve().parent / "results/2026-09-06-decoder-review"
PARAMS = dict(max_iterations=100, osd_order=10, osd_method="osd_cs",
              bp_method="min_sum", ms_scaling_factor=0)


def job_id(c):
    return f"d{c['d']}_p{c['p']:.7g}_{c['profile']}"


def build(d, p):
    ideal = MemoryExperiment(qec_patch=BaconShorCode(distance=d), basis="Z", rounds=d).build()
    assert not ideal.compile_detector_sampler(seed=735).sample(256, append_observables=True).any()
    return reference_noise(ideal, p, d*d)


def detector_rows(circuit):
    records, keep, index = [], [], 0
    for op in circuit.flattened():
        if op.name == "DETECTOR":
            if all(records[len(records) + t.value] == "Z" for t in op.targets_copy()):
                keep.append(index)
            index += 1
        else:
            single = stim.Circuit()
            single.append(op)
            if single.num_measurements:
                assert op.name in {"M", "MX"}
                records.extend(["X" if op.name == "MX" else "Z"] * single.num_measurements)
    return np.asarray(keep)


def run_job(c):
    jid = job_id(c)
    path = OUT / (jid + ".json")
    state = json.loads(path.read_text()) if path.exists() else dict(
        config=c, shots=0, counts={}, batches=[], elapsed_seconds=0.)
    state["config"] = c
    start = time.perf_counter()
    full = build(c['d'], c['p'])
    projected = select_basis_detectors(full, "Z")
    keep = detector_rows(full)
    dem = (projected if c['profile'] == 'z_pair' else full).detector_error_model()
    histogram = {}
    for op in dem.flattened():
        if op.type == "error":
            w = sum(t.is_relative_detector_id() for t in op.targets_copy())
            histogram[w] = histogram.get(w, 0) + 1
    if c['profile'] == 'z_pair':
        assert max(histogram) <= 2
        configs = dict(mwpm_z=("pymatching", {}), bposd_z_serial=("bposd", dict(PARAMS, schedule="serial")))
    else:
        configs = {f"bposd_full_{c['profile']}": ("bposd", dict(PARAMS, schedule=c['profile']))}
    decoders = {name: get_decoder(decoder, **params).compile_decoder_for_dem(dem=dem)
                for name, (decoder, params) in configs.items()}
    for name in decoders:
        state['counts'].setdefault(name, 0)
    state.update(decoder_configs=configs, backend="cpu", detector_weight_histogram=histogram,
                 num_detectors=dem.num_detectors, num_errors=dem.num_errors, status="running")
    full.to_file(OUT / (jid + '.stim'))
    dem.to_file(OUT / (jid + '.dem'))
    state['circuit_sha256'] = sha256(OUT / (jid + '.stim'))
    save_json(path, state)
    base_seed = int.from_bytes(hashlib.sha256(f"dedicated_review_d{c['d']}_p{c['p']}".encode()).digest()[:6], "little")
    while (min(state['counts'].values()) < c['target_errors'] and state['shots'] < c['max_shots']
           and state['elapsed_seconds'] + time.perf_counter() - start < c['max_seconds']):
        batch = len(state['batches'])
        seed = base_seed + batch
        n = min(1000, c['max_shots'] - state['shots'])
        dets, obs = full.compile_detector_sampler(seed=seed).sample(n, separate_observables=True, bit_packed=True)
        ds = np.packbits(np.unpackbits(dets, axis=1, bitorder='little')[:, keep], axis=1, bitorder='little') if c['profile'] == 'z_pair' else dets
        predictions, failed, seconds = {}, {}, {}
        for name, decoder in decoders.items():
            tick = time.perf_counter()
            predictions[name] = decoder.decode_shots_bit_packed(bit_packed_detection_event_data=ds)
            seconds[name] = time.perf_counter() - tick
            failed[name] = ((predictions[name] ^ obs) & 1).any(axis=1)
            state['counts'][name] += int(failed[name].sum())
        if batch == 0:
            np.savez_compressed(OUT / (jid + '_first_batch.npz'), full_detectors=dets,
                                selected_detector_indices=keep, decoder_detectors=ds, observables=obs,
                                seed=seed, **predictions)
        counts = {name: int(f.sum()) for name, f in failed.items()}
        paired = {}
        if c['profile'] == 'z_pair':
            a, b = failed['mwpm_z'], failed['bposd_z_serial']
            paired = dict(both=int((a & b).sum()), mwpm_only=int((a & ~b).sum()), bposd_only=int((b & ~a).sum()))
        state['batches'].append(dict(seed=seed, shots=n, counts=counts, decode_seconds=seconds, paired=paired))
        state['shots'] += n
        state['results'] = {name: dict(errors=k, ler=k/state['shots'], ci95=interval(k, state['shots']))
                            for name, k in state['counts'].items()}
        save_json(path, {**state, 'elapsed_seconds': state['elapsed_seconds'] + time.perf_counter() - start})
    state['elapsed_seconds'] += time.perf_counter() - start
    state['status'] = ('error_target' if min(state['counts'].values()) >= c['target_errors'] else
                       'shot_cap' if state['shots'] >= c['max_shots'] else 'time_cap')
    save_json(path, state)
    print(jid, state['status'], state['shots'], state['counts'], flush=True)


def launch(c):
    jid = job_id(c)
    cfg = OUT / (jid + '_config.json')
    save_json(cfg, c)
    with (OUT / (jid + '.log')).open('a') as log:
        try:
            result = subprocess.run([sys.executable, __file__, '--job', str(cfg)], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=c['max_seconds'] + 90)
            print(jid, 'exit', result.returncode, flush=True)
            if result.returncode:
                raise RuntimeError(f'Worker exited {result.returncode}; see {jid}.log')
        except subprocess.TimeoutExpired:
            path = OUT / (jid + '.json')
            state = json.loads(path.read_text()) if path.exists() else dict(config=c)
            state['status'] = 'hard_timeout'
            save_json(path, state)
            print(jid, 'hard_timeout', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-seconds', type=int, default=180)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.job:
        run_job(json.loads(args.job.read_text()))
    else:
        assert 1 <= args.workers <= 48
        source_names = ['lightstim/qec_code/bacon_shor/SE_block.py', 'lightstim/qec_code/bacon_shor/code_patch.py',
                        'lightstim/simulation/decoder_backend/decoders/bposd.py', str(Path(__file__).relative_to(ROOT))]
        save_json(OUT / 'manifest.json', dict(
            source_sha256={s: sha256(ROOT/s) for s in source_names},
            versions={p: importlib.metadata.version(p) for p in ['stim', 'stimbposd', 'ldpc', 'pymatching', 'numpy']},
            default_memory_decoder='pymatching, Z-record detectors for Z memory',
            noise='Reset and ancilla readout flips p; CX DEPOLARIZE2(p); no idle noise; ideal final data readout',
            metric='Logical Z memory failure probability per complete shot with d XZ pairs',
            sampling='Same full-circuit samples across decoders; seeds exclude decoder and projection. 1000 shots/batch.'))
        jobs = [dict(d=d, p=p, profile='z_pair') for d in [3, 5, 7, 9] for p in [.0005, .001, .002]]
        jobs += [dict(d=d, p=.001, profile=profile) for d in [3, 5, 7] for profile in ['parallel', 'serial']]
        for c in jobs:
            c.update(target_errors=100, max_shots=2_000_000, max_seconds=args.max_seconds)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch, jobs))
