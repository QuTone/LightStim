"""
LS 7-to-1 distillation — full circuit-level noise sweep.

Builds circuits on first run, caches to circuits/LS_7to1_d{d}_rnd{rounds}.stim
+ _obs.json, then loads from cache on subsequent runs.

Saves results to LS_full_noise_results.csv with append-on-complete
checkpointing (one row written per task completion).

CSV schema matches benchmarks/distillation_results.ipynb (run_distillation_simulations.py):
    KEY : d, rounds, p, p_injected
    DATA: p_in, ler_ps, post_selection_rate, shots, errors

Usage (from repo root):
    venv/bin/python benchmarks/logical_circuits/distillation/ls_7to1/run_ls_full_sweep.py
    venv/bin/python benchmarks/logical_circuits/distillation/ls_7to1/run_ls_full_sweep.py \\
        -d 3 5 7 --p-values 1e-5 6e-5 4e-4 3e-3 2e-2 1e-1 --num-workers 8
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import stim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..')))

from eval.logical_circuit_benchmark.distillation.ls_7to1.LS_distillation_7_to_1 import (
    build_distillation_circuit as build_ls,
)
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector
from lightstim.simulation.decoder_backend.config import DecoderConfig
from lightstim.simulation.decoder_backend.pipeline import SimulationPipeline
from lightstim.simulation.observable_analysis import (
    build_obs_patch_matrix,
    identify_distillation_observables,
)

KEY_COLS  = ['d', 'rounds', 'p', 'p_injected']
DATA_COLS = ['p_in', 'ler_ps', 'post_selection_rate', 'shots', 'errors']
ALL_COLS  = KEY_COLS + DATA_COLS


def _circ_dir():
    d = os.path.join(SCRIPT_DIR, 'circuits')
    os.makedirs(d, exist_ok=True)
    return d


def build_and_cache(d: int, rounds: int):
    """Build LS circuit, run observable analysis, save .stim + _obs.json."""
    circ_dir  = _circ_dir()
    stim_path = os.path.join(circ_dir, f'LS_7to1_d{d}_rnd{rounds}.stim')
    obs_path  = os.path.join(circ_dir, f'LS_7to1_d{d}_rnd{rounds}_obs.json')

    if os.path.exists(stim_path) and os.path.exists(obs_path):
        print(f'  Loading cached circuit: {os.path.basename(stim_path)}')
        circuit = stim.Circuit.from_file(stim_path)
        with open(obs_path) as f:
            meta = json.load(f)
        return circuit, meta['post_select_obs'], meta['target_obs']

    print(f'  Building LS d={d} rounds={rounds} (first time — will be cached)...')
    t0 = time.perf_counter()
    circuit, _, system = build_ls(d=d, rounds=rounds)
    t_build = time.perf_counter() - t0
    print(f'  Built in {t_build:.1f}s — '
          f'qubits={circuit.num_qubits}  det={circuit.num_detectors}')

    matrix, pn = build_obs_patch_matrix(circuit, system)
    _, tgt, ps = identify_distillation_observables(matrix, pn, ['W4'])
    ps  = list(ps)
    tgt = list(tgt)

    circuit.to_file(stim_path)
    with open(obs_path, 'w') as f:
        json.dump({'post_select_obs': ps, 'target_obs': tgt,
                   'd': d, 'rounds': rounds}, f, indent=2)

    print(f'  Cached → {stim_path}')
    print(f'  post_select_obs={ps}  target_obs={tgt}')
    return circuit, ps, tgt


def make_noisy(circuit: stim.Circuit, p: float) -> stim.Circuit:
    all_qubits = list(range(circuit.num_qubits))
    cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p)
    return NoiseInjector.from_circuit_level(cfg, all_qubits).inject_noise(circuit)


def run_one(circuit, ps_obs, target_obs, p, decoder, max_shots, max_errors,
            batch_size, num_workers):
    noisy = make_noisy(circuit, p)
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig(decoder),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        post_select_corrected_observable_indices=ps_obs,
        target_observable_indices=target_obs,
        print_progress=True,
        progress_interval_sec=30.0,
        num_workers=num_workers,
    )
    return pipeline.run(noisy)


def load_done_keys(csv_path):
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                done.add((int(row['d']), int(row['rounds']),
                          float(row['p']), float(row['p_injected'])))
            except (KeyError, ValueError):
                pass
    return done


def append_row(csv_path, row: dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ALL_COLS)
        if write_header:
            w.writeheader()
        w.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--distances', type=int, nargs='+', default=[3, 5, 7])
    parser.add_argument('--decoder', choices=['pymatching', 'bposd', 'mwpf'],
                        default='pymatching')
    parser.add_argument('--rounds', type=int, default=None,
                        help='SE rounds per cycle (default: d for each distance)')
    parser.add_argument('--p-values', type=float, nargs='+',
                        default=np.logspace(-5, -1, 6).tolist(),
                        help='Circuit-level p values (default: logspace(-5,-1,6))')
    parser.add_argument('--max-shots',   type=int, default=100_000_000)
    parser.add_argument('--max-errors',  type=int, default=100)
    parser.add_argument('--batch-size',  type=int, default=5_000)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    # CSV goes in SCRIPT_DIR so the notebook (distillation_results.ipynb) can find it:
    #   LS_DIR / 'LS_full_noise_results.csv'
    csv_path = os.path.join(SCRIPT_DIR, 'LS_full_noise_results.csv')

    done_keys = load_done_keys(csv_path)
    if done_keys:
        print(f'Checkpoint: {len(done_keys)} task(s) already done, skipping.')

    for d in args.distances:
        rounds = args.rounds if args.rounds is not None else d
        print(f'\n{"="*60}')
        print(f'LS d={d}  rounds={rounds}')
        print(f'{"="*60}')

        circuit, ps_obs, target_obs = build_and_cache(d, rounds)
        print(f'  qubits={circuit.num_qubits}  det={circuit.num_detectors}  '
              f'obs={circuit.num_observables}')

        # Noiseless sanity check
        dets, obs = circuit.compile_detector_sampler().sample(
            200, separate_observables=True)
        ok = not np.any(dets) and not np.any(obs)
        print(f'  Noiseless check: {"OK" if ok else "FAIL"}')

        for p in args.p_values:
            key = (d, rounds, p, 0.0)
            if key in done_keys:
                print(f'  d={d} p={p:.1e} — SKIPPED (checkpoint)')
                continue

            print(f'\n  d={d} rounds={rounds} p={p:.1e}  '
                  f'(max_shots={args.max_shots:.0e}  max_errors={args.max_errors})')
            t0 = time.perf_counter()
            stats = run_one(circuit, ps_obs, target_obs, p, args.decoder,
                            args.max_shots, args.max_errors,
                            args.batch_size, args.num_workers)
            elapsed = time.perf_counter() - t0

            row = {
                'd':                   d,
                'rounds':              rounds,
                'p':                   p,
                'p_injected':          0.0,
                'p_in':                0.0,
                'ler_ps':              stats.logical_error_rate,
                'post_selection_rate': stats.post_selection_rate,
                'shots':               stats.shots,
                'errors':              stats.errors,
            }
            append_row(csv_path, row)
            done_keys.add(key)

            print(f'  shots={stats.shots:,}  errors={stats.errors}  '
                  f'LER={stats.logical_error_rate:.3e}  '
                  f'accept={stats.post_selection_rate:.3f}  '
                  f'time={elapsed:.1f}s')

    print(f'\nDone. Results → {csv_path}')
