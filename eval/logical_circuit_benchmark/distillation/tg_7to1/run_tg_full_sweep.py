"""
TG 7-to-1 distillation — full circuit-level noise sweep.

Uses pre-built circuits from circuits/ (TG_7to1_d{d}_r1.stim + _obs.json).
No rebuild needed. Saves results to TG_full_noise_results.csv with
append-on-complete checkpointing (one row written per task completion).

CSV schema matches eval/distillation_results.ipynb (run_distillation_simulations.py):
    KEY : d, rounds, r, p, p_injected
    DATA: p_in, ler_ps, post_selection_rate, shots, errors

Usage (from repo root):
    venv/bin/python eval/logical_circuit_benchmark/distillation/tg_7to1/run_tg_full_sweep.py
    venv/bin/python eval/logical_circuit_benchmark/distillation/tg_7to1/run_tg_full_sweep.py \\
        -d 5 7 --p-values 1e-4 3e-4 1e-3 3e-3 1e-2 --num-workers 8
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

from src.noise.config import NoiseConfig
from src.noise.injector import NoiseInjector
from src.simulation.decoder_backend.config import DecoderConfig
from src.simulation.decoder_backend.pipeline import SimulationPipeline

# Rounds convention: rounds = d  (used when the circuits/ cache was built)
ROUNDS_FOR_D = {3: 3, 5: 5, 7: 7}
R = 1  # transversal-gate SE rounds (always 1 in the cached circuits)

KEY_COLS  = ['d', 'rounds', 'r', 'p', 'p_injected']
DATA_COLS = ['p_in', 'ler_ps', 'post_selection_rate', 'shots', 'errors']
ALL_COLS  = KEY_COLS + DATA_COLS


def load_circuit(d: int):
    """Load pre-built noiseless TG circuit and observable metadata from circuits/."""
    circ_dir = os.path.join(SCRIPT_DIR, 'circuits')
    stim_path = os.path.join(circ_dir, f'TG_7to1_d{d}_r{R}.stim')
    obs_path  = os.path.join(circ_dir, f'TG_7to1_d{d}_r{R}_obs.json')

    if not os.path.exists(stim_path):
        raise FileNotFoundError(
            f'Cached circuit not found: {stim_path}\n'
            f'Build it first with TG_distillation_7_to_1.py --build-only -d {d}')

    circuit = stim.Circuit.from_file(stim_path)
    with open(obs_path) as f:
        meta = json.load(f)

    ps_obs     = meta['post_select_obs']   # e.g. [1, 2, 3]
    target_obs = meta['target_obs']        # e.g. [0]
    return circuit, ps_obs, target_obs


def make_noisy(circuit: stim.Circuit, p: float) -> stim.Circuit:
    """Apply uniform circuit-level noise to all qubits."""
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
                done.add((int(row['d']), int(row['rounds']), int(row['r']),
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
    parser.add_argument('--p-values', type=float, nargs='+',
                        default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    parser.add_argument('--decoder', choices=['pymatching', 'bposd', 'mwpf'],
                        default='pymatching')
    parser.add_argument('--max-shots',   type=int, default=100_000_000)
    parser.add_argument('--max-errors',  type=int, default=100)
    parser.add_argument('--batch-size',  type=int, default=5_000)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    # CSV goes in SCRIPT_DIR so the notebook (distillation_results.ipynb) can find it:
    #   TG_DIR / 'TG_full_noise_results.csv'
    csv_path = os.path.join(SCRIPT_DIR, 'TG_full_noise_results.csv')

    done_keys = load_done_keys(csv_path)
    if done_keys:
        print(f'Checkpoint: {len(done_keys)} task(s) already done, skipping.')

    for d in args.distances:
        rounds = ROUNDS_FOR_D.get(d, d)
        print(f'\n{"="*60}')
        print(f'Loading TG d={d}  rounds={rounds}  r={R}')
        print(f'{"="*60}')

        circuit, ps_obs, target_obs = load_circuit(d)
        print(f'  qubits={circuit.num_qubits}  det={circuit.num_detectors}  '
              f'obs={circuit.num_observables}')
        print(f'  post_select_obs={ps_obs}  target_obs={target_obs}')

        # Noiseless sanity check
        dets, obs = circuit.compile_detector_sampler().sample(
            200, separate_observables=True)
        ok = not np.any(dets) and not np.any(obs)
        print(f'  Noiseless check: {"OK" if ok else "FAIL"}')

        for p in args.p_values:
            key = (d, rounds, R, p, 0.0)
            if key in done_keys:
                print(f'  d={d} p={p:.0e} — SKIPPED (checkpoint)')
                continue

            print(f'\n  d={d} p={p:.1e}  '
                  f'(max_shots={args.max_shots:.0e}  max_errors={args.max_errors})')
            t0 = time.perf_counter()
            stats = run_one(circuit, ps_obs, target_obs, p, args.decoder,
                            args.max_shots, args.max_errors,
                            args.batch_size, args.num_workers)
            elapsed = time.perf_counter() - t0

            row = {
                'd':                  d,
                'rounds':             rounds,
                'r':                  R,
                'p':                  p,
                'p_injected':         0.0,
                'p_in':               0.0,
                'ler_ps':             stats.logical_error_rate,
                'post_selection_rate': stats.post_selection_rate,
                'shots':              stats.shots,
                'errors':             stats.errors,
            }
            append_row(csv_path, row)
            done_keys.add(key)

            print(f'  shots={stats.shots:,}  errors={stats.errors}  '
                  f'LER={stats.logical_error_rate:.3e}  '
                  f'accept={stats.post_selection_rate:.3f}  '
                  f'time={elapsed:.1f}s')

    print(f'\nDone. Results → {csv_path}')
