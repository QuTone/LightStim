"""
TG 7-to-1 distillation — injection-only noise sweep.

Builds circuits fresh (rounds=1, r=1) on first run, caches magic_qubit_indices
alongside the .stim file. Runs injection-only noise sweep and saves to
TG_injection_results.csv with per-row checkpointing.

CSV schema (matches run_distillation_simulations.py TG format):
    KEY : d, rounds, r, p, p_injected
    DATA: p_in, ler_ps, post_selection_rate, shots, errors

Usage (from repo root):
    venv/bin/python eval/logical_circuit_benchmark/distillation/tg_7to1/run_tg_injection.py
    venv/bin/python eval/logical_circuit_benchmark/distillation/tg_7to1/run_tg_injection.py \\
        -d 3 5 --rounds 1 --r 1 \\
        --p-injected 1e-3 5.6e-3 3.2e-2 1.78e-1 \\
        --num-workers 8
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

from eval.logical_circuit_benchmark.distillation.tg_7to1.TG_distillation_7_to_1 import (
    build_distillation_circuit as build_tg,
    inject_noise as inject_noise_tg,
    estimate_p_in as estimate_p_in_tg,
    _TG_MAGIC_NAMES,
)
from src.simulation.observable_analysis import (
    build_obs_patch_matrix,
    identify_distillation_observables,
)
from src.simulation.decoder_backend.config import DecoderConfig
from src.simulation.decoder_backend.pipeline import SimulationPipeline

import stim as _stim  # noqa — already imported above, alias for clarity

KEY_COLS  = ['d', 'rounds', 'r', 'p', 'p_injected']
DATA_COLS = ['p_in', 'ler_ps', 'post_selection_rate', 'shots', 'errors']
ALL_COLS  = KEY_COLS + DATA_COLS


def _circ_dir():
    d = os.path.join(SCRIPT_DIR, 'circuits_inj')
    os.makedirs(d, exist_ok=True)
    return d


def build_and_cache(d: int, rounds: int, r: int):
    """Build TG circuit, extract magic_qubits, save .stim + _obs.json."""
    circ_dir  = _circ_dir()
    tag       = f'TG_inj_d{d}_rnd{rounds}_r{r}'
    stim_path = os.path.join(circ_dir, f'{tag}.stim')
    obs_path  = os.path.join(circ_dir, f'{tag}_obs.json')

    if os.path.exists(stim_path) and os.path.exists(obs_path):
        print(f'  Loading cached: {os.path.basename(stim_path)}')
        circuit = stim.Circuit.from_file(stim_path)
        with open(obs_path) as f:
            meta = json.load(f)
        return circuit, meta['post_select_obs'], meta['target_obs'], set(meta['magic_qubit_indices'])

    print(f'  Building TG d={d} rounds={rounds} r={r} (will cache)...')
    t0 = time.perf_counter()
    circuit, _, system = build_tg(d=d, rounds=rounds, r=r)
    t_build = time.perf_counter() - t0
    print(f'  Built in {t_build:.1f}s — qubits={circuit.num_qubits}  det={circuit.num_detectors}')

    matrix, pn = build_obs_patch_matrix(circuit, system)
    T, tgt, ps = identify_distillation_observables(matrix, pn, ['W0'])
    ps  = list(ps)
    tgt = list(tgt)

    magic_qubits = sorted({q for q, owner in system.index_to_owner_map.items()
                           if owner in _TG_MAGIC_NAMES})

    circuit.to_file(stim_path)
    with open(obs_path, 'w') as f:
        json.dump({
            'post_select_obs':    ps,
            'target_obs':         tgt,
            'magic_qubit_indices': magic_qubits,
            'd': d, 'rounds': rounds, 'r': r,
        }, f, indent=2)

    print(f'  Cached → {stim_path}')
    print(f'  post_select_obs={ps}  target_obs={tgt}  magic_qubits={len(magic_qubits)}')
    return circuit, ps, tgt, set(magic_qubits)


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
    parser.add_argument('-d', '--distances', type=int, nargs='+', default=[3, 5])
    parser.add_argument('--rounds', type=int, default=1,
                        help='SE rounds per cycle (default: 1)')
    parser.add_argument('--r', type=int, default=1,
                        help='SE rounds between transversal gates (default: 1)')
    parser.add_argument('--decoder', choices=['pymatching', 'bposd', 'mwpf'],
                        default='pymatching',
                        help='Decoder (default: pymatching; bposd is very slow to init)')
    parser.add_argument('--p-injected', type=float, nargs='+',
                        default=[1e-3, 5.6e-3, 3.2e-2, 1.78e-1])
    parser.add_argument('--max-shots',   type=int, default=100_000_000)
    parser.add_argument('--max-errors',  type=int, default=100)
    parser.add_argument('--batch-size',  type=int, default=5_000)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    csv_path  = os.path.join(SCRIPT_DIR, 'TG_injection_results.csv')
    done_keys = load_done_keys(csv_path)
    if done_keys:
        print(f'Checkpoint: {len(done_keys)} row(s) already done, skipping.')

    for d in args.distances:
        rounds = args.rounds
        r      = args.r
        print(f'\n{"="*60}')
        print(f'TG injection  d={d}  rounds={rounds}  r={r}')
        print(f'{"="*60}')

        circuit, ps_obs, target_obs, magic_qubits = build_and_cache(d, rounds, r)
        print(f'  qubits={circuit.num_qubits}  det={circuit.num_detectors}  '
              f'obs={circuit.num_observables}  magic_qubits={len(magic_qubits)}')

        # Noiseless sanity check
        dets, obs = circuit.compile_detector_sampler().sample(
            200, separate_observables=True)
        ok = not np.any(dets) and not np.any(obs)
        print(f'  Noiseless check: {"OK" if ok else "FAIL"}')

        # Calibrate p_in for each p_injected
        print(f'\n-- Calibrating p_in (injection mode, rounds={rounds}) --')
        p_in_map = {}
        for p_inj in args.p_injected:
            p_in_est = estimate_p_in_tg(d, rounds, p_inj,
                                        p_background=0.0,
                                        max_shots=args.max_shots,
                                        max_errors=args.max_errors,
                                        batch_size=args.batch_size)
            p_in_map[p_inj] = p_in_est
            print(f'    p_injected={p_inj:.2e}  →  p_in={p_in_est:.3e}')

        # Build pipeline once (reused for all p_injected values)
        pipeline = SimulationPipeline(
            decoder_config=DecoderConfig(args.decoder),
            max_shots=args.max_shots,
            max_errors=args.max_errors,
            batch_size=args.batch_size,
            post_select_corrected_observable_indices=ps_obs,
            target_observable_indices=target_obs,
            print_progress=True,
            progress_interval_sec=30.0,
            num_workers=args.num_workers,
        )

        print(f'\n-- Injection-only noise sweep --')
        for p_inj in args.p_injected:
            key = (d, rounds, r, 0.0, p_inj)
            if key in done_keys:
                print(f'  SKIPPED (checkpoint): d={d} p_injected={p_inj:.1e}')
                continue

            noisy = inject_noise_tg(circuit, magic_qubits,
                                    p=0.0, p_injected=p_inj, mode='injection')
            t0 = time.perf_counter()
            stats = pipeline.run(noisy)
            elapsed = time.perf_counter() - t0

            row = {
                'd':                   d,
                'rounds':              rounds,
                'r':                   r,
                'p':                   0.0,
                'p_injected':          p_inj,
                'p_in':                p_in_map[p_inj],
                'ler_ps':              stats.logical_error_rate,
                'post_selection_rate': stats.post_selection_rate,
                'shots':               stats.shots,
                'errors':              stats.errors,
            }
            append_row(csv_path, row)
            done_keys.add(key)

            print(f'  d={d} p_injected={p_inj:.1e}  '
                  f'p_in={p_in_map[p_inj]:.3e}  '
                  f'LER={stats.logical_error_rate:.3e}  '
                  f'accept={stats.post_selection_rate:.3f}  '
                  f'shots={stats.shots:,}  errors={stats.errors}  '
                  f'time={elapsed:.1f}s')

    print(f'\nDone. Results → {csv_path}')
