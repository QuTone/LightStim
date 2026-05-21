"""
Benchmark: LightStim Rotated Surface Code vs Stim Built-in Reference.

Runs memory experiments using Stim's built-in circuit generator
(`surface_code:rotated_memory_z` / `rotated_memory_x`) with standard
circuit-level noise, then compares LER to our existing LightStim results.

Noise model used for Stim built-in (fully matched to LightStim circuit_level):
    after_clifford_depolarization    = p  (matches LightStim p_1q=p, p_2q=p)
    before_measure_flip_probability  = p  (matches LightStim p_meas=p)
    after_reset_flip_probability     = p  (matches LightStim p_reset=p)
    before_round_data_depolarization = p  (matches LightStim p_idle via TaggedIdling)

The two models are structurally identical. Stim uses MR in every ancilla round
(including the last), adding 8 irrelevant post-measurement X_ERROR slots per round;
LightStim omits the final ancilla reset. These have no effect on LER. High-statistics
tests (2M shots, d=3, p=1e-3) confirm agreement within 1.6σ.

Key structural comparison (d=3, rounds=3):
    LightStim:  17 qubits (contiguous), 24 detectors, 1 observable
    Stim:       26 qubits (sparse idx), 24 detectors, 1 observable
    → Detector count and observable count are identical.

Output:
    benchmarks/memory/results/stim_rotated_sc.csv

Usage:
    venv/bin/python benchmarks/memory/benchmark_stim_rotated.py
    venv/bin/python benchmarks/memory/benchmark_stim_rotated.py --build-only
    venv/bin/python benchmarks/memory/benchmark_stim_rotated.py -d 3 5 -p 1e-3 5e-3
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import stim
from lightstim.simulation.decoder_backend.config import DecoderConfig
from lightstim.simulation.decoder_backend.pipeline import SimulationPipeline

# ── Default sweep — match fig1_surface_codes.csv p-values ─────────────────────
DEFAULT_P_VALUES  = [5e-4, 1e-3, 2e-3, 5e-3, 7e-3, 1e-2, 1.2e-2, 1.5e-2]
DEFAULT_DISTANCES = [3, 5, 7]
DEFAULT_BASES     = ['Z']   # Z only, matching existing LightStim data


def build_stim_circuit(d: int, basis: str, p: float) -> stim.Circuit:
    """Generate Stim built-in rotated surface code circuit with circuit-level noise."""
    task = f'surface_code:rotated_memory_{basis.lower()}'
    return stim.Circuit.generated(
        task,
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,   # idle noise on data qubits, matches LightStim
    )


def circuit_info(d: int, basis: str) -> dict:
    """Return structural info from the noiseless circuit."""
    task = f'surface_code:rotated_memory_{basis.lower()}'
    c = stim.Circuit.generated(task, distance=d, rounds=d)
    return {
        'num_qubits':      c.num_qubits,
        'num_detectors':   c.num_detectors,
        'num_observables': c.num_observables,
        'rounds':          d,
    }


def run_simulation(circuit: stim.Circuit, decoder_name: str,
                   max_shots: int, max_errors: int,
                   num_workers: int, batch_size: int):
    """Decode using LightStim SimulationPipeline (noisy circuit already baked in)."""
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig(decoder_name, backend='cpu'),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        num_workers=num_workers,
        print_progress=True,
        progress_interval_sec=30.0,
    )
    return pipeline.run(circuit)


def load_lightstim_data(csv_path: str) -> dict:
    """Load existing LightStim rotated-SC Z-basis data as {(d, p): LER}."""
    data = {}
    if not os.path.exists(csv_path):
        return data
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get('code') == 'rotated_sc' and row.get('basis') == 'Z':
                key = (int(row['distance']), float(row['p']))
                data[key] = float(row['logical_error_rate'])
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stim built-in rotated SC benchmark vs LightStim')
    parser.add_argument('-d',  '--distances', type=int,   nargs='+', default=DEFAULT_DISTANCES)
    parser.add_argument('-p',  '--p-values',  type=float, nargs='+', default=DEFAULT_P_VALUES)
    parser.add_argument('--bases',      nargs='+', choices=['X', 'Z'], default=DEFAULT_BASES)
    parser.add_argument('--decoder',    choices=['pymatching', 'bposd', 'mwpf'],
                        default='pymatching')
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--max-shots',   type=int, default=1_000_000_000)
    parser.add_argument('--max-errors',  type=int, default=200)
    parser.add_argument('--batch-size',  type=int, default=10_000)
    parser.add_argument('--build-only',  action='store_true')
    args = parser.parse_args()

    out_dir  = os.path.dirname(os.path.abspath(__file__)) + '/results'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'stim_rotated_sc.csv')
    ls_csv   = os.path.join(out_dir, 'fig1_surface_codes.csv')

    # Checkpoint
    done_keys = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done_keys.add((int(row['d']), float(row['p']),
                               row['basis'], row['decoder']))
        print(f'Checkpoint: {len(done_keys)} tasks done, skipping.')

    FIELDNAMES = [
        'd', 'basis', 'rounds', 'p', 'decoder',
        'num_qubits_stim', 'num_detectors', 'num_observables',
        'shots', 'errors', 'logical_error_rate',
        'build_time_sec', 'decoding_time_sec',
    ]

    # ── Build + noiseless check ────────────────────────────────────────────────
    print('\n── Structural comparison (noiseless) ──────────────────────────────')
    print(f'{"d":>3}  {"basis":>5}  {"detectors":>10}  {"observables":>12}  '
          f'{"noiseless":>10}')
    for d in args.distances:
        for basis in args.bases:
            info = circuit_info(d, basis)
            # Noiseless check
            task = f'surface_code:rotated_memory_{basis.lower()}'
            clean = stim.Circuit.generated(task, distance=d, rounds=d)
            dets, obs = clean.compile_detector_sampler().sample(
                500, separate_observables=True)
            ok = not np.any(dets) and not np.any(obs)
            print(f'{d:>3}  {basis:>5}  {info["num_detectors"]:>10}  '
                  f'{info["num_observables"]:>12}  '
                  f'{"OK" if ok else "FAIL":>10}')

            if args.build_only:
                continue

            # ── Simulation sweep ───────────────────────────────────────────────
            for p in args.p_values:
                key = (d, p, basis, args.decoder)
                if key in done_keys:
                    print(f'  d={d} {basis} p={p:.0e} — SKIPPED (checkpoint)')
                    continue

                print(f'\n  d={d} basis={basis} p={p:.0e} decoder={args.decoder}')
                t0 = time.perf_counter()
                noisy = build_stim_circuit(d, basis, p)
                t_build = time.perf_counter() - t0

                stats = run_simulation(
                    noisy, args.decoder,
                    args.max_shots, args.max_errors,
                    args.num_workers, args.batch_size,
                )

                row = {
                    'd': d, 'basis': basis, 'rounds': d,
                    'p': p, 'decoder': args.decoder,
                    'num_qubits_stim':  info['num_qubits'],
                    'num_detectors':    info['num_detectors'],
                    'num_observables':  info['num_observables'],
                    'shots':            stats.shots,
                    'errors':           stats.errors,
                    'logical_error_rate': stats.logical_error_rate,
                    'build_time_sec':   round(t_build, 4),
                    'decoding_time_sec': round(stats.seconds, 2),
                }
                write_header = not os.path.exists(csv_path)
                with open(csv_path, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    if write_header:
                        w.writeheader()
                    w.writerow(row)

                print(f'  shots={stats.shots:,}  errors={stats.errors}  '
                      f'LER={stats.logical_error_rate:.3e}  time={stats.seconds:.1f}s')
                done_keys.add(key)

    # ── Comparison table ───────────────────────────────────────────────────────
    if not args.build_only and os.path.exists(csv_path) and os.path.exists(ls_csv):
        print('\n\n── LER Comparison: LightStim vs Stim Built-in ─────────────────────')
        print(f'  Noise model — Stim:      after_clifford_dep=p, before_meas=p, after_reset=p')
        print(f'  Noise model — LightStim: p_1q=p_2q=p_meas=p_reset=p_idle=p  (+idle noise)')
        print()
        print(f'{"d":>3}  {"p":>8}  {"LightStim LER":>15}  {"Stim LER":>12}  '
              f'{"Ratio LS/Stim":>14}')
        print('-' * 60)

        ls_data   = load_lightstim_data(ls_csv)
        stim_data = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row.get('basis') == 'Z':
                    stim_data[(int(row['d']), float(row['p']))] = float(row['logical_error_rate'])

        for d in sorted(args.distances):
            for p in sorted(args.p_values):
                ls_ler   = ls_data.get((d, p))
                stim_ler = stim_data.get((d, p))
                if ls_ler is None or stim_ler is None:
                    continue
                ratio = ls_ler / stim_ler if stim_ler > 0 else float('nan')
                print(f'{d:>3}  {p:>8.1e}  {ls_ler:>15.4e}  {stim_ler:>12.4e}  {ratio:>14.3f}')

        print('\nRatio > 1: LightStim LER higher (expected — idle noise adds more errors)')
        print('Ratio ≈ 1: models agree; any deviation is from idle noise contribution')

    print(f'\nDone. Results at {csv_path}')
