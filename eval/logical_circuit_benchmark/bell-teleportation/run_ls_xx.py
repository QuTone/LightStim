"""
Bell State Teleportation — XX lattice surgery (horizontal patch row).

Two sequential XX merges: first patch2–patch3 (Bell prep), then patch1–patch2
(entangle source with Bell pair). Matches `notebooks/test_bell_teleport.ipynb`.

Decoder: CPU PyMatching (default).
States: X, Z  (no Y)

Usage:
    python run_ls_xx.py --build-only
    python run_ls_xx.py -d 3 5 7 -p 5e-4 1e-3 2e-3 5e-3 --states X Z
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.ir.builder import CircuitBuilder
from src.ir.qec_system import QECSystem
from src.ir.tracker import SyndromeTracker
from src.noise.config import NoiseConfig
from src.noise.injector import NoiseInjector
from src.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedTwoPatchCoupler,
)
from src.simulation.decoder_backend.config import DecoderConfig
from src.simulation.decoder_backend.pipeline import SimulationPipeline


def build_circuit(d: int, teleport_state: str):
    """
    XX-LS Bell teleportation. Layout: patch1, patch2, patch3 left-to-right, step = 2d.
    """
    rounds_pre = d
    rounds_ls = d
    gap = 1
    d_size = 2 * d - 1
    step = d_size + gap

    patch1_local = UnrotatedSurfaceCode(distance=d)
    patch2_local = UnrotatedSurfaceCode(distance=d)
    patch3_local = UnrotatedSurfaceCode(distance=d)

    system = QECSystem()
    system.add_patch(patch1_local, name='patch1')
    system.add_patch(patch2_local, name='patch2', offset=(step, 0))
    system.add_patch(patch3_local, name='patch3', offset=(2 * step, 0))

    coupler_proto = UnrotatedTwoPatchCoupler()
    system.register_coupler(
        coupler_proto,
        patch_names=['patch2', 'patch3'],
        name='coupler_23',
        interaction_type='XX',
    )
    system.register_coupler(
        coupler_proto,
        patch_names=['patch1', 'patch2'],
        name='coupler_12',
        interaction_type='XX',
    )

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    init_xx = {'patch1': teleport_state, 'patch2': 'Z', 'patch3': 'Z'}
    init_dict = {
        q: init_xx[system.index_to_owner_map[q]]
        for q in system.data_indices
        if system.index_to_owner_map[q] in init_xx
    }
    builder.initialize(init_dict=init_dict, n=system.num_qubits)

    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_pre)

    builder.activate_coupler('coupler_23')
    cp23_local = system.coupler_patches['coupler_23'].data_indices
    cp23_global = [system.local_to_global_map['coupler_23'][q] for q in cp23_local]
    builder.initialize(
        init_dict={q: 'Z' for q in cp23_global},
        n=system.num_qubits,
    )
    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_ls)
    builder.deactivate_coupler('coupler_23')
    # Measure coupler_23 data qubits immediately at deactivation (same bug as CNOT_LS)
    builder.apply_data_readout(final_measurements={q: 'Z' for q in cp23_global})

    builder.activate_coupler('coupler_12')
    cp12_local = system.coupler_patches['coupler_12'].data_indices
    cp12_global = [system.local_to_global_map['coupler_12'][q] for q in cp12_local]
    builder.initialize(
        init_dict={q: 'Z' for q in cp12_global},
        n=system.num_qubits,
    )
    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_ls)

    meas_xx = {'patch1': 'Z', 'patch2': 'Z', 'patch3': teleport_state}
    meas_dict = {
        q: meas_xx[system.index_to_owner_map[q]]
        for q in system.data_indices
        if system.index_to_owner_map[q] in meas_xx
    }
    meas_dict.update({q: 'Z' for q in cp12_global})
    # cp23_global already measured at deactivation
    builder.apply_data_readout(final_measurements=meas_dict)

    circuit = builder.circuit
    info = {
        'num_qubits': circuit.num_qubits,
        'num_detectors': circuit.num_detectors,
        'num_observables': circuit.num_observables,
        'rounds_pre': rounds_pre,
        'rounds_ls': rounds_ls,
    }
    return circuit, info


def run_simulation(circuit, p: float, decoder_name: str,
                   max_shots: int, max_errors: int,
                   num_workers: int, batch_size: int):
    noise_config = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    injector = NoiseInjector.from_circuit_level(
        noise_config, list(range(circuit.num_qubits))
    )
    noisy = injector.inject_noise(circuit)

    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig(decoder_name, backend='cpu'),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        num_workers=num_workers,
        print_progress=True,
        progress_interval_sec=30.0,
    )
    return pipeline.run(noisy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='XX-LS Bell teleportation sweep — CPU PyMatching')
    parser.add_argument('-d', '--distances', type=int, nargs='+', default=[3, 5, 7])
    parser.add_argument('-p', '--p-values', type=float, nargs='+',
                        default=[5e-4, 1e-3, 2e-3, 5e-3])
    parser.add_argument('--states', nargs='+', choices=['X', 'Z'], default=['X', 'Z'])
    parser.add_argument('--decoder', choices=['pymatching', 'bposd', 'mwpf'],
                        default='pymatching')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-shots', type=int, default=1_000_000_000)
    parser.add_argument('--max-errors', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=10_000)
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'ls_xx_results.csv')

    done_keys = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done_keys.add((int(row['d']), float(row['p']), row['state'], row['decoder']))
        print(f'Checkpoint: {len(done_keys)} tasks done, skipping.')

    fieldnames = [
        'd', 'state', 'rounds_pre', 'rounds_ls',
        'p', 'decoder',
        'num_qubits', 'num_detectors', 'num_observables',
        'shots', 'errors', 'logical_error_rate',
        'build_time_sec', 'decoding_time_sec',
    ]

    for d in args.distances:
        for state in args.states:
            print(f'\n{"="*60}')
            print(f'Building d={d}, state=|{state}⟩ (XX-LS)')
            print(f'{"="*60}')

            t0 = time.perf_counter()
            circuit, info = build_circuit(d, state)
            t_build = time.perf_counter() - t0

            print(f'  {info["num_qubits"]} qubits, {info["num_detectors"]} det, '
                  f'{info["num_observables"]} obs  (built in {t_build:.1f}s)')

            dets, obs = circuit.compile_detector_sampler().sample(
                shots=100, separate_observables=True)
            ok = not np.any(dets) and not np.any(obs)
            print(f'  Noiseless check: {"OK" if ok else "FAIL"}')

            if args.build_only:
                continue

            for p in args.p_values:
                key = (d, p, state, args.decoder)
                if key in done_keys:
                    print(f'\n  d={d} state={state} p={p:.0e} — SKIPPED (checkpoint)')
                    continue

                print(f'\n  d={d} state={state} p={p:.0e} decoder={args.decoder}')
                stats = run_simulation(
                    circuit, p, args.decoder,
                    args.max_shots, args.max_errors,
                    args.num_workers, args.batch_size,
                )

                result = {
                    'd': d,
                    'state': state,
                    'rounds_pre': info['rounds_pre'],
                    'rounds_ls': info['rounds_ls'],
                    'p': p,
                    'decoder': args.decoder,
                    'num_qubits': info['num_qubits'],
                    'num_detectors': info['num_detectors'],
                    'num_observables': info['num_observables'],
                    'shots': stats.shots,
                    'errors': stats.errors,
                    'logical_error_rate': stats.logical_error_rate,
                    'build_time_sec': round(t_build, 2),
                    'decoding_time_sec': round(stats.seconds, 2),
                }

                write_header = not os.path.exists(csv_path)
                with open(csv_path, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        w.writeheader()
                    w.writerow(result)

                print(f'  shots={stats.shots:,}  errors={stats.errors}  '
                      f'LER={stats.logical_error_rate:.2e}  '
                      f'time={stats.seconds:.1f}s')

    print(f'\nDone. Results at {csv_path}')
