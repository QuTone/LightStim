"""
Bell State Teleportation — Transversal-Gate (TG) protocol.

Three patches in a horizontal row; two sequential transversal CNOTs implement
a Bell-state teleportation from patch1 to patch3 via patch2.

Protocol:
    Init: patch1=|ψ⟩, patch2=|+⟩, patch3=|0⟩
    SE(d) → CNOT(patch2→patch3) → SE(1) → CNOT(patch1→patch2) → SE(1) → Meas

Decoder: CPU BP+OSD
States: X, Z  (no Y)

Usage:
    python run_tg.py --build-only
    python run_tg.py -d 3 5 7 -p 5e-4 1e-3 2e-3 5e-3 --states X Z
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.ir.operation import CSSLogicalOpSet
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)
from lightstim.simulation.decoder_backend.config import DecoderConfig
from lightstim.simulation.decoder_backend.pipeline import SimulationPipeline


# =============================================================================
# Circuit construction
# =============================================================================

def build_circuit(d: int, teleport_state: str):
    """
    Build TG Bell teleportation circuit.

    Layout: patch1 -- patch2 -- patch3  (horizontal, dx = 2*(2d-1)-2 apart)

    Args:
        d: Code distance.
        teleport_state: 'X' or 'Z' — basis of the state being teleported.

    Returns:
        (circuit, info_dict)
    """
    rounds_pre  = d
    rounds_mid  = 1
    rounds_post = 1
    dx = 2 * (2 * d - 1) - 2  # spacing between patch origins

    patch1_local = UnrotatedSurfaceCode(distance=d)
    patch2_local = UnrotatedSurfaceCode(distance=d)
    patch3_local = UnrotatedSurfaceCode(distance=d)

    system = QECSystem()
    patch1 = system.add_patch(patch1_local, name='patch1')
    patch2 = system.add_patch(patch2_local, name='patch2', offset=(dx, 0))
    patch3 = system.add_patch(patch3_local, name='patch3', offset=(2 * dx, 0))

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    INIT = {'patch1': teleport_state, 'patch2': 'X', 'patch3': 'Z'}
    init_dict = {q: INIT[system.index_to_owner_map[q]] for q in system.data_indices}
    builder.initialize(init_dict=init_dict, n=system.num_qubits)

    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_pre)

    executor = LogicalExecutor(builder=builder)
    executor.register_op_set(UnrotatedSurfaceCode, CSSLogicalOpSet())

    executor.apply_logical_operation('transversal_cnot', patches=[patch2, patch3])

    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_mid)

    executor.apply_logical_operation('transversal_cnot', patches=[patch1, patch2])

    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_post)

    MEAS = {'patch1': 'X', 'patch2': 'Z', 'patch3': teleport_state}
    meas_dict = {q: MEAS[system.index_to_owner_map[q]] for q in system.data_indices}
    builder.apply_data_readout(final_measurements=meas_dict)

    circuit = builder.circuit
    info = {
        'num_qubits': circuit.num_qubits,
        'num_detectors': circuit.num_detectors,
        'num_observables': circuit.num_observables,
        'rounds_pre': rounds_pre,
        'rounds_mid': rounds_mid,
        'rounds_post': rounds_post,
    }
    return circuit, info


# =============================================================================
# Simulation
# =============================================================================

def run_simulation(circuit, p: float, decoder_name: str,
                   max_shots: int, max_errors: int,
                   num_workers: int, batch_size: int):
    noise_config = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    injector = NoiseInjector.from_circuit_level(
        noise_config, list(range(circuit.num_qubits))
    )
    noisy = injector.inject_noise(circuit)

    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig(decoder_name, backend='gpu' if decoder_name == 'nv-qldpc-decoder' else 'cpu'),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        num_workers=num_workers,
        print_progress=True,
        progress_interval_sec=30.0,
    )
    return pipeline.run(noisy)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TG Bell teleportation sweep — CPU BP+OSD')
    parser.add_argument('-d', '--distances', type=int, nargs='+', default=[3, 5, 7])
    parser.add_argument('-p', '--p-values', type=float, nargs='+',
                        default=[5e-4, 1e-3, 2e-3, 5e-3])
    parser.add_argument('--states', nargs='+', choices=['X', 'Z'], default=['X', 'Z'],
                        help='States to teleport (default: X Z)')
    parser.add_argument('--decoder', choices=['bposd', 'pymatching', 'mwpf', 'nv-qldpc-decoder'],
                        default='bposd')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-shots', type=int, default=1_000_000_000)
    parser.add_argument('--max-errors', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=10_000)
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'tg_results.csv')

    # Load checkpoint
    done_keys = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done_keys.add((int(row['d']), float(row['p']), row['state'], row['decoder']))
        print(f'Checkpoint: {len(done_keys)} tasks done, skipping.')

    FIELDNAMES = [
        'd', 'state', 'rounds_pre', 'rounds_mid', 'rounds_post',
        'p', 'decoder',
        'num_qubits', 'num_detectors', 'num_observables',
        'shots', 'errors', 'logical_error_rate',
        'build_time_sec', 'decoding_time_sec',
    ]

    for d in args.distances:
        for state in args.states:
            print(f'\n{"="*60}')
            print(f'Building d={d}, state=|{state}⟩')
            print(f'{"="*60}')

            t0 = time.perf_counter()
            circuit, info = build_circuit(d, state)
            t_build = time.perf_counter() - t0

            print(f'  {info["num_qubits"]} qubits, {info["num_detectors"]} det, '
                  f'{info["num_observables"]} obs  (built in {t_build:.1f}s)')

            # Noiseless check
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
                    'rounds_mid': info['rounds_mid'],
                    'rounds_post': info['rounds_post'],
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
                    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    if write_header:
                        w.writeheader()
                    w.writerow(result)

                print(f'  shots={stats.shots:,}  errors={stats.errors}  '
                      f'LER={stats.logical_error_rate:.2e}  '
                      f'time={stats.seconds:.1f}s')

    print(f'\nDone. Results at {csv_path}')
