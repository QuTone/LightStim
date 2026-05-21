"""
Bell State ZZ Teleportation — varying coupler_23 gap as proxy for routing distance.

The Bell pair is prepared between patch2 and patch3. When those two patches are
physically far apart (longer routing), coupler_23 spans more qubits — but SE rounds
stay d (single stabilizer measurement, just wider). coupler_12 (source → Bell anchor)
always stays 1× adjacent.

routing_mult=2: patch2–patch3 gap = 2× normal  → coupler_23 spans 2 patch-widths
routing_mult=4: patch2–patch3 gap = 4× normal  → coupler_23 spans 4 patch-widths
routing_mult=8: patch2–patch3 gap = 8× normal  → coupler_23 spans 8 patch-widths

Sub-experiments: teleport state |Z⟩ and |X⟩.

Usage:
    venv/bin/python eval/logical_circuit_benchmark/bell-teleportation/run_ls_zz_dist.py --build-only
    venv/bin/python eval/logical_circuit_benchmark/bell-teleportation/run_ls_zz_dist.py \\
        -d 3 5 7 -p 5e-4 1e-3 2e-3 5e-3 1e-2 --mults 2 4 8
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedTwoPatchCoupler,
)
from lightstim.simulation.decoder_backend.config import DecoderConfig
from lightstim.simulation.decoder_backend.pipeline import SimulationPipeline


def build_circuit(d: int, teleport_state: str, routing_mult: int):
    """
    ZZ-LS Bell teleportation with long-range coupler_23 (Bell pair).

    coupler_23 spans routing_mult × step_1x  (Bell pair preparation over longer distance).
    coupler_12 always stays 1× (source patch adjacent to Bell anchor).
    SE rounds = d throughout (single stabilizer measurement, just wider coupler).
    """
    rounds_pre = d
    rounds_ls  = d
    gap     = 1
    step_1x = (2 * d - 1) + gap  # one patch-width

    patch1_local = UnrotatedSurfaceCode(distance=d)
    patch2_local = UnrotatedSurfaceCode(distance=d)
    patch3_local = UnrotatedSurfaceCode(distance=d)

    system = QECSystem()
    system.add_patch(patch1_local, name='patch1')
    # patch2 always 1× from patch1
    system.add_patch(patch2_local, name='patch2', offset=(0, step_1x))
    # patch3 is routing_mult × step_1x from patch2
    system.add_patch(patch3_local, name='patch3', offset=(0, step_1x + routing_mult * step_1x))

    coupler_proto = UnrotatedTwoPatchCoupler()
    system.register_coupler(
        coupler_proto,
        patch_names=['patch2', 'patch3'],
        name='coupler_23',
        interaction_type='ZZ',
    )
    system.register_coupler(
        coupler_proto,
        patch_names=['patch1', 'patch2'],
        name='coupler_12',
        interaction_type='ZZ',
    )

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    # Initialise patches: patch1 = teleport_state, patch2/3 = |+⟩ (X eigenstates)
    init_map = {'patch1': teleport_state, 'patch2': 'X', 'patch3': 'X'}
    init_dict = {
        q: init_map[system.index_to_owner_map[q]]
        for q in system.data_indices
        if system.index_to_owner_map[q] in init_map
    }
    builder.initialize(init_dict=init_dict, n=system.num_qubits)

    # Pre-rounds SE
    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_pre)

    # First LS: coupler_23 (Bell prep — long-range, routing_mult × gap)
    builder.activate_coupler('coupler_23')
    cp23_local  = system.coupler_patches['coupler_23'].data_indices
    cp23_global = [system.local_to_global_map['coupler_23'][q] for q in cp23_local]
    builder.initialize(init_dict={q: 'X' for q in cp23_global}, n=system.num_qubits)
    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_ls)
    builder.deactivate_coupler('coupler_23')
    # Measure coupler_23 data qubits immediately at deactivation
    builder.apply_data_readout(final_measurements={q: 'X' for q in cp23_global})

    # Second LS: coupler_12 (teleport — always 1× gap)
    builder.activate_coupler('coupler_12')
    cp12_local  = system.coupler_patches['coupler_12'].data_indices
    cp12_global = [system.local_to_global_map['coupler_12'][q] for q in cp12_local]
    builder.initialize(init_dict={q: 'X' for q in cp12_global}, n=system.num_qubits)
    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_ls)

    # Final readout: patch1→X (discard), patch2→X (discard), patch3→teleport_state
    meas_map = {'patch1': 'X', 'patch2': 'X', 'patch3': teleport_state}
    meas_dict = {
        q: meas_map[system.index_to_owner_map[q]]
        for q in system.data_indices
        if system.index_to_owner_map[q] in meas_map
    }
    meas_dict.update({q: 'X' for q in cp12_global})
    # cp23_global already measured at deactivation
    builder.apply_data_readout(final_measurements=meas_dict)

    circuit = builder.circuit
    info = {
        'num_qubits':      circuit.num_qubits,
        'num_detectors':   circuit.num_detectors,
        'num_observables': circuit.num_observables,
        'rounds_pre':      rounds_pre,
        'rounds_ls':       rounds_ls,
        'routing_mult':    routing_mult,
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
        description='Bell ZZ-LS teleportation — varying coupler_23 routing distance')
    parser.add_argument('-d',  '--distances',  type=int,   nargs='+', default=[3, 5, 7])
    parser.add_argument('-p',  '--p-values',   type=float, nargs='+',
                        default=[5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    parser.add_argument('--states',    nargs='+', choices=['X', 'Z'], default=['X', 'Z'])
    parser.add_argument('--mults',     type=int,   nargs='+', default=[2, 4, 8],
                        help='coupler_23 gap = mult × step_1x (default: 2 4 8)')
    parser.add_argument('--decoder',   choices=['pymatching', 'bposd', 'mwpf'],
                        default='pymatching')
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--max-shots',   type=int, default=1_000_000_000)
    parser.add_argument('--max-errors',  type=int, default=100)
    parser.add_argument('--batch-size',  type=int, default=10_000)
    parser.add_argument('--build-only',  action='store_true')
    args = parser.parse_args()

    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'ls_zz_dist_results.csv')

    # Checkpoint
    done_keys = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done_keys.add((
                    int(row['d']), float(row['p']),
                    row['state'], int(row['routing_mult']), row['decoder'],
                ))
        print(f'Checkpoint: {len(done_keys)} tasks done, skipping.')

    FIELDNAMES = [
        'd', 'state', 'rounds_pre', 'rounds_ls', 'routing_mult',
        'p', 'decoder',
        'num_qubits', 'num_detectors', 'num_observables',
        'shots', 'errors', 'logical_error_rate',
        'build_time_sec', 'decoding_time_sec',
    ]

    for d in args.distances:
        for state in args.states:
            for mult in args.mults:
                label = f'd={d} state={state} mult={mult}x'
                print(f'\n{"="*60}')
                print(f'Building {label}  (coupler_23 gap={mult}×step)')
                print(f'{"="*60}')

                t0 = time.perf_counter()
                circuit, info = build_circuit(d, state, mult)
                t_build = time.perf_counter() - t0

                print(f'  {info["num_qubits"]} qubits, {info["num_detectors"]} det, '
                      f'{info["num_observables"]} obs  (built in {t_build:.2f}s)')

                # Noiseless check
                dets, obs = circuit.compile_detector_sampler().sample(
                    500, separate_observables=True)
                ok = not np.any(dets) and not np.any(obs)
                print(f'  Noiseless check: {"OK" if ok else "FAIL ← check circuit!"}')

                # DEM weight-1 logical mechanisms
                dem = circuit.detector_error_model(decompose_errors=False)
                w1 = sum(1 for inst in dem.flattened()
                         if inst.type == "error"
                         and len(inst.targets_copy()) == 1
                         and any(t.is_logical_observable_id() for t in inst.targets_copy()))
                print(f'  DEM weight-1 logical mechanisms: {w1}')

                if args.build_only:
                    continue

                for p in args.p_values:
                    key = (d, p, state, mult, args.decoder)
                    if key in done_keys:
                        print(f'  {label} p={p:.0e} — SKIPPED (checkpoint)')
                        continue

                    print(f'\n  {label} p={p:.0e} decoder={args.decoder}')
                    stats = run_simulation(
                        circuit, p, args.decoder,
                        args.max_shots, args.max_errors,
                        args.num_workers, args.batch_size,
                    )

                    row = {
                        'd':               d,
                        'state':           state,
                        'rounds_pre':      info['rounds_pre'],
                        'rounds_ls':       info['rounds_ls'],
                        'routing_mult':    mult,
                        'p':               p,
                        'decoder':         args.decoder,
                        'num_qubits':      info['num_qubits'],
                        'num_detectors':   info['num_detectors'],
                        'num_observables': info['num_observables'],
                        'shots':           stats.shots,
                        'errors':          stats.errors,
                        'logical_error_rate': stats.logical_error_rate,
                        'build_time_sec':  round(t_build, 2),
                        'decoding_time_sec': round(stats.seconds, 2),
                    }
                    write_header = not os.path.exists(csv_path)
                    with open(csv_path, 'a', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                        if write_header:
                            w.writeheader()
                        w.writerow(row)

                    print(f'  shots={stats.shots:,}  errors={stats.errors}  '
                          f'LER={stats.logical_error_rate:.2e}  time={stats.seconds:.1f}s')
                    done_keys.add(key)

    print(f'\nDone. Results at {csv_path}')
