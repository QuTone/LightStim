"""
Two-Qubit ZZ Lattice Surgery Teleportation.

Protocol:
    patch1 = |ψ⟩_L,  patch2 = |+⟩_L (X basis)
    ZZ merge for d rounds  →  measure patch1 in X  →  patch2 holds |ψ⟩_L

Two sub-experiments:
    teleport_Z : init patch1=Z, measure patch2=Z  (Z-eigenstate teleportation)
    teleport_X : init patch1=X, measure patch2=X  (X-eigenstate teleportation)

The strong Z/X LER asymmetry is expected here because:
  - ZZ coupler detects Z errors during the merge window.
  - X errors on the ancilla only become detectable at the final X measurement.
  → Different detector-chain depths for Z and X logical channels.

Layout: patch1 at (0,0) rotated π, patch2 at (0, 2d) — vertical ZZ coupling.

Usage:
    venv/bin/python eval/logical_circuit_benchmark/two-qubit-teleportation/run_zz_tele.py
    venv/bin/python eval/logical_circuit_benchmark/two-qubit-teleportation/run_zz_tele.py -d 3 5 -p 5e-4 1e-3 2e-3 5e-3 1e-2
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector
from lightstim.qec_code.surface_code.unrotated import UnrotatedSurfaceCodeExtractionBlock
from lightstim.simulation.decoder_backend.config import DecoderConfig
from lightstim.simulation.decoder_backend.pipeline import SimulationPipeline

# Sub-experiments: (label, init_patch1, meas_patch2)
# patch2 always init X (|+⟩ for ZZ), patch1 always measured X (discard)
SUB_EXPERIMENTS = [
    ("teleport_Z", "Z", "Z"),
    ("teleport_X", "X", "X"),
]


def build_circuit(d: int, init_p1: str, meas_p2: str):
    """Build ZZ-LS teleportation circuit."""
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        exp = TwoPatchLSExperiment(
            patch1_config={"distance": d},
            patch2_config={"distance": d},
            offset=(0, 2 * d),           # vertical ZZ coupling
            interaction_type="ZZ",
            initial_state_patch1=init_p1,
            initial_state_patch2="X",    # |+⟩ ancilla for ZZ
            measure_state_patch1="X",    # discard patch1
            measure_state_patch2=meas_p2,
            extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
            rounds=d,                    # d rounds pre + d rounds during
            rotate_patch1=True,
        )
        circuit = exp.build()

    info = {
        "num_qubits":     circuit.num_qubits,
        "num_detectors":  circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "rounds": d,
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
        decoder_config=DecoderConfig(decoder_name,
                                     backend='gpu' if decoder_name == 'nv-qldpc-decoder' else 'cpu'),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        num_workers=num_workers,
        print_progress=True,
        progress_interval_sec=30.0,
    )
    return pipeline.run(noisy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ZZ-LS two-qubit teleportation sweep')
    parser.add_argument('-d', '--distances',  type=int,   nargs='+', default=[3, 5, 7])
    parser.add_argument('-p', '--p-values',   type=float, nargs='+',
                        default=[5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    parser.add_argument('--decoder',  choices=['pymatching', 'bposd', 'mwpf', 'nv-qldpc-decoder'],
                        default='pymatching')
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--max-shots',   type=int, default=1_000_000_000)
    parser.add_argument('--max-errors',  type=int, default=100)
    parser.add_argument('--batch-size',  type=int, default=10_000)
    parser.add_argument('--build-only',  action='store_true')
    args = parser.parse_args()

    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'zz_tele_results.csv')

    # Checkpoint
    done_keys = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done_keys.add((int(row['d']), float(row['p']), row['state'], row['decoder']))
        print(f'Checkpoint: {len(done_keys)} tasks done, skipping.')

    FIELDNAMES = [
        'd', 'state', 'rounds',
        'p', 'decoder',
        'num_qubits', 'num_detectors', 'num_observables',
        'shots', 'errors', 'logical_error_rate',
        'build_time_sec', 'decoding_time_sec',
    ]

    for d in args.distances:
        for label, init_p1, meas_p2 in SUB_EXPERIMENTS:
            print(f'\n{"="*60}')
            print(f'Building d={d}, state={label}')
            print(f'{"="*60}')

            t0 = time.perf_counter()
            circuit, info = build_circuit(d, init_p1, meas_p2)
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
                key = (d, p, label, args.decoder)
                if key in done_keys:
                    print(f'  d={d} {label} p={p:.0e} — SKIPPED (checkpoint)')
                    continue

                print(f'\n  d={d} {label} p={p:.0e} decoder={args.decoder}')
                stats = run_simulation(
                    circuit, p, args.decoder,
                    args.max_shots, args.max_errors,
                    args.num_workers, args.batch_size,
                )

                row = {
                    'd': d, 'state': label, 'rounds': info['rounds'],
                    'p': p, 'decoder': args.decoder,
                    'num_qubits': info['num_qubits'],
                    'num_detectors': info['num_detectors'],
                    'num_observables': info['num_observables'],
                    'shots': stats.shots, 'errors': stats.errors,
                    'logical_error_rate': stats.logical_error_rate,
                    'build_time_sec': round(t_build, 2),
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
