"""
Transversal-Gate 7-to-1 |Y⟩ Distillation — sweep over d and p.

Protocol implementation lives in lightstim/protocols/tg_distillation.py.
This script is a CLI driver only.

Usage:
    python benchmarks/logical_circuits/distillation/tg_7to1/TG_distillation_7_to_1.py --build-only
    python benchmarks/logical_circuits/distillation/tg_7to1/TG_distillation_7_to_1.py -d 3 5 7 -p 1e-3
    python benchmarks/logical_circuits/distillation/tg_7to1/TG_distillation_7_to_1.py -d 3 --noise-mode injection --p-injected 1e-3 5e-3
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import stim

from lightstim.protocols.tg_distillation import (
    build_distillation_circuit,
    inject_noise,
    estimate_p_in,
    run_simulation,
    analyze_observables,
    _TG_MAGIC_NAMES,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transversal-gate 7-to-1 |Y⟩ distillation experiment")
    parser.add_argument("-d", "--distances", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("-p", "--p-values", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--p-injected", type=float, nargs="+",
                        default=[1e-3, 2e-3, 5e-3, 1e-2])
    parser.add_argument("--noise-mode", choices=["injection", "full", "both"], default="both")
    parser.add_argument("--rounds-gate", type=int, default=1)
    parser.add_argument("--decoder", choices=["bposd", "mwpf", "pymatching"],
                        default="pymatching")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--max-shots", type=int, default=10_000_000)
    parser.add_argument("--max-errors", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--load-circuits", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    circuit_dir = os.path.join(out_dir, "circuits")
    csv_path = os.path.join(out_dir, "TG_distillation_7_to_1_results.csv")
    all_results = []

    for d in args.distances:
        rounds_init = d
        print(f"\n{'='*60}")
        print(f"Building d={d}, rounds_init={rounds_init}, rounds_gate={args.rounds_gate}")
        print(f"{'='*60}")

        circuit_path = os.path.join(circuit_dir, f"TG_7to1_d{d}_rg{args.rounds_gate}.stim")
        transform_path = os.path.join(circuit_dir,
                                      f"TG_7to1_d{d}_rg{args.rounds_gate}_obs.json")

        if args.load_circuits and os.path.exists(circuit_path) and os.path.exists(transform_path):
            t_build_start = time.perf_counter()
            circuit = stim.Circuit.from_file(circuit_path)
            with open(transform_path) as f:
                obs_info = json.load(f)
            T = np.array(obs_info['T'], dtype=int)
            target_obs = obs_info['target_obs']
            post_select_obs = obs_info['post_select_obs']
            magic_qubits = set(obs_info['magic_qubit_indices'])
            t_build = time.perf_counter() - t_build_start
            circuit_info = {
                'num_qubits': circuit.num_qubits,
                'num_detectors': circuit.num_detectors,
                'num_observables': circuit.num_observables,
                'rounds_gate': args.rounds_gate,
            }
            print(f"Loaded {circuit_info['num_qubits']} qubits, "
                  f"target obs: {target_obs}, PS obs: {post_select_obs}")

        else:
            t_build_start = time.perf_counter()
            circuit, circuit_info, system = build_distillation_circuit(
                d, rounds_init=rounds_init, rounds_gate=args.rounds_gate)
            t_build = time.perf_counter() - t_build_start

            print(f"Circuit: {circuit_info['num_qubits']} qubits, "
                  f"{circuit_info['num_detectors']} det, "
                  f"{circuit_info['num_observables']} obs (built in {t_build:.1f}s)")

            dets, obs = circuit.compile_detector_sampler().sample(100, separate_observables=True)
            print(f"Noiseless check: {'OK' if not np.any(dets) and not np.any(obs) else 'FAIL'}")

            T, target_obs, post_select_obs, _, _ = analyze_observables(
                circuit, system, target_patch_names=['W0'])

            magic_qubits = {q for q, owner in system.index_to_owner_map.items()
                            if owner in _TG_MAGIC_NAMES}

            os.makedirs(circuit_dir, exist_ok=True)
            with open(circuit_path, "w") as f:
                f.write(str(circuit))
            with open(transform_path, "w") as f:
                json.dump({
                    'T': T.tolist(),
                    'target_obs': target_obs,
                    'post_select_obs': post_select_obs,
                    'magic_qubit_indices': sorted(magic_qubits),
                }, f, indent=2)
            print(f"Saved circuit to {circuit_path}")

        if args.build_only:
            continue

        if args.noise_mode == "injection":
            sweep_pairs = [(0.0, p_inj) for p_inj in args.p_injected]
        elif args.noise_mode == "full":
            sweep_pairs = [(p, 0.0) for p in args.p_values]
        else:
            sweep_pairs = [(p, p_inj) for p in args.p_values for p_inj in args.p_injected]

        for p, p_inj in sweep_pairs:
            label = (f"p={p:.1e}, p_injected={p_inj:.1e}"
                     if args.noise_mode != "full" else f"p={p:.1e}")
            print(f"\n--- d={d}, {label}, "
                  f"mode={args.noise_mode}, decoder={args.decoder}, backend={args.backend} ---")

            stats = run_simulation(
                circuit, magic_qubits, p, p_inj, args.noise_mode,
                T, post_select_obs, target_obs, args.decoder,
                args.max_shots, args.max_errors,
                num_workers=args.num_workers, backend=args.backend,
                batch_size=args.batch_size,
            )

            result = {
                'd': d, 'rounds_init': rounds_init, 'rounds_gate': args.rounds_gate,
                'p': p, 'p_injected': p_inj, 'noise_mode': args.noise_mode,
                'decoder': args.decoder,
                'num_qubits': circuit_info['num_qubits'],
                'num_detectors': circuit_info['num_detectors'],
                'num_observables': circuit_info['num_observables'],
                'post_select_obs': post_select_obs,
                'target_obs': target_obs,
                'shots': stats.shots,
                'post_selected_shots': stats.post_selected_shots,
                'post_selection_rate': stats.post_selection_rate,
                'errors': stats.errors,
                'logical_error_rate': stats.logical_error_rate,
                'decoding_time_sec': stats.seconds,
                'build_time_sec': round(t_build, 2),
            }
            all_results.append(result)

            import csv as _csv
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a") as f:
                w = _csv.DictWriter(f, fieldnames=list(result.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(result)

            print(f"  shots={stats.shots:,}, kept={stats.post_selected_shots:,}, "
                  f"PS={stats.post_selection_rate*100:.2f}%  "
                  f"errors={stats.errors}  LER={stats.logical_error_rate:.2e}  "
                  f"time={stats.seconds:.1f}s")

    if all_results:
        json_path = os.path.join(out_dir, "TG_distillation_7_to_1_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {json_path}")
