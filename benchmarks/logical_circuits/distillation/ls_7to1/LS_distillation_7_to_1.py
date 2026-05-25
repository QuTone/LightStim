"""
Steane 7-to-1 |Y⟩ Distillation — sweep over d and p.

Protocol implementation lives in lightstim/protocols/ls_distillation.py.
This script is a CLI driver only.

Usage:
    python benchmarks/logical_circuits/distillation/ls_7to1/LS_distillation_7_to_1.py
    python benchmarks/logical_circuits/distillation/ls_7to1/LS_distillation_7_to_1.py --noise-mode injection --p-injected 1e-3 5e-3
    python benchmarks/logical_circuits/distillation/ls_7to1/LS_distillation_7_to_1.py -d 3 5 -p 1e-3 1e-4 1e-5
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import stim

from lightstim.protocols.ls_distillation import (
    build_distillation_circuit,
    inject_noise,
    estimate_p_in,
    run_simulation,
    _LS_MAGIC_NAMES,
)
from lightstim.simulation.observable_analysis import (
    build_obs_patch_matrix,
    identify_distillation_observables,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steane 7-to-1 |Y⟩ distillation experiment")
    parser.add_argument("-d", "--distances", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("-p", "--p-values", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--p-injected", type=float, nargs="+",
                        default=[1e-3, 2e-3, 5e-3, 1e-2])
    parser.add_argument("--noise-mode", choices=["injection", "full", "both"], default="both")
    parser.add_argument("--y-prep", choices=["fold_transversal_s"],
                        default="fold_transversal_s")
    parser.add_argument("--decoder", choices=["bposd", "mwpf", "pymatching"],
                        default="pymatching")
    parser.add_argument("--max-shots", type=int, default=10_000_000)
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--load-circuits", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    circuit_dir = os.path.join(out_dir, "circuits")
    suffix = f"_{args.y_prep}" if args.y_prep != "fold_transversal_s" else ""
    csv_path = os.path.join(out_dir, f"LS_distillation_7_to_1_results{suffix}.csv")
    all_results = []

    for d in args.distances:
        rounds = d
        print(f"\n{'='*60}")
        print(f"Building d={d}, rounds={rounds}, y_prep={args.y_prep}")
        print(f"{'='*60}")

        circuit_path = os.path.join(circuit_dir, f"LS_7to1_d{d}{suffix}.stim")
        transform_path = os.path.join(circuit_dir, f"LS_7to1_d{d}{suffix}_obs.json")

        if args.load_circuits and os.path.exists(circuit_path) and os.path.exists(transform_path):
            t_build_start = time.perf_counter()
            circuit = stim.Circuit.from_file(circuit_path)
            with open(transform_path) as f:
                obs_info = json.load(f)
            ps_obs = obs_info['ps_obs']
            target_obs = obs_info['target_obs']
            magic_qubits = set(obs_info['magic_qubit_indices'])
            magic_data_qubits = set(obs_info.get('magic_data_qubit_indices',
                                                  obs_info['magic_qubit_indices']))
            t_build = time.perf_counter() - t_build_start
            circuit_info = {
                'num_qubits': circuit.num_qubits,
                'num_detectors': circuit.num_detectors,
                'num_observables': circuit.num_observables,
                'corridor_internal_width': obs_info.get('corridor_internal_width', '?'),
                'y_prep': args.y_prep,
            }
            print(f"Loaded {circuit_info['num_qubits']} qubits, "
                  f"target obs: {target_obs}, PS obs: {ps_obs}")

        else:
            t_build_start = time.perf_counter()
            circuit, circuit_info, system = build_distillation_circuit(d, rounds, args.y_prep)
            t_build = time.perf_counter() - t_build_start

            print(f"Circuit: {circuit_info['num_qubits']} qubits, "
                  f"{circuit_info['num_detectors']} det, "
                  f"{circuit_info['num_observables']} obs (built in {t_build:.1f}s)")

            dets, obs = circuit.compile_detector_sampler().sample(100, separate_observables=True)
            print(f"Noiseless check: {'OK' if not np.any(dets) and not np.any(obs) else 'FAIL'}")

            matrix, patch_names = build_obs_patch_matrix(circuit, system)
            _, target_obs, ps_obs = identify_distillation_observables(
                matrix, patch_names, ["W4"])
            print(f"  Target obs: {target_obs}, Post-select obs: {ps_obs}")

            magic_qubits = {q for q, owner in system.index_to_owner_map.items()
                            if owner in _LS_MAGIC_NAMES}
            magic_data_qubits = magic_qubits & system.data_indices

            os.makedirs(circuit_dir, exist_ok=True)
            with open(circuit_path, "w") as f:
                f.write(str(circuit))
            with open(transform_path, "w") as f:
                json.dump({
                    'ps_obs': ps_obs,
                    'target_obs': target_obs,
                    'magic_qubit_indices': sorted(magic_qubits),
                    'magic_data_qubit_indices': sorted(magic_data_qubits),
                    'corridor_internal_width': circuit_info['corridor_internal_width'],
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
            print(f"\n--- d={d}, {label}, mode={args.noise_mode}, decoder={args.decoder} ---")

            stats = run_simulation(
                circuit, magic_qubits, p, p_inj, args.noise_mode,
                ps_obs, target_obs, args.decoder,
                args.max_shots, args.max_errors, args.batch_size,
                data_indices=magic_data_qubits,
            )

            result = {
                'd': d, 'rounds': rounds, 'p': p, 'p_injected': p_inj,
                'noise_mode': args.noise_mode, 'y_prep': args.y_prep,
                'decoder': args.decoder,
                'num_qubits': circuit_info['num_qubits'],
                'num_detectors': circuit_info['num_detectors'],
                'num_observables': circuit_info['num_observables'],
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
        json_path = os.path.join(out_dir, f"LS_distillation_7_to_1_results{suffix}.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {json_path}")
