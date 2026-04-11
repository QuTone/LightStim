"""
TG 7-to-1 Distillation — No-Injection Verification Circuit.

Simplified version of TG_distillation_7_to_1.py that replaces magic state
teleportation with direct fold-transversal S gates:
  - W1..W7: fold-transversal S  (simulates teleportation outcome with perfect |Y>)
  - W0:     fold-transversal S_dag (output correction)

All qubits measured in X basis at the end.

Purpose:
    Verify that the hypercube [[7,1,3]] encoding + transversal S produces the
    correct 4-observable structure WITHOUT any Y state injection noise, isolating
    injection-independent circuit bugs.

Expected:
    - 4 observables (all W patches only, no M patches)
    - Noiseless check: all observables = 0
    - With noise (circuit-level): LER should suppress as ~p^3 with d

Usage:
    python eval/logical_circuit_benchmark/distillation/tg_7to1/TG_distillation_no_inject.py --build-only
    python eval/logical_circuit_benchmark/distillation/tg_7to1/TG_distillation_no_inject.py -d 3 5 7 -p 1e-3
"""
import argparse
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import stim
from src.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedSurfaceCodeLogicalOpSet,
)
from src.qec_code.surface_code.unrotated.operation import _get_fold_yx_pairs
from src.ir.qec_system import QECSystem
from src.ir.builder import CircuitBuilder
from src.ir.tracker import SyndromeTracker
from src.simulation.decoder_backend.pipeline import SimulationPipeline
from src.simulation.decoder_backend.config import DecoderConfig
from src.noise.config import NoiseConfig
from src.noise.injector import NoiseInjector
from src.simulation.observable_analysis import (
    build_obs_patch_matrix,
    identify_distillation_observables,
    transform_observables,
)


# =============================================================================
# Helpers
# =============================================================================

def _apply_fold_s(circuit: stim.Circuit, system, patch, dag: bool = False):
    """
    Append fold-transversal S (or S_dag if dag=True) for the given patch.

    For S:     diagonal-even → S,    diagonal-odd → S_dag, mirror pairs → CZ
    For S_dag: diagonal-even → S_dag, diagonal-odd → S,   mirror pairs → CZ
    """
    diag_s, diag_sdag, mirror_pairs = _get_fold_yx_pairs(system, patch)
    if dag:
        diag_s, diag_sdag = diag_sdag, diag_s  # swap for S_dag
    if diag_s:
        circuit.append("S", sorted(diag_s))
    if diag_sdag:
        circuit.append("S_DAG", sorted(diag_sdag))
    for a, b in mirror_pairs:
        circuit.append("CZ", [a, b])


# =============================================================================
# Circuit construction
# =============================================================================

def build_no_inject_circuit(d, rounds, r=1):
    """
    Build the no-injection distillation verification circuit.

    Steps:
        1. Initialize 8 working patches W0-W7 (same init as full distillation)
        2. d rounds SE (initialization)
        3. Hypercube CNOT encoding: 3 ticks of transversal CNOTs, r SE each
        4. Apply fold-transversal S on W1-W7 + fold-transversal S_dag on W0
           (all in one unitary block, so noise injection applies uniformly)
        5. r SE rounds
        6. Measure all data qubits in X basis

    Args:
        d:      Code distance.
        rounds: SE rounds after initialization (typically d).
        r:      SE rounds after each gate layer.

    Returns:
        (circuit, circuit_info, system)
    """
    patch_size = 2 * (d - 1)
    gap = 2
    col_sp = patch_size + gap
    row_sp = patch_size + gap

    working_layout = {
        'W0': (0,      0),
        'W1': (0,      row_sp),
        'W2': (col_sp, 0),
        'W3': (col_sp, row_sp),
        'W4': (0,      2 * row_sp),
        'W5': (0,      3 * row_sp),
        'W6': (col_sp, 2 * row_sp),
        'W7': (col_sp, 3 * row_sp),
    }

    system = QECSystem()
    gp = {}
    for name, offset in working_layout.items():
        p = UnrotatedSurfaceCode(distance=d)
        gp[name] = system.add_patch(p, name=name, offset=offset)

    lp = {name: system.patches[name][0] for name in working_layout}

    num_qubits = system.num_qubits
    tracker = SyndromeTracker(
        num_qubits=num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    def do_se(n_rounds):
        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=n_rounds)

    # --- Step 1: Initialize W patches ---
    x_patches = {'W0', 'W1', 'W2', 'W4'}   # |+>
    z_patches = {'W3', 'W5', 'W6', 'W7'}   # |0>

    init_dict = {}
    for name in x_patches:
        for q in system.data_indices:
            if system.index_to_owner_map[q] == name:
                init_dict[q] = 'X'
    for name in z_patches:
        for q in system.data_indices:
            if system.index_to_owner_map[q] == name:
                init_dict[q] = 'Z'
    builder.initialize(init_dict=init_dict, n=num_qubits)

    # --- Step 2: Initialization SE ---
    print(f"  Init SE ({rounds} rounds)...")
    do_se(rounds)

    # --- Step 3: Hypercube CNOT encoding ---
    cnot_ticks = [
        [('W0', 'W4'), ('W1', 'W5'), ('W2', 'W6'), ('W3', 'W7')],
        [('W0', 'W2'), ('W1', 'W3'), ('W4', 'W6'), ('W5', 'W7')],
        [('W0', 'W1'), ('W2', 'W3'), ('W4', 'W5'), ('W6', 'W7')],
    ]
    for tick_idx, tick in enumerate(cnot_ticks):
        cnot_circuit = stim.Circuit()
        for ctrl_name, tgt_name in tick:
            c_qubits = sorted(gp[ctrl_name].data_indices)
            t_qubits = sorted(gp[tgt_name].data_indices)
            targets = []
            for c, t in zip(c_qubits, t_qubits):
                targets.extend([c, t])
            cnot_circuit.append("CNOT", targets)
        builder.apply_unitary_block(cnot_circuit)
        print(f"  CNOT tick {tick_idx+1} + SE ({r} rounds)...")
        do_se(r)

    # --- Step 4: Transversal S on W1-W7, S_dag on W0 (all noisy) ---
    # Simulates the teleportation outcome when all M patches are perfect |Y>.
    s_block = stim.Circuit()
    for i in range(1, 8):
        _apply_fold_s(s_block, system, lp[f'W{i}'], dag=False)  # S on W1..W7
    _apply_fold_s(s_block, system, lp['W0'], dag=True)           # S_dag on W0
    builder.apply_unitary_block(s_block)
    print(f"  Transversal S (W1-W7) + S_dag (W0) + SE ({r} rounds)...")
    do_se(r)

    # --- Step 5: Measure all W patches in X ---
    final_meas = {q: 'X' for q in system.data_indices}
    builder.apply_data_readout(final_measurements=final_meas)

    circuit = builder.circuit
    circuit_info = {
        'num_qubits': circuit.num_qubits,
        'num_detectors': circuit.num_detectors,
        'num_observables': circuit.num_observables,
        'r': r,
    }
    return circuit, circuit_info, system


# =============================================================================
# Observable analysis (same as v2)
# =============================================================================

def analyze_observables(circuit, system, target_patch_names=None):
    if target_patch_names is None:
        target_patch_names = ['W0']

    matrix, patch_names = build_obs_patch_matrix(circuit, system)
    T, target, ps = identify_distillation_observables(matrix, patch_names, target_patch_names)

    print(f"  Observables: {circuit.num_observables}")
    print(f"  Obs-to-patch matrix:")
    for i in range(matrix.shape[0]):
        involved = [patch_names[j] for j in range(len(patch_names)) if matrix[i, j]]
        print(f"    L{i}: {involved}")

    M_new = (T @ matrix) % 2
    print(f"  After GF(2) elimination (target={target_patch_names}):")
    for i in range(M_new.shape[0]):
        involved = [patch_names[j] for j in range(len(patch_names)) if M_new[i, j]]
        label = 'TARGET' if i in target else 'PS'
        print(f"    L{i}': {involved} [{label}]")

    print(f"  -> Target obs: {target}, Post-select obs: {ps}")
    return T, target, ps, matrix, patch_names


# =============================================================================
# Simulation
# =============================================================================

def run_simulation(circuit, p, T, ps_indices, target_indices, decoder_name,
                   max_shots=10_000_000, max_errors=200,
                   num_workers=32, backend="cpu", batch_size=50_000):
    noise_config = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    injector = NoiseInjector.from_circuit_level(noise_config, list(range(circuit.num_qubits)))
    noisy = injector.inject_noise(circuit)

    n_obs = circuit.num_observables
    n_det = circuit.num_detectors

    if np.array_equal(T, np.eye(T.shape[0], dtype=int)):
        pipeline = SimulationPipeline(
            decoder_config=DecoderConfig(decoder_name, backend=backend),
            max_shots=max_shots,
            max_errors=max_errors,
            batch_size=batch_size,
            num_workers=num_workers,
            post_select_observable_indices=ps_indices,
            target_observable_indices=target_indices,
            print_progress=True,
            progress_interval_sec=30.0,
        )
        return pipeline.run(noisy)

    import math
    from src.simulation.decoder_backend.registry import get_decoder

    decoder = get_decoder(decoder_name, backend=backend)
    dem = noisy.detector_error_model(
        decompose_errors=getattr(decoder, "decompose_errors", False),
        approximate_disjoint_errors=True,
    )
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    sampler = noisy.compile_detector_sampler()

    total_shots = 0
    kept_shots = 0
    errors = 0
    t0 = time.perf_counter()

    while total_shots < max_shots and errors < max_errors:
        dets, obs = sampler.sample(shots=batch_size, separate_observables=True)
        total_shots += batch_size
        obs_t = transform_observables(obs, T)
        mask = np.all(obs_t[:, ps_indices] == 0, axis=1)
        if not np.any(mask):
            elapsed = time.perf_counter() - t0
            print(f"shots={total_shots:,} kept={kept_shots:,} errors={errors} "
                  f"LER={errors/max(kept_shots,1):.2e} elapsed={elapsed:.1f}s")
            continue

        dets_kept = dets[mask]
        obs_t_kept = obs_t[mask]
        kept_shots += dets_kept.shape[0]

        n_det_bytes = math.ceil(n_det / 8)
        dets_packed = np.packbits(dets_kept, axis=1, bitorder="little")[:, :n_det_bytes]
        pred_packed = compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=dets_packed
        )
        pred_unpacked = np.unpackbits(pred_packed, axis=1, bitorder="little")[:, :n_obs]
        pred_t = transform_observables(pred_unpacked, T)

        corrected = (obs_t_kept[:, target_indices] ^ pred_t[:, target_indices])
        errors += int(np.sum(np.any(corrected, axis=1)))

        elapsed = time.perf_counter() - t0
        ler = errors / kept_shots if kept_shots > 0 else 0
        print(f"shots={total_shots:,} kept={kept_shots:,} errors={errors} "
              f"LER={ler:.2e} elapsed={elapsed:.1f}s")

    class _Stats:
        pass
    stats = _Stats()
    stats.shots = total_shots
    stats.post_selected_shots = kept_shots
    stats.post_selection_rate = kept_shots / total_shots if total_shots > 0 else 0
    stats.errors = errors
    stats.logical_error_rate = errors / kept_shots if kept_shots > 0 else 0
    stats.seconds = time.perf_counter() - t0
    stats.decoder = decoder_name
    return stats


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TG distillation no-injection verification circuit")
    parser.add_argument("-d", "--distances", type=int, nargs="+",
                        default=[3, 5, 7])
    parser.add_argument("-p", "--p-values", type=float, nargs="+",
                        default=[1e-3])
    parser.add_argument("-r", "--gate-se-rounds", type=int, default=1)
    parser.add_argument("--decoder", choices=["bposd", "mwpf", "pymatching"],
                        default="bposd")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--max-shots", type=int, default=10_000_000)
    parser.add_argument("--max-errors", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "TG_no_inject_results.csv")
    circuit_dir = os.path.join(out_dir, "circuits_no_inject")
    os.makedirs(circuit_dir, exist_ok=True)

    done_keys = set()
    if os.path.exists(csv_path):
        import csv as _csv
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                done_keys.add((int(row["d"]), float(row["p"]),
                               row.get("decoder", ""), int(row.get("r", 1))))
        print(f"Checkpoint: {len(done_keys)} tasks already done.")

    all_results = []

    for d in args.distances:
        rounds = d
        print(f"\n{'='*60}")
        print(f"Building d={d}, init_rounds={rounds}, r={args.gate_se_rounds}")
        print(f"{'='*60}")

        circuit_path = os.path.join(circuit_dir, f"TG_noinject_d{d}_r{args.gate_se_rounds}.stim")
        obs_path = os.path.join(circuit_dir, f"TG_noinject_d{d}_r{args.gate_se_rounds}_obs.json")

        t_build_start = time.perf_counter()
        circuit, circuit_info, system = build_no_inject_circuit(
            d, rounds, r=args.gate_se_rounds)
        t_build = time.perf_counter() - t_build_start

        print(f"Circuit: {circuit_info['num_qubits']} qubits, "
              f"{circuit_info['num_detectors']} det, "
              f"{circuit_info['num_observables']} obs "
              f"(built in {t_build:.1f}s)")

        dets, obs = circuit.compile_detector_sampler().sample(
            shots=100, separate_observables=True)
        noiseless_ok = not np.any(dets) and not np.any(obs)
        print(f"Noiseless check: {'OK' if noiseless_ok else 'FAIL'}")
        if not noiseless_ok:
            print(f"  WARNING: {np.sum(dets)} detector triggers, {np.sum(obs)} obs triggers")

        T, target_obs, post_select_obs, _, _ = analyze_observables(
            circuit, system, target_patch_names=['W0'])

        with open(circuit_path, "w") as f:
            f.write(str(circuit))
        with open(obs_path, "w") as f:
            json.dump({'T': T.tolist(), 'target_obs': target_obs,
                       'post_select_obs': post_select_obs}, f, indent=2)
        print(f"Saved {circuit_path}")

        if args.build_only:
            continue

        for p in args.p_values:
            if (d, p, args.decoder, args.gate_se_rounds) in done_keys:
                print(f"  d={d}, p={p:.0e} — SKIPPED (checkpoint)")
                continue
            print(f"\n--- d={d}, p={p:.0e}, decoder={args.decoder}, backend={args.backend} ---")
            stats = run_simulation(
                circuit, p, T, post_select_obs, target_obs, args.decoder,
                args.max_shots, args.max_errors,
                num_workers=args.num_workers, backend=args.backend,
                batch_size=args.batch_size,
            )

            result = {
                'd': d, 'rounds': rounds, 'r': args.gate_se_rounds,
                'p': p, 'decoder': args.decoder,
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

            import csv as _csv2
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a") as f:
                w = _csv2.DictWriter(f, fieldnames=list(result.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(result)

            print(f"  shots={stats.shots:,}, kept={stats.post_selected_shots:,}, "
                  f"PS_rate={stats.post_selection_rate*100:.2f}%")
            print(f"  errors={stats.errors}, LER={stats.logical_error_rate:.2e}, "
                  f"time={stats.seconds:.1f}s")

    if all_results:
        print(f"\n{'='*80}")
        print(f"SUMMARY (r={args.gate_se_rounds}, decoder={args.decoder})")
        print(f"{'='*80}")
        print(f"{'d':>3} {'p':>8} {'det':>6} {'obs':>4} {'shots':>12} "
              f"{'kept':>12} {'PS%':>8} {'errors':>7} {'LER':>10}")
        print("-" * 80)
        for r in all_results:
            print(f"{r['d']:>3} {r['p']:>8.0e} {r['num_detectors']:>6} "
                  f"{r['num_observables']:>4} {r['shots']:>12,} "
                  f"{r['post_selected_shots']:>12,} "
                  f"{r['post_selection_rate']*100:>7.2f}% {r['errors']:>7} "
                  f"{r['logical_error_rate']:>10.2e}")
