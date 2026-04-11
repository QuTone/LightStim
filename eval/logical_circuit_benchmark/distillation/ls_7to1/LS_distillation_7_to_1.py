"""
Steane 7-to-1 |Y⟩ Distillation — sweep over d and p.

Noise modes:
    injection   p_injected only on magic-patch resets (W1,W2,W3,W5); circuit otherwise noiseless.
    full        p applied uniformly to all noise channels (default).
    both        p on all channels + p_injected extra on magic-patch resets independently.

Usage:
    # Default full noise sweep:
    python eval/logical_circuit_benchmark/distillation/ls_7to1/LS_distillation_7_to_1.py

    # Injection-only noise:
    python eval/logical_circuit_benchmark/distillation/ls_7to1/LS_distillation_7_to_1.py --noise-mode injection --p-injected 1e-3 5e-3

    # Both modes:
    python eval/logical_circuit_benchmark/distillation/ls_7to1/LS_distillation_7_to_1.py --noise-mode both -p 1e-3 --p-injected 5e-3

    # Custom sweep:
    python eval/logical_circuit_benchmark/distillation/ls_7to1/LS_distillation_7_to_1.py -d 3 5 -p 1e-3 1e-4 1e-5

Outputs detailed CSV + JSON results to eval/logical_circuit_benchmark/distillation/ls_7to1/
"""
import argparse
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from src.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedMultiPatchCoupler,
    UnrotatedSurfaceCodeLogicalOpSet,
)
from src.ir.qec_system import QECSystem
from src.ir.builder import CircuitBuilder
from src.ir.tracker import SyndromeTracker
from src.simulation.decoder_backend.pipeline import SimulationPipeline
from src.simulation.decoder_backend.config import DecoderConfig
from src.noise.config import NoiseConfig
from src.noise.injector import NoiseInjector
from src.noise.rules import FlipAfterResetFiltered
from src.simulation.observable_analysis import (
    build_obs_patch_matrix,
    identify_distillation_observables,
)

import stim

# Magic patch names: the four input |Y⟩ patches
_LS_MAGIC_NAMES = {"W1", "W2", "W3", "W5"}


# =============================================================================
# Circuit construction
# =============================================================================

def build_distillation_circuit(d, rounds, y_prep="fold_transversal_s"):
    """
    Build Steane 7-to-1 distillation circuit (noiseless).

    Args:
        d: Code distance.
        rounds: Number of SE rounds per measurement cycle (typically d).
        y_prep: |Y⟩ state preparation method:
            - "fold_transversal_s": RX + fold-transversal S (deterministic, default)

    Returns:
        (circuit, circuit_info, system) where circuit_info has keys:
            num_qubits, num_detectors, num_observables, corridor_internal_width, y_prep
    """
    # --- Parameterized layout ---
    patch_size = 2 * (d - 1)
    gap = 2 * d + 2
    right_x = patch_size + gap
    y_spacing = gap
    center = patch_size + gap / 2

    patch_layout = {
        'W1': (0, 0),
        'W3': (right_x, 0),
        'W2': (0, y_spacing),
        'W4': (right_x, y_spacing),
        'W5': (0, 2 * y_spacing),
    }

    system = QECSystem()
    for name, offset in patch_layout.items():
        p = UnrotatedSurfaceCode(distance=d)
        p.transpose_coords()
        system.add_patch(p, name=name, offset=offset)

    op_set = UnrotatedSurfaceCodeLogicalOpSet(UnrotatedSurfaceCodeExtractionBlock)

    # Register first coupler upfront
    system.register_coupler(UnrotatedMultiPatchCoupler(),
        patch_names=['W1','W2','W3','W5'], name='meas_1',
        path_axis='vertical', center_axis=center)

    num_qubits = system.num_qubits
    tracker = SyndromeTracker(num_qubits=num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    # --- Helper: prepare |Y⟩ on a patch ---
    def prepare_y(patch_name):
        patch = system.patches[patch_name][0]
        if y_prep == "fold_transversal_s":
            wd = {q: 'X' for q in system.data_indices
                  if system.index_to_owner_map[q] == patch_name}
            builder.initialize(init_dict=wd, n=system.num_qubits)
            op_set.fold_transversal_s(builder, patch)
        else:
            raise ValueError(f"Unknown y_prep method: {y_prep}")

    # --- Step 1: State preparation ---
    for wname in ['W1', 'W2', 'W3']:
        prepare_y(wname)

    # W4: |+⟩ (output register)
    w4d = {q: 'X' for q in system.data_indices
           if system.index_to_owner_map[q] == 'W4'}
    builder.initialize(init_dict=w4d, n=num_qubits)

    # W5: |Y⟩ (reusable ancilla)
    prepare_y('W5')

    # --- Step 2: Pre-coupler syndrome extraction ---
    se = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds)

    # --- Step 3: Four sequential ZZZZ measurements ---
    subsets = [
        ['W1', 'W2', 'W3', 'W5'],
        ['W1', 'W2', 'W4', 'W5'],
        ['W1', 'W3', 'W4', 'W5'],
        ['W2', 'W3', 'W4', 'W5'],
    ]

    for i, subset in enumerate(subsets):
        coupler_name = f'meas_{i+1}'

        if coupler_name not in system.coupler_patches:
            system.register_coupler(UnrotatedMultiPatchCoupler(),
                patch_names=subset, name=coupler_name,
                path_axis='vertical', center_axis=center)
            n = system.num_qubits
            if n > tracker.num_qubits:
                tracker.expand(n - tracker.num_qubits)
            builder.write_coordinates()

        builder.activate_coupler(coupler_name)
        cp = system.coupler_patches[coupler_name]
        cd = sorted([system.local_to_global_map[coupler_name][q]
                     for q in cp.data_indices])
        builder.initialize(init_dict={q: 'X' for q in cd}, n=system.num_qubits)

        se2 = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se2.circuit, rounds=rounds)

        measure_mid = {q: 'X' for q in cd}
        for q in system.data_indices:
            if system.index_to_owner_map.get(q) == 'W5':
                measure_mid[q] = 'X'
        builder.apply_data_readout(final_measurements=measure_mid)

        builder.deactivate_coupler(coupler_name)

        if i < len(subsets) - 1:
            prepare_y('W5')

    # --- Step 4: Final readout ---
    op_set.fold_transversal_s_dag(builder, system.patches['W4'][0], noiseless=True)

    measure_final = {q: 'X' for q in system.data_indices
                     if system.index_to_owner_map.get(q) in ('W1', 'W2', 'W3', 'W4')}
    builder.apply_data_readout(final_measurements=measure_final)

    circuit = builder.circuit
    dem = circuit.detector_error_model(decompose_errors=True)
    circuit_info = {
        'num_qubits': circuit.num_qubits,
        'num_detectors': circuit.num_detectors,
        'num_observables': circuit.num_observables,
        'dem_detectors': dem.num_detectors,
        'dem_observables': dem.num_observables,
        'corridor_internal_width': gap - 1,
        'y_prep': y_prep,
    }
    return circuit, circuit_info, system


# =============================================================================
# Noise injection
# =============================================================================

def inject_noise(circuit, magic_qubits, p, p_injected, mode="full", data_indices=None):
    """
    Inject noise into a clean LS 7-to-1 distillation circuit.

    Args:
        circuit:      Noiseless stim.Circuit from build_distillation_circuit().
        magic_qubits: Set of global qubit indices belonging to magic patches (W1,W2,W3,W5).
                      Obtained from system.index_to_owner_map.
        p:            Circuit-level depolarizing rate (1q, 2q gates, meas, reset).
                      Active in modes 'full' and 'both'.
        p_injected:   Injection noise rate on magic-patch DATA qubit resets only.
                      Active in modes 'injection' and 'both'.
        mode:         'injection' — p_injected on magic DATA qubit resets; circuit otherwise
                                    noiseless. Matches paper (arXiv:2406.17653): Z_ERROR only
                                    on data qubit RX resets (state prep), skipping ancilla
                                    resets in SE rounds.
                      'full'      — p on all noise channels uniformly.
                      'both'      — p on everything + p_injected extra on magic DATA qubit
                                    resets.
        data_indices: Set of global DATA qubit indices for the full system (system.data_indices).
                      When provided, injection noise in 'injection' and 'both' modes is
                      restricted to magic_qubits ∩ data_indices, excluding ancilla qubits.
                      If None, injection noise applies to all magic_qubits (old behaviour).

    Returns:
        Noisy stim.Circuit.
    """
    all_qubits = list(range(circuit.num_qubits))

    # Injection targets: restrict to data qubits only when data_indices is available.
    # Magic DATA qubits only get RX (X-basis) resets, so FlipAfterReset applies
    # Z_ERROR after RX — a phase flip on |+⟩ before fold-transversal-S, which maps
    # directly to a logical input error on |Y⟩. Ancilla resets (RZ in SE rounds)
    # are excluded, matching the paper's model of noisy magic state inputs.
    if data_indices is not None:
        injection_targets = magic_qubits & frozenset(data_indices)
    else:
        injection_targets = magic_qubits

    if mode == "injection":
        cfg = NoiseConfig(p_reset=p_injected)
        inj = NoiseInjector(cfg)
        inj.add_rule(FlipAfterResetFiltered(injection_targets, param_name="p_reset"))
        return inj.inject_noise(circuit)

    elif mode == "full":
        cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p)
        return NoiseInjector.from_circuit_level(cfg, all_qubits).inject_noise(circuit)

    elif mode == "both":
        # Circuit-level p on everything, plus extra p_injected on magic DATA qubit resets.
        cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p,
                          custom_params={"p_injected": p_injected})
        inj = NoiseInjector.from_circuit_level(cfg, all_qubits)
        inj.add_rule(FlipAfterResetFiltered(injection_targets, param_name="p_injected"))
        return inj.inject_noise(circuit)

    else:
        raise ValueError(f"Unknown noise mode: {mode!r}. Choose from 'injection', 'full', 'both'.")


# =============================================================================
# P_in calibration
# =============================================================================

def estimate_p_in(d, rounds, p_injected, p_background=0.0,
                  max_shots=10_000_000, max_errors=100, batch_size=5_000):
    """
    Estimate the effective logical input infidelity P_in for a |Y⟩ magic state
    prepared with physical injection noise rate p_injected.

    Simulates a single-patch calibration circuit:
        RX (data qubits) → fold_transversal_S → SE(rounds)
        → noiseless fold_transversal_S† → MX logical readout

    Noise model mirrors the corresponding distillation noise mode:
        p_background=0   → injection-only  (Z_ERROR on data qubits only)
        p_background>0   → both modes      (circuit-level p_background + extra
                                            Z_ERROR(p_injected) on data qubits)

    Returns:
        p_in (float): estimated logical error rate of the prepared |Y⟩ state.
    """
    patch = UnrotatedSurfaceCode(distance=d)
    patch.transpose_coords()
    sys1 = QECSystem()
    sys1.add_patch(patch, name='cal', offset=(0, 0))

    op_set = UnrotatedSurfaceCodeLogicalOpSet(UnrotatedSurfaceCodeExtractionBlock)
    tracker = SyndromeTracker(num_qubits=sys1.num_qubits,
                              expected_num_logicals=sys1.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=sys1, if_detector=True)
    builder.write_coordinates()

    # |Y⟩ = S_L|+⟩_L preparation
    wd = {q: 'X' for q in sys1.data_indices}
    builder.initialize(init_dict=wd, n=sys1.num_qubits)
    op_set.fold_transversal_s(builder, sys1.patches['cal'][0])

    # SE round(s): partially correct preparation errors
    se = UnrotatedSurfaceCodeExtractionBlock(sys1)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds)

    # Noiseless S†_L + MX: |Y⟩ → |+⟩_L; logical Z error on |Y⟩ becomes logical X error here
    op_set.fold_transversal_s_dag(builder, sys1.patches['cal'][0], noiseless=True)
    builder.apply_data_readout(final_measurements={q: 'X' for q in sys1.data_indices})

    circuit = builder.circuit
    all_qubits = list(range(circuit.num_qubits))

    if p_background > 0:
        # Both-mode calibration: circuit-level background + extra injection on data qubits
        cfg = NoiseConfig(p_1q=p_background, p_2q=p_background,
                          p_meas=p_background, p_reset=p_background,
                          custom_params={"p_injected": p_injected})
        inj = NoiseInjector.from_circuit_level(cfg, all_qubits)
        inj.add_rule(FlipAfterResetFiltered(set(sys1.data_indices), param_name="p_injected"))
    else:
        # Injection-only calibration: Z_ERROR on data-qubit RX resets only
        cfg = NoiseConfig(p_reset=p_injected)
        inj = NoiseInjector(cfg)
        inj.add_rule(FlipAfterResetFiltered(set(sys1.data_indices), param_name="p_reset"))

    noisy = inj.inject_noise(circuit)

    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("pymatching"),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        target_observable_indices=[0],
        print_progress=False,
    )
    return pipeline.run(noisy).logical_error_rate


# =============================================================================
# Simulation
# =============================================================================

def run_simulation(circuit, magic_qubits, p, p_injected, mode,
                   ps_obs, target_obs,
                   decoder_name="pymatching",
                   max_shots=10_000_000, max_errors=100,
                   batch_size=50_000, num_workers=1,
                   data_indices=None):
    """
    Run noisy LS simulation with post-decode post-selection.

    Args:
        circuit:      Noiseless stim.Circuit from build_distillation_circuit().
        magic_qubits: Magic qubit index set (data + ancilla of W1/W2/W3/W5).
        p:            Circuit-level error rate.
        p_injected:   Injection noise rate on magic DATA qubit resets.
        mode:         'injection', 'full', or 'both'.
        ps_obs:       Observable indices to post-select on (corrected == 0 after decoding).
        target_obs:   Observable index to measure LER on.
        decoder_name: Decoder to use.
        data_indices: system.data_indices — restricts injection noise to data qubits only.
    """
    noisy = inject_noise(circuit, magic_qubits, p, p_injected, mode,
                         data_indices=data_indices)

    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig(decoder_name, backend="cpu"),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        num_workers=num_workers,
        post_select_corrected_observable_indices=ps_obs,
        target_observable_indices=[target_obs[0]],
        print_progress=True,
        progress_interval_sec=30.0,
    )
    return pipeline.run(noisy)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Steane 7-to-1 |Y⟩ distillation experiment")
    parser.add_argument("-d", "--distances", type=int, nargs="+",
                        default=[3, 5, 7], help="Code distances (default: 3 5 7)")
    parser.add_argument("-p", "--p-values", type=float, nargs="+",
                        default=[1e-3],
                        help="Circuit-level error rates (used in 'full' and 'both' modes)")
    parser.add_argument("--p-injected", type=float, nargs="+",
                        default=[1e-3, 2e-3, 5e-3, 1e-2],
                        help="Injection noise rates on magic-patch resets "
                             "(used in 'injection' and 'both' modes)")
    parser.add_argument("--noise-mode", choices=["injection", "full", "both"],
                        default="both",
                        help="Noise model: injection (p_injected only on magic resets), "
                             "full (p everywhere), both (p + p_injected independently). "
                             "Default: both.")
    parser.add_argument("--y-prep", choices=["fold_transversal_s"],
                        default="fold_transversal_s",
                        help="Y state preparation method (default: fold_transversal_s)")
    parser.add_argument("--decoder", choices=["bposd", "mwpf", "pymatching"],
                        default="pymatching", help="Decoder (default: pymatching)")
    parser.add_argument("--max-shots", type=int, default=10_000_000,
                        help="Max shots per run (default: 10M)")
    parser.add_argument("--max-errors", type=int, default=100,
                        help="Max errors for early stopping (default: 100)")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--build-only", action="store_true",
                        help="Only build the circuit (no simulation)")
    parser.add_argument("--load-circuits", action="store_true",
                        help="Load pre-built circuits from eval/logical_circuit_benchmark/distillation/ls_7to1/circuits/ "
                             "instead of building")
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
            # Fall back to magic_qubits for JSON files saved before this field was added
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
            print(f"Circuit: {circuit_info['num_qubits']} qubits, "
                  f"{circuit_info['num_detectors']} det, "
                  f"{circuit_info['num_observables']} obs "
                  f"(loaded from {circuit_path} in {t_build:.2f}s)")
            print(f"  Target obs: {target_obs}, Post-select obs: {ps_obs}")

        else:
            t_build_start = time.perf_counter()
            circuit, circuit_info, system = build_distillation_circuit(d, rounds, args.y_prep)
            t_build = time.perf_counter() - t_build_start

            print(f"Circuit: {circuit_info['num_qubits']} qubits, "
                  f"{circuit_info['num_detectors']} det, "
                  f"{circuit_info['num_observables']} obs "
                  f"(built in {t_build:.1f}s)")

            # Noiseless verification
            dets, obs = circuit.compile_detector_sampler().sample(
                shots=100, separate_observables=True)
            noiseless_ok = not np.any(dets) and not np.any(obs)
            print(f"Noiseless check: {'OK' if noiseless_ok else 'FAIL'}")

            # Observable analysis (GF(2) elimination targeting W4)
            matrix, patch_names = build_obs_patch_matrix(circuit, system)
            _, target_obs, ps_obs = identify_distillation_observables(
                matrix, patch_names, ["W4"])
            print(f"  Target obs: {target_obs}, Post-select obs: {ps_obs}")

            # Magic qubit indices (data + ancilla) and data-only subset
            magic_qubits = {q for q, owner in system.index_to_owner_map.items()
                            if owner in _LS_MAGIC_NAMES}
            magic_data_qubits = magic_qubits & system.data_indices

            # Save circuit + transform for reuse
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
            print("(build-only mode, skipping simulation)")
            continue

        # Build sweep pairs based on noise mode
        if args.noise_mode == "injection":
            sweep_pairs = [(0.0, p_inj) for p_inj in args.p_injected]
        elif args.noise_mode == "full":
            sweep_pairs = [(p, 0.0) for p in args.p_values]
        else:  # "both"
            sweep_pairs = [(p, p_inj)
                           for p in args.p_values
                           for p_inj in args.p_injected]

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
                'd': d,
                'rounds': rounds,
                'p': p,
                'p_injected': p_inj,
                'noise_mode': args.noise_mode,
                'y_prep': args.y_prep,
                'decoder': args.decoder,
                'num_qubits': circuit_info['num_qubits'],
                'num_detectors': circuit_info['num_detectors'],
                'num_observables': circuit_info['num_observables'],
                'corridor_width': circuit_info['corridor_internal_width'],
                'shots': stats.shots,
                'post_selected_shots': stats.post_selected_shots,
                'post_selection_rate': stats.post_selection_rate,
                'errors': stats.errors,
                'logical_error_rate': stats.logical_error_rate,
                'decoding_time_sec': stats.seconds,
                'decoder': stats.decoder,
                'build_time_sec': round(t_build, 2),
            }
            all_results.append(result)

            # Append immediately so kill/OOM never loses this result
            keys = list(result.keys())
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a") as f:
                import csv as _csv2
                w = _csv2.DictWriter(f, fieldnames=keys)
                if write_header:
                    w.writeheader()
                w.writerow(result)

            print(f"  shots={stats.shots:,}, kept={stats.post_selected_shots:,}, "
                  f"PS_rate={stats.post_selection_rate*100:.2f}%")
            print(f"  errors={stats.errors}, LER={stats.logical_error_rate:.2e}, "
                  f"time={stats.seconds:.1f}s")

    # Save JSON
    json_path = os.path.join(out_dir, f"LS_distillation_7_to_1_results{suffix}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {json_path}")

    # Save CSV (csv_path defined before the loop, above)
    if all_results:
        keys = all_results[0].keys()
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in all_results:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"Saved {csv_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"SUMMARY (y_prep={args.y_prep}, mode={args.noise_mode}, decoder={args.decoder})")
    print(f"{'='*100}")
    print(f"{'d':>3} {'p':>8} {'p_inj':>8} {'qubits':>7} {'detectors':>10} "
          f"{'shots':>12} {'kept':>12} {'PS_rate':>8} {'errors':>7} {'LER':>10} {'time':>7}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['d']:>3} {r['p']:>8.0e} {r['p_injected']:>8.0e} "
              f"{r['num_qubits']:>7} {r['num_detectors']:>10} "
              f"{r['shots']:>12,} {r['post_selected_shots']:>12,} "
              f"{r['post_selection_rate']*100:>7.2f}% {r['errors']:>7} "
              f"{r['logical_error_rate']:>10.2e} {r['decoding_time_sec']:>6.1f}s")
