"""
Steane 7-to-1 |Y⟩ Distillation — sweep over d and p.

Usage:
    # Default (fold-transversal S for |Y⟩ preparation):
    python eval/LS_distillation/LS_distillation_7_to_1.py

    # With corner state injection for |Y⟩ preparation:
    python eval/LS_distillation/LS_distillation_7_to_1.py --y-prep injection

    # Custom sweep:
    python eval/LS_distillation/LS_distillation_7_to_1.py -d 3 5 -p 1e-3 1e-4 1e-5

Outputs detailed CSV + JSON results to eval/LS_distillation/
See setup.md for experiment details.
"""
import argparse
import sys, os, json, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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


# =============================================================================
# Circuit construction
# =============================================================================

def build_distillation_circuit(d, rounds, y_prep="fold_transversal_s"):
    """
    Build Steane 7-to-1 distillation circuit.

    Args:
        d: Code distance.
        rounds: Number of SE rounds per measurement cycle (typically d).
        y_prep: |Y⟩ state preparation method:
            - "fold_transversal_s": RX + fold-transversal S (deterministic, default)
            - "injection": Corner state injection (non-deterministic, adds
              detector-level post-selection)

    Returns:
        (circuit, circuit_info) tuple.
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
        elif y_prep == "injection":
            # Corner state injection: non-deterministic |Y⟩ preparation with
            # diagonal X/Z split + post-selection on first SE round.
            #
            # NOTE: Currently requires single-patch SE (state_injection runs
            # SE internally on the full system). In the multi-patch distillation
            # circuit, we initialize all patches first then run system-wide SE,
            # which conflicts with injection's non-code-space initial state.
            #
            # Workaround: use injection only for W5 (the reusable magic state)
            # while using fold_transversal_s for W1-W4. This mirrors a realistic
            # scenario where the main patches are fault-tolerantly prepared and
            # the ancilla magic state is injected.
            raise NotImplementedError(
                "Corner state injection is not yet compatible with multi-patch "
                "distillation circuits. The tracker cannot handle syndrome "
                "extraction on a non-code-space state across multiple patches "
                "simultaneously. Use --y-prep fold_transversal_s (default) or "
                "implement per-patch SE isolation."
            )
        else:
            raise ValueError(f"Unknown y_prep method: {y_prep}")

    # --- Step 1: State preparation ---
    # W1, W2, W3: |Y⟩ (noisy magic states)
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

        # Register coupler (reuses qubit indices if corridor already exists)
        if coupler_name not in system.coupler_patches:
            system.register_coupler(UnrotatedMultiPatchCoupler(),
                patch_names=subset, name=coupler_name,
                path_axis='vertical', center_axis=center)
            n = system.num_qubits
            if n > tracker.num_qubits:
                tracker.expand(n - tracker.num_qubits)
            builder.write_coordinates()

        # Activate coupler + init corridor data in X
        builder.activate_coupler(coupler_name)
        cp = system.coupler_patches[coupler_name]
        cd = sorted([system.local_to_global_map[coupler_name][q]
                     for q in cp.data_indices])
        builder.initialize(init_dict={q: 'X' for q in cd}, n=system.num_qubits)

        # SE with coupler active
        se2 = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se2.circuit, rounds=rounds)

        # Mid-circuit MX on coupler data + W5 data
        measure_mid = {q: 'X' for q in cd}
        for q in system.data_indices:
            if system.index_to_owner_map.get(q) == 'W5':
                measure_mid[q] = 'X'
        builder.apply_data_readout(final_measurements=measure_mid)

        # Deactivate coupler
        builder.deactivate_coupler(coupler_name)

        # Re-inject W5 for next cycle
        if i < len(subsets) - 1:
            prepare_y('W5')

    # --- Step 4: Final readout ---
    # S† on W4: |Y⟩ → |+⟩
    op_set.fold_transversal_s_dag(builder, system.patches['W4'][0], noiseless=True)

    # MX on W1-W4
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
# Simulation
# =============================================================================

def run_simulation(circuit, p, max_shots=10_000_000, max_errors=200):
    """Run noisy simulation with post-selection. Returns stats."""
    noise_config = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    all_qubits = set()
    for inst in circuit.flattened():
        if inst.name in ("H", "S", "S_DAG", "CX", "CZ", "R", "RX", "M", "MX"):
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    all_qubits.add(t.value)
    injector = NoiseInjector.from_circuit_level(noise_config, sorted(all_qubits))
    noisy = injector.inject_noise(circuit)

    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("pymatching", backend="cpu"),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=50_000,
        num_workers=32,
        post_select_observable_indices=[0, 1, 3],
        target_observable_indices=[2],
        print_progress=True,
        progress_interval_sec=30.0,
    )
    stats = pipeline.run(noisy)
    return stats


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Steane 7-to-1 |Y⟩ distillation experiment")
    parser.add_argument("-d", "--distances", type=int, nargs="+",
                        default=[3, 5, 7], help="Code distances (default: 3 5 7)")
    parser.add_argument("-p", "--p-values", type=float, nargs="+",
                        default=[1e-3, 1e-4],
                        help="Physical error rates (default: 1e-3 1e-4)")
    parser.add_argument("--y-prep", choices=["fold_transversal_s", "injection"],
                        default="fold_transversal_s",
                        help="Y state preparation method (default: fold_transversal_s)")
    parser.add_argument("--max-shots", type=int, default=10_000_000,
                        help="Max shots per run (default: 10M)")
    parser.add_argument("--max-errors", type=int, default=200,
                        help="Max errors for early stopping (default: 200)")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    suffix = f"_{args.y_prep}" if args.y_prep != "fold_transversal_s" else ""
    csv_path = os.path.join(out_dir, f"LS_distillation_7_to_1_results{suffix}.csv")

    # Load checkpoint: skip (d, p, y_prep) combos already in CSV
    done_keys = set()
    if os.path.exists(csv_path):
        import csv as _csv
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                done_keys.add((int(row["d"]), float(row["p"]), row.get("y_prep", "")))
        print(f"Checkpoint: {len(done_keys)} tasks already done, skipping.")

    all_results = []

    for d in args.distances:
        rounds = d
        print(f"\n{'='*60}")
        print(f"Building d={d}, rounds={rounds}, y_prep={args.y_prep}")
        print(f"{'='*60}")

        t_build_start = time.perf_counter()
        circuit, circuit_info, system = build_distillation_circuit(d, rounds, args.y_prep)
        t_build = time.perf_counter() - t_build_start

        print(f"Circuit: {circuit_info['num_qubits']} qubits, "
              f"{circuit_info['num_detectors']} det, "
              f"{circuit_info['num_observables']} obs "
              f"(built in {t_build:.1f}s)")

        for p in args.p_values:
            if (d, p, args.y_prep) in done_keys:
                print(f"\n--- d={d}, p={p:.0e} --- SKIPPED (checkpoint)")
                continue
            print(f"\n--- d={d}, p={p:.0e} ---")
            stats = run_simulation(circuit, p, args.max_shots, args.max_errors)

            result = {
                'd': d,
                'rounds': rounds,
                'p': p,
                'y_prep': args.y_prep,
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

    # Save CSV
    csv_path = os.path.join(out_dir, f"LS_distillation_7_to_1_results{suffix}.csv")
    if all_results:
        keys = all_results[0].keys()
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in all_results:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"Saved {csv_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"SUMMARY (y_prep={args.y_prep})")
    print(f"{'='*90}")
    print(f"{'d':>3} {'p':>8} {'qubits':>7} {'detectors':>10} {'shots':>12} {'kept':>12} "
          f"{'PS_rate':>8} {'errors':>7} {'LER':>10} {'time':>7}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['d']:>3} {r['p']:>8.0e} {r['num_qubits']:>7} {r['num_detectors']:>10} "
              f"{r['shots']:>12,} {r['post_selected_shots']:>12,} "
              f"{r['post_selection_rate']*100:>7.2f}% {r['errors']:>7} "
              f"{r['logical_error_rate']:>10.2e} {r['decoding_time_sec']:>6.1f}s")
