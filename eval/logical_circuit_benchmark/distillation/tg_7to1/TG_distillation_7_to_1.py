"""
Transversal-Gate 7-to-1 |Y⟩ Distillation (Zhou et al. Fig 4c).

Hypercube encoding of [[7,1,3]] PQRM code via 8 unrotated surface code patches,
with Y state injection on 8 magic patches and transversal S-bar gate application.

Protocol (algorithmic fault tolerance):
    1. Init all 16 patches + fold-transversal S on magic patches
    2. d rounds SE (initialization)
    3. Hypercube encoding: 3 ticks of transversal CNOTs, r SE rounds after each
    4. Teleportation: CNOT(W_i, M_i) for i=1..7 + r SE, then H on M1..M7 + r SE
    5. Noiseless fold-transversal S + H on W0
    6. Measure all data qubits in Z basis

Usage:
    python eval/TG_distillation/TG_distillation_7_to_1.py --build-only
    python eval/TG_distillation/TG_distillation_7_to_1.py -d 3 5 7 -p 1e-3
    python eval/TG_distillation/TG_distillation_7_to_1.py -d 3 -r 2 --decoder bposd
"""
import argparse
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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
# Circuit construction
# =============================================================================

def build_distillation_circuit(d, rounds, r=1, injection_noise_only=False):
    """
    Build transversal-gate 7-to-1 distillation circuit.

    Layout: 8 working patches (W0-W7) in 2x4 grid + 8 magic patches (M0-M7).

    Working patch init states (hypercube encoding of [[7,1,3]] Steane code):
        W0(|+>), W2(|+>)
        W1(|+>), W3(|0>)
        W4(|+>), W6(|0>)
        W5(|0>), W7(|0>)

    Magic patches: |Y⟩ via RX + fold-transversal S.

    Args:
        d: Code distance.
        rounds: Number of SE rounds after initialization (typically d).
        r: Number of SE rounds after each transversal gate layer.

    Returns:
        (circuit, circuit_info) tuple.
    """
    # --- Layout ---
    patch_size = 2 * (d - 1)
    gap = 2
    col_sp = patch_size + gap
    row_sp = patch_size + gap

    # Working patches: 2 columns x 4 rows
    working_layout = {
        'W0': (0,          0),
        'W1': (0,          row_sp),
        'W2': (col_sp,     0),
        'W3': (col_sp,     row_sp),
        'W4': (0,          2 * row_sp),
        'W5': (0,          3 * row_sp),
        'W6': (col_sp,     2 * row_sp),
        'W7': (col_sp,     3 * row_sp),
    }

    # Magic patches M1-M7: to the right of working patches (no M0 needed)
    magic_offset_x = 2 * col_sp
    magic_layout = {
        'M1': (magic_offset_x,          row_sp),
        'M2': (magic_offset_x + col_sp, 0),
        'M3': (magic_offset_x + col_sp, row_sp),
        'M4': (magic_offset_x,          2 * row_sp),
        'M5': (magic_offset_x,          3 * row_sp),
        'M6': (magic_offset_x + col_sp, 2 * row_sp),
        'M7': (magic_offset_x + col_sp, 3 * row_sp),
    }

    # --- Create system ---
    system = QECSystem()
    # Global patches (data_indices remapped to global) — for transversal_cnot
    gp = {}
    for name, offset in working_layout.items():
        p = UnrotatedSurfaceCode(distance=d)
        gp[name] = system.add_patch(p, name=name, offset=offset)

    for name, offset in magic_layout.items():
        p = UnrotatedSurfaceCode(distance=d)
        gp[name] = system.add_patch(p, name=name, offset=offset)

    # Local patches (shifted, with consistent qubit_coords) — for fold-transversal ops
    lp = {name: system.patches[name][0] for name in list(working_layout) + list(magic_layout)}

    op_set = UnrotatedSurfaceCodeLogicalOpSet(UnrotatedSurfaceCodeExtractionBlock)

    # --- Tracker + Builder ---
    num_qubits = system.num_qubits
    tracker = SyndromeTracker(
        num_qubits=num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    # Helper: create fresh SE block and apply
    def do_se(n_rounds):
        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=n_rounds,
                                           noiseless=injection_noise_only)

    # --- Step 1: Initialize all 15 patches (1 tick) ---
    # Working patches: transversal X or Z
    x_patches = {'W0', 'W1', 'W2', 'W4'}   # |+>
    z_patches = {'W3', 'W5', 'W6', 'W7'}    # |0>
    magic_names = [f'M{i}' for i in range(1, 8)]  # Only M1-M7 (no M0)

    init_dict = {}
    for name in x_patches:
        for q in system.data_indices:
            if system.index_to_owner_map[q] == name:
                init_dict[q] = 'X'
    for name in z_patches:
        for q in system.data_indices:
            if system.index_to_owner_map[q] == name:
                init_dict[q] = 'Z'
    builder.initialize(init_dict=init_dict, n=num_qubits, noiseless=injection_noise_only)

    # Magic patches M1-M7: Y state injection (corner protocol, rounds=0)
    # Use global patches (gp) so data_indices are global.
    # post_select_coords=set() → full QEC (no post-selection on injection),
    # because this is a patch-growth scenario.
    for mname in magic_names:
        op_set.state_injection(builder, gp[mname], inject_state='Y',
                               protocol='corner', rounds=0,
                               post_select_coords=set())

    # --- Step 2: d rounds of syndrome extraction (initialization) ---
    print(f"  Init SE ({rounds} rounds)...")
    do_se(rounds)

    # --- Step 3: Hypercube encoding — 3 ticks of parallel transversal CNOTs ---
    cnot_ticks = [
        [('W0', 'W4'), ('W1', 'W5'), ('W2', 'W6'), ('W3', 'W7')],
        [('W0', 'W2'), ('W1', 'W3'), ('W4', 'W6'), ('W5', 'W7')],
        [('W0', 'W1'), ('W2', 'W3'), ('W4', 'W5'), ('W6', 'W7')],
    ]

    for tick_idx, tick in enumerate(cnot_ticks):
        # Batch all parallel CNOTs into a single unitary block (1 TICK)
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

    # --- Step 4: Teleportation CNOT(W_i, M_i) for i=1..7 ---
    # Simultaneously apply noiseless S on W0 (output correction)
    tele_cnot = stim.Circuit()
    for i in range(1, 8):
        c_qubits = sorted(gp[f'W{i}'].data_indices)
        t_qubits = sorted(gp[f'M{i}'].data_indices)
        targets = []
        for c, t in zip(c_qubits, t_qubits):
            targets.extend([c, t])
        tele_cnot.append("CNOT", targets)
    # Noiseless S on W0 in the same tick (disjoint qubits)
    diag_s, diag_sdag, mirror_pairs_w0 = _get_fold_yx_pairs(system, lp['W0'])
    if diag_s:
        tele_cnot.append("S", sorted(diag_s))
    if diag_sdag:
        tele_cnot.append("S_DAG", sorted(diag_sdag))
    for a, b in mirror_pairs_w0:
        tele_cnot.append("CZ", [a, b])
    builder.apply_unitary_block(tele_cnot)
    print(f"  Teleportation CNOT + S(W0) + SE ({r} rounds)...")
    do_se(r)

    # --- Step 5: Measure working patches W0-W7 in X, magic M1-M7 in Z ---
    final_meas = {}
    for q in system.data_indices:
        owner = system.index_to_owner_map[q]
        if owner.startswith('W'):
            final_meas[q] = 'X'
        else:  # M1-M7
            final_meas[q] = 'Z'
    builder.apply_data_readout(final_measurements=final_meas)

    # --- Build circuit info ---
    circuit = builder.circuit
    circuit_info = {
        'num_qubits': circuit.num_qubits,
        'num_detectors': circuit.num_detectors,
        'num_observables': circuit.num_observables,
        'r': r,
    }
    return circuit, circuit_info, system


# =============================================================================
# Simulation
# =============================================================================

def run_simulation(circuit, p, T, ps_indices, target_indices, decoder_name,
                   max_shots=10_000_000, max_errors=200,
                   num_workers=32, backend="cpu", batch_size=50_000):
    """
    Run noisy simulation with GF(2)-transformed post-selection.

    Pipeline:
        1. Sample raw (detectors, observables)
        2. Transform raw obs: obs' = (T @ obs) % 2
        3. Post-select: keep shots where obs'[ps_indices] == 0
        4. Decode surviving shots (batched via decode_shots_bit_packed)
        5. Transform predictions: pred' = (T @ pred) % 2
        6. LER = errors on obs'[target_indices]
    """
    import math

    noise_config = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    all_qubits = list(range(circuit.num_qubits))
    injector = NoiseInjector.from_circuit_level(noise_config, all_qubits)
    noisy = injector.inject_noise(circuit)

    n_obs = circuit.num_observables
    n_det = circuit.num_detectors

    if np.array_equal(T, np.eye(T.shape[0], dtype=int)):
        # Identity transform → use standard pipeline
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

    # Non-trivial transform → custom decode loop with batched decoding
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

        # Transform raw observables
        obs_t = transform_observables(obs, T)

        # Post-select on ps_indices == 0
        mask = np.all(obs_t[:, ps_indices] == 0, axis=1)
        if not np.any(mask):
            elapsed = time.perf_counter() - t0
            print(f"shots={total_shots:,} kept={kept_shots:,} errors={errors} "
                  f"LER={errors/max(kept_shots,1):.2e} elapsed={elapsed:.1f}s")
            continue

        dets_kept = dets[mask]
        obs_t_kept = obs_t[mask]
        kept_shots += dets_kept.shape[0]

        # Batch decode: pack detector data → decode → unpack predictions
        n_det_bytes = math.ceil(n_det / 8)
        dets_packed = np.packbits(dets_kept, axis=1, bitorder="little")[:, :n_det_bytes]
        pred_packed = compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=dets_packed
        )
        # Unpack predictions
        pred_unpacked = np.unpackbits(pred_packed, axis=1, bitorder="little")[:, :n_obs]

        # Transform predictions
        pred_t = transform_observables(pred_unpacked, T)

        # Count errors on target observables
        corrected = (obs_t_kept[:, target_indices] ^ pred_t[:, target_indices])
        errors += int(np.sum(np.any(corrected, axis=1)))

        elapsed = time.perf_counter() - t0
        ler = errors / kept_shots if kept_shots > 0 else 0
        print(f"shots={total_shots:,} kept={kept_shots:,} errors={errors} "
              f"LER={ler:.2e} elapsed={elapsed:.1f}s")

    # Build result object
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
# Observable analysis
# =============================================================================

def analyze_observables(circuit, system, target_patch_names=None):
    """
    Analyze observables using the obs-to-patch matrix and GF(2) elimination.

    Args:
        circuit: Built stim circuit.
        system: QECSystem used to build the circuit.
        target_patch_names: Patch names for the distilled output.
            Defaults to ['W0'].

    Returns:
        (T, target_indices, ps_indices, obs_patch_matrix, patch_names)
    """
    if target_patch_names is None:
        target_patch_names = ['W0']

    matrix, patch_names = build_obs_patch_matrix(circuit, system)
    T, target, ps = identify_distillation_observables(matrix, patch_names, target_patch_names)

    print(f"  Observables: {circuit.num_observables}")
    print(f"  Obs-to-patch matrix (working patches only):")
    w_cols = [i for i, n in enumerate(patch_names) if n.startswith('W')]
    for i in range(matrix.shape[0]):
        involved_w = [patch_names[j] for j in w_cols if matrix[i, j]]
        print(f"    L{i}: {involved_w}")

    M_new = (T @ matrix) % 2
    print(f"  After GF(2) elimination (target={target_patch_names}):")
    for i in range(M_new.shape[0]):
        involved_w = [patch_names[j] for j in w_cols if M_new[i, j]]
        label = 'TARGET' if i in target else 'PS'
        print(f"    L{i}': {involved_w} [{label}]")

    print(f"  -> Target obs: {target}, Post-select obs: {ps}")
    return T, target, ps, matrix, patch_names


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transversal-gate 7-to-1 |Y⟩ distillation experiment")
    parser.add_argument("-d", "--distances", type=int, nargs="+",
                        default=[3, 5, 7], help="Code distances")
    parser.add_argument("-p", "--p-values", type=float, nargs="+",
                        default=[1e-3], help="Physical error rates")
    parser.add_argument("-r", "--gate-se-rounds", type=int, default=1,
                        help="SE rounds after each gate layer (default: 1)")
    parser.add_argument("--decoder", choices=["bposd", "mwpf", "pymatching"],
                        default="bposd", help="Decoder (default: bposd)")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu",
                        help="Decoder backend (default: cpu)")
    parser.add_argument("--num-workers", type=int, default=32,
                        help="Number of parallel workers (default: 32)")
    parser.add_argument("--max-shots", type=int, default=10_000_000)
    parser.add_argument("--max-errors", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--build-only", action="store_true",
                        help="Only build the circuit (no simulation)")
    parser.add_argument("--load-circuits", action="store_true",
                        help="Load pre-built circuits from eval/TG_distillation/circuits/ instead of building")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "TG_distillation_7_to_1_results.json")
    csv_path_ck = os.path.join(out_dir, "TG_distillation_7_to_1_results.csv")

    # Load checkpoint: skip (d, p, decoder, r) combos already saved
    done_keys = set()
    if os.path.exists(csv_path_ck):
        import csv as _csv
        with open(csv_path_ck) as f:
            for row in _csv.DictReader(f):
                done_keys.add((int(row["d"]), float(row["p"]),
                               row.get("decoder", ""), int(row.get("r", 1))))
        print(f"Checkpoint: {len(done_keys)} tasks already done, skipping.")

    all_results = []

    for d in args.distances:
        rounds = d
        print(f"\n{'='*60}")
        print(f"Building d={d}, init_rounds={rounds}, gate_se_rounds={args.gate_se_rounds}")
        print(f"{'='*60}")

        circuit_dir = os.path.join(out_dir, "circuits")
        circuit_path = os.path.join(circuit_dir, f"TG_7to1_d{d}_r{args.gate_se_rounds}.stim")
        transform_path = os.path.join(circuit_dir, f"TG_7to1_d{d}_r{args.gate_se_rounds}_obs.json")

        if args.load_circuits and os.path.exists(circuit_path) and os.path.exists(transform_path):
            # Load pre-built circuit + observable transform
            t_build_start = time.perf_counter()
            circuit = stim.Circuit.from_file(circuit_path)
            with open(transform_path) as f:
                obs_info = json.load(f)
            T = np.array(obs_info['T'], dtype=int)
            target_obs = obs_info['target_obs']
            post_select_obs = obs_info['post_select_obs']
            t_build = time.perf_counter() - t_build_start

            circuit_info = {
                'num_qubits': circuit.num_qubits,
                'num_detectors': circuit.num_detectors,
                'num_observables': circuit.num_observables,
                'r': args.gate_se_rounds,
            }
            print(f"Circuit: {circuit_info['num_qubits']} qubits, "
                  f"{circuit_info['num_detectors']} det, "
                  f"{circuit_info['num_observables']} obs "
                  f"(loaded from {circuit_path} in {t_build:.2f}s)")
            print(f"  Target obs: {target_obs}, Post-select obs: {post_select_obs}")
        else:
            # Build circuit from scratch
            t_build_start = time.perf_counter()
            circuit, circuit_info, system = build_distillation_circuit(
                d, rounds, r=args.gate_se_rounds)
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

            # Observable analysis with GF(2) elimination
            T, target_obs, post_select_obs, _, _ = analyze_observables(
                circuit, system, target_patch_names=['W0'])

            # Save circuit + transform for reuse
            os.makedirs(circuit_dir, exist_ok=True)
            with open(circuit_path, "w") as f:
                f.write(str(circuit))
            with open(transform_path, "w") as f:
                json.dump({
                    'T': T.tolist(),
                    'target_obs': target_obs,
                    'post_select_obs': post_select_obs,
                }, f, indent=2)
            print(f"Saved circuit to {circuit_path}")

        if args.build_only:
            print("(build-only mode, skipping simulation)")
            continue

        for p in args.p_values:
            if (d, p, args.decoder, args.gate_se_rounds) in done_keys:
                print(f"\n--- d={d}, p={p:.0e} --- SKIPPED (checkpoint)")
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

            # Append immediately so kill/OOM never loses this result
            import csv as _csv2
            write_header = not os.path.exists(csv_path_ck)
            with open(csv_path_ck, "a") as f:
                w = _csv2.DictWriter(f, fieldnames=list(result.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(result)

            print(f"  shots={stats.shots:,}, kept={stats.post_selected_shots:,}, "
                  f"PS_rate={stats.post_selection_rate*100:.2f}%")
            print(f"  errors={stats.errors}, LER={stats.logical_error_rate:.2e}, "
                  f"time={stats.seconds:.1f}s")

    # Save results
    if all_results:
        json_path = os.path.join(out_dir, "TG_distillation_7_to_1_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {json_path}")

    # Print summary table
    if all_results:
        print(f"\n{'='*100}")
        print(f"SUMMARY (r={args.gate_se_rounds}, decoder={args.decoder})")
        print(f"{'='*100}")
        print(f"{'d':>3} {'p':>8} {'qubits':>7} {'det':>6} {'obs':>4} "
              f"{'shots':>12} {'kept':>12} {'PS%':>8} {'errors':>7} {'LER':>10}")
        print("-" * 100)
        for r in all_results:
            print(f"{r['d']:>3} {r['p']:>8.0e} {r['num_qubits']:>7} "
                  f"{r['num_detectors']:>6} {r['num_observables']:>4} "
                  f"{r['shots']:>12,} {r['post_selected_shots']:>12,} "
                  f"{r['post_selection_rate']*100:>7.2f}% {r['errors']:>7} "
                  f"{r['logical_error_rate']:>10.2e}")
