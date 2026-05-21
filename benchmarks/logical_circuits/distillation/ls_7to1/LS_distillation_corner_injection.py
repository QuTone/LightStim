"""
Steane 7-to-1 |Y⟩ Distillation — Corner State Injection — Injection Noise Sweep.

Magic state preparation via corner state injection (replacing fold-transversal-S).
Injection noise model: FlipAfterResetFiltered on W1/W2/W3/W5 data qubits.
  - RX qubits (upper triangle): Z_ERROR(p_inj)
  - RY corner qubit:            Z_ERROR(p_inj)
  - RZ qubits (lower triangle): X_ERROR(p_inj)

The effective logical input infidelity p_in is calibrated via a single-patch
corner-injection circuit (corner init → SE(rounds) → noiseless S†_L → MX).

NOTE: fold_transversal_s_dag is intentionally skipped in the final distillation
readout.  With corner injection the tracker's canonical W4 logical is already X_L
(equivalent to Y_L modulo stabilizers). Applying S_dag would add Z components
and make W4 unreadable by MX.

Usage:
    venv/bin/python eval/logical_circuit_benchmark/distillation/ls_7to1/LS_distillation_corner_injection.py

    # Custom p_inj sweep:
    venv/bin/python ... --p-injected 1e-3 5e-3 1e-2 5e-2 1e-1 -d 3 5

Outputs: CSV with per-task checkpointing in the same directory.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedMultiPatchCoupler,
    UnrotatedSurfaceCodeLogicalOpSet,
)
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector
from lightstim.noise.rules import FlipAfterResetFiltered
from lightstim.simulation.decoder_backend.pipeline import SimulationPipeline
from lightstim.simulation.decoder_backend.config import DecoderConfig
from lightstim.simulation.observable_analysis import (
    build_obs_patch_matrix,
    identify_distillation_observables,
)

_MAGIC_NAMES = {"W1", "W2", "W3", "W5"}


# =============================================================================
# Circuit construction
# =============================================================================

def build_corner_injection_circuit(d, rounds):
    """
    Build Steane 7-to-1 distillation circuit with corner state injection (noiseless).

    Magic state preparation:
        - Corner data qubit (min x,y global coord): RY  (|+i⟩ = |Y⟩)
        - Upper triangle (rel_y >= rel_x):           RX  (|+⟩)
        - Lower triangle (rel_y <  rel_x):           RZ  (|0⟩)

    Final readout: MX on W1–W4 directly (no fold_transversal_s_dag).
    The tracker reduces W4's logical to X_L form with corner injection
    stabilizers, so MX reads it correctly.

    Returns:
        (circuit, system, magic_data_qubits, ps_obs, target_obs)
    """
    patch_size = 2 * (d - 1)
    gap        = 2 * d + 2
    right_x    = patch_size + gap
    y_spacing  = gap
    center     = patch_size + gap / 2

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
    system.register_coupler(UnrotatedMultiPatchCoupler(),
        patch_names=['W1', 'W2', 'W3', 'W5'], name='meas_1',
        path_axis='vertical', center_axis=center)

    num_qubits = system.num_qubits
    tracker    = SyndromeTracker(num_qubits=num_qubits,
                                  expected_num_logicals=system.num_logicals)
    builder    = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    def get_injection_init(patch_name, inject_state='Y'):
        """Corner injection init dict (global qubit indices)."""
        patch_data = sorted(q for q in system.data_indices
                            if system.index_to_owner_map.get(q) == patch_name)
        coords = [(system.qubit_coords[q], q) for q in patch_data]
        corner_coord, corner_gidx = min(coords, key=lambda t: (t[0][0], t[0][1]))
        ox, oy = corner_coord
        d_map = {}
        for gidx in patch_data:
            cx, cy = system.qubit_coords[gidx]
            rel_x, rel_y = cx - ox, cy - oy
            if gidx == corner_gidx:
                d_map[gidx] = inject_state
            elif rel_y >= rel_x:
                d_map[gidx] = 'X'
            else:
                d_map[gidx] = 'Z'
        return d_map

    # --- Step 1: Simultaneous corner injection init ---
    combined_init = {}
    for wname in ['W1', 'W2', 'W3', 'W5']:
        combined_init.update(get_injection_init(wname, 'Y'))
    for q in system.data_indices:
        if system.index_to_owner_map.get(q) == 'W4':
            combined_init[q] = 'X'
    builder.initialize(init_dict=combined_init, n=num_qubits)

    # --- Step 2: d SE rounds ---
    se = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds)

    def reinit_w5():
        builder.initialize(init_dict=get_injection_init('W5', 'Y'), n=system.num_qubits)

    # --- Step 3: Four sequential ZZZZ measurements ---
    subsets = [
        ['W1', 'W2', 'W3', 'W5'],
        ['W1', 'W2', 'W4', 'W5'],
        ['W1', 'W3', 'W4', 'W5'],
        ['W2', 'W3', 'W4', 'W5'],
    ]
    for i, subset in enumerate(subsets):
        cname = f'meas_{i+1}'
        if cname not in system.coupler_patches:
            system.register_coupler(UnrotatedMultiPatchCoupler(),
                patch_names=subset, name=cname,
                path_axis='vertical', center_axis=center)
            n = system.num_qubits
            if n > tracker.num_qubits:
                tracker.expand(n - tracker.num_qubits)
            builder.write_coordinates()

        builder.activate_coupler(cname)
        cp = system.coupler_patches[cname]
        cd = sorted([system.local_to_global_map[cname][q] for q in cp.data_indices])
        builder.initialize(init_dict={q: 'X' for q in cd}, n=system.num_qubits)

        se2 = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se2.circuit, rounds=rounds)

        measure_mid = {q: 'X' for q in cd}
        for q in system.data_indices:
            if system.index_to_owner_map.get(q) == 'W5':
                measure_mid[q] = 'X'
        builder.apply_data_readout(final_measurements=measure_mid)
        builder.deactivate_coupler(cname)

        if i < len(subsets) - 1:
            reinit_w5()

    # --- Step 4: Final MX readout (no S_dag — see module docstring) ---
    measure_final = {q: 'X' for q in system.data_indices
                     if system.index_to_owner_map.get(q) in ('W1', 'W2', 'W3', 'W4')}
    builder.apply_data_readout(final_measurements=measure_final)

    circuit = builder.circuit

    # Observable analysis
    matrix, patch_names = build_obs_patch_matrix(circuit, system)
    _, target_obs, ps_obs = identify_distillation_observables(
        matrix, patch_names, ['W4'])

    # Magic DATA qubits only (exclude ancilla SE qubits)
    magic_data_qubits = frozenset(
        q for q in system.data_indices
        if system.index_to_owner_map.get(q) in _MAGIC_NAMES
    )

    return circuit, system, magic_data_qubits, ps_obs, target_obs


# =============================================================================
# p_in calibration
# =============================================================================

def estimate_p_in_corner(d, rounds, p_injected,
                          max_shots=5_000_000, max_errors=100, batch_size=50_000):
    """
    Estimate the effective logical input infidelity p_in for a |Y⟩ magic state
    prepared via corner state injection at physical noise rate p_injected.

    Single-patch calibration circuit:
        corner_init → SE(rounds) → noiseless S†_L → MX logical readout

    Noise: FlipAfterResetFiltered on all data qubits
        (Z_ERROR after RX/RY, X_ERROR after RZ) at rate p_injected.

    Returns:
        p_in (float): effective logical error rate of the injected |Y⟩ state.
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

    # Corner injection init for the calibration patch
    patch_data = sorted(q for q in sys1.data_indices
                        if sys1.index_to_owner_map.get(q) == 'cal')
    coords = [(sys1.qubit_coords[q], q) for q in patch_data]
    corner_coord, corner_gidx = min(coords, key=lambda t: (t[0][0], t[0][1]))
    ox, oy = corner_coord
    init_dict = {}
    for gidx in patch_data:
        cx, cy = sys1.qubit_coords[gidx]
        rel_x, rel_y = cx - ox, cy - oy
        if gidx == corner_gidx:
            init_dict[gidx] = 'Y'
        elif rel_y >= rel_x:
            init_dict[gidx] = 'X'
        else:
            init_dict[gidx] = 'Z'
    builder.initialize(init_dict=init_dict, n=sys1.num_qubits)

    # SE rounds to project into code space
    se = UnrotatedSurfaceCodeExtractionBlock(sys1)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds)

    # Noiseless S†_L: converts |Y_L⟩ → |+_L⟩ so MX gives deterministic result
    op_set.fold_transversal_s_dag(builder, sys1.patches['cal'][0], noiseless=True)

    # MX readout: logical X_L = +1 in noiseless case
    builder.apply_data_readout(
        final_measurements={q: 'X' for q in sys1.data_indices})

    circuit = builder.circuit

    # Injection noise: FlipAfterResetFiltered on all data qubits
    cfg = NoiseConfig(p_reset=p_injected)
    inj = NoiseInjector(cfg)
    inj.add_rule(FlipAfterResetFiltered(set(sys1.data_indices), param_name='p_reset'))
    noisy = inj.inject_noise(circuit)

    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig('pymatching'),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        target_observable_indices=[0],
        print_progress=False,
    )
    return pipeline.run(noisy).logical_error_rate


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Steane 7-to-1 distillation — corner state injection noise sweep")
    parser.add_argument("-d", "--distances", type=int, nargs="+", default=[3],
                        help="Code distances (default: 3)")
    parser.add_argument("--p-injected", type=float, nargs="+",
                        default=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
                        help="Injection noise rates (default: 1e-3 5e-3 1e-2 5e-2 1e-1)")
    parser.add_argument("--decoder", choices=["pymatching", "bposd", "mwpf"],
                        default="pymatching")
    parser.add_argument("--max-shots", type=int, default=10_000_000)
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip p_in calibration (use p_injected as p_in)")
    args = parser.parse_args()

    out_dir  = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "LS_distillation_corner_injection_results.csv")

    fieldnames = [
        'd', 'rounds', 'p_injected', 'p_in', 'y_prep', 'decoder',
        'num_qubits', 'num_detectors', 'num_observables',
        'shots', 'post_selected_shots', 'post_selection_rate',
        'errors', 'logical_error_rate', 'p_in_cubed_x7',
        'suppression_ratio', 'decoding_time_sec', 'build_time_sec',
    ]

    for d in args.distances:
        rounds = d
        print(f"\n{'='*60}")
        print(f"Building d={d}, rounds={rounds}, y_prep=corner_injection")
        print(f"{'='*60}")

        t_build = time.perf_counter()
        circuit, system, magic_data_qubits, ps_obs, target_obs = \
            build_corner_injection_circuit(d, rounds)
        t_build = time.perf_counter() - t_build

        print(f"Circuit: {circuit.num_qubits} qubits, {circuit.num_detectors} det, "
              f"{circuit.num_observables} obs  (built in {t_build:.1f}s)")
        print(f"Magic data qubits: {len(magic_data_qubits)}")
        print(f"Target obs: {target_obs}  |  Post-select obs: {ps_obs}")

        # Noiseless check
        dets, obs = circuit.compile_detector_sampler().sample(
            shots=200, separate_observables=True)
        ok = not np.any(dets) and not np.any(obs)
        print(f"Noiseless check: {'OK' if ok else 'FAIL'}")
        if not ok:
            print("  WARNING: noiseless circuit has detector/observable errors!")

        # Calibrate p_in for each p_injected
        p_in_map = {}
        if not args.skip_calibration:
            print(f"\n-- Calibrating p_in (single-patch corner injection) --")
            for p_inj in args.p_injected:
                p_in_est = estimate_p_in_corner(d, rounds, p_inj,
                                                max_shots=args.max_shots,
                                                max_errors=args.max_errors,
                                                batch_size=args.batch_size)
                p_in_map[p_inj] = p_in_est
                print(f"  p_injected={p_inj:.2e}  →  p_in={p_in_est:.3e}")
        else:
            for p_inj in args.p_injected:
                p_in_map[p_inj] = p_inj

        pipeline = SimulationPipeline(
            decoder_config=DecoderConfig(args.decoder, backend="cpu"),
            max_shots=args.max_shots,
            max_errors=args.max_errors,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            post_select_corrected_observable_indices=ps_obs,
            target_observable_indices=[target_obs[0]],
            print_progress=True,
            progress_interval_sec=15.0,
        )

        for p_inj in args.p_injected:
            p_in = p_in_map[p_inj]
            print(f"\n--- d={d}, p_inj={p_inj:.1e}, p_in={p_in:.3e}, decoder={args.decoder} ---")

            # Injection noise: FlipAfterResetFiltered on all magic data qubits
            # Z_ERROR after RX/RY, X_ERROR after RZ — same model as fold-S version
            cfg = NoiseConfig(p_reset=p_inj)
            inj = NoiseInjector(cfg)
            inj.add_rule(FlipAfterResetFiltered(magic_data_qubits, param_name="p_reset"))
            noisy = inj.inject_noise(circuit)

            t0    = time.perf_counter()
            stats = pipeline.run(noisy)
            elapsed = time.perf_counter() - t0

            ler      = stats.logical_error_rate
            expected = 7.0 * p_in ** 3
            ratio    = ler / expected if expected > 0 else float('nan')

            print(f"  shots={stats.shots:,}  kept={stats.post_selected_shots:,}  "
                  f"PS_rate={stats.post_selection_rate*100:.2f}%")
            print(f"  errors={stats.errors}  LER={ler:.3e}  "
                  f"7·p_in³={expected:.3e}  ratio={ratio:.2f}  time={elapsed:.1f}s")

            row = {
                'd':                    d,
                'rounds':               rounds,
                'p_injected':           p_inj,
                'p_in':                 p_in,
                'y_prep':               'corner_injection',
                'decoder':              args.decoder,
                'num_qubits':           circuit.num_qubits,
                'num_detectors':        circuit.num_detectors,
                'num_observables':      circuit.num_observables,
                'shots':                stats.shots,
                'post_selected_shots':  stats.post_selected_shots,
                'post_selection_rate':  stats.post_selection_rate,
                'errors':               stats.errors,
                'logical_error_rate':   ler,
                'p_in_cubed_x7':        expected,
                'suppression_ratio':    ratio,
                'decoding_time_sec':    round(elapsed, 2),
                'build_time_sec':       round(t_build, 2),
            }

            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                w.writerow(row)
            print(f"  → saved to {csv_path}")

    print(f"\nDone. Results in {csv_path}")


if __name__ == "__main__":
    main()
