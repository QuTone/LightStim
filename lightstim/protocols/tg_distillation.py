"""
Transversal-Gate (TG) 7-to-1 |Y⟩ distillation protocol.

Hypercube encoding of [[7,1,3]] PQRM code via 8 unrotated surface code patches,
with Y state injection on 7 magic patches (M1-M7) and transversal S-bar gate.

Public API
----------
build_distillation_circuit(d, rounds_init, rounds_gate) → (circuit, circuit_info, system)
inject_noise(circuit, magic_qubits, p, p_injected, mode) → circuit
estimate_p_in(d, rounds_init, p_injected, p_background, ...) → float
run_simulation(circuit, magic_qubits, p, p_injected, mode, T, ...) → stats
analyze_observables(circuit, system, target_patch_names) → (T, target, ps, matrix, names)

Round parameters
----------------
rounds_init  : SE rounds after state preparation (working-patch init + magic corner
               injection). Paper value: d. Defaults to None → d.
rounds_gate  : SE rounds after each transversal gate layer (3 CNOT encoding ticks +
               1 teleportation CNOT). Paper value: 1. Defaults to 1.
"""
import time
import numpy as np
import stim

from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedSurfaceCodeLogicalOpSet,
)
from lightstim.qec_code.surface_code.unrotated.operation import _get_fold_yx_pairs
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker
from lightstim.simulation.decoder_backend.pipeline import SimulationPipeline
from lightstim.simulation.decoder_backend.config import DecoderConfig
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector
from lightstim.noise.rules import FlipAfterYResetFiltered
from lightstim.simulation.observable_analysis import (
    build_obs_patch_matrix,
    identify_distillation_observables,
    transform_observables,
)

# Magic patch names (M0 is output; only M1-M7 receive injection noise)
TG_MAGIC_NAMES = {f'M{i}' for i in range(1, 8)}
# Keep underscore alias for backward compatibility
_TG_MAGIC_NAMES = TG_MAGIC_NAMES


def build_distillation_circuit(d, rounds_init=None, rounds_gate=1):
    """
    Build TG 7-to-1 distillation circuit (noiseless).

    Layout: 8 working patches (W0-W7) in 2×4 grid + 7 magic patches (M1-M7).

    Args:
        d:            Code distance.
        rounds_init:  SE rounds after state preparation (working-patch init +
                      magic corner injection). Defaults to d (paper setting).
        rounds_gate:  SE rounds after each transversal gate layer (3 CNOT encoding
                      ticks + 1 teleportation CNOT). Paper value: 1.

    Returns:
        (circuit, circuit_info, system)
    """
    if rounds_init is None:
        rounds_init = d
    patch_size = 2 * (d - 1)
    gap = 2
    col_sp = patch_size + gap
    row_sp = patch_size + gap

    working_layout = {
        'W0': (0,       0),
        'W1': (0,       row_sp),
        'W2': (col_sp,  0),
        'W3': (col_sp,  row_sp),
        'W4': (0,       2 * row_sp),
        'W5': (0,       3 * row_sp),
        'W6': (col_sp,  2 * row_sp),
        'W7': (col_sp,  3 * row_sp),
    }

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

    system = QECSystem()
    gp = {}
    for name, offset in working_layout.items():
        gp[name] = system.add_patch(UnrotatedSurfaceCode(distance=d), name=name, offset=offset)
    for name, offset in magic_layout.items():
        gp[name] = system.add_patch(UnrotatedSurfaceCode(distance=d), name=name, offset=offset)

    lp = {name: system.patches[name][0]
          for name in list(working_layout) + list(magic_layout)}

    op_set = UnrotatedSurfaceCodeLogicalOpSet(UnrotatedSurfaceCodeExtractionBlock)

    num_qubits = system.num_qubits
    tracker = SyndromeTracker(num_qubits=num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    def do_se(n_rounds):
        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=n_rounds)

    # Step 1: Initialize all patches
    x_patches = {'W0', 'W1', 'W2', 'W4'}   # |+⟩
    z_patches = {'W3', 'W5', 'W6', 'W7'}    # |0⟩
    magic_names = [f'M{i}' for i in range(1, 8)]

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

    for mname in magic_names:
        op_set.state_injection(builder, gp[mname], inject_state='Y',
                               protocol='corner', rounds=0, post_select_coords=set())

    # Step 2: Initialization SE (rounds_init = d by default)
    do_se(rounds_init)

    # Step 3: Hypercube encoding — 3 ticks of parallel transversal CNOTs
    cnot_ticks = [
        [('W0', 'W4'), ('W1', 'W5'), ('W2', 'W6'), ('W3', 'W7')],
        [('W0', 'W2'), ('W1', 'W3'), ('W4', 'W6'), ('W5', 'W7')],
        [('W0', 'W1'), ('W2', 'W3'), ('W4', 'W5'), ('W6', 'W7')],
    ]

    for tick in cnot_ticks:
        cnot_circuit = stim.Circuit()
        for ctrl_name, tgt_name in tick:
            c_qubits = sorted(gp[ctrl_name].data_indices)
            t_qubits = sorted(gp[tgt_name].data_indices)
            targets = []
            for c, t in zip(c_qubits, t_qubits):
                targets.extend([c, t])
            cnot_circuit.append("CNOT", targets)
        builder.apply_unitary_block(cnot_circuit)
        do_se(rounds_gate)

    # Step 4: Teleportation CNOT(W_i, M_i) + noiseless S on W0
    tele_cnot = stim.Circuit()
    for i in range(1, 8):
        c_qubits = sorted(gp[f'W{i}'].data_indices)
        t_qubits = sorted(gp[f'M{i}'].data_indices)
        targets = []
        for c, t in zip(c_qubits, t_qubits):
            targets.extend([c, t])
        tele_cnot.append("CNOT", targets)
    diag_s, diag_sdag, mirror_pairs_w0 = _get_fold_yx_pairs(system, lp['W0'])
    if diag_s:
        tele_cnot.append("S", sorted(diag_s))
    if diag_sdag:
        tele_cnot.append("S_DAG", sorted(diag_sdag))
    for a, b in mirror_pairs_w0:
        tele_cnot.append("CZ", [a, b])
    builder.apply_unitary_block(tele_cnot)
    do_se(rounds_gate)

    # Step 5: Final readout — W in X, M in Z
    final_meas = {}
    for q in system.data_indices:
        owner = system.index_to_owner_map[q]
        final_meas[q] = 'X' if owner.startswith('W') else 'Z'
    builder.apply_data_readout(final_measurements=final_meas)

    circuit = builder.circuit
    circuit_info = {
        'num_qubits': circuit.num_qubits,
        'num_detectors': circuit.num_detectors,
        'num_observables': circuit.num_observables,
        'rounds_init': rounds_init,
        'rounds_gate': rounds_gate,
    }
    return circuit, circuit_info, system


def inject_noise(circuit, magic_qubits, p, p_injected, mode="full"):
    """
    Inject noise into a clean TG 7-to-1 distillation circuit.

    Args:
        magic_qubits: Qubit indices of magic patches (M1-M7).
        p:            Circuit-level depolarizing rate ('full' and 'both' modes).
        p_injected:   Injection noise rate on magic corner qubits only.
        mode:         'injection' | 'full' | 'both'.
    """
    all_qubits = list(range(circuit.num_qubits))

    if mode == "injection":
        cfg = NoiseConfig(p_reset=p_injected)
        inj = NoiseInjector(cfg)
        inj.add_rule(FlipAfterYResetFiltered(magic_qubits, param_name="p_reset"))
        return inj.inject_noise(circuit)

    elif mode == "full":
        cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p)
        return NoiseInjector.from_circuit_level(cfg, all_qubits).inject_noise(circuit)

    elif mode == "both":
        cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p,
                          custom_params={"p_injected": p_injected})
        inj = NoiseInjector.from_circuit_level(cfg, all_qubits)
        inj.add_rule(FlipAfterYResetFiltered(magic_qubits, param_name="p_injected"))
        return inj.inject_noise(circuit)

    else:
        raise ValueError(f"Unknown noise mode: {mode!r}. Choose from 'injection', 'full', 'both'.")


def estimate_p_in(d, rounds_init=None, p_injected=1e-3, p_background=0.0,
                  max_shots=10_000_000, max_errors=100, batch_size=5_000):
    """
    Estimate effective logical input infidelity P_in for TG |Y⟩ magic state.

    Calibration circuit: corner injection → SE(rounds_init)
    → noiseless S†_L → MX logical readout.

    Args:
        rounds_init: SE rounds after corner injection. Defaults to d (paper setting).

    Returns:
        p_in (float): logical error rate of the corner-injected |Y⟩ state.
    """
    if rounds_init is None:
        rounds_init = d

    patch = UnrotatedSurfaceCode(distance=d)
    sys1 = QECSystem()
    gp = sys1.add_patch(patch, name='cal', offset=(0, 0))

    op_set = UnrotatedSurfaceCodeLogicalOpSet(UnrotatedSurfaceCodeExtractionBlock)
    tracker = SyndromeTracker(num_qubits=sys1.num_qubits,
                              expected_num_logicals=sys1.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=sys1, if_detector=True)
    builder.write_coordinates()

    op_set.state_injection(builder, gp, inject_state='Y',
                           protocol='corner', rounds=0, post_select_coords=set())

    se = UnrotatedSurfaceCodeExtractionBlock(sys1)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds_init)

    op_set.fold_transversal_s_dag(builder, sys1.patches['cal'][0], noiseless=True)
    builder.apply_data_readout(final_measurements={q: 'X' for q in sys1.data_indices})

    circuit = builder.circuit
    all_qubits = list(range(circuit.num_qubits))

    if p_background > 0:
        cfg = NoiseConfig(p_1q=p_background, p_2q=p_background,
                          p_meas=p_background, p_reset=p_background,
                          custom_params={"p_injected": p_injected})
        inj = NoiseInjector.from_circuit_level(cfg, all_qubits)
        inj.add_rule(FlipAfterYResetFiltered(set(sys1.data_indices), param_name="p_injected"))
    else:
        cfg = NoiseConfig(p_reset=p_injected)
        inj = NoiseInjector(cfg)
        inj.add_rule(FlipAfterYResetFiltered(set(sys1.data_indices), param_name="p_reset"))

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


def run_simulation(circuit, magic_qubits, p, p_injected, mode,
                   T, ps_indices, target_indices, decoder_name,
                   max_shots=10_000_000, max_errors=200,
                   num_workers=32, backend="cpu", batch_size=50_000):
    """
    Run noisy TG simulation with GF(2)-transformed post-selection.

    If T is the identity matrix, uses standard SimulationPipeline.
    Otherwise, uses a custom decode loop with observable transformation.
    """
    import math

    noisy = inject_noise(circuit, magic_qubits, p, p_injected, mode)
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

    from lightstim.simulation.decoder_backend.registry import get_decoder

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
                  f"LER={errors/max(kept_shots, 1):.2e} elapsed={elapsed:.1f}s")
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

        corrected = obs_t_kept[:, target_indices] ^ pred_t[:, target_indices]
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


def analyze_observables(circuit, system, target_patch_names=None):
    """
    Analyze observables using obs-to-patch matrix and GF(2) elimination.

    Returns:
        (T, target_indices, ps_indices, obs_patch_matrix, patch_names)
    """
    if target_patch_names is None:
        target_patch_names = ['W0']

    matrix, patch_names = build_obs_patch_matrix(circuit, system)
    T, target, ps = identify_distillation_observables(matrix, patch_names, target_patch_names)

    print(f"  Observables: {circuit.num_observables}")
    w_cols = [i for i, n in enumerate(patch_names) if n.startswith('W')]
    print(f"  Obs-to-patch matrix (working patches only):")
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
