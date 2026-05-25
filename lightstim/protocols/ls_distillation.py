"""
Steane (LS) 7-to-1 |Y⟩ distillation protocol.

Circuit: 4 input |Y⟩ patches (W1,W2,W3,W5) + 1 output patch (W4).
Four sequential ZZZZ measurements via UnrotatedMultiPatchCoupler.

Public API
----------
build_distillation_circuit(d, rounds, y_prep) → (circuit, circuit_info, system)
inject_noise(circuit, magic_qubits, p, p_injected, mode, data_indices) → circuit
estimate_p_in(d, rounds, p_injected, p_background, ...) → float
run_simulation(circuit, magic_qubits, p, p_injected, mode, ...) → stats
"""
import time
import numpy as np

from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedMultiPatchCoupler,
    UnrotatedSurfaceCodeLogicalOpSet,
)
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker
from lightstim.simulation.decoder_backend.pipeline import SimulationPipeline
from lightstim.simulation.decoder_backend.config import DecoderConfig
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector
from lightstim.noise.rules import FlipAfterResetFiltered

# Magic patch names: the four input |Y⟩ patches
LS_MAGIC_NAMES = {"W1", "W2", "W3", "W5"}
# Keep underscore alias for backward compatibility
_LS_MAGIC_NAMES = LS_MAGIC_NAMES


def build_distillation_circuit(d, rounds, y_prep="fold_transversal_s"):
    """
    Build Steane 7-to-1 distillation circuit (noiseless).

    Args:
        d:      Code distance.
        rounds: SE rounds per measurement cycle (typically d).
        y_prep: |Y⟩ preparation method ('fold_transversal_s').

    Returns:
        (circuit, circuit_info, system)
    """
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

    system.register_coupler(
        UnrotatedMultiPatchCoupler(),
        patch_names=['W1', 'W2', 'W3', 'W5'],
        name='meas_1',
        path_axis='vertical',
        center_axis=center,
    )

    num_qubits = system.num_qubits
    tracker = SyndromeTracker(num_qubits=num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    def prepare_y(patch_name):
        patch = system.patches[patch_name][0]
        if y_prep == "fold_transversal_s":
            wd = {q: 'X' for q in system.data_indices
                  if system.index_to_owner_map[q] == patch_name}
            builder.initialize(init_dict=wd, n=system.num_qubits)
            op_set.fold_transversal_s(builder, patch)
        else:
            raise ValueError(f"Unknown y_prep method: {y_prep}")

    for wname in ['W1', 'W2', 'W3']:
        prepare_y(wname)

    w4d = {q: 'X' for q in system.data_indices if system.index_to_owner_map[q] == 'W4'}
    builder.initialize(init_dict=w4d, n=num_qubits)
    prepare_y('W5')

    se = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds)

    subsets = [
        ['W1', 'W2', 'W3', 'W5'],
        ['W1', 'W2', 'W4', 'W5'],
        ['W1', 'W3', 'W4', 'W5'],
        ['W2', 'W3', 'W4', 'W5'],
    ]

    for i, subset in enumerate(subsets):
        coupler_name = f'meas_{i+1}'

        if coupler_name not in system.coupler_patches:
            system.register_coupler(
                UnrotatedMultiPatchCoupler(),
                patch_names=subset,
                name=coupler_name,
                path_axis='vertical',
                center_axis=center,
            )
            n = system.num_qubits
            if n > tracker.num_qubits:
                tracker.expand(n - tracker.num_qubits)
            builder.write_coordinates()

        builder.activate_coupler(coupler_name)
        cp = system.coupler_patches[coupler_name]
        cd = sorted([system.local_to_global_map[coupler_name][q] for q in cp.data_indices])
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

    op_set.fold_transversal_s_dag(builder, system.patches['W4'][0], noiseless=True)
    measure_final = {
        q: 'X' for q in system.data_indices
        if system.index_to_owner_map.get(q) in ('W1', 'W2', 'W3', 'W4')
    }
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


def inject_noise(circuit, magic_qubits, p, p_injected, mode="full", data_indices=None):
    """
    Inject noise into a clean LS 7-to-1 distillation circuit.

    Args:
        magic_qubits: Qubit indices of magic patches (W1,W2,W3,W5).
        p:            Circuit-level depolarizing rate ('full' and 'both' modes).
        p_injected:   Injection noise rate on magic DATA qubit resets.
        mode:         'injection' | 'full' | 'both'.
        data_indices: system.data_indices — restricts injection to data qubits only.
    """
    all_qubits = list(range(circuit.num_qubits))

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
        cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p,
                          custom_params={"p_injected": p_injected})
        inj = NoiseInjector.from_circuit_level(cfg, all_qubits)
        inj.add_rule(FlipAfterResetFiltered(injection_targets, param_name="p_injected"))
        return inj.inject_noise(circuit)

    else:
        raise ValueError(f"Unknown noise mode: {mode!r}. Choose from 'injection', 'full', 'both'.")


def estimate_p_in(d, rounds, p_injected, p_background=0.0,
                  max_shots=10_000_000, max_errors=100, batch_size=5_000):
    """
    Estimate effective logical input infidelity P_in for LS |Y⟩ magic state.

    Calibration circuit: RX → fold_transversal_S → SE(rounds)
    → noiseless S†_L → MX logical readout.

    Returns:
        p_in (float): logical error rate of the prepared |Y⟩ state.
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

    wd = {q: 'X' for q in sys1.data_indices}
    builder.initialize(init_dict=wd, n=sys1.num_qubits)
    op_set.fold_transversal_s(builder, sys1.patches['cal'][0])

    se = UnrotatedSurfaceCodeExtractionBlock(sys1)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds)

    op_set.fold_transversal_s_dag(builder, sys1.patches['cal'][0], noiseless=True)
    builder.apply_data_readout(final_measurements={q: 'X' for q in sys1.data_indices})

    circuit = builder.circuit
    all_qubits = list(range(circuit.num_qubits))

    if p_background > 0:
        cfg = NoiseConfig(p_1q=p_background, p_2q=p_background,
                          p_meas=p_background, p_reset=p_background,
                          custom_params={"p_injected": p_injected})
        inj = NoiseInjector.from_circuit_level(cfg, all_qubits)
        inj.add_rule(FlipAfterResetFiltered(set(sys1.data_indices), param_name="p_injected"))
    else:
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


def run_simulation(circuit, magic_qubits, p, p_injected, mode,
                   ps_obs, target_obs,
                   decoder_name="pymatching",
                   max_shots=10_000_000, max_errors=100,
                   batch_size=50_000, num_workers=1,
                   data_indices=None):
    """
    Run noisy LS simulation with post-decode post-selection.

    Args:
        ps_obs:       Observable indices to post-select on.
        target_obs:   Observable indices to measure LER on.
        data_indices: system.data_indices — restricts injection to data qubits.
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
