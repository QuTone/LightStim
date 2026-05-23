"""
P4-A: Comprehensive Compilation Performance Benchmark
Covers all protocols in the paper: Memory, TG CNOT, LS CNOT,
Bell Teleportation (TG + LS), Steane Distillation (TG + LS), CrossLS.

Measures: compile time, qubit count, detector count, observable count,
          annotation LoC (DETECTOR + OBSERVABLE_INCLUDE in expanded circuit).

Usage:
    venv/bin/python benchmarks/correctness_check/comprehensive_benchmark.py
"""
import sys, time, json
sys.path.insert(0, '/home/xiang/workspace/LightStim')

import numpy as np
from pathlib import Path

out = Path('/home/xiang/workspace/LightStim/benchmarks/correctness_check/results')
out.mkdir(parents=True, exist_ok=True)

N_TRIALS = 3
results = {}


def count_annotation_loc(circuit):
    """Count DETECTOR + OBSERVABLE_INCLUDE lines in fully expanded circuit."""
    expanded = circuit.flattened()
    lines = str(expanded).split('\n')
    n_det = sum(1 for l in lines if l.strip().startswith('DETECTOR'))
    n_obs = sum(1 for l in lines if l.strip().startswith('OBSERVABLE_INCLUDE'))
    return n_det + n_obs, n_det, n_obs


def bench(label, build_fn, d_label=""):
    """Run N_TRIALS timing trials, return median time + circuit info."""
    times = []
    circuit = None
    for trial in range(N_TRIALS):
        t0 = time.perf_counter()
        circuit = build_fn()
        times.append(time.perf_counter() - t0)
    t_med = np.median(times)
    ann_loc, n_det, n_obs = count_annotation_loc(circuit)
    info = {
        'label': label,
        'd': d_label,
        'compile_ms': round(t_med * 1000, 1),
        'num_qubits': circuit.num_qubits,
        'num_detectors': circuit.num_detectors,
        'num_observables': circuit.num_observables,
        'annotation_loc': ann_loc,
    }
    print(f"  {label} d={d_label}: {circuit.num_qubits}q, "
          f"{circuit.num_detectors}det, {circuit.num_observables}obs, "
          f"annot={ann_loc}, compile={t_med*1000:.1f}ms")
    return info


# =============================================================================
# 1. Surface Code Memory (Rotated)
# =============================================================================
print("\n=== 1. Surface Code Memory (Rotated) ===")
from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode
from lightstim.qec_code.surface_code.rotated.SE_block import RotatedSurfaceCodeExtractionBlock as RotatedSEBlock
from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.noise.config import NoiseConfig

for d in [3, 5, 7, 9, 11]:
    def build_surface_mem(d=d):
        patch = RotatedSurfaceCode(distance=d)
        system = QECSystem()
        system.add_patch(patch)
        noise_cfg = NoiseConfig(p_1q=0, p_2q=0, p_meas=0, p_reset=0)
        exp = MemoryExperiment(system, RotatedSEBlock, rounds=d,
                               noise_params=noise_cfg, noise_model='circuit_level', basis='Z')
        return exp.build()

    info = bench("Surface Mem", build_surface_mem, d_label=d)
    results[f'surface_mem_d{d}'] = info

# =============================================================================
# 2. Transversal Gate CNOT (Unrotated SC)
# =============================================================================
print("\n=== 2. TG CNOT (Unrotated SC) ===")
from lightstim.protocols.cnot_trans import CNOTTransExperiment
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)

for d in [3, 5, 7]:
    def build_tg_cnot(d=d):
        exp = CNOTTransExperiment(
            code_patch_class=UnrotatedSurfaceCode,
            extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
            code_params_control={'distance': d},
            offset_target=(4 * d, 0),
            initial_basis_control='Z',
            initial_basis_target='Z',
            measure_basis_control='Z',
            measure_basis_target='Z',
            rounds_before=d,
            rounds_after=1,
            noise_params=NoiseConfig(p_1q=0, p_2q=0, p_meas=0, p_reset=0),
            noise_model='circuit_level',
        )
        return exp.build()

    info = bench("TG CNOT", build_tg_cnot, d_label=d)
    results[f'tg_cnot_d{d}'] = info

# =============================================================================
# 3. LS CNOT (Unrotated SC)
# =============================================================================
print("\n=== 3. LS CNOT (Unrotated SC) ===")
from lightstim.protocols.cnot_ls import CNOTLSExperiment

for d in [3, 5, 7]:
    def build_ls_cnot(d=d):
        step = 2 * (2 * d - 1)  # patch footprint + gap
        patch_configs = {
            'c': {'distance': d},
            't': {'distance': d},
            'a': {'distance': d},
        }
        exp = CNOTLSExperiment(
            patch_configs=patch_configs,
            offset_ta=(step, 0),
            offset_ca=(0, step),
            initial_state_dict={'a': 'X', 'c': 'X', 't': 'Z'},
            measure_state_dict={'a': 'Z', 'c': 'X', 't': 'X'},
            extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
            rounds=d,
            noise_params=NoiseConfig(p_1q=0, p_2q=0, p_meas=0, p_reset=0),
            noise_model='circuit_level',
        )
        return exp.build()

    info = bench("LS CNOT", build_ls_cnot, d_label=d)
    results[f'ls_cnot_d{d}'] = info

# =============================================================================
# 4. Bell Teleportation — TG variant
# =============================================================================
print("\n=== 4. Bell Teleportation TG ===")

def _build_bell_tg_circuit(d, teleport_state='Z'):
    from lightstim.ir.builder import CircuitBuilder
    from lightstim.ir.logical_executor import LogicalExecutor
    from lightstim.ir.operation import CSSLogicalOpSet
    from lightstim.ir.tracker import SyndromeTracker

    rounds_pre  = d
    rounds_mid  = 1
    rounds_post = 1
    dx = 2 * (2 * d - 1) - 2

    patch1_local = UnrotatedSurfaceCode(distance=d)
    patch2_local = UnrotatedSurfaceCode(distance=d)
    patch3_local = UnrotatedSurfaceCode(distance=d)

    system = QECSystem()
    patch1 = system.add_patch(patch1_local, name='patch1')
    patch2 = system.add_patch(patch2_local, name='patch2', offset=(dx, 0))
    patch3 = system.add_patch(patch3_local, name='patch3', offset=(2 * dx, 0))

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    INIT = {'patch1': teleport_state, 'patch2': 'X', 'patch3': 'Z'}
    init_dict = {q: INIT[system.index_to_owner_map[q]] for q in system.data_indices}
    builder.initialize(init_dict=init_dict, n=system.num_qubits)

    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_pre)

    executor = LogicalExecutor(builder=builder)
    executor.register_op_set(UnrotatedSurfaceCode, CSSLogicalOpSet())

    executor.apply_logical_operation('transversal_cnot', patches=[patch2, patch3])

    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_mid)

    executor.apply_logical_operation('transversal_cnot', patches=[patch1, patch2])

    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_post)

    MEAS = {'patch1': 'X', 'patch2': 'Z', 'patch3': teleport_state}
    meas_dict = {q: MEAS[system.index_to_owner_map[q]] for q in system.data_indices}
    builder.apply_data_readout(final_measurements=meas_dict)
    return builder.circuit

for d in [3, 5, 7]:
    def build_bell_tg(d=d):
        return _build_bell_tg_circuit(d, 'Z')

    info = bench("Bell TG", build_bell_tg, d_label=d)
    results[f'bell_tg_d{d}'] = info

# =============================================================================
# 5. Bell Teleportation — LS ZZ variant
# =============================================================================
print("\n=== 5. Bell Teleportation LS-ZZ ===")
from lightstim.qec_code.surface_code.unrotated import UnrotatedTwoPatchCoupler
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker

def _build_bell_ls_zz_circuit(d, teleport_state='Z'):
    rounds_pre = d
    rounds_ls = d
    gap = 1
    d_size = 2 * d - 1
    step = d_size + gap

    patch1_local = UnrotatedSurfaceCode(distance=d)
    patch2_local = UnrotatedSurfaceCode(distance=d)
    patch3_local = UnrotatedSurfaceCode(distance=d)

    system = QECSystem()
    system.add_patch(patch1_local, name='patch1')
    system.add_patch(patch2_local, name='patch2', offset=(0, step))
    system.add_patch(patch3_local, name='patch3', offset=(0, 2 * step))

    coupler_proto = UnrotatedTwoPatchCoupler()
    system.register_coupler(coupler_proto, patch_names=['patch2', 'patch3'],
                            name='coupler_23', interaction_type='ZZ')
    system.register_coupler(coupler_proto, patch_names=['patch1', 'patch2'],
                            name='coupler_12', interaction_type='ZZ')

    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    init_zz = {'patch1': teleport_state, 'patch2': 'X', 'patch3': 'X'}
    init_dict = {q: init_zz[system.index_to_owner_map[q]]
                 for q in system.data_indices
                 if system.index_to_owner_map[q] in init_zz}
    builder.initialize(init_dict=init_dict, n=system.num_qubits)

    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_pre)

    builder.activate_coupler('coupler_23')
    cp23_local = system.coupler_patches['coupler_23'].data_indices
    cp23_global = [system.local_to_global_map['coupler_23'][q] for q in cp23_local]
    builder.initialize(init_dict={q: 'X' for q in cp23_global}, n=system.num_qubits)
    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_ls)
    builder.deactivate_coupler('coupler_23')
    builder.apply_data_readout(final_measurements={q: 'X' for q in cp23_global})

    builder.activate_coupler('coupler_12')
    cp12_local = system.coupler_patches['coupler_12'].data_indices
    cp12_global = [system.local_to_global_map['coupler_12'][q] for q in cp12_local]
    builder.initialize(init_dict={q: 'X' for q in cp12_global}, n=system.num_qubits)
    se_block = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds_ls)

    meas_zz = {'patch1': 'X', 'patch2': 'X', 'patch3': teleport_state}
    meas_dict = {q: meas_zz[system.index_to_owner_map[q]]
                 for q in system.data_indices
                 if system.index_to_owner_map[q] in meas_zz}
    meas_dict.update({q: 'X' for q in cp12_global})
    builder.apply_data_readout(final_measurements=meas_dict)
    return builder.circuit

for d in [3, 5, 7]:
    def build_bell_ls_zz(d=d):
        return _build_bell_ls_zz_circuit(d, 'Z')

    info = bench("Bell LS-ZZ", build_bell_ls_zz, d_label=d)
    results[f'bell_ls_zz_d{d}'] = info

# =============================================================================
# 6. Steane 7-to-1 Distillation — TG variant
# =============================================================================
print("\n=== 6. Steane 7-to-1 Distillation TG ===")
try:
    sys.path.insert(0, '/home/xiang/workspace/LightStim/benchmarks/logical_circuits/distillation/tg_7to1')
    from TG_distillation_7_to_1 import build_distillation_circuit as build_tg_dist

    for d in [3, 5]:
        def build_tg_7to1(d=d):
            result = build_tg_dist(d=d, rounds=d, r=1)
            return result[0]  # circuit is first element

        info = bench("TG Distillation", build_tg_7to1, d_label=d)
        results[f'tg_dist_d{d}'] = info
except Exception as e:
    print(f"  TG Distillation skipped: {e}")
    import traceback; traceback.print_exc()
finally:
    if '/home/xiang/workspace/LightStim/benchmarks/logical_circuits/distillation/tg_7to1' in sys.path:
        sys.path.remove('/home/xiang/workspace/LightStim/benchmarks/logical_circuits/distillation/tg_7to1')

# =============================================================================
# 7. Steane 7-to-1 Distillation — LS variant
# =============================================================================
print("\n=== 7. Steane 7-to-1 Distillation LS ===")
try:
    sys.path.insert(0, '/home/xiang/workspace/LightStim/benchmarks/logical_circuits/distillation/ls_7to1')
    from LS_distillation_7_to_1 import build_distillation_circuit as build_ls_dist

    for d in [3, 5]:
        def build_ls_7to1(d=d):
            circuit, _, _ = build_ls_dist(d=d, rounds=d)
            return circuit

        info = bench("LS Distillation", build_ls_7to1, d_label=d)
        results[f'ls_dist_d{d}'] = info
except Exception as e:
    print(f"  LS Distillation skipped: {e}")
    import traceback; traceback.print_exc()
finally:
    if '/home/xiang/workspace/LightStim/benchmarks/logical_circuits/distillation/ls_7to1' in sys.path:
        sys.path.remove('/home/xiang/workspace/LightStim/benchmarks/logical_circuits/distillation/ls_7to1')

# =============================================================================
# 8. CrossLS (Surface ↔ PQRM)
# =============================================================================
print("\n=== 8. CrossLS (Surface–PQRM) ===")
try:
    from lightstim.protocols.cross_ls.cross_ls_experiment import CrossLSExperiment

    for d_surf in [3, 5, 7]:
        for pqrm_para, pqrm_label in [([1, 2, 4], '1-2-4')]:
            def build_cross_ls(d_surf=d_surf, pqrm_para=pqrm_para):
                exp = CrossLSExperiment(
                    PQRM_para=pqrm_para,
                    d_surf=d_surf,
                    rounds=d_surf,
                    PQRM_state='Z',
                    surf_state='X',
                    noise_params=NoiseConfig(p_1q=0, p_2q=0, p_meas=0, p_reset=0),
                    noise_model='circuit_level',
                )
                return exp.build()

            info = bench(f"CrossLS PQRM({pqrm_label})", build_cross_ls, d_label=d_surf)
            results[f'cross_ls_pqrm{pqrm_label}_dsurf{d_surf}'] = info
except Exception as e:
    print(f"  CrossLS skipped: {e}")
    import traceback; traceback.print_exc()

# =============================================================================
# Save
# =============================================================================
out_path = out / 'comprehensive_benchmark.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

# Pretty print table
print("\n" + "="*90)
print(f"{'Protocol':<28} {'d':>3} {'Qubits':>7} {'Det':>7} {'Obs':>5} {'AnnLoc':>8} {'ms':>8}")
print("="*90)
for k, v in results.items():
    print(f"  {v['label']:<26} {str(v['d']):>3} {v['num_qubits']:>7} "
          f"{v['num_detectors']:>7} {v['num_observables']:>5} "
          f"{v['annotation_loc']:>8} {v['compile_ms']:>8.1f}")
print("="*90)
