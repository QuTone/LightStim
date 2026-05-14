"""
Extended compilation benchmark for Table 4.
New instances: TG Distill d=7, LS Distill d=11, SC Mem d=31,
               TG CNOT d=15, LS CNOT d=15, Bell Tele (TG/LS-ZZ) d=15.
No hard timeout — runs to completion.
"""
import sys, time, json
sys.path.insert(0, '/home/xiang/workspace/LightStim')
import numpy as np
from pathlib import Path

out = Path('/home/xiang/workspace/LightStim/eval/correctness_check/results')
out.mkdir(exist_ok=True)

N_TRIALS = 1
# Load existing checkpoint to skip already-completed tasks
_chk = out / 'extended_benchmark.json'
results = json.loads(_chk.read_text()) if _chk.exists() else {}

def count_ann_loc(circuit):
    lines = str(circuit.flattened()).split('\n')
    return sum(1 for l in lines if l.strip().startswith('DETECTOR') or l.strip().startswith('OBSERVABLE_INCLUDE'))

def bench(label, build_fn, d_label, autodem_loc, result_key):
    if result_key in results:
        print(f"  {label} d={d_label} — ALREADY DONE (skipping)", flush=True)
        return results[result_key]
    print(f"  {label} d={d_label}...", flush=True)
    times = []
    circuit = None
    for trial in range(N_TRIALS):
        t0 = time.perf_counter()
        circuit = build_fn()
        times.append(time.perf_counter() - t0)
        print(f"    trial {trial+1}: {times[-1]*1000:.0f}ms", flush=True)
    t_med = np.median(times)
    ann = count_ann_loc(circuit)
    info = {
        'label': label, 'd': d_label,
        'compile_ms': round(t_med * 1000, 1),
        'num_qubits': circuit.num_qubits,
        'num_detectors': circuit.num_detectors,
        'annotation_loc': ann,
        'autodem_loc': autodem_loc,
    }
    print(f"  => {circuit.num_qubits}q, {circuit.num_detectors}det, {t_med*1000:.0f}ms")
    results[result_key] = info
    # checkpoint
    with open(out / 'extended_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    return info

# ============================================================
from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode
from lightstim.qec_code.surface_code.rotated.SE_block import RotatedSurfaceCodeExtractionBlock as RotatedSEBlock
from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.noise.config import NoiseConfig

print("\n=== Surface Memory d=31 ===")
def build_sc31():
    patch = RotatedSurfaceCode(distance=31)
    system = QECSystem()
    system.add_patch(patch)
    nc = NoiseConfig(p_1q=0,p_2q=0,p_meas=0,p_reset=0)
    return MemoryExperiment(system, RotatedSEBlock, rounds=31, noise_params=nc, noise_model='circuit_level', basis='Z').build()
bench("Surface Mem", build_sc31, 31, 4, "sc_mem_d31")

# ============================================================
print("\n=== TG CNOT d=15 ===")
from lightstim.qec_code.surface_code.unrotated import UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock, UnrotatedTwoPatchCoupler
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.ir.operation import CSSLogicalOpSet

def build_tg_cnot(d):
    dx = 2 * (2*d - 1) - 2  # same spacing as run_tg.py
    system = QECSystem()
    patch1 = system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch1')
    patch2 = system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch2', offset=(dx, 0))
    tracker = SyndromeTracker(num_qubits=system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()
    init_dict = {q: 'Z' for q in system.data_indices}
    builder.initialize(init_dict=init_dict, n=system.num_qubits)
    se = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=d)
    executor = LogicalExecutor(builder=builder)
    executor.register_op_set(UnrotatedSurfaceCode, CSSLogicalOpSet())
    executor.apply_logical_operation('transversal_cnot', patches=[patch1, patch2])
    se2 = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se2.circuit, rounds=1)
    meas_dict = {q: 'Z' for q in system.data_indices}
    builder.apply_data_readout(final_measurements=meas_dict)
    return builder.circuit

bench("TG CNOT", lambda: build_tg_cnot(15), 15, 8, "tg_cnot_d15")

# ============================================================
print("\n=== LS CNOT d=15 ===")
# Use CNOTLSExperiment (3-patch protocol) for consistency with existing d=7 table row
from lightstim.protocols.cnot_ls import CNOTLSExperiment

def build_ls_cnot(d):
    step = 2 * (2 * d - 1)
    exp = CNOTLSExperiment(
        patch_configs={'c': {'distance': d}, 't': {'distance': d}, 'a': {'distance': d}},
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

bench("LS CNOT", lambda: build_ls_cnot(15), 15, 6, "ls_cnot_d15")

# ============================================================
print("\n=== Bell Teleportation TG d=15 ===")
# CSSLogicalOpSet already imported above

def build_bell_tg(d, teleport_state='Z'):
    rounds_pre = d; rounds_mid = 1; rounds_post = 1
    dx = 2 * (2*d - 1) - 2
    system = QECSystem()
    patch1 = system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch1')
    patch2 = system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch2', offset=(dx, 0))
    patch3 = system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch3', offset=(2*dx, 0))
    tracker = SyndromeTracker(num_qubits=system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()
    INIT = {'patch1': teleport_state, 'patch2': 'X', 'patch3': 'Z'}
    init_dict = {q: INIT[system.index_to_owner_map[q]] for q in system.data_indices}
    builder.initialize(init_dict=init_dict, n=system.num_qubits)
    se = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds_pre)
    executor = LogicalExecutor(builder=builder)
    executor.register_op_set(UnrotatedSurfaceCode, CSSLogicalOpSet())
    executor.apply_logical_operation('transversal_cnot', patches=[patch2, patch3])
    se2 = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se2.circuit, rounds=rounds_mid)
    executor.apply_logical_operation('transversal_cnot', patches=[patch1, patch2])
    se3 = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se3.circuit, rounds=rounds_post)
    MEAS = {'patch1': 'X', 'patch2': 'Z', 'patch3': teleport_state}
    meas_dict = {q: MEAS[system.index_to_owner_map[q]] for q in system.data_indices}
    builder.apply_data_readout(final_measurements=meas_dict)
    return builder.circuit

bench("Bell Tele.(TG)", lambda: build_bell_tg(15), 15, 10, "bell_tg_d15")

# ============================================================
print("\n=== Bell Teleportation LS-ZZ d=15 ===")
def build_bell_ls_zz(d, teleport_state='Z'):
    rounds_pre = d; rounds_ls = d
    d_size = 2*d - 1; step = d_size + 1
    system = QECSystem()
    patch1 = system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch1')
    patch2 = system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch2', offset=(0, step))
    patch3 = system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch3', offset=(0, 2*step))
    coupler_proto = UnrotatedTwoPatchCoupler()
    system.register_coupler(coupler_proto, patch_names=['patch2','patch3'], name='coupler_23', interaction_type='ZZ')
    system.register_coupler(coupler_proto, patch_names=['patch1','patch2'], name='coupler_12', interaction_type='ZZ')
    tracker = SyndromeTracker(num_qubits=system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()
    init_zz = {'patch1': teleport_state, 'patch2': 'X', 'patch3': 'X'}
    init_dict = {q: init_zz[system.index_to_owner_map[q]] for q in system.data_indices if system.index_to_owner_map[q] in init_zz}
    builder.initialize(init_dict=init_dict, n=system.num_qubits)
    se = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds_pre)
    builder.activate_coupler('coupler_23')
    cp23 = [system.local_to_global_map['coupler_23'][q] for q in system.coupler_patches['coupler_23'].data_indices]
    builder.initialize(init_dict={q:'X' for q in cp23}, n=system.num_qubits)
    se2 = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se2.circuit, rounds=rounds_ls)
    builder.deactivate_coupler('coupler_23')
    builder.apply_data_readout(final_measurements={q:'X' for q in cp23})
    builder.activate_coupler('coupler_12')
    cp12 = [system.local_to_global_map['coupler_12'][q] for q in system.coupler_patches['coupler_12'].data_indices]
    builder.initialize(init_dict={q:'X' for q in cp12}, n=system.num_qubits)
    se3 = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se3.circuit, rounds=rounds_ls)
    builder.deactivate_coupler('coupler_12')
    meas_dict = {q:'Z' for q in system.data_indices}
    builder.apply_data_readout(final_measurements=meas_dict)
    return builder.circuit

bench("Bell Tele.(LS-ZZ)", lambda: build_bell_ls_zz(15), 15, 10, "bell_ls_zz_d15")

# ============================================================
print("\n=== TG Distillation d=7 (no timeout) ===")
sys.path.insert(0, '/home/xiang/workspace/LightStim/eval/logical_circuit_benchmark/distillation/tg_7to1')
try:
    from TG_distillation_7_to_1 import build_distillation_circuit as build_tg_dist
    def build_tg_d7():
        result = build_tg_dist(d=7, rounds=7, r=1)
        return result[0]
    bench("TG Distill.", build_tg_d7, 7, 15, "tg_dist_d7")
except Exception as e:
    print(f"  TG Distill d=7 error: {e}")
    import traceback; traceback.print_exc()
finally:
    p_ = '/home/xiang/workspace/LightStim/eval/logical_circuit_benchmark/distillation/tg_7to1'
    if p_ in sys.path: sys.path.remove(p_)

# ============================================================
print("\n=== LS Distillation d=9 ===")
sys.path.insert(0, '/home/xiang/workspace/LightStim/eval/logical_circuit_benchmark/distillation/ls_7to1')
try:
    from LS_distillation_7_to_1 import build_distillation_circuit as build_ls_dist
    def build_ls_d9():
        circuit, _, _ = build_ls_dist(d=9, rounds=9)
        return circuit
    bench("LS Distill.", build_ls_d9, 9, 15, "ls_dist_d9")
except Exception as e:
    print(f"  LS Distill d=9 error: {e}")
    import traceback; traceback.print_exc()
finally:
    p_ = '/home/xiang/workspace/LightStim/eval/logical_circuit_benchmark/distillation/ls_7to1'
    if p_ in sys.path: sys.path.remove(p_)

print("\n=== Done ===")
with open(out / 'extended_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)
for k, v in results.items():
    print(f"  {k}: {v['num_qubits']}q {v['num_detectors']}det {v['compile_ms']:.0f}ms")
