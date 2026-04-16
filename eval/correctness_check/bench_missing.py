"""Quick 1-trial benchmark for missing table entries:
  - BB [[144,12,12]] r=12
  - CrossLS d_surf=7 + PQRM(1,4,6)
"""
import sys, time, json
sys.path.insert(0, '/home/xiang/workspace/LightStim')

from pathlib import Path
out = Path('/home/xiang/workspace/LightStim/eval/correctness_check/results')

def count_ann_loc(circuit):
    lines = str(circuit.flattened()).split('\n')
    return sum(1 for l in lines if l.strip().startswith('DETECTOR') or l.strip().startswith('OBSERVABLE_INCLUDE'))

results = {}

# ── BB [[144,12,12]] r=12 ────────────────────────────────────────────────────
print("\n=== BB [[144,12,12]] r=12 ===")
try:
    from src.qec_code.BB_code.code_patch import BBCode
    from src.qec_code.BB_code.SE_block import BBCodeExtractionBlock
    from src.ir.qec_system import QECSystem
    from experiments.memory import MemoryExperiment
    from src.noise.config import NoiseConfig

    t0 = time.perf_counter()
    A = [[3,0],[0,1],[0,2]]
    B = [[0,3],[1,0],[2,0]]
    patch = BBCode(l=12, m=6, A=A, B=B)
    system = QECSystem()
    system.add_patch(patch)
    nc = NoiseConfig(p_1q=0, p_2q=0, p_meas=0, p_reset=0)
    circuit = MemoryExperiment(system, BBCodeExtractionBlock, rounds=12,
                               noise_params=nc, noise_model='circuit_level', basis='Z').build()
    t = time.perf_counter() - t0
    ann = count_ann_loc(circuit)
    results['bb_144_r12'] = {
        'num_qubits': circuit.num_qubits,
        'num_detectors': circuit.num_detectors,
        'annotation_loc': ann,
        'compile_ms': round(t * 1000, 1),
    }
    print(f"  => {circuit.num_qubits}q, {circuit.num_detectors}det, {ann} Ann LoC, {t*1000:.0f}ms")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ── CrossLS d_surf=7 + PQRM(1,4,6) ─────────────────────────────────────────
print("\n=== CrossLS d_surf=7 + PQRM(1,4,6) ===")
try:
    from experiments.cross_ls.cross_ls_experiment import CrossLSExperiment

    t0 = time.perf_counter()
    exp = CrossLSExperiment(
        d_surf=7,
        PQRM_para=[1, 4, 6],
        PQRM_state='Z',
        rounds=7,
        noise_params=NoiseConfig(p_1q=0, p_2q=0, p_meas=0, p_reset=0),
        noise_model='circuit_level',
    )
    circuit = exp.build()
    t = time.perf_counter() - t0
    ann = count_ann_loc(circuit)
    results['cross_ls_146_dsurf7'] = {
        'num_qubits': circuit.num_qubits,
        'num_detectors': circuit.num_detectors,
        'annotation_loc': ann,
        'compile_ms': round(t * 1000, 1),
    }
    print(f"  => {circuit.num_qubits}q, {circuit.num_detectors}det, {ann} Ann LoC, {t*1000:.0f}ms")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

print("\n=== Results ===")
for k, v in results.items():
    print(f"  {k}: {v}")

with open(out / 'bench_missing.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved to bench_missing.json")
