"""
Addendum A2: Extended compilation benchmark with larger instances.
New: BB [[72,12,6]], BB [[144,12,12]], LS Distillation d=7, TG Distillation d=7 (timeout expected).
Also logs AutoDEM LoC (Python API lines) for each protocol.
"""
import sys, time, json
sys.path.insert(0, '/home/xiang/workspace/LightStim')

import numpy as np
from pathlib import Path

out = Path('/home/xiang/workspace/LightStim/eval/correctness_check/results')
out.mkdir(parents=True, exist_ok=True)

N_TRIALS = 3
TIMEOUT_SEC = 600  # 10 minutes

results = {}

def count_annotation_loc(circuit):
    expanded = circuit.flattened()
    lines = str(expanded).split('\n')
    n_det = sum(1 for l in lines if l.strip().startswith('DETECTOR'))
    n_obs = sum(1 for l in lines if l.strip().startswith('OBSERVABLE_INCLUDE'))
    return n_det + n_obs

def bench(label, build_fn, d_label="", autodem_loc=None):
    import signal
    def timeout_handler(signum, frame):
        raise TimeoutError("Compilation timed out")

    times = []
    circuit = None
    timed_out = False

    for trial in range(N_TRIALS):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SEC)
        try:
            t0 = time.perf_counter()
            circuit = build_fn()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        except TimeoutError:
            timed_out = True
            print(f"  {label} d={d_label}: TIMEOUT (>{TIMEOUT_SEC//60}min) after trial {trial+1}")
            break
        finally:
            signal.alarm(0)

    if timed_out:
        info = {
            'label': label,
            'd': d_label,
            'compile_ms': None,
            'timeout': True,
            'timeout_sec': TIMEOUT_SEC,
            'num_qubits': None,
            'num_detectors': None,
            'num_observables': None,
            'annotation_loc': None,
            'autodem_loc': autodem_loc,
        }
        return info

    t_med = np.median(times)
    ann_loc = count_annotation_loc(circuit)
    info = {
        'label': label,
        'd': d_label,
        'compile_ms': round(t_med * 1000, 1),
        'timeout': False,
        'num_qubits': circuit.num_qubits,
        'num_detectors': circuit.num_detectors,
        'num_observables': circuit.num_observables,
        'annotation_loc': ann_loc,
        'autodem_loc': autodem_loc,
    }
    print(f"  {label} d={d_label}: {circuit.num_qubits}q, "
          f"{circuit.num_detectors}det, annot={ann_loc}, "
          f"autodem_loc={autodem_loc}, compile={t_med*1000:.1f}ms")
    return info

# ============================================================
# 1. Rotated Surface Code Memory — extended to d=31 if feasible
# ============================================================
print("\n=== 1. Rotated SC Memory (extended) ===")
from src.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode
from src.qec_code.surface_code.rotated.SE_block import RotatedSurfaceCodeExtractionBlock as RotatedSEBlock
from src.ir.qec_system import QECSystem
from experiments.memory import MemoryExperiment
from src.noise.config import NoiseConfig

for d in [3, 7, 11, 15, 21]:
    def build_sc_mem(d=d):
        patch = RotatedSurfaceCode(distance=d)
        system = QECSystem()
        system.add_patch(patch)
        noise_cfg = NoiseConfig(p_1q=0, p_2q=0, p_meas=0, p_reset=0)
        exp = MemoryExperiment(system, RotatedSEBlock, rounds=d,
                               noise_params=noise_cfg, noise_model='circuit_level', basis='Z')
        return exp.build()

    info = bench("Rotated SC Memory", build_sc_mem, d_label=d, autodem_loc=4)
    results[f'sc_mem_d{d}'] = info
    # Stop if close to timeout
    if info['compile_ms'] and info['compile_ms'] > 30000:
        print(f"  Stopping SC memory at d={d} ({info['compile_ms']}ms, approaching timeout)")
        break

# ============================================================
# 2. BB Code Memory
# ============================================================
print("\n=== 2. BB Code Memory ===")
from src.qec_code.BB_code.code_patch import BBCode
from src.qec_code.BB_code.SE_block import BBCodeExtractionBlock

BB_PARAMS = {
    '72,12,6': {'l': 6, 'm': 6, 'A': [[3,0],[0,1],[0,2]], 'B': [[0,3],[1,0],[2,0]]},
    '144,12,12': {'l': 12, 'm': 6, 'A': [[3,0],[0,1],[0,2]], 'B': [[0,3],[1,0],[2,0]]},
}

for label, params in BB_PARAMS.items():
    def build_bb_mem(params=params):
        patch = BBCode(**params)
        system = QECSystem()
        system.add_patch(patch)
        noise_cfg = NoiseConfig(p_1q=0, p_2q=0, p_meas=0, p_reset=0)
        exp = MemoryExperiment(system, BBCodeExtractionBlock, rounds=6,
                               noise_params=noise_cfg, noise_model='circuit_level', basis='Z')
        return exp.build()

    info = bench(f"BB [[{label}]] Memory", build_bb_mem, d_label='—', autodem_loc=5)
    results[f'bb_{label}'] = info

# ============================================================
# 3. LS Distillation d=5, 7
# ============================================================
print("\n=== 3. LS Distillation (extended) ===")
sys.path.insert(0, '/home/xiang/workspace/LightStim/eval/logical_circuit_benchmark/distillation/ls_7to1')
try:
    from LS_distillation_7_to_1 import build_distillation_circuit as build_ls_dist

    for d in [5, 7]:
        def build_ls_7to1(d=d):
            circuit, _, _ = build_ls_dist(d=d, rounds=d)
            return circuit

        info = bench("LS Distillation", build_ls_7to1, d_label=d, autodem_loc=15)
        results[f'ls_dist_d{d}_addendum'] = info
        if info.get('timeout'):
            break
except Exception as e:
    print(f"  LS Distillation error: {e}")
finally:
    if '/home/xiang/workspace/LightStim/eval/logical_circuit_benchmark/distillation/ls_7to1' in sys.path:
        sys.path.remove('/home/xiang/workspace/LightStim/eval/logical_circuit_benchmark/distillation/ls_7to1')

# ============================================================
# 4. TG Distillation d=7 (expect timeout)
# ============================================================
print("\n=== 4. TG Distillation d=7 (may timeout) ===")
sys.path.insert(0, '/home/xiang/workspace/LightStim/eval/logical_circuit_benchmark/distillation/tg_7to1')
try:
    from TG_distillation_7_to_1 import build_distillation_circuit as build_tg_dist

    d = 7
    def build_tg_7to1_d7():
        result = build_tg_dist(d=7, rounds=7, r=1)
        return result[0]

    info = bench("TG Distillation", build_tg_7to1_d7, d_label=7, autodem_loc=15)
    results[f'tg_dist_d7_addendum'] = info
except Exception as e:
    print(f"  TG Distillation d=7 error: {e}")
    import traceback; traceback.print_exc()
finally:
    if '/home/xiang/workspace/LightStim/eval/logical_circuit_benchmark/distillation/tg_7to1' in sys.path:
        sys.path.remove('/home/xiang/workspace/LightStim/eval/logical_circuit_benchmark/distillation/tg_7to1')

# ============================================================
# Save
# ============================================================
# Merge with existing comprehensive benchmark
existing_path = out / 'comprehensive_benchmark.json'
if existing_path.exists():
    with open(existing_path) as f:
        existing = json.load(f)
    existing.update(results)
    results_all = existing
else:
    results_all = results

with open(out / 'addendum_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)
with open(existing_path, 'w') as f:
    json.dump(results_all, f, indent=2)

print(f"\nSaved addendum results to {out}/addendum_benchmark.json")

# Print summary
print("\n" + "="*90)
print(f"{'Protocol':<35} {'d':>5} {'Qubits':>7} {'Det':>7} {'AnnLoC':>8} {'ADLoC':>6} {'ms':>10}")
print("="*90)
for k, v in results.items():
    if v.get('timeout'):
        print(f"  {v['label']:<33} {str(v['d']):>5} {'—':>7} {'—':>7} {'—':>8} "
              f"{str(v['autodem_loc']):>6} {'TIMEOUT':>10}")
    elif v.get('num_qubits') is not None:
        print(f"  {v['label']:<33} {str(v['d']):>5} {v['num_qubits']:>7} "
              f"{v['num_detectors']:>7} {v['annotation_loc']:>8} "
              f"{str(v['autodem_loc'] or '?'):>6} {v['compile_ms']:>10.1f}")
print("="*90)
