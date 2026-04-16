"""
P2-3: Compilation Performance Benchmark
Measures AutoDEM compilation time vs Stim circuit generation time.
Protocols: surface memory, LS CNOT, TG CNOT (approx), BB memory.
"""
import sys, time, json
sys.path.insert(0, '/home/xiang/workspace/LightStim')

import stim
import numpy as np
from pathlib import Path

out = Path('/home/xiang/workspace/LightStim/eval/correctness_check/results')
out.mkdir(parents=True, exist_ok=True)

from src.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode
from src.qec_code.surface_code.rotated.SE_block import RotatedSurfaceCodeExtractionBlock as RotatedSEBlock
from src.ir.qec_system import QECSystem
from experiments.memory import MemoryExperiment
from src.noise.config import NoiseConfig

N_TRIALS = 3
results = {}

# ─── Surface code memory ─────────────────────────────────────────────────────
print("=== Surface Code Memory ===")
for d in [3, 5, 7, 9, 11]:
    rounds = d

    # Stim built-in
    times_stim = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        circ = stim.Circuit.generated(
            'surface_code:rotated_memory_z', rounds=rounds, distance=d,
            after_clifford_depolarization=1e-3, before_round_data_depolarization=1e-3,
            before_measure_flip_probability=1e-3, after_reset_flip_probability=1e-3,
        )
        times_stim.append(time.perf_counter() - t0)

    # AutoDEM
    times_autodem = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        patch = RotatedSurfaceCode(distance=d)
        system = QECSystem()
        system.add_patch(patch)
        noise_cfg = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3)
        exp = MemoryExperiment(system, RotatedSEBlock, rounds=rounds,
                               noise_params=noise_cfg, noise_model='circuit_level', basis='Z')
        _ = exp.build()
        times_autodem.append(time.perf_counter() - t0)

    n_qubits = d * d + (d-1)*(d-1)
    stim_t = np.median(times_stim)
    auto_t = np.median(times_autodem)
    print(f"  d={d:2d} ({n_qubits:3d}q, {rounds}r): Stim={stim_t*1000:.1f}ms, AutoDEM={auto_t*1000:.1f}ms, ratio={auto_t/stim_t:.1f}x")
    results[f'surface_memory_d{d}'] = {
        'protocol': 'surface_memory',
        'd': d, 'rounds': rounds, 'n_qubits': n_qubits,
        'stim_ms': round(stim_t*1000, 2),
        'autodem_ms': round(auto_t*1000, 2),
        'ratio': round(auto_t/stim_t, 2),
    }

# ─── BB code memory ────────────────────────────────────────────────────────
print("\n=== BB Code Memory ===")
try:
    from src.qec_code.BB_code.code_patch import BBCode
    from src.qec_code.BB_code.SE_block import BBCodeExtractionBlock

    for (l, m), code_label in [((6, 6), '[[72,12,6]]'), ((9, 6), '[[108,8,10]]')]:
        rounds = 6
        times_autodem = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            patch = BBCode(l=l, m=m)
            system = QECSystem()
            system.add_patch(patch)
            noise_cfg = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3)
            exp = MemoryExperiment(system, BBCodeExtractionBlock, rounds=rounds,
                                   noise_params=noise_cfg, noise_model='circuit_level', basis='Z')
            _ = exp.build()
            times_autodem.append(time.perf_counter() - t0)

        auto_t = np.median(times_autodem)
        n_qubits = 2 * l * m
        print(f"  {code_label} ({n_qubits}q, {rounds}r): AutoDEM={auto_t*1000:.1f}ms")
        results[f'bb_{code_label}'] = {
            'protocol': 'bb_memory',
            'code': code_label, 'rounds': rounds, 'n_qubits': n_qubits,
            'autodem_ms': round(auto_t*1000, 2),
        }
except Exception as e:
    print(f"  BB code skipped: {e}")

with open(out / 'compilation_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out}/compilation_benchmark.json")
