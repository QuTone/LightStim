"""
P0-2: Correctness verification — AutoDEM vs Stim built-in
Compares rotated surface code memory circuits for d=3,5,7.
Checks: detector count, observable count, noise instruction count, LER agreement.
"""
import sys
sys.path.insert(0, '/home/xiang/workspace/LightStim')

import stim
import numpy as np
import json
from pathlib import Path

from src.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode
from src.qec_code.surface_code.rotated.SE_block import RotatedSurfaceCodeExtractionBlock as RotatedSEBlock
from src.ir.qec_system import QECSystem
from experiments.memory import MemoryExperiment
from src.noise.config import NoiseConfig

out_dir = Path('/home/xiang/workspace/LightStim/eval/correctness_check/results')
out_dir.mkdir(parents=True, exist_ok=True)

results = []

for d in [3, 5, 7]:
    print(f"\n{'='*50}")
    print(f"d = {d}")
    print('='*50)

    rounds = d
    p = 1e-3

    # ---- Stim built-in ----
    stim_circ = stim.Circuit.generated(
        'surface_code:rotated_memory_z',
        rounds=rounds,
        distance=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    stim_detectors = stim_circ.num_detectors
    stim_observables = stim_circ.num_observables
    stim_noise_instrs = [str(inst) for inst in stim_circ.flattened()
                         if inst.name in ['DEPOLARIZE1','DEPOLARIZE2','X_ERROR','Z_ERROR','PAULI_CHANNEL_1']]

    # ---- AutoDEM ----
    patch = RotatedSurfaceCode(distance=d)
    system = QECSystem()
    system.add_patch(patch)
    noise_cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p)
    exp = MemoryExperiment(
        qec_system=system,
        extraction_block_class=RotatedSEBlock,
        rounds=rounds,
        noise_params=noise_cfg,
        noise_model='circuit_level',
        basis='Z',
    )
    autodem_circ = exp.build()
    autodem_detectors = autodem_circ.num_detectors
    autodem_observables = autodem_circ.num_observables
    autodem_noise_instrs = [str(inst) for inst in autodem_circ.flattened()
                            if inst.name in ['DEPOLARIZE1','DEPOLARIZE2','X_ERROR','Z_ERROR','PAULI_CHANNEL_1']]

    det_match = stim_detectors == autodem_detectors
    obs_match = stim_observables == autodem_observables
    noise_match = len(stim_noise_instrs) == len(autodem_noise_instrs)

    print(f"  Stim:    detectors={stim_detectors}, observables={stim_observables}, noise_instr={len(stim_noise_instrs)}")
    print(f"  AutoDEM: detectors={autodem_detectors}, observables={autodem_observables}, noise_instr={len(autodem_noise_instrs)}")
    print(f"  Detector match: {det_match}")
    print(f"  Observable match: {obs_match}")
    print(f"  Noise instruction count match: {noise_match}")

    # ---- LER comparison (fixed seed) ----
    n_shots = 200_000
    print(f"  Running {n_shots} shots LER comparison...")

    ler_results = {}
    import pymatching
    for label, circ in [('Stim', stim_circ), ('AutoDEM', autodem_circ)]:
        try:
            dem = circ.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            sampler = circ.compile_detector_sampler(seed=42)
            det_samples, obs_actual = sampler.sample(n_shots, separate_observables=True)
            predictions = matcher.decode_batch(det_samples)
            errors = int(np.sum(predictions != obs_actual))
            ler = errors / n_shots
            print(f"    {label}: LER = {ler:.4e} ({errors}/{n_shots} errors)")
            ler_results[label] = {'ler': ler, 'errors': errors, 'shots': n_shots}
        except Exception as e:
            print(f"    {label}: ERROR — {e}")
            ler_results[label] = {'error': str(e)}

    if 'Stim' in ler_results and 'AutoDEM' in ler_results:
        s = ler_results['Stim'].get('ler', None)
        a = ler_results['AutoDEM'].get('ler', None)
        if s and a and s > 0:
            ratio = a / s
            print(f"    LER ratio (AutoDEM/Stim) = {ratio:.3f}")
            ler_results['ratio'] = ratio

    results.append({
        'd': d,
        'stim_detectors': stim_detectors,
        'autodem_detectors': autodem_detectors,
        'det_match': det_match,
        'stim_observables': stim_observables,
        'autodem_observables': autodem_observables,
        'obs_match': obs_match,
        'noise_count_match': noise_match,
        'ler': ler_results,
    })

with open(out_dir / 'surface_code_correctness.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_dir}/surface_code_correctness.json")
