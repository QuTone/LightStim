"""
High-quality LER verification: AutoDEM vs Stim built-in.
Runs 1e9 shots for d=3,5 and 1e8 shots for d=7 to get precise LER ratios.
Removes need for footnote about statistical uncertainty.
"""
import sys, json
sys.path.insert(0, '/home/xiang/workspace/LightStim')

import stim, numpy as np, pymatching
from pathlib import Path
from src.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode
from src.qec_code.surface_code.rotated.SE_block import RotatedSurfaceCodeExtractionBlock as RotatedSEBlock
from src.ir.qec_system import QECSystem
from experiments.memory import MemoryExperiment
from src.noise.config import NoiseConfig

out = Path('/home/xiang/workspace/LightStim/eval/correctness_check/results')
out.mkdir(exist_ok=True)

SHOTS = {3: 1_000_000_000, 5: 1_000_000_000, 7: 100_000_000}
p = 1e-3
results = []

for d in [3, 5, 7]:
    n_shots = SHOTS[d]
    print(f"\n=== d={d}, {n_shots:.0e} shots ===")

    stim_circ = stim.Circuit.generated(
        'surface_code:rotated_memory_z',
        rounds=d, distance=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    patch = RotatedSurfaceCode(distance=d)
    system = QECSystem()
    system.add_patch(patch)
    noise_cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    autodem_circ = MemoryExperiment(
        qec_system=system,
        extraction_block_class=RotatedSEBlock,
        rounds=d,
        noise_params=noise_cfg,
        noise_model='circuit_level',
        basis='Z',
    ).build()

    det_match = stim_circ.num_detectors == autodem_circ.num_detectors
    obs_match = stim_circ.num_observables == autodem_circ.num_observables
    print(f"  Det: {stim_circ.num_detectors}/{autodem_circ.num_detectors} match={det_match}")

    lers = {}
    for label, circ in [('Stim', stim_circ), ('AutoDEM', autodem_circ)]:
        dem = circ.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)

        BATCH = 10_000_000
        total_errors = 0
        rng = np.random.default_rng(42)
        sampler = circ.compile_detector_sampler(seed=42)

        batches_done = 0
        shots_done = 0
        while shots_done < n_shots:
            b = min(BATCH, n_shots - shots_done)
            det_samples, obs_actual = sampler.sample(b, separate_observables=True)
            preds = matcher.decode_batch(det_samples)
            total_errors += int(np.sum(preds != obs_actual))
            shots_done += b
            batches_done += 1
            if batches_done % 10 == 0:
                print(f"    {label}: {shots_done:.2e}/{n_shots:.0e} shots, errors={total_errors}")

        ler = total_errors / n_shots
        print(f"  {label}: LER={ler:.4e} ({total_errors}/{n_shots})")
        lers[label] = {'ler': ler, 'errors': total_errors, 'shots': n_shots}

    ratio = lers['AutoDEM']['ler'] / lers['Stim']['ler'] if lers['Stim']['ler'] > 0 else None
    print(f"  Ratio (AutoDEM/Stim) = {ratio:.4f}" if ratio else "  Ratio: N/A")

    results.append({
        'd': d, 'shots': n_shots,
        'stim_det': stim_circ.num_detectors, 'autodem_det': autodem_circ.num_detectors,
        'det_match': det_match, 'obs_match': obs_match,
        'stim_ler': lers['Stim']['ler'], 'autodem_ler': lers['AutoDEM']['ler'],
        'stim_errors': lers['Stim']['errors'], 'autodem_errors': lers['AutoDEM']['errors'],
        'ratio': ratio,
    })

    # Checkpoint after each d
    with open(out / 'hq_verification.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved checkpoint.")

print("\nFinal results:")
for r in results:
    print(f"  d={r['d']}: ratio={r['ratio']:.4f}, stim_ler={r['stim_ler']:.3e}, autodem_ler={r['autodem_ler']:.3e}, errors={r['autodem_errors']}")
