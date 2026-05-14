#!/usr/bin/env python3
"""
CrossLS rounds sweep: PQRM(1,2,4) + d_surf={3,5,7} x rounds={3,5,7}.
MWPF CPU 32w, hybrid post-selection, PQRM_state=Z.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from lightstim.protocols.cross_ls import CrossLSExperiment
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig
from lightstim.simulation.decoder_backend.post_select import get_post_select_detector_indices

P = 1e-3
NOISE = NoiseConfig(p_1q=1e-6, p_2q=P, p_meas=P, p_reset=P)

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("mwpf", backend="cpu"),
    max_shots=2_000_000,
    max_errors=200,
    num_workers=32,
    print_progress=False,
)

print(f"Rounds sweep: PQRM(1,2,4), p={P}, PQRM_state=Z, hybrid PS, MWPF CPU 32w")
print(f"{'d_surf':<7} {'rounds':<7} {'#PS':>5} {'#det':>6} {'Shots':>10} {'Kept':>10} {'PS Rate':>8} {'Errors':>8} {'LER':>12} {'Time':>7}")
print("-" * 95)
sys.stdout.flush()

for d in [3, 5, 7]:
    for r in [3, 5, 7]:
        exp = CrossLSExperiment(
            PQRM_para=[1, 2, 4], d_surf=d, rounds=r,
            PQRM_state="Z", surf_state="X",
            noise_params=NOISE, if_detector=True,
            post_select_hybrid=True,
        )
        circ = exp.build()
        n_ps = len(get_post_select_detector_indices(circ))
        n_det = circ.num_detectors

        stats = pipeline.run(circuit=circ)
        ler = stats.logical_error_rate
        ps_rate = stats.post_selection_rate
        print(f"{d:<7} {r:<7} {n_ps:>5} {n_det:>6} {stats.shots:>10,} {stats.post_selected_shots:>10,} {ps_rate:>7.1%} {stats.errors:>8,} {ler:>12.2e} {stats.seconds:>6.0f}s")
        sys.stdout.flush()
    print()
    sys.stdout.flush()

print("Done.")
