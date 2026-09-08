"""Fixed-shot BB144 paper-point check using a separately built upstream checkout.

Run from the LightStim checkout (or after installing LightStim). No downloads,
compilation, or third-party code redistribution are performed by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from importlib.metadata import version
from statistics import NormalDist

import numpy as np
import stim


def wilson(errors, shots, z):
    p = errors / shots
    denom = 1 + z*z/shots
    center = (p + z*z/(2*shots))/denom
    radius = z*np.sqrt(p*(1-p)/shots + z*z/(4*shots*shots))/denom
    return [float(max(0, center-radius)), float(min(1, center+radius))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--shots', type=int, default=2000, help='Fixed shots per X/Z memory')
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--p', type=float, default=0.003, choices=[0.001,0.002,0.003,0.004,0.005,0.006])
    parser.add_argument('--compare-shots', type=int, default=64)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/decoding/results/ionq_beam_search.json'))
    args = parser.parse_args()
    if args.shots <= 0 or args.compare_shots < 0:
        parser.error('shots must be positive and compare-shots nonnegative')
    root = args.upstream.resolve()
    # Explicit user-supplied checkout; also expose upstream's beamsearch.py.
    sys.path[:0] = [str(root / 'decoder'), str(root)]
    from beamsearch import BeamSearch
    from stimbposd.dem_to_matrices import detector_error_model_to_check_matrices
    from lightstim.simulation.decoder_backend.registry import get_decoder
    from lightstim.simulation.decoder_backend.dem_matrices import dem_to_matrices

    params = dict(max_rounds=10, beam_width=8, num_results=1,
                  initial_iters=30, iters_per_round=20)
    report = dict(
        upstream_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip(),
        code='[[144,12,12]]', rounds=12, physical_error_rate=args.p,
        decoder='ionq-beam-search', paper_configuration='beam8_230iters', params=params,
        failure_policy='ignore (upstream paper scoring); strict counts also recorded',
        versions={name: version(name) for name in ['numpy','scipy','stim','sinter','ldpc','stimbposd']},
        python=sys.version, bases={},
    )
    for index, basis in enumerate('XZ'):
        circuit_path = root / 'StimCircuit' / f'BB[[144,12,12]],memory_{basis},error_rate={args.p},syndrome_rounds=12.stim'
        circuit = stim.Circuit.from_file(circuit_path)
        dem = circuit.detector_error_model(decompose_errors=False)
        H, O, priors = dem_to_matrices(dem, sparse=True, merge_duplicates=True)
        matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
        assert H.shape == matrices.check_matrix.shape
        assert (H != matrices.check_matrix).nnz == 0
        assert (O != matrices.observables_matrix).nnz == 0
        np.testing.assert_array_equal(priors, matrices.priors)
        decoder = get_decoder('ionq-beam-search', **params)
        compiled = decoder.compile_decoder_for_dem(dem=dem)
        reference = BeamSearch(dem, **params)
        # Sample once so changing the decoding batch size doesn't change samples.
        syndromes, observables = circuit.compile_detector_sampler(seed=args.seed+index).sample(
            args.shots, separate_observables=True)
        errors = failures = strict_errors = mismatches = compared = 0
        decode_seconds = 0.0
        for start in range(0, args.shots, 100):
            stop = min(start+100, args.shots)
            dets = syndromes[start:stop]
            t0 = time.perf_counter()
            packed = compiled.decode_shots_bit_packed(
                bit_packed_detection_event_data=np.packbits(dets, axis=1, bitorder='little'))
            decode_seconds += time.perf_counter()-t0
            predictions = np.unpackbits(packed, axis=1, bitorder='little')[:, :circuit.num_observables]
            wrong = np.any(predictions != observables[start:stop], axis=1)
            flags = compiled.last_flags
            failed = np.zeros(len(dets), dtype=bool) if flags is None else ~flags
            errors += int(wrong.sum())
            failures += int(failed.sum())
            strict_errors += int((wrong | failed).sum())
            take = min(len(dets), max(0, args.compare_shots-compared))
            if take:
                expected = reference.decode_batch(dets[:take].copy())
                mismatches += int(np.any(predictions[:take] != expected, axis=1).sum())
                compared += take
            print(f'{basis}: {stop}/{args.shots} shots, {errors} errors, {failures} invalid corrections', flush=True)
        # 97.5% per-basis Wilson intervals, summed/divided by rounds: a
        # conservative approximate 95% joint interval (Bonferroni).
        interval = wilson(errors, args.shots, NormalDist().inv_cdf(0.9875))
        report['bases'][basis] = dict(
            shots=args.shots, errors=errors, invalid_corrections=failures,
            strict_policy_errors=strict_errors, seed=args.seed+index,
            shot_ler=errors/args.shots, shot_ler_wilson_97_5=interval,
            upstream_compared_shots=compared, upstream_prediction_mismatches=mismatches,
            matrices_equal=True, matrix_shape=list(H.shape), decode_seconds=decode_seconds,
            circuit_sha256=hashlib.sha256(circuit_path.read_bytes()).hexdigest(),
        )
        assert mismatches == 0, 'Adapter predictions differ from upstream'
    report['paper_ler_per_round'] = sum(b['shot_ler'] for b in report['bases'].values())/12
    report['paper_ler_approx_95_interval'] = [
        sum(b['shot_ler_wilson_97_5'][i] for b in report['bases'].values())/12 for i in (0,1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
