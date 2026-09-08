"""Resumable balanced X/Z paper-point sampling until at least N logical errors.

Processes use independent, recorded seeds. Each job samples equal X/Z counts.
In-flight jobs are included when the target is reached. JSONL checkpoints allow
resumption without discarding completed jobs. No native decoder is rebuilt.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
from importlib.metadata import version
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import stim
from scipy.optimize import brentq
from scipy.special import betaln

_CONTEXT = None


def confidence_sequence(errors, shots, alpha=0.025):
    """Beta(1/2,1/2) mixture Bernoulli confidence sequence, valid at stopping.

    Each basis gets alpha=0.025; sum the endpoints / rounds for a joint 95%
    confidence sequence for the paper's sum-of-basis error rates per round.
    """
    if not shots:
        return [0., 1.]
    constant = betaln(errors+0.5, shots-errors+0.5)-betaln(0.5,0.5)
    def f(p):
        value = constant + np.log(alpha)
        if errors:
            value -= errors*np.log(p)
        if shots-errors:
            value -= (shots-errors)*np.log1p(-p)
        return value
    mle = errors/shots
    lower = brentq(f, 1e-300, mle, xtol=1e-18) if errors else 0.
    upper = brentq(f, mle, 1-1e-15, xtol=1e-18) if errors < shots else 1.
    return [lower, upper]


def initialize(root, physical_rate):
    global _CONTEXT
    sys.path[:0] = [str(Path(root)/'decoder'), root]
    from beamsearch import BeamSearch
    from stimbposd.dem_to_matrices import detector_error_model_to_check_matrices
    from lightstim.simulation.decoder_backend.registry import get_decoder
    from lightstim.simulation.decoder_backend.dem_matrices import dem_to_matrices
    _CONTEXT = {}
    for basis in 'XZ':
        path = Path(root)/'StimCircuit'/f'BB[[144,12,12]],memory_{basis},error_rate={physical_rate},syndrome_rounds=12.stim'
        circuit = stim.Circuit.from_file(path)
        dem = circuit.detector_error_model(decompose_errors=False)
        h,o,p = dem_to_matrices(dem, sparse=True)
        reference = detector_error_model_to_check_matrices(dem)
        assert h.shape == reference.check_matrix.shape
        assert (h != reference.check_matrix).nnz == 0
        assert (o != reference.observables_matrix).nnz == 0
        np.testing.assert_array_equal(p, reference.priors)
        decoder = get_decoder('ionq-beam-search').compile_decoder_for_dem(dem=dem)
        _CONTEXT[basis] = (circuit, decoder, BeamSearch(dem))


def run_job(job_id, shots, master_seed):
    result = dict(job_id=job_id, bases={})
    started = time.monotonic()
    for index,basis in enumerate('XZ'):
        # uint64 Stim seed derived deterministically from job/basis/master seed.
        seed = int(np.random.SeedSequence([master_seed, job_id, index]).generate_state(1, dtype=np.uint64)[0])
        circuit,decoder,reference = _CONTEXT[basis]
        detections,observables = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
        wrong_indices = []
        errors = invalid = strict_errors = compared = mismatches = 0
        for start in range(0,shots,250):
            stop = min(start+250,shots)
            dets = detections[start:stop]
            packed = decoder.decode_shots_bit_packed(bit_packed_detection_event_data=np.packbits(dets,axis=1,bitorder='little'))
            preds = np.unpackbits(packed,axis=1,bitorder='little')[:,:circuit.num_observables]
            wrong = np.any(preds != observables[start:stop],axis=1)
            failed = np.zeros(len(dets),dtype=bool) if decoder.last_flags is None else ~decoder.last_flags
            errors += int(wrong.sum()); invalid += int(failed.sum()); strict_errors += int((wrong|failed).sum())
            wrong_indices.extend((start+np.flatnonzero(wrong)).tolist())
            # Verify every observed logical error, plus the first 16 shots/job.
            check = wrong.copy()
            if start == 0:
                check[:16] = True
            if check.any():
                expected = reference.decode_batch(dets[check].copy())
                compared += int(check.sum())
                mismatches += int(np.any(expected != preds[check],axis=1).sum())
        if mismatches:
            raise RuntimeError(f'Upstream mismatch in job {job_id} basis {basis}: {mismatches}')
        result['bases'][basis] = dict(shots=shots, errors=errors, invalid_corrections=invalid,
            strict_policy_errors=strict_errors, seed=seed, error_shot_indices=wrong_indices,
            upstream_compared_shots=compared, upstream_prediction_mismatches=mismatches)
    result['seconds'] = time.monotonic()-started
    return result


def summarize(rows):
    out={b:{k:0 for k in ['shots','errors','invalid_corrections','strict_policy_errors',
                         'upstream_compared_shots','upstream_prediction_mismatches']} for b in 'XZ'}
    for row in rows:
        for b in 'XZ':
            for key in out[b]:
                out[b][key] += row['bases'][b][key]
    for b in 'XZ':
        n=out[b]['shots']; e=out[b]['errors']
        out[b]['shot_ler'] = e/n if n else None
        out[b]['shot_ler_97_5_confidence_sequence'] = confidence_sequence(e,n)
    n=out['X']['shots']
    ids={row['job_id'] for row in rows}
    complete_prefix = ids == set(range(max(ids, default=-1)+1))
    # Completion order depends on decoder runtime. Only a complete fixed job
    # prefix supports the advertised anytime coverage; never report it for a
    # subset selected by which worker happened to finish first.
    if not complete_prefix:
        for basis in 'XZ':
            out[basis]['shot_ler_97_5_confidence_sequence'] = None
    return dict(completed_jobs=len(rows), complete_job_prefix=complete_prefix, bases=out,
        logical_errors=sum(out[b]['errors'] for b in 'XZ'),
        paper_ler_per_round=sum(out[b]['shot_ler'] for b in 'XZ')/12 if n else None,
        paper_ler_95_confidence_sequence=[sum(out[b]['shot_ler_97_5_confidence_sequence'][i] for b in 'XZ')/12 for i in (0,1)] if complete_prefix else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True,type=Path)
    parser.add_argument('--p',type=float,default=0.001,choices=[0.001,0.002,0.003])
    parser.add_argument('--target-errors',type=int,default=20)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--shots-per-job',type=int,default=5000,help='Per basis')
    parser.add_argument('--seed',type=int,default=20260910)
    parser.add_argument('--output-dir',type=Path,default=Path('benchmarks/decoding/results/ionq_low_ler_p001'))
    args=parser.parse_args()
    if min(args.target_errors,args.workers,args.shots_per_job)<=0:
        parser.error('error target, workers, and shots-per-job must be positive')
    root=args.upstream.resolve(); dest=args.output_dir
    dest.mkdir(parents=True,exist_ok=True)
    configuration=dict(upstream_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        p=args.p,seed=args.seed,shots_per_job=args.shots_per_job,code='[[144,12,12]]',rounds=12,
        decoder='beam8_230iters',params=dict(max_rounds=10,beam_width=8,num_results=1,initial_iters=30,iters_per_round=20),
        versions={x:version(x) for x in ['numpy','scipy','stim','sinter','ldpc','stimbposd']},
        circuit_sha256={b:hashlib.sha256((root/'StimCircuit'/f'BB[[144,12,12]],memory_{b},error_rate={args.p},syndrome_rounds=12.stim').read_bytes()).hexdigest() for b in 'XZ'})
    config_path=dest/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        parser.error('Existing checkpoint configuration differs; choose another output directory')
    config_path.write_text(json.dumps(configuration,indent=2)+'\n')
    checkpoint=dest/'jobs.jsonl'
    rows=[json.loads(line) for line in checkpoint.read_text().splitlines()] if checkpoint.exists() else []
    done={r['job_id'] for r in rows}
    if len(done)!=len(rows):
        raise ValueError('Duplicate job ids in checkpoint')
    resume_max_id=max(done, default=-1)
    next_id=0; started=time.monotonic()
    def save():
        summary=summarize(rows)
        summary.update(target_errors=args.target_errors,target_reached=summary['logical_errors']>=args.target_errors,
            workers=args.workers,current_run_seconds=time.monotonic()-started,
            failure_policy='ignore (paper scoring); strict errors also recorded',
            configuration=configuration)
        temporary=dest/'summary.tmp'
        temporary.write_text(json.dumps(summary,indent=2)+'\n')
        temporary.replace(dest/'summary.json')
        print(f"jobs={len(rows)} shots/basis={summary['bases']['X']['shots']} errors={summary['logical_errors']} X={summary['bases']['X']['errors']} Z={summary['bases']['Z']['errors']} LER/round={summary['paper_ler_per_round']} elapsed={time.monotonic()-started:.0f}s",flush=True)
        return summary
    summary=save()
    if summary['target_reached'] and summary['complete_job_prefix']:
        return
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(str(root),args.p)) as pool:
        pending=set()
        def submit():
            nonlocal next_id
            while next_id in done:
                next_id+=1
            # After an interrupted run, fill missing IDs even if the saved
            # error target is already met, restoring a complete job prefix.
            if summary['target_reached'] and next_id > resume_max_id:
                return
            pending.add(pool.submit(run_job,next_id,args.shots_per_job,args.seed))
            next_id+=1
        for _ in range(args.workers):
            submit()
        while pending:
            completed,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in completed:
                row=future.result()
                with checkpoint.open('a') as f:
                    f.write(json.dumps(row)+'\n'); f.flush()
                rows.append(row)
            summary=save()
            for _ in completed:
                submit()
    save()


if __name__=='__main__':
    main()
