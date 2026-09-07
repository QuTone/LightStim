"""Replay MWPF jobs with native panics raised, and compare every batch count.

mwpf 0.2.12 defaults to catching native failures and returning random predictions.
The main experiments now request RAISE. This separate replay also audits jobs
started before that explicit policy was added; failed/partial audits are visible.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

for var in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[var]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/lightstim-crosscheck-mpl')

import numpy as np
import stim
from mwpf.sinter_decoders import PanicAction
from common import RESULTS, ROOT, save_json
from lightstim.simulation.decoder_backend import get_decoder
from run import circuit_id


def audit(path):
    state=json.loads(Path(path).read_text());c=state['config']
    out=RESULTS/'mwpf_audits'/(state['job_id']+'.json')
    if out.exists():
        previous=json.loads(out.read_text())
        if previous.get('shots_verified')==state['shots'] and previous.get('status')=='passed':return
    result=dict(job_id=state['job_id'],shots_verified=0,batches_verified=0,status='running')
    t=time.perf_counter()
    try:
        circuit=stim.Circuit.from_file(RESULTS/'assets/circuits'/(circuit_id(c)+'.stim'))
        compiled=get_decoder('mwpf',**c['decoder_params'],panic_action=PanicAction.RAISE).compile_decoder_for_dem(dem=circuit.detector_error_model())
        for batch in state['batches']:
            d,o=circuit.compile_detector_sampler(seed=batch['seed']).sample(batch['shots'],separate_observables=True,bit_packed=True)
            predicted=compiled.decode_shots_bit_packed(bit_packed_detection_event_data=d)
            errors=int(np.any(predicted^o,axis=1).sum())
            assert not compiled.panic_cases
            assert errors==batch['errors'],(batch['seed'],errors,batch['errors'])
            result['shots_verified']+=batch['shots'];result['batches_verified']+=1
            result['elapsed_seconds']=time.perf_counter()-t
            save_json(out,result)
        result['status']='passed'
    except BaseException:
        result.update(status='failed',exception=traceback.format_exc())
    result['elapsed_seconds']=time.perf_counter()-t
    save_json(out,result)
    print(result['job_id'],result['status'],result['shots_verified'],flush=True)


def launch(path):
    logs=RESULTS/'logs';logs.mkdir(exist_ok=True)
    with (logs/(Path(path).stem+'_mwpf_audit.log')).open('a') as f:
        subprocess.run([sys.executable,__file__,'--job',str(path)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1200)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job');p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    if args.job:audit(args.job)
    else:
        paths=[]
        for path in (RESULTS/'jobs').glob('*.json'):
            s=json.loads(path.read_text())
            if s['config']['decoder']=='mwpf' and s.get('shots',0) and s['status']!='running':paths.append(path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch,paths))
