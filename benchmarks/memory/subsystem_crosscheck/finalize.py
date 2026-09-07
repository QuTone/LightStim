"""Validate saved counts/hashes, then archive the completed review bundle."""
import datetime
import json
from pathlib import Path
import tarfile

import nbformat
import numpy as np
import stim

from common import HERE, RESULTS, ROOT, save_json, sha256
from run import circuit_id


def validate():
    jobs=[json.loads(p.read_text()) for p in (RESULTS/'jobs').glob('*.json')]
    assert not [j['job_id'] for j in jobs if j['status'] in {'running','building'}]
    signatures={}
    sample_checks=0; audited_mwpf=0
    for job in jobs:
        if not job.get('shots'):continue
        c=job['config'];batches=job['batches']
        for key in ['shots','errors','decode_failures']:
            assert sum(b[key] for b in batches)==job[key],(job['job_id'],key)
        assert len(batches)==job['next_batch']
        assert len(set(b['seed'] for b in batches))==len(batches)
        for batch in batches:
            assert 0<=batch['decode_failures']<=batch['errors']<=batch['shots']
            assert max(batch['per_observable_errors'],default=0)<=batch['errors']
            assert batch['errors']<=sum(batch['per_observable_errors'])+batch['decode_failures']
        cid=circuit_id(c);path=RESULTS/'assets/circuits'/(cid+'.stim')
        digest=signatures.setdefault(cid,sha256(path))
        assert digest==job['circuit_sha256'],job['job_id']
        sample=RESULTS/'assets/samples'/(job['job_id']+'.npz')
        with np.load(sample) as data:
            assert int(data['seed'])==batches[0]['seed']
            predicted=data['predictions'];actual=data['observables']
            residual=np.unpackbits(predicted^actual,axis=1,bitorder='little')[:,:job['num_observables']]
            failures=residual.any(axis=1)|data['decode_failures']
            assert len(failures)==batches[0]['shots']
            assert int(failures.sum())==batches[0]['errors'],job['job_id']
        sample_checks+=1
        if c['decoder']=='mwpf':
            audit=json.loads((RESULTS/'mwpf_audits'/(job['job_id']+'.json')).read_text())
            assert audit['status']=='passed',audit
            assert audit['shots_verified']==job['shots'],job['job_id']
            audited_mwpf+=1
    nbpath=ROOT/'playground/subsystem/subsystem_crosscheck.ipynb'
    nb=nbformat.read(nbpath,as_version=4)
    cells=[cell for cell in nb.cells if cell.cell_type=='code']
    assert all(cell.execution_count for cell in cells)
    assert all(out.output_type!='error' for cell in cells for out in cell.outputs)
    assert 'crumble' not in nbpath.read_text().lower()
    summary_path=RESULTS/'run_summary.json'
    summary=json.loads(summary_path.read_text())
    summary['code_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in HERE.glob('*.py')}
    summary['mwpf_audits_passed']=audited_mwpf
    assert summary['unfinished_groups']==0
    save_json(summary_path,summary)
    all_assets=[p for p in RESULTS.rglob('*') if p.is_file() and p.name not in {'asset_checksums.json','integrity_check.json'}]
    save_json(RESULTS/'asset_checksums.json',{str(p.relative_to(RESULTS)):sha256(p) for p in all_assets})
    receipt=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 attempted_jobs=len(jobs),counted_jobs=sample_checks,
                 decoded_shots=sum(j.get('shots',0) for j in jobs),
                 unique_circuit_hashes=len(signatures),first_sample_batches_verified=sample_checks,
                 mwpf_jobs_fully_replayed=audited_mwpf,
                 notebook_code_cells_executed=len(cells),notebook_sha256=sha256(nbpath),
                 zero_shot_jobs=[j['job_id'] for j in jobs if not j.get('shots')],
                 status='passed')
    save_json(RESULTS/'integrity_check.json',receipt)
    return receipt


def archive():
    parent=ROOT.parent/'LightStim-artifacts';parent.mkdir(exist_ok=True)
    dest=parent/'subsystem_crosscheck_2026-09-06.tar.gz'
    temp=dest.with_suffix('.tar.gz.tmp')
    with tarfile.open(temp,'w:gz') as bundle:
        for path in sorted(HERE.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                bundle.add(path,arcname='subsystem_crosscheck/'+str(path.relative_to(HERE)),recursive=False)
        bundle.add(ROOT/'playground/subsystem/subsystem_crosscheck.ipynb',arcname='subsystem_crosscheck/review_notebook.ipynb')
    temp.replace(dest)
    # Verify the compressed stream and required members after closing it.
    with tarfile.open(dest,'r:gz') as bundle:
        names=set(bundle.getnames())
        assert 'subsystem_crosscheck/review_notebook.ipynb' in names
        assert 'subsystem_crosscheck/results/2026-09-06/REPORT.md' in names
        assert 'subsystem_crosscheck/results/2026-09-06/integrity_check.json' in names
    checksum=sha256(dest)
    dest.with_suffix(dest.suffix+'.sha256').write_text(checksum+'  '+dest.name+'\n')
    print('Archive:',dest,'bytes:',dest.stat().st_size,'sha256:',checksum,flush=True)


if __name__=='__main__':
    print(validate(),flush=True)
    archive()
