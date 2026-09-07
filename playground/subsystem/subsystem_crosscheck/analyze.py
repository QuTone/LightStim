"""Export counts, comparisons, and standalone figures; no simulations here."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from common import RESULTS, ROOT, HERE, interval, bs_theory, save_json, sha256

plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":150,"savefig.dpi":200})
FIGURES=RESULTS/"figures"
FIGURES.mkdir(exist_ok=True)


def collect():
    grouped=defaultdict(list)
    raw=[]
    for path in sorted((RESULTS/"jobs").glob("*.json")):
        row=json.loads(path.read_text()); c=row["config"]
        if not row.get("shots"):
            raw.append(dict(job_id=row["job_id"],status=row["status"],shots=0))
            continue
        variant=c.get("variant","default").split("_shard")[0]
        key=(c["experiment"],c["size"],c["p"],c["decoder"],variant)
        grouped[key].append(row)
        raw.append(dict(job_id=row["job_id"],status=row["status"],shots=row["shots"],
                        errors=row["errors"],decode_failures=row["decode_failures"],
                        elapsed_seconds=row["elapsed_seconds"],**{k:c[k] for k in ('experiment','size','p','decoder')}))
    rows=[]
    for (experiment,size,p,decoder,variant),jobs in grouped.items():
        n=sum(j["shots"] for j in jobs); k=sum(j["errors"] for j in jobs)
        low,high=interval(k,n)
        fails=sum(j["decode_failures"] for j in jobs)
        # Verify independent sample-seed ranges before combining replicates.
        seen=set()
        for j in jobs:
            for b in j["batches"]:
                assert b["seed"] not in seen, "Duplicate samples in aggregate"
                seen.add(b["seed"])
        row=dict(experiment=experiment,size=size,distance=size if experiment.startswith("bacon") else 2**(size-1),
                 p=p,decoder=decoder,variant=variant,shots=n,errors=k,decode_failures=fails,
                 converged_wrong=k-fails,ler=k/n,ci95_low=low,ci95_high=high,
                 zero_error_upper_bound=(high if not k else np.nan),
                 jobs=len(jobs),all_finished=all(j['status']!='running' for j in jobs),
                 statuses=','.join(sorted(set(j['status'] for j in jobs))),
                 decoder_params=json.dumps(jobs[0]['config']['decoder_params'],sort_keys=True),
                 cpu_wall_seconds_sum=sum(j['elapsed_seconds'] for j in jobs),
                 num_detectors=jobs[0]['num_detectors'],num_observables=jobs[0]['num_observables'])
        if experiment=='bacon_capacity':
            theory=bs_theory(size,p);lo,hi=interval(k,n,alpha=.05/16)
            row.update(theory_ler=theory,ratio_to_theory=(k/n)/theory,
                       theory_in_ci95=low<=theory<=high,
                       theory_in_family95=lo<=theory<=hi)
        rows.append(row)
    pd.DataFrame(raw).to_csv(RESULTS/'job_summary.csv',index=False)
    df=pd.DataFrame(rows).sort_values(['experiment','decoder','variant','size','p'])
    df.to_csv(RESULTS/'results.csv',index=False)
    return df


def paper_data():
    frames=[]
    for r in [3,4]:
        df=pd.read_csv(RESULTS/f'reference_snapshot/manuscript_figure_raw_data/main_figure_2_shyps_r{r}_memory.csv')
        df=df.rename(columns={'physical_error_rate':'p','num_errors':'errors','num_shots':'shots'})
        df['size']=r;df['distance']=2**(r-1);df['ler']=df.errors/df.shots
        ci=[interval(int(k),int(n)) for k,n in zip(df.errors,df.shots)]
        df['ci95_low']=[x[0] for x in ci];df['ci95_high']=[x[1] for x in ci]
        def normalize(ler,s):
            if ler >= 1-2**(-r*r):
                return np.nan  # Outside the independent-observable BSC inverse domain.
            single=-np.expm1(np.log1p(-ler)/(r*r))
            per=-np.expm1(np.log1p(-2*single)/s)/2
            return -np.expm1(r*r*np.log1p(-per))
        df['eq52_s_d']=[normalize(x,2**(r-1)) for x in df.ler]
        df['eq52_s_d_plus_1']=[normalize(x,2**(r-1)+1) for x in df.ler]
        frames.append(df)
    df=pd.concat(frames,ignore_index=True)
    df.to_csv(RESULTS/'paper_reference_rates.csv',index=False)
    return df


def points(ax,frame,label,color=None,marker='o',linestyle='-'):
    frame=frame.sort_values('p'); positive=frame.errors>0
    # Draw only measured nonzero rates as estimates; zero counts are bounds.
    f=frame[positive]
    if len(f):
        ax.errorbar(f.p,f.ler,yerr=[f.ler-f.ci95_low,f.ci95_high-f.ler],
                    label=label,color=color,marker=marker,linestyle=linestyle,
                    capsize=2,linewidth=1.2,markersize=4)
    z=frame[~positive]
    if len(z):
        ax.scatter(z.p,z.ci95_high,marker='v',facecolors='none',edgecolors=color or 'black',
                   label=label+' (95% upper bound)' if not len(f) else None)
    ax.set_xscale('log');ax.set_yscale('log');ax.grid(True,which='major',alpha=.2)
    ax.set_xlabel('Physical error probability p')
    ax.set_ylabel('Logical block failure per shot')


def save(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf','svg']:
        fig.savefig(FIGURES/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)


def plots(df,paper):
    colors=plt.get_cmap('tab10').colors
    cc=df[(df.experiment=='bacon_capacity')&(df.decoder=='bposd')]
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    for i,d in enumerate([3,5,7,9]):
        f=cc[cc['size']==d];points(axs[0],f,f'd={d}, BPOSD',colors[i],linestyle='none')
        ps=np.geomspace(.004,.045,150)
        axs[0].plot(ps,[bs_theory(d,p) for p in ps],color=colors[i],lw=1)
    axs[0].set_title('Bacon–Shor: exact publication model\nIndependent X/Z data errors; perfect extraction')
    axs[0].legend(fontsize=8)
    for i,p in enumerate([.005,.01,.02,.04]):
        f=cc[np.isclose(cc.p,p)].sort_values('size')
        ds=np.arange(3,42,2)
        axs[1].plot(ds,[bs_theory(int(d),p) for d in ds],color=colors[i],label=f'p={p:g}')
        axs[1].errorbar(f['size'],f.ler,yerr=[f.ler-f.ci95_low,f.ci95_high-f.ler],
                        linestyle='none',marker='o',color=colors[i],capsize=2,markersize=4)
    axs[1].set_yscale('log');axs[1].set_xlabel('Odd square distance d')
    axs[1].set_ylabel('One-sector logical failure');axs[1].legend(fontsize=8)
    axs[1].grid(True,alpha=.2);axs[1].set_title('Finite optimal size\nLines: exact formula; markers: LightStim sampling')
    save(fig,'bacon_capacity')

    fig,axs=plt.subplots(1,3,figsize=(14,4.2),sharey=True)
    for ax,(decoder,variant,title) in zip(axs,[('bposd','default','BPOSD: parallel / dynamic scaling'),
                       ('bposd','serial','BPOSD: serial / dynamic scaling'),('mwpf','default','MWPF: cluster limit 50')]):
        for i,d in enumerate([3,5,7]):
            f=df[(df.experiment=='bacon_circuit')&(df['size']==d)&(df.decoder==decoder)&(df.variant==variant)]
            if len(f):points(ax,f,f'd={d}',colors[i])
        ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('Native Bacon–Shor memory: d XZ pairs, circuit noise, all automatic detectors',y=1.04)
    save(fig,'bacon_circuit_decoders')

    fig,axs=plt.subplots(1,2,figsize=(11,4.4))
    for ax,r in zip(axs,[3,4]):
        points(ax,paper[paper['size']==r],'Paper public CSV (SW BP+LSD)',colors[0],marker='s')
        for i,(decoder,variant,label) in enumerate([
            ('bposd','default','Global BPOSD: defaults'),
            ('bposd','scale01','Global BPOSD: scaling 0.1'),
            ('bposd','serial85','Global BPOSD: serial, scaling 0.85'),
            ('mwpf','default','Global MWPF')]):
            f=df[(df.experiment=='shyps_reference_z')&(df['size']==r)&(df.decoder==decoder)&(df.variant==variant)]
            if len(f):points(ax,f,label,colors[i+1],linestyle='--')
        ax.set_title(f'SHYPS r={r}, [[{(2**r-1)**2}, {r*r}, {2**(r-1)}]]\nAuthors’ exact physical circuit and Z detectors')
        ax.legend(fontsize=7.5)
    save(fig,'shyps_reference_decoders')

    fig,axs=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    for ax,variant,title in zip(axs,['default','scale01'],['Default BPOSD','BPOSD, min-sum scaling 0.1']):
        points(ax,paper[paper['size']==3],'Paper public CSV',colors[0],marker='s')
        for i,(experiment,label) in enumerate([('shyps_reference_z','Authors’ Z detectors'),
                                               ('shyps_auto_z','Authors’ gates + LightStim Z detectors'),
                                               ('shyps_generic_z','Native generic XZ + LightStim Z detectors')]):
            f=df[(df.experiment==experiment)&(df['size']==3)&(df.decoder=='bposd')&(df.variant==variant)]
            if len(f):points(ax,f,label,colors[i+1])
        ax.set_title(f'SHYPS r=3: {title}');ax.legend(fontsize=7.5)
    save(fig,'shyps_integration_comparison')

    fig,ax=plt.subplots(figsize=(8.5,4.6))
    points(ax,paper[paper['size']==4],'Paper public CSV (SW BP+LSD)',colors[0],marker='s')
    for i,(experiment,variant,label) in enumerate([
        ('shyps_reference_z','serial85','Authors’ circuit/detectors: BPOSD serial, scale 0.85'),
        ('shyps_auto_z','serial85','Authors’ gates + automatic Z: serial, scale 0.85'),
        ('shyps_generic_z','serial85','Native generic XZ + automatic Z: serial, scale 0.85'),
        ('shyps_auto_z','scale01_iter100','Automatic Z: parallel, scale 0.1, 100 iterations'),
        ('shyps_auto_z','scale01_iter1','Automatic Z: parallel, scale 0.1, 1 iteration')]):
        f=df[(df.experiment==experiment)&(df['size']==4)&(df.decoder=='bposd')&(df.variant==variant)]
        if len(f):points(ax,f,label,colors[i+1],linestyle='--' if i>=3 else '-')
    ax.set_title('SHYPS r=4: detector / decoder compatibility remains unresolved')
    ax.legend(fontsize=7.5)
    save(fig,'shyps_r4_integration')

    fig,ax=plt.subplots(figsize=(8,3.4))
    names=[];reference=[];extra=[]
    for r in [3,4]:
        meta=json.loads((RESULTS/f'shyps_r{r}_equivalence.json').read_text())
        names.append(f'r={r}');reference.append(meta['z_detector_space']['rank_a'])
        extra.append(meta['z_detector_space']['rank_b']-reference[-1])
    pos=np.arange(2)
    ax.barh(pos,reference,label='Authors’ Z detector span',color=colors[0])
    ax.barh(pos,extra,left=reference,label='Additional independent LightStim constraints',color=colors[2])
    for i,(a,b) in enumerate(zip(reference,extra)):
        ax.text(a/2,i,str(a),va='center',ha='center',color='white')
        ax.text(a+b/2,i,f'+{b}',va='center',ha='center',color='white')
    ax.set_yticks(pos,names);ax.set_xlabel('Rank over GF(2), affine measurement-record parities')
    ax.set_title('Same physical circuit; automatic detector space is a strict superset')
    ax.legend(fontsize=8,loc='lower right');save(fig,'shyps_detector_spaces')


def summaries(df,paper):
    rows=[]
    for _,row in df[df.experiment.str.startswith('shyps')].iterrows():
        matching=paper[(paper['size']==row['size'])&np.isclose(paper.p,row.p)]
        if len(matching)!=1:continue
        ref=matching.iloc[0]
        rows.append(dict(row, paper_errors=int(ref.errors),paper_shots=int(ref.shots),
                         paper_ler=ref.ler,paper_ci95_low=ref.ci95_low,paper_ci95_high=ref.ci95_high,
                         ratio_to_paper=row.ler/ref.ler,
                         ci95_overlap=max(row.ci95_low,ref.ci95_low)<=min(row.ci95_high,ref.ci95_high)))
    pd.DataFrame(rows).to_csv(RESULTS/'comparison_to_paper.csv',index=False)
    scaling=[]
    for (experiment,decoder,variant,p),f in df[df.experiment.str.startswith('bacon')].groupby(['experiment','decoder','variant','p']):
        if len(f)<3 or not (f.errors>0).all():continue
        f=f.sort_values('size');fit=linregress(f['size'],np.log(f.ler))
        scaling.append(dict(experiment=experiment,decoder=decoder,variant=variant,p=p,
                            d_min=int(f['size'].min()),d_max=int(f['size'].max()),
                            finite_range_suppression=f.iloc[0].ler/f.iloc[-1].ler,
                            log_ler_slope=fit.slope,r_squared=fit.rvalue**2,
                            note='Descriptive finite-range fit, not an asymptotic threshold claim'))
    pd.DataFrame(scaling).to_csv(RESULTS/'bacon_scaling.csv',index=False)
    # This manifest covers output assets, code versions, and current aggregate counts.
    job_states=[json.loads(p.read_text()) for p in (RESULTS/'jobs').glob('*.json')]
    unfinished=[j for j in job_states if j['status'] in {'running','building'}]
    audits=[json.loads(p.read_text()) for p in (RESULTS/'mwpf_audits').glob('*.json')]
    save_json(RESULTS/'run_summary.json',dict(aggregate_points=len(df),
        total_decoded_shots=int(df.shots.sum()),decoder_failures=int(df.decode_failures.sum()),
        unfinished_groups=len(set((j['config']['experiment'],j['config']['size'],j['config']['p'],
                                   j['config']['decoder'],j['config'].get('variant','default').split('_shard')[0]) for j in unfinished)),
        attempted_jobs=len(job_states),zero_shot_jobs=sum(not j.get('shots') for j in job_states),
        mwpf_audits_passed=sum(a['status']=='passed' for a in audits),
        mwpf_audits_failed=sum(a['status']=='failed' for a in audits),
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in HERE.glob('*.py')},
        note='Different decoders reuse samples where circuit, seed offset, and batch size match; total shots is decoding workload, not unique physical samples.'))


if __name__=='__main__':
    df=collect();paper=paper_data();summaries(df,paper);plots(df,paper)
    print('Exported',len(df),'aggregate points and figures to',RESULTS)
