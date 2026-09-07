"""Human-readable physical operation counts and extra-gauge-relation examples."""
from collections import Counter
import csv
import json
from common import *


def counts(c):
    histogram=Counter()
    for op in c.flattened():
        if op.name in ANNOTATIONS:continue
        targets=op.targets_copy()
        number=len(targets)//2 if op.name in {'CX','DEPOLARIZE2'} else len(targets)
        key=op.name+('('+str(op.gate_args_copy()[0])+')' if op.gate_args_copy() else '')
        histogram[key]+=number
    return dict(histogram)


def examples(r):
    meta=json.loads((RESULTS/f'shyps_r{r}_equivalence.json').read_text())
    source=stim.Circuit.from_file(RESULTS/f'assets/shyps_r{r}_reference_mapped.stim')
    full=stim.Circuit.from_file(RESULTS/f'assets/shyps_r{r}_auto_z.stim')
    patch=SHYPSCode(r=r);n=patch.num_data_qubits;m=patch.simplex_length
    gauges={g['syn_idx']:set(g['pauli']) for g in patch.gauges}
    measured=[];groups=[];group=-1;out=[]
    for op in full.flattened():
        if op.name=='R' and op.targets_copy()[0].value>=n:group+=1
        elif op.name in {'M','MX'}:
            measured.extend(t.value for t in op.targets_copy());groups.extend([group]*len(op.targets_copy()))
        elif op.name=='DETECTOR':
            ids=[len(measured)+t.value for t in op.targets_copy()]
            qs=[measured[i] for i in ids]
            if len(out)==0 and groups[ids[0]]==0 and len(ids)==1:
                out.append(dict(kind='initial known Z gauge',absolute_measurement_indices=ids,
                                measured_ancillas=qs,data_pauli_support=sorted(gauges[qs[0]])))
            if len(out)==1 and len(set(groups[i] for i in ids))==1 and groups[ids[0]]==1:
                product=set()
                for q in qs:product.symmetric_difference_update(gauges[q])
                if not product:
                    out.append(dict(kind='within-round product of Z gauges equals identity',
                                    absolute_measurement_indices=ids,measured_ancillas=qs,
                                    data_pauli_product_support=sorted(product)))
    assert len(out)==2
    d=2**(r-1)
    result=dict(r=r,physical_counts_reference=counts(source),physical_counts_regenerated=counts(full),
                examples=out,extra_rank_explanation=dict(
                initial_extra=n-r*m,subsequent_extra_per_round=r*(m-r),
                subsequent_rounds=d,total=(n-r*m)+d*r*(m-r),
                initial_gauge_fixing_dimension=(m-r)**2,
                initial_missing_identity_relations=r*(m-r)))
    assert result['extra_rank_explanation']['total']==meta['z_detector_space']['rank_b']-meta['z_detector_space']['rank_a']
    assert result['physical_counts_reference']==result['physical_counts_regenerated']
    with (RESULTS/f'shyps_r{r}_qubit_mapping.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['reference_qubit','lightstim_qubit','role','lightstim_x','lightstim_y'])
        writer.writeheader()
        for old,new in sorted((int(k),v) for k,v in meta['reference_to_lightstim_qubit_map'].items()):
            role='data' if new<n else 'X gauge ancilla' if new<2*n else 'Z gauge ancilla'
            x,y=patch.qubit_coords[new]
            writer.writerow(dict(reference_qubit=old,lightstim_qubit=new,role=role,lightstim_x=x,lightstim_y=y))
    save_json(RESULTS/f'shyps_r{r}_physical_audit.json',result)
    return result


if __name__=='__main__':
    for r in (3,4):
        print(examples(r))
