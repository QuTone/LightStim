"""Save all generated-circuit comparisons before starting decoder jobs."""
import argparse
import time
from common import *
from run import configuration, circuit_id


def retune(circuit, old_p, new_p):
    out=stim.Circuit()
    for op in circuit.flattened():
        args=op.gate_args_copy()
        if op.name in {"X_ERROR","Z_ERROR","DEPOLARIZE1","DEPOLARIZE2","M","MX"} and args:
            assert args == [old_p], (op,args)
            args=[new_p]
        out.append(op.name,op.targets_copy(),args)
    return out


def prepare(r):
    t=time.perf_counter()
    jobs=[]
    generic_xz=generic_memory("shyps",r,.001,rounds=2**(r-1)+1)
    generic_z=z_detectors(generic_xz)
    generic_zx_z=generic_memory("shyps",r,.001,rounds=2**(r-1)+1,order=("Z","X"),z_only=True)
    auto_z=stim.Circuit.from_file(RESULTS/"assets"/f"shyps_r{r}_auto_z.stim")
    auto_xz=stim.Circuit.from_file(RESULTS/"assets"/f"shyps_r{r}_auto_xz.stim")
    circuits=dict(shyps_generic_z=generic_z,shyps_generic_xz=generic_xz,
                  shyps_generic_zx_z=generic_zx_z,shyps_auto_z=auto_z,shyps_auto_xz=auto_xz)
    ps=(.0005,.001,.002,.003,.005) if r==3 else (.0005,.001,.002,.003)
    for experiment,c in circuits.items():
        # Full-XZ and gate-order controls use fewer points because global OSD
        # on both syndrome sectors is much more expensive.
        for p in ps if experiment in {"shyps_auto_z","shyps_generic_z"} else (.001,.003):
            config=configuration(experiment,r,p)
            config.update(batch_size=100 if r==4 else 500,max_seconds=600)
            asset=RESULTS/"assets/circuits"/(circuit_id(config)+".stim")
            retune(c,.001,p).to_file(asset)
            jobs.append(config)
    save_json(RESULTS/f"comparison_suite_r{r}.json",jobs)
    print('Prepared',r,len(jobs),'jobs in',time.perf_counter()-t,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--r",type=int,required=True)
    prepare(parser.parse_args().r)
