"""
隔离 221 个危险 error 的来源。
逐步把不同电路片段设为 noiseless，观察危险 error 数的变化。

电路结构：
  [INIT SE (d轮)] [CNOT1 + SE] [CNOT2 + SE] [CNOT3 + SE] [S + SE] [MEAS]

危险 error = 在 transformed frame 中翻转 target 但通过 PS 的 DEM error。
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
import numpy as np
import stim
from src.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock,
)
from src.qec_code.surface_code.unrotated.operation import _get_fold_yx_pairs
from src.ir.qec_system import QECSystem
from src.ir.builder import CircuitBuilder
from src.ir.tracker import SyndromeTracker
from src.noise.config import NoiseConfig
from src.noise.injector import NoiseInjector
from src.simulation.observable_analysis import (
    build_obs_patch_matrix, identify_distillation_observables,
)


def _apply_fold_s(circuit, system, patch, dag=False):
    diag_s, diag_sdag, mirror_pairs = _get_fold_yx_pairs(system, patch)
    if dag:
        diag_s, diag_sdag = diag_sdag, diag_s
    if diag_s:
        circuit.append("S", sorted(diag_s))
    if diag_sdag:
        circuit.append("S_DAG", sorted(diag_sdag))
    for a, b in mirror_pairs:
        circuit.append("CZ", [a, b])


def build_circuit(d, r=1,
                  cnot1_nl=False, cnot2_nl=False, cnot3_nl=False,
                  s_nl=False):
    """
    构建 no-inject 电路，可分别控制每个 CNOT tick 和 S block 是否 noiseless。
    SE 轮次始终 noisy（由 NoiseInjector 注入，不支持 tag）。
    """
    ps = 2*(d-1); gap=2; cs=ps+gap; rs=ps+gap
    layout = {
        'W0':(0,0), 'W1':(0,rs), 'W2':(cs,0), 'W3':(cs,rs),
        'W4':(0,2*rs), 'W5':(0,3*rs), 'W6':(cs,2*rs), 'W7':(cs,3*rs),
    }
    system = QECSystem()
    gp = {}
    for name, off in layout.items():
        p = UnrotatedSurfaceCode(distance=d)
        gp[name] = system.add_patch(p, name=name, offset=off)
    lp = {n: system.patches[n][0] for n in layout}

    nq = system.num_qubits
    tracker = SyndromeTracker(num_qubits=nq, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    def do_se(n):
        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=n)

    # Init
    x_patches = {'W0','W1','W2','W4'}; z_patches = {'W3','W5','W6','W7'}
    init_dict = {}
    for name in x_patches:
        for q in system.data_indices:
            if system.index_to_owner_map[q] == name: init_dict[q] = 'X'
    for name in z_patches:
        for q in system.data_indices:
            if system.index_to_owner_map[q] == name: init_dict[q] = 'Z'
    builder.initialize(init_dict=init_dict, n=nq)
    do_se(d)

    cnot_ticks = [
        [('W0','W4'),('W1','W5'),('W2','W6'),('W3','W7')],
        [('W0','W2'),('W1','W3'),('W4','W6'),('W5','W7')],
        [('W0','W1'),('W2','W3'),('W4','W5'),('W6','W7')],
    ]
    nls = [cnot1_nl, cnot2_nl, cnot3_nl]
    for tick, nl in zip(cnot_ticks, nls):
        cc = stim.Circuit()
        for ctrl_name, tgt_name in tick:
            cq = sorted(gp[ctrl_name].data_indices)
            tq = sorted(gp[tgt_name].data_indices)
            tgts = []
            for c, t in zip(cq, tq):
                tgts.extend([c, t])
            cc.append("CNOT", tgts)
        builder.apply_unitary_block(cc, noiseless=nl)
        do_se(r)

    sb = stim.Circuit()
    for i in range(1, 8):
        _apply_fold_s(sb, system, lp[f'W{i}'], dag=False)
    _apply_fold_s(sb, system, lp['W0'], dag=True)
    builder.apply_unitary_block(sb, noiseless=s_nl)
    do_se(r)

    final_meas = {q: 'X' for q in system.data_indices}
    builder.apply_data_readout(final_measurements=final_meas)
    return builder.circuit, system


def count_dangerous(circuit, system, p=1e-3):
    matrix, pn = build_obs_patch_matrix(circuit, system)
    T, ta, ps = identify_distillation_observables(matrix, pn, ['W0'])
    n = circuit.num_observables
    nc = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    inj = NoiseInjector.from_circuit_level(nc, list(range(circuit.num_qubits)))
    noisy = inj.inject_noise(circuit)
    # noiseless check
    ds, obs = circuit.compile_detector_sampler().sample(shots=10, separate_observables=True)
    if np.any(ds) or np.any(obs):
        return -1  # FAIL
    dem = noisy.detector_error_model(approximate_disjoint_errors=True)
    cnt = 0; by_nd = {}
    for inst in dem:
        if not (isinstance(inst, stim.DemInstruction) and inst.type == 'error'):
            continue
        tgts = inst.targets_copy()
        fl = [t.val for t in tgts if t.is_logical_observable_id()]
        nd = sum(1 for t in tgts if t.is_relative_detector_id())
        rv = np.zeros(n, dtype=int)
        for i in fl: rv[i] = 1
        tv = (T @ rv) % 2
        if any(tv[i]==1 for i in ta) and all(tv[i]==0 for i in ps):
            cnt += 1
            by_nd[nd] = by_nd.get(nd, 0) + 1
    return cnt, by_nd


def main():
    d, r = 3, 1
    p = 1e-3
    print(f"d={d}, r={r}, p={p:.0e}")
    print(f"{'配置':<45}  {'危险errors':>10}  {'n_dets分布'}")
    print("=" * 80)

    configs = [
        ("全部 noisy (baseline)",                     False,False,False,False),
        ("CNOT1 noiseless",                           True, False,False,False),
        ("CNOT2 noiseless",                           False,True, False,False),
        ("CNOT3 noiseless",                           False,False,True, False),
        ("CNOT1+2 noiseless",                         True, True, False,False),
        ("CNOT1+3 noiseless",                         True, False,True, False),
        ("CNOT2+3 noiseless",                         False,True, True, False),
        ("全部 CNOT noiseless",                        True, True, True, False),
        ("全部 CNOT + S noiseless",                   True, True, True, True),
    ]

    for label, c1, c2, c3, s in configs:
        circuit, system = build_circuit(d, r=r, cnot1_nl=c1, cnot2_nl=c2,
                                        cnot3_nl=c3, s_nl=s)
        result = count_dangerous(circuit, system, p=p)
        if result == -1:
            print(f"  {label:<45}  NOISELESS FAIL!")
            continue
        cnt, by_nd = result
        nd_str = " ".join(f"d{k}:{v}" for k,v in sorted(by_nd.items()))
        print(f"  {label:<45}  {cnt:>10}  {nd_str}")

    print("=" * 80)


if __name__ == "__main__":
    main()
