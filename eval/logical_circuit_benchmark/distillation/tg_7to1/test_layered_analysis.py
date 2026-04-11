"""
逐层分析 no-inject 蒸馏电路的危险 error 来源。

每一步都构建一个完整电路（Init + d SE + N层CNOT + 可选S门 + X测量），
分析 DEM 中的危险 error（transformed frame 中 flip target, pass PS）。

步骤：
  Step 1: Init + d SE + CNOT1 + X meas
  Step 2: Init + d SE + CNOT1 + SE(r) + CNOT2 + X meas
  Step 3: Init + d SE + CNOT1 + SE(r) + CNOT2 + SE(r) + CNOT3 + X meas
  Step 4: Step3 + SE(r) + S(W1-7) + S_dag(W0) + X meas
  Step 4b: Step3 + SE(r) + S(W1-7) + S_dag(W0) + SE(r) + X meas  (完整 no-inject)
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


def setup_system(d):
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
    return system, gp, lp


def make_init_and_builder(system, d):
    nq = system.num_qubits
    tracker = SyndromeTracker(num_qubits=nq, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    x_patches = {'W0','W1','W2','W4'}
    z_patches = {'W3','W5','W6','W7'}
    init_dict = {}
    for name in x_patches:
        for q in system.data_indices:
            if system.index_to_owner_map[q] == name: init_dict[q] = 'X'
    for name in z_patches:
        for q in system.data_indices:
            if system.index_to_owner_map[q] == name: init_dict[q] = 'Z'
    builder.initialize(init_dict=init_dict, n=nq)
    return builder


def do_se(builder, system, n):
    se = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=n)


def make_cnot_circuit(gp, tick):
    cc = stim.Circuit()
    for ctrl_name, tgt_name in tick:
        cq = sorted(gp[ctrl_name].data_indices)
        tq = sorted(gp[tgt_name].data_indices)
        tgts = []
        for c, t in zip(cq, tq): tgts.extend([c, t])
        cc.append("CNOT", tgts)
    return cc


def analyze_circuit(circuit, system, p=1e-3, label=""):
    """分析一个完整电路的 observable 结构和危险 error。"""
    # Noiseless check
    ds, obs = circuit.compile_detector_sampler().sample(shots=20, separate_observables=True)
    noiseless_ok = not np.any(ds) and not np.any(obs)

    # Observable analysis
    matrix, patch_names = build_obs_patch_matrix(circuit, system)
    T, target_obs, ps_obs = identify_distillation_observables(matrix, patch_names, ['W0'])
    n_obs = circuit.num_observables

    # Observable structure (which patches in each obs)
    w_patches = ['W0','W1','W2','W3','W4','W5','W6','W7']
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  qubits={circuit.num_qubits}, detectors={circuit.num_detectors}, "
          f"observables={n_obs}, measurements={circuit.num_measurements}")
    print(f"  Noiseless: {'OK' if noiseless_ok else 'FAIL'}")
    print(f"  T = {T.tolist()}")
    print(f"  target_obs={target_obs}, ps_obs={ps_obs}")

    print(f"\n  Raw observables (W patches):")
    for i in range(matrix.shape[0]):
        patches = [patch_names[j] for j in range(len(patch_names)) if matrix[i,j]]
        w_in = [p for p in patches if p.startswith('W')]
        print(f"    L{i}: {w_in}")

    M_new = (T @ matrix) % 2
    print(f"\n  Transformed observables:")
    for i in range(M_new.shape[0]):
        patches = [patch_names[j] for j in range(len(patch_names)) if M_new[i,j]]
        w_in = [p for p in patches if p.startswith('W')]
        label2 = 'TARGET' if i in target_obs else 'PS'
        print(f"    L{i}': {w_in}  [{label2}]")

    # DEM dangerous error analysis
    nc = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    inj = NoiseInjector.from_circuit_level(nc, list(range(circuit.num_qubits)))
    noisy = inj.inject_noise(circuit)
    dem = noisy.detector_error_model(approximate_disjoint_errors=True)

    dangerous = []
    raw_patterns = {}
    for inst in dem:
        if not (isinstance(inst, stim.DemInstruction) and inst.type == 'error'): continue
        tgts = inst.targets_copy()
        fl = [t.val for t in tgts if t.is_logical_observable_id()]
        nd = sum(1 for t in tgts if t.is_relative_detector_id())
        rv = np.zeros(n_obs, dtype=int)
        for i in fl: rv[i] = 1
        tv = (T @ rv) % 2
        if any(tv[i]==1 for i in target_obs) and all(tv[i]==0 for i in ps_obs):
            dangerous.append((inst.args_copy()[0], nd, fl))
            key = tuple(sorted(fl))
            raw_patterns[key] = raw_patterns.get(key, 0) + 1

    by_nd = {}
    for _, nd, _ in dangerous:
        by_nd[nd] = by_nd.get(nd, 0) + 1

    print(f"\n  DEM: {dem.num_errors} errors total")
    print(f"  Dangerous errors (escape PS in transformed frame): {len(dangerous)}")
    if by_nd:
        print(f"  n_dets distribution: {dict(sorted(by_nd.items()))}")
    if raw_patterns:
        print(f"  Raw obs flip patterns:")
        for k, cnt in sorted(raw_patterns.items(), key=lambda x: -x[1]):
            rv = np.zeros(n_obs, dtype=int)
            for i in k: rv[i] = 1
            tv = (T @ rv) % 2
            print(f"    raw={list(k)}  trans={tv.tolist()}  count={cnt}")

    return len(dangerous), by_nd


def main():
    d, r, p = 3, 1, 1e-3

    cnot_ticks = [
        [('W0','W4'),('W1','W5'),('W2','W6'),('W3','W7')],
        [('W0','W2'),('W1','W3'),('W4','W6'),('W5','W7')],
        [('W0','W1'),('W2','W3'),('W4','W5'),('W6','W7')],
    ]

    print(f"d={d}, r={r}, p={p:.0e}")
    print("逐层分析 no-inject 蒸馏电路的危险 error\n")

    # ----------------------------------------------------------------
    # Step 1: Init + d SE + CNOT1 + SE(r) + X meas
    # ----------------------------------------------------------------
    system, gp, lp = setup_system(d)
    builder = make_init_and_builder(system, d)
    do_se(builder, system, d)
    builder.apply_unitary_block(make_cnot_circuit(gp, cnot_ticks[0]))
    do_se(builder, system, r)
    builder.apply_data_readout(final_measurements={q:'X' for q in system.data_indices})
    analyze_circuit(builder.circuit, system, p,
                    "Step 1: Init + d SE + CNOT1 + SE(r) + X meas")

    # ----------------------------------------------------------------
    # Step 2: Init + d SE + CNOT1 + SE(r) + CNOT2 + SE(r) + X meas
    # ----------------------------------------------------------------
    system, gp, lp = setup_system(d)
    builder = make_init_and_builder(system, d)
    do_se(builder, system, d)
    builder.apply_unitary_block(make_cnot_circuit(gp, cnot_ticks[0]))
    do_se(builder, system, r)
    builder.apply_unitary_block(make_cnot_circuit(gp, cnot_ticks[1]))
    do_se(builder, system, r)
    builder.apply_data_readout(final_measurements={q:'X' for q in system.data_indices})
    analyze_circuit(builder.circuit, system, p,
                    "Step 2: + CNOT1 + SE(r) + CNOT2 + SE(r) + X meas")

    # ----------------------------------------------------------------
    # Step 3: Init + d SE + CNOT1 + SE(r) + CNOT2 + SE(r) + CNOT3 + SE(r) + X meas
    # ----------------------------------------------------------------
    system, gp, lp = setup_system(d)
    builder = make_init_and_builder(system, d)
    do_se(builder, system, d)
    for tick in cnot_ticks:
        builder.apply_unitary_block(make_cnot_circuit(gp, tick))
        do_se(builder, system, r)
    builder.apply_data_readout(final_measurements={q:'X' for q in system.data_indices})
    analyze_circuit(builder.circuit, system, p,
                    "Step 3: + CNOT1+SE + CNOT2+SE + CNOT3+SE + X meas")

    # ----------------------------------------------------------------
    # Step 4: Step3 + S(W1-7)+S_dag(W0) + X meas  (no SE after S)
    # ----------------------------------------------------------------
    system, gp, lp = setup_system(d)
    builder = make_init_and_builder(system, d)
    do_se(builder, system, d)
    for tick in cnot_ticks:
        builder.apply_unitary_block(make_cnot_circuit(gp, tick))
        do_se(builder, system, r)
    sb = stim.Circuit()
    for i in range(1, 8): _apply_fold_s(sb, system, lp[f'W{i}'], dag=False)
    _apply_fold_s(sb, system, lp['W0'], dag=True)
    builder.apply_unitary_block(sb)
    builder.apply_data_readout(final_measurements={q:'X' for q in system.data_indices})
    analyze_circuit(builder.circuit, system, p,
                    "Step 4: Step3 + S(W1-7)+S_dag(W0) + X meas  [no SE after S]")

    # ----------------------------------------------------------------
    # Step 4b: 完整 no-inject（Step4 + SE(r) after S）
    # ----------------------------------------------------------------
    system, gp, lp = setup_system(d)
    builder = make_init_and_builder(system, d)
    do_se(builder, system, d)
    for tick in cnot_ticks[:2]:
        builder.apply_unitary_block(make_cnot_circuit(gp, tick))
        do_se(builder, system, r)
    builder.apply_unitary_block(make_cnot_circuit(gp, cnot_ticks[2]))
    do_se(builder, system, r)
    sb = stim.Circuit()
    for i in range(1, 8): _apply_fold_s(sb, system, lp[f'W{i}'], dag=False)
    _apply_fold_s(sb, system, lp['W0'], dag=True)
    builder.apply_unitary_block(sb)
    do_se(builder, system, r)
    builder.apply_data_readout(final_measurements={q:'X' for q in system.data_indices})
    analyze_circuit(builder.circuit, system, p,
                    "Step 4b: 完整 no-inject（+ SE(r) after S）")

    print(f"\n{'='*70}")
    print("分析完成")


if __name__ == "__main__":
    main()
