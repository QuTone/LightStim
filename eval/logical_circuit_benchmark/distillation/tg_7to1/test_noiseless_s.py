"""
临时测试脚本：分析哪些部分引起"全局翻转"危险 error。

四个测试版本（d=3, p=1e-3）:
  A. 全部 noisy（baseline）
  B. 仅 S_dag on W0 noiseless
  C. 仅 S on W1-W7 noiseless
  D. 全部 S/S_dag noiseless

对比四个版本中 dangerous weight-limited errors（flip target, pass PS）的数量。
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


def build_circuit(d, rounds, r=1,
                  w0_sdag_noiseless=False,
                  w1_7_s_noiseless=False):
    """构建 no-inject 电路，可分别控制 W0 S_dag 和 W1-W7 S 是否 noiseless。"""
    patch_size = 2 * (d - 1)
    gap = 2
    col_sp = patch_size + gap
    row_sp = patch_size + gap

    working_layout = {
        'W0': (0,      0),
        'W1': (0,      row_sp),
        'W2': (col_sp, 0),
        'W3': (col_sp, row_sp),
        'W4': (0,      2 * row_sp),
        'W5': (0,      3 * row_sp),
        'W6': (col_sp, 2 * row_sp),
        'W7': (col_sp, 3 * row_sp),
    }

    system = QECSystem()
    gp = {}
    for name, offset in working_layout.items():
        p = UnrotatedSurfaceCode(distance=d)
        gp[name] = system.add_patch(p, name=name, offset=offset)
    lp = {name: system.patches[name][0] for name in working_layout}

    num_qubits = system.num_qubits
    tracker = SyndromeTracker(num_qubits=num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    def do_se(n):
        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=n)

    # Initialize
    x_patches = {'W0', 'W1', 'W2', 'W4'}
    z_patches = {'W3', 'W5', 'W6', 'W7'}
    init_dict = {}
    for name in x_patches:
        for q in system.data_indices:
            if system.index_to_owner_map[q] == name:
                init_dict[q] = 'X'
    for name in z_patches:
        for q in system.data_indices:
            if system.index_to_owner_map[q] == name:
                init_dict[q] = 'Z'
    builder.initialize(init_dict=init_dict, n=num_qubits)
    do_se(rounds)

    # CNOT encoding
    cnot_ticks = [
        [('W0','W4'),('W1','W5'),('W2','W6'),('W3','W7')],
        [('W0','W2'),('W1','W3'),('W4','W6'),('W5','W7')],
        [('W0','W1'),('W2','W3'),('W4','W5'),('W6','W7')],
    ]
    for tick in cnot_ticks:
        cnot_circuit = stim.Circuit()
        for ctrl_name, tgt_name in tick:
            c_q = sorted(gp[ctrl_name].data_indices)
            t_q = sorted(gp[tgt_name].data_indices)
            targets = []
            for c, t in zip(c_q, t_q):
                targets.extend([c, t])
            cnot_circuit.append("CNOT", targets)
        builder.apply_unitary_block(cnot_circuit)
        do_se(r)

    # S gates — 分成两块，分别控制 noiseless
    if w1_7_s_noiseless and w0_sdag_noiseless:
        # 全部 noiseless：一个块
        s_block = stim.Circuit()
        for i in range(1, 8):
            _apply_fold_s(s_block, system, lp[f'W{i}'], dag=False)
        _apply_fold_s(s_block, system, lp['W0'], dag=True)
        builder.apply_unitary_block(s_block, noiseless=True)
    elif w1_7_s_noiseless and not w0_sdag_noiseless:
        # W1-W7 noiseless, W0 noisy
        s17_block = stim.Circuit()
        for i in range(1, 8):
            _apply_fold_s(s17_block, system, lp[f'W{i}'], dag=False)
        builder.apply_unitary_block(s17_block, noiseless=True)
        s0_block = stim.Circuit()
        _apply_fold_s(s0_block, system, lp['W0'], dag=True)
        builder.apply_unitary_block(s0_block, noiseless=False)
    elif not w1_7_s_noiseless and w0_sdag_noiseless:
        # W1-W7 noisy, W0 noiseless
        s17_block = stim.Circuit()
        for i in range(1, 8):
            _apply_fold_s(s17_block, system, lp[f'W{i}'], dag=False)
        builder.apply_unitary_block(s17_block, noiseless=False)
        s0_block = stim.Circuit()
        _apply_fold_s(s0_block, system, lp['W0'], dag=True)
        builder.apply_unitary_block(s0_block, noiseless=True)
    else:
        # 全部 noisy
        s_block = stim.Circuit()
        for i in range(1, 8):
            _apply_fold_s(s_block, system, lp[f'W{i}'], dag=False)
        _apply_fold_s(s_block, system, lp['W0'], dag=True)
        builder.apply_unitary_block(s_block, noiseless=False)

    do_se(r)

    final_meas = {q: 'X' for q in system.data_indices}
    builder.apply_data_readout(final_measurements=final_meas)

    return builder.circuit, system


def count_dangerous_errors(circuit, system, p=1e-3, max_dets=None):
    """分析 DEM，统计 dangerous errors（在变换后帧中 flip target, pass PS）。

    注意：DEM 中的 observable 索引是 raw frame 的。
    必须先把 raw flip vector 用 T 矩阵变换到 transformed frame，
    然后才能判断是否 escape post-selection。
    """
    matrix, patch_names = build_obs_patch_matrix(circuit, system)
    T, target_obs, ps_obs = identify_distillation_observables(
        matrix, patch_names, ['W0'])

    n_obs = circuit.num_observables

    noise_config = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    injector = NoiseInjector.from_circuit_level(noise_config, list(range(circuit.num_qubits)))
    noisy = injector.inject_noise(circuit)

    dem = noisy.detector_error_model(approximate_disjoint_errors=True)

    dangerous = []
    for inst in dem:
        if not (isinstance(inst, stim.DemInstruction) and inst.type == 'error'):
            continue
        targets = inst.targets_copy()
        raw_flipped_ids = [t.val for t in targets if t.is_logical_observable_id()]
        flipped_dets = [t.val for t in targets if t.is_relative_detector_id()]

        # 构建 raw flip vector
        raw_flip = np.zeros(n_obs, dtype=int)
        for idx in raw_flipped_ids:
            raw_flip[idx] = 1

        # 变换到 transformed frame：trans = (T @ raw_flip) % 2
        trans_flip = (T @ raw_flip) % 2

        # 危险条件：target 观测量被翻转，所有 PS 观测量未被翻转
        target_flipped = any(trans_flip[i] == 1 for i in target_obs)
        ps_all_zero = all(trans_flip[i] == 0 for i in ps_obs)

        if target_flipped and ps_all_zero:
            if max_dets is None or len(flipped_dets) <= max_dets:
                dangerous.append({
                    'prob': inst.args_copy()[0],
                    'n_dets': len(flipped_dets),
                    'raw_obs': raw_flipped_ids,
                    'trans_flip': trans_flip.tolist(),
                })

    by_ndets = {}
    for e in dangerous:
        n = e['n_dets']
        by_ndets[n] = by_ndets.get(n, 0) + 1

    return len(dangerous), by_ndets


def main():
    d, r, p = 3, 1, 1e-3
    rounds = d

    configs = [
        ("A. 全部 noisy (baseline)",   False, False),
        ("B. 仅 W0 S_dag noiseless",   True,  False),
        ("C. 仅 W1-W7 S noiseless",    False, True),
        ("D. 全部 S noiseless",         True,  True),
    ]

    print(f"d={d}, r={r}, p={p:.0e}")
    print("=" * 60)
    print(f"{'配置':<28}  {'危险errors':>10}  {'n_dets分布'}")
    print("-" * 60)

    for label, w0_nl, w17_nl in configs:
        circuit, system = build_circuit(d, rounds, r=r,
                                        w0_sdag_noiseless=w0_nl,
                                        w1_7_s_noiseless=w17_nl)
        # noiseless check
        dets0, obs0 = circuit.compile_detector_sampler().sample(
            shots=20, separate_observables=True)
        if np.any(dets0) or np.any(obs0):
            print(f"{label:<28}  NOISELESS FAIL!")
            continue

        total, by_ndets = count_dangerous_errors(circuit, system, p=p)
        ndets_str = " ".join(f"d{k}:{v}" for k, v in sorted(by_ndets.items()))
        print(f"{label:<28}  {total:>10}  {ndets_str}")

    print("=" * 60)


if __name__ == "__main__":
    main()
