"""
深入分析 Step 3（纯 CNOT 编码）的 DEM 结构。

关键问题：
  1. 为什么 Step 3 的 dangerous errors 都是 raw=[0]，但 BPOSD 仍然给 LER≈p？
  2. max_obs_flip=4 的 DEM 条目来自哪里？
  3. 危险 error 的 n_dets 分布和概率是多少？
  4. 对于每个危险 error D[i] L0，是否存在竞争解释 D[i]（无 logical flip）概率更高？
  5. d 增大时，危险 error 数量如何变化？

测试 d=3,5 的 DEM 结构（分析，不跑完整 decode）。
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
import numpy as np
import stim
from src.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock,
)
from src.ir.qec_system import QECSystem
from src.ir.builder import CircuitBuilder
from src.ir.tracker import SyndromeTracker
from src.noise.config import NoiseConfig
from src.noise.injector import NoiseInjector
from src.simulation.observable_analysis import (
    build_obs_patch_matrix, identify_distillation_observables,
)
from collections import defaultdict


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


def make_step3_circuit(d, r=1):
    """Init + d SE + 3×(CNOT + r SE) + X meas（无 S gates）"""
    system, gp, lp = setup_system(d)
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

    se = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=d)

    cnot_ticks = [
        [('W0','W4'),('W1','W5'),('W2','W6'),('W3','W7')],
        [('W0','W2'),('W1','W3'),('W4','W6'),('W5','W7')],
        [('W0','W1'),('W2','W3'),('W4','W5'),('W6','W7')],
    ]
    for tick in cnot_ticks:
        cc = stim.Circuit()
        for ctrl_name, tgt_name in tick:
            cq = sorted(gp[ctrl_name].data_indices)
            tq = sorted(gp[tgt_name].data_indices)
            tgts = []
            for c, t in zip(cq, tq): tgts.extend([c, t])
            cc.append("CNOT", tgts)
        builder.apply_unitary_block(cc)
        builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=r)

    builder.apply_data_readout(final_measurements={q:'X' for q in system.data_indices})
    return builder.circuit, system


def analyze_dem_structure(d, r=1, p=1e-3):
    print(f"\n{'='*70}")
    print(f"  Step 3 DEM Analysis  d={d}, r={r}, p={p:.0e}")
    print(f"{'='*70}")

    circuit, system = make_step3_circuit(d, r)
    print(f"  qubits={circuit.num_qubits}, det={circuit.num_detectors}, "
          f"obs={circuit.num_observables}, meas={circuit.num_measurements}")

    # Noiseless check
    dts, obs = circuit.compile_detector_sampler().sample(shots=10, separate_observables=True)
    print(f"  Noiseless: {'OK' if not np.any(dts) and not np.any(obs) else 'FAIL'}")

    matrix, patch_names = build_obs_patch_matrix(circuit, system)
    T, target_obs, ps_obs = identify_distillation_observables(matrix, patch_names, ['W0'])
    n_obs = circuit.num_observables
    print(f"  T=identity: {np.array_equal(T, np.eye(n_obs, dtype=int))}")
    print(f"  target_obs={target_obs}, ps_obs={ps_obs}")

    # Get noisy DEM
    nc = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    inj = NoiseInjector.from_circuit_level(nc, list(range(circuit.num_qubits)))
    noisy = inj.inject_noise(circuit)
    dem = noisy.detector_error_model(approximate_disjoint_errors=True)

    total_errors = dem.num_errors
    print(f"\n  DEM: {total_errors} total errors, {circuit.num_detectors} detectors")

    # Full DEM scan
    dangerous = []
    not_dangerous = []
    max_obs_flip = 0

    for inst in dem:
        if not (isinstance(inst, stim.DemInstruction) and inst.type == 'error'):
            continue
        prob = inst.args_copy()[0]
        tgts = inst.targets_copy()
        fl = sorted(t.val for t in tgts if t.is_logical_observable_id())
        nd = sum(1 for t in tgts if t.is_relative_detector_id())
        det_ids = tuple(sorted(t.val for t in tgts if t.is_relative_detector_id()))
        max_obs_flip = max(max_obs_flip, len(fl))

        rv = np.zeros(n_obs, dtype=int)
        for i in fl: rv[i] = 1
        tv = (T @ rv) % 2
        is_danger = (any(tv[i]==1 for i in target_obs) and
                     all(tv[i]==0 for i in ps_obs))

        if is_danger:
            dangerous.append((prob, nd, fl, det_ids))
        else:
            not_dangerous.append((prob, nd, fl, det_ids))

    print(f"  max_obs_flip={max_obs_flip}")
    print(f"  Dangerous: {len(dangerous)} errors")
    print(f"  Not dangerous: {len(not_dangerous)} errors")

    # n_dets distribution for dangerous errors
    by_nd = defaultdict(int)
    by_nd_prob = defaultdict(list)
    for prob, nd, fl, _ in dangerous:
        by_nd[nd] += 1
        by_nd_prob[nd].append(prob)
    print(f"\n  Dangerous errors by n_dets:")
    for nd in sorted(by_nd.keys()):
        probs = by_nd_prob[nd]
        print(f"    n_dets={nd}: count={by_nd[nd]}, "
              f"p_min={min(probs):.2e}, p_max={max(probs):.2e}, "
              f"p_sum={sum(probs):.2e}")

    # For n_dets=1 dangerous errors: look for competing non-dangerous explanations
    print(f"\n  Analyzing n_dets=1 dangerous entries vs competing safe explanations:")
    nd1_dang = [(prob, fl, det_ids) for prob, nd, fl, det_ids in dangerous if nd == 1]
    nd1_safe = [(prob, fl, det_ids) for prob, nd, fl, det_ids in not_dangerous if nd == 1]

    # Build map from det_id to safe alternatives
    safe_by_det = defaultdict(list)
    for prob, fl, det_ids in nd1_safe:
        for d_id in det_ids:
            safe_by_det[d_id].append((prob, fl))

    total_dang_prob = 0.0
    decoder_wrong_count = 0
    for prob_dang, fl_dang, det_ids in nd1_dang[:10]:  # sample first 10
        total_dang_prob += prob_dang
        det_id = det_ids[0]
        competing_safe = safe_by_det.get(det_id, [])
        safe_total_prob = sum(p for p,_ in competing_safe)
        ratio = safe_total_prob / max(prob_dang, 1e-30)
        will_fail = (safe_total_prob > prob_dang)  # simplified: if safe > dang, decoder likely chooses safe
        if will_fail:
            decoder_wrong_count += 1
        print(f"    dang: p={prob_dang:.2e}, det={det_id}, flip={fl_dang}")
        print(f"      competing safe (same det): n={len(competing_safe)}, "
              f"p_sum={safe_total_prob:.2e}, ratio={ratio:.1f}x -> "
              f"{'DECODER LIKELY WRONG' if will_fail else 'decoder OK'}")

    if len(nd1_dang) > 10:
        rest_probs = [p for p,_,_ in nd1_dang[10:]]
        print(f"    ... ({len(nd1_dang)-10} more, p_sum={sum(rest_probs):.2e})")

    # Total dangerous probability (sum of all dangerous DEM entry probs)
    total_dang = sum(p for p,_,_,_ in dangerous)
    print(f"\n  Total dangerous DEM probability sum: {total_dang:.3e}")
    print(f"  (If decoder fixes all: LER floor = 0)")
    print(f"  (If decoder fixes none: LER floor = {total_dang:.3e})")

    # Multi-obs DEM entries (max_obs_flip > 1)
    multi_obs = [(prob, nd, fl) for prob, nd, fl, _ in not_dangerous if len(fl) > 1]
    print(f"\n  Non-dangerous multi-obs DEM entries: {len(multi_obs)}")
    by_nfl = defaultdict(int)
    for prob, nd, fl in multi_obs:
        by_nfl[len(fl)] += 1
    print(f"  by n_obs_flip: {dict(sorted(by_nfl.items()))}")

    # Check which patches contribute to W0-only dangerous errors
    print(f"\n  PS coverage per patch:")
    M, pn = build_obs_patch_matrix(circuit, system)
    for wname in ['W0','W1','W2','W3','W4','W5','W6','W7']:
        # Find which observables contain this patch
        in_obs = []
        for obs_i in range(M.shape[0]):
            for col_j in range(len(pn)):
                if pn[col_j] == wname and M[obs_i, col_j]:
                    in_obs.append(obs_i)
                    break
        in_ps = [i for i in in_obs if i in ps_obs]
        in_target_flag = any(i in target_obs for i in in_obs)
        print(f"    {wname}: in obs={in_obs}, ps_coverage={in_ps} -> "
              f"{'UNPROTECTED (only target)' if not in_ps and in_target_flag else 'protected' if in_ps else 'not in any'}")

    # Analyze multi-detector dangerous entries vs safe alternatives
    print(f"\n  Multi-detector dangerous entries vs safe competitors:")
    safe_by_detset = defaultdict(list)
    for prob, nd, fl, det_ids in not_dangerous:
        safe_by_detset[det_ids].append((prob, fl))

    decoder_fail_estimate = 0.0
    decoder_ok_estimate = 0.0
    by_nd_fail = defaultdict(float)
    for prob_dang, nd, fl_dang, det_ids in dangerous:
        competing_safe = safe_by_detset.get(det_ids, [])
        safe_total_prob = sum(pp for pp,_ in competing_safe)
        if safe_total_prob > prob_dang:
            decoder_fail_estimate += prob_dang
            by_nd_fail[nd] += prob_dang
        else:
            decoder_ok_estimate += prob_dang

    print(f"  Dangerous where safe competitor wins (decoder fails): p_sum={decoder_fail_estimate:.3e}")
    print(f"  Dangerous where decoder wins: p_sum={decoder_ok_estimate:.3e}")
    print(f"  By n_dets (decoder fail contribution):")
    for nd in sorted(by_nd_fail.keys()):
        print(f"    n_dets={nd}: fail_p_sum={by_nd_fail[nd]:.3e}")

    return len(dangerous), total_dang


def main():
    p = 1e-3
    r = 1

    print("Step 3 (CNOT encoding + SE, no S gates) DEM 结构分析")
    print("目标: 理解为什么 BPOSD 解码器无法修正 dangerous errors")
    print(f"p={p:.0e}, r={r}")

    results = {}
    for d in [3, 5]:
        n_dang, total_prob = analyze_dem_structure(d, r, p)
        results[d] = (n_dang, total_prob)

    print(f"\n{'='*70}")
    print(f"  d 扩展对比")
    print(f"{'='*70}")
    print(f"  {'d':>4}  {'#dangerous':>12}  {'p_sum':>12}  {'expected LER floor':>20}")
    for d, (n, prob) in sorted(results.items()):
        print(f"  {d:>4}  {n:>12}  {prob:>12.3e}  {'p*n='+str(f'{n*p:.3e}'):>20}")


if __name__ == "__main__":
    main()
