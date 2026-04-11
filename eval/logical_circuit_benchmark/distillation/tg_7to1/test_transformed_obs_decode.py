"""
把 T 变换 embed 进 Stim circuit 的 OBSERVABLE_INCLUDE，
让 DEM 中的 dangerous errors 从 4-obs flip 变成 1-obs flip（graphlike），
然后对比原始 vs 变换后的解码 LER。
"""
import sys, os, math, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
import numpy as np
import stim
from eval.logical_circuit_benchmark.distillation.tg_7to1.TG_distillation_no_inject import build_no_inject_circuit
from src.simulation.observable_analysis import build_obs_patch_matrix, identify_distillation_observables, transform_observables
from src.noise.config import NoiseConfig
from src.noise.injector import NoiseInjector
from src.simulation.decoder_backend.registry import get_decoder


def transform_circuit_observables(circuit: stim.Circuit, T: np.ndarray) -> stim.Circuit:
    """把 GF(2) 变换矩阵 T embed 进 OBSERVABLE_INCLUDE 指令。"""
    n_obs = T.shape[0]
    flat = circuit.flattened()
    obs_bits = {i: set() for i in range(n_obs)}
    total_meas = 0
    for inst in flat:
        if not isinstance(inst, stim.CircuitInstruction):
            continue
        if inst.name == 'OBSERVABLE_INCLUDE':
            oi = int(inst.gate_args_copy()[0])
            for t in inst.targets_copy():
                obs_bits[oi].add(total_meas + t.value)
        elif inst.num_measurements > 0:
            total_meas += inst.num_measurements

    total_meas_all = circuit.num_measurements
    new_obs_bits = {}
    for new_i in range(n_obs):
        combined = set()
        for old_j in range(n_obs):
            if T[new_i, old_j]:
                combined ^= obs_bits[old_j]
        new_obs_bits[new_i] = combined

    new_circuit = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            new_circuit.append(inst)
        elif isinstance(inst, stim.CircuitInstruction):
            if inst.name != 'OBSERVABLE_INCLUDE':
                new_circuit.append(inst)

    for new_i in range(n_obs):
        bits = sorted(new_obs_bits[new_i])
        if not bits:
            continue
        targets = [stim.target_rec(b - total_meas_all) for b in bits]
        new_circuit.append("OBSERVABLE_INCLUDE", targets, new_i)

    return new_circuit


def decode_with_ps(noisy_circuit, decoder_name, target_obs, ps_obs,
                   shots=30_000, backend='cpu'):
    """
    用指定 decoder 解码，post-select on ps_obs == 0，统计 target_obs 的 LER。
    在变换后的电路里，target_obs=[0], ps_obs=[1,2,3]，直接用原始 obs（无需 transform）。
    """
    n_obs = noisy_circuit.num_observables
    n_det = noisy_circuit.num_detectors
    n_bytes = math.ceil(n_det / 8)

    decoder = get_decoder(decoder_name, backend=backend)
    dem = noisy_circuit.detector_error_model(approximate_disjoint_errors=True)
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    sampler = noisy_circuit.compile_detector_sampler()

    total = kept = errors = 0
    t0 = time.perf_counter()

    dts, obs = sampler.sample(shots=shots, separate_observables=True)
    total = shots

    # PS: obs 已在 transformed frame，直接用 ps_obs 列
    ps_vals = obs[:, ps_obs]
    mask = np.all(ps_vals == 0, axis=1)
    kept = int(np.sum(mask))

    if kept == 0:
        return 0, 0, 0, 0.0

    dts_k = dts[mask]
    obs_k = obs[mask]

    packed = np.packbits(dts_k, axis=1, bitorder='little')[:, :n_bytes]
    pred_packed = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
    pred = np.unpackbits(pred_packed, axis=1, bitorder='little')[:, :n_obs]

    corr = obs_k[:, target_obs] ^ pred[:, target_obs]
    errors = int(np.sum(np.any(corr, axis=1)))
    elapsed = time.perf_counter() - t0

    return total, kept, errors, elapsed


def main():
    d, r = 3, 1
    p = 1e-3
    shots = 5_000

    print(f"d={d}, r={r}, p={p:.0e}, shots={shots:,}")
    print("=" * 60)

    # 构建电路
    circuit, _, system = build_no_inject_circuit(d, rounds=d, r=r)
    matrix, pn = build_obs_patch_matrix(circuit, system)
    T, target_obs, ps_obs = identify_distillation_observables(matrix, pn, ['W0'])

    # 变换后电路
    circuit_t = transform_circuit_observables(circuit, T)

    # noiseless check
    ds, obs = circuit_t.compile_detector_sampler().sample(shots=20, separate_observables=True)
    assert not np.any(ds) and not np.any(obs), "Noiseless check FAIL"
    print(f"Noiseless check: OK")
    print(f"target_obs={target_obs}, ps_obs={ps_obs}")
    print()

    p_val = p
    nc = NoiseConfig(p_1q=p_val, p_2q=p_val, p_meas=p_val, p_reset=p_val, p_idle=p_val)
    inj = NoiseInjector.from_circuit_level(nc, list(range(circuit.num_qubits)))

    noisy_orig  = inj.inject_noise(circuit)
    noisy_trans = inj.inject_noise(circuit_t)

    # ---- 原始电路：需要 post-process with T ----
    print("原始电路 + BPOSD (T post-process):")
    n_obs = circuit.num_observables
    n_det = circuit.num_detectors
    n_bytes = math.ceil(n_det / 8)
    decoder = get_decoder('bposd', backend='cpu')
    dem_orig = noisy_orig.detector_error_model(approximate_disjoint_errors=True)
    compiled_orig = decoder.compile_decoder_for_dem(dem=dem_orig)
    sampler_orig = noisy_orig.compile_detector_sampler()
    t0 = time.perf_counter()
    dts, obs_raw = sampler_orig.sample(shots=shots, separate_observables=True)
    obs_t = transform_observables(obs_raw, T)
    mask = np.all(obs_t[:, ps_obs] == 0, axis=1)
    kept = int(np.sum(mask))
    dts_k = dts[mask]; obs_k = obs_t[mask]
    packed = np.packbits(dts_k, axis=1, bitorder='little')[:, :n_bytes]
    pred_packed = compiled_orig.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
    pred = np.unpackbits(pred_packed, axis=1, bitorder='little')[:, :n_obs]
    pred_t = transform_observables(pred, T)
    corr = obs_k[:, target_obs] ^ pred_t[:, target_obs]
    errors_orig = int(np.sum(np.any(corr, axis=1)))
    elapsed_orig = time.perf_counter() - t0
    ler_orig = errors_orig / kept if kept > 0 else 0
    ps_rate_orig = kept / shots
    print(f"  kept={kept:,}/{shots:,} ({ps_rate_orig*100:.1f}%), errors={errors_orig}, LER={ler_orig:.3e}  ({elapsed_orig:.1f}s)")

    # ---- 变换后电路：obs 已在 transformed frame，直接解码 ----
    print("\n变换后电路 + BPOSD (obs 已在 transformed frame):")
    t, k, e, elapsed_t = decode_with_ps(
        noisy_trans, 'bposd',
        target_obs=[0], ps_obs=[1, 2, 3],
        shots=shots, backend='cpu'
    )
    ler_trans = e / k if k > 0 else 0
    print(f"  kept={k:,}/{t:,} ({k/t*100:.1f}%), errors={e}, LER={ler_trans:.3e}  ({elapsed_t:.1f}s)")

    print()
    print(f"LER 对比:")
    print(f"  原始 DEM (4-obs flip):    {ler_orig:.3e}")
    print(f"  变换后 DEM (1-obs flip):  {ler_trans:.3e}")
    if ler_trans < ler_orig:
        print(f"  => 改善 {ler_orig/ler_trans:.1f}x")
    else:
        print(f"  => 没有改善")


if __name__ == "__main__":
    main()
