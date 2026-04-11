"""测试 BPOSD 能否正确解码含 all-4-obs flip 的 dangerous errors。"""
import sys, os, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
import numpy as np
import stim
from eval.logical_circuit_benchmark.distillation.tg_7to1.TG_distillation_no_inject import build_no_inject_circuit
from src.simulation.observable_analysis import build_obs_patch_matrix, identify_distillation_observables, transform_observables
from src.noise.config import NoiseConfig
from src.noise.injector import NoiseInjector
from src.simulation.decoder_backend.registry import get_decoder

d, r = 3, 1
print(f"Building d={d}...")
circuit, _, system = build_no_inject_circuit(d, rounds=d, r=r)
matrix, patch_names = build_obs_patch_matrix(circuit, system)
T, target_obs, ps_obs = identify_distillation_observables(matrix, patch_names, ['W0'])
n_obs = circuit.num_observables

p = 1e-3
noise_config = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
injector = NoiseInjector.from_circuit_level(noise_config, list(range(circuit.num_qubits)))
noisy = injector.inject_noise(circuit)

# 统计 DEM 中的危险 errors（transformed frame）
dem = noisy.detector_error_model(approximate_disjoint_errors=True)
dangerous = []
for inst in dem:
    if not (isinstance(inst, stim.DemInstruction) and inst.type == 'error'):
        continue
    targets = inst.targets_copy()
    raw_fl = [t.val for t in targets if t.is_logical_observable_id()]
    n_dets = sum(1 for t in targets if t.is_relative_detector_id())
    rv = np.zeros(n_obs, dtype=int)
    for i in raw_fl: rv[i] = 1
    tv = (T @ rv) % 2
    if any(tv[i] == 1 for i in target_obs) and all(tv[i] == 0 for i in ps_obs):
        dangerous.append((inst.args_copy()[0], n_dets, raw_fl))

by_nd = {}
for _, nd, _ in dangerous:
    by_nd[nd] = by_nd.get(nd, 0) + 1
print(f"Dangerous errors (transformed frame): {len(dangerous)}")
print(f"  n_dets distribution: {dict(sorted(by_nd.items()))}")

# 打印 n_dets=1 的样本
print("\nn_dets=1 样本：")
for p_val, nd, obs in dangerous:
    if nd == 1:
        print(f"  error({p_val:.3e}) obs={obs}")
        break

# 运行 BPOSD 解码 20000 shots
print("\nRunning BPOSD decode (20000 shots)...")
decoder = get_decoder('bposd', backend='cpu')
compiled = decoder.compile_decoder_for_dem(dem=dem)
sampler = noisy.compile_detector_sampler()

t0 = time.perf_counter()
dts, obs = sampler.sample(shots=20000, separate_observables=True)
obs_t = transform_observables(obs, T)
mask = np.all(obs_t[:, ps_obs] == 0, axis=1)
kept = int(np.sum(mask))
print(f"  kept={kept}/{20000} ({kept/200:.1f}%) after PS")

if kept > 0:
    dts_k = dts[mask]; obs_t_k = obs_t[mask]
    n_det = dem.num_detectors
    n_bytes = math.ceil(n_det / 8)
    packed = np.packbits(dts_k, axis=1, bitorder='little')[:, :n_bytes]
    pred_packed = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
    pred = np.unpackbits(pred_packed, axis=1, bitorder='little')[:, :n_obs]
    pred_t = transform_observables(pred, T)
    corr = obs_t_k[:, target_obs] ^ pred_t[:, target_obs]
    errors = int(np.sum(np.any(corr, axis=1)))
    ler = errors / kept
    elapsed = time.perf_counter() - t0
    print(f"  errors={errors}, LER={ler:.3e}  ({elapsed:.1f}s)")
    if ler < 5e-3:
        print("  => LER 正常（有抑制）")
    else:
        print("  => LER 偏高！解码器可能对 multi-obs DEM entry 处理有问题")
