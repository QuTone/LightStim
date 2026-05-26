"""
Comprehensive Compilation Performance Benchmark — Table 3

Covers all protocols in the paper: Memory, BB Code, TG CNOT, LS CNOT,
Bell Teleportation (TG + LS), Steane Distillation (TG + LS), CrossLS.

Measures: compile time (median of N_TRIALS), qubit count, detector count,
          observable count, annotation LoC (DETECTOR + OBSERVABLE_INCLUDE).

Results saved per-entry to precompute/table3.json (checkpoint: never loses
completed work on restart). To regenerate a single entry, delete its key.

Usage:
    venv/bin/python paper_artifact/table/comprehensive_benchmark.py
"""
import sys, time, json, signal
sys.path.insert(0, '/home/xiang/workspace/LightStim')

import numpy as np
from pathlib import Path

# ── Output ───────────────────────────────────────────────────────────────────
# results/ is gitignored — this is the reviewer's local output.
# The canonical reference data lives in precompute/table3.json (committed).
OUT_DIR = Path('/home/xiang/workspace/LightStim/paper_artifact/table/results')
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = OUT_DIR / 'table3.json'

N_TRIALS = 3
TIMEOUT_SEC = 600  # 10 minutes; TG Distill d=7 is expected to timeout

# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint():
    if CKPT_PATH.exists():
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {}

def save_checkpoint(results):
    with open(CKPT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

results = load_checkpoint()

# ── Utilities ─────────────────────────────────────────────────────────────────

def count_annotation_loc(circuit):
    lines = str(circuit.flattened()).split('\n')
    n_det = sum(1 for l in lines if l.strip().startswith('DETECTOR'))
    n_obs = sum(1 for l in lines if l.strip().startswith('OBSERVABLE_INCLUDE'))
    return n_det + n_obs, n_det, n_obs


def bench(key, label, build_fn, d_label="", timeout=False):
    """Run N_TRIALS timing trials. Saves checkpoint after completion.
    Skips if key already in results. If timeout=True, wraps with SIGALRM."""
    if key in results:
        print(f"  [skip] {label} d={d_label} (already in checkpoint)")
        return results[key]

    def _timeout_handler(signum, frame):
        raise TimeoutError("Compilation timed out")

    times = []
    circuit = None
    timed_out = False

    for trial in range(N_TRIALS):
        if timeout:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TIMEOUT_SEC)
        try:
            t0 = time.perf_counter()
            circuit = build_fn()
            times.append(time.perf_counter() - t0)
        except TimeoutError:
            timed_out = True
            print(f"  {label} d={d_label}: TIMEOUT (>{TIMEOUT_SEC//60} min) after trial {trial+1}")
            break
        finally:
            if timeout:
                signal.alarm(0)

    if timed_out:
        info = {
            'label': label, 'd': d_label,
            'compile_ms': None, 'timeout': True, 'timeout_sec': TIMEOUT_SEC,
            'num_qubits': None, 'num_detectors': None,
            'num_observables': None, 'annotation_loc': None,
        }
    else:
        t_med = np.median(times)
        ann_loc, n_det, n_obs = count_annotation_loc(circuit)
        info = {
            'label': label, 'd': d_label,
            'compile_ms': round(t_med * 1000, 1), 'timeout': False,
            'num_qubits': circuit.num_qubits,
            'num_detectors': circuit.num_detectors,
            'num_observables': circuit.num_observables,
            'annotation_loc': ann_loc,
        }
        print(f"  {label} d={d_label}: {circuit.num_qubits}q, "
              f"{circuit.num_detectors}det, {circuit.num_observables}obs, "
              f"annot={ann_loc}, compile={t_med*1000:.1f}ms")

    results[key] = info
    save_checkpoint(results)
    return info


# =============================================================================
# 1. Surface Code Memory (Rotated)
# =============================================================================
print("\n=== 1. Surface Code Memory (Rotated) ===")
from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode
from lightstim.qec_code.surface_code.rotated.SE_block import RotatedSurfaceCodeExtractionBlock as RotatedSEBlock
from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.noise.config import NoiseConfig

_NC = NoiseConfig(p_1q=0, p_2q=0, p_meas=0, p_reset=0)

for d in [3, 5, 7, 9, 11]:
    def build_surface_mem(d=d):
        patch = RotatedSurfaceCode(distance=d)
        system = QECSystem()
        system.add_patch(patch)
        exp = MemoryExperiment(system, RotatedSEBlock, rounds=d,
                               noise_params=_NC, noise_model='circuit_level', basis='Z')
        return exp.build()
    bench(f'surface_mem_d{d}', 'Surface Mem', build_surface_mem, d_label=d)

# =============================================================================
# 2. BB Code Memory  ([[72,12,6]] r=6  +  [[144,12,12]] r=12)
# =============================================================================
print("\n=== 2. BB Code Memory ===")
from lightstim.qec_code.BB_code.code_patch import BBCode
from lightstim.qec_code.BB_code.SE_block import BBCodeExtractionBlock

_BB_A = [[3, 0], [0, 1], [0, 2]]
_BB_B = [[0, 3], [1, 0], [2, 0]]

BB_CONFIGS = [
    ('72,12,6',   dict(l=6,  m=6, A=_BB_A, B=_BB_B), 6),   # rounds = r = 6
    ('144,12,12', dict(l=12, m=6, A=_BB_A, B=_BB_B), 12),  # rounds = r = 12
]

for code_label, bb_params, bb_rounds in BB_CONFIGS:
    def build_bb_mem(bp=bb_params, br=bb_rounds):
        patch = BBCode(**bp)
        system = QECSystem()
        system.add_patch(patch)
        exp = MemoryExperiment(system, BBCodeExtractionBlock, rounds=br,
                               noise_params=_NC, noise_model='circuit_level', basis='Z')
        return exp.build()
    bench(f'bb_{code_label}_r{bb_rounds}', f'BB [[{code_label}]]', build_bb_mem, d_label='—')

# =============================================================================
# 3. Transversal Gate CNOT (Unrotated SC)  — d=7, 15
# =============================================================================
print("\n=== 3. TG CNOT (Unrotated SC) ===")
from lightstim.protocols.cnot_trans import CNOTTransExperiment
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)

for d in [7, 15]:
    def build_tg_cnot(d=d):
        exp = CNOTTransExperiment(
            code_patch_class=UnrotatedSurfaceCode,
            extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
            code_params_control={'distance': d},
            offset_target=(4 * d, 0),
            initial_basis_control='Z', initial_basis_target='Z',
            measure_basis_control='Z', measure_basis_target='Z',
            rounds_before=d, rounds_after=1,
            noise_params=_NC, noise_model='circuit_level',
        )
        return exp.build()
    bench(f'tg_cnot_d{d}', 'TG CNOT', build_tg_cnot, d_label=d)

# =============================================================================
# 4. LS CNOT (Unrotated SC)  — d=7, 15
# =============================================================================
print("\n=== 4. LS CNOT (Unrotated SC) ===")
from lightstim.protocols.cnot_ls import CNOTLSExperiment

for d in [7, 15]:
    def build_ls_cnot(d=d):
        step = 2 * (2 * d - 1)
        patch_configs = {'c': {'distance': d}, 't': {'distance': d}, 'a': {'distance': d}}
        exp = CNOTLSExperiment(
            patch_configs=patch_configs,
            offset_ta=(step, 0), offset_ca=(0, step),
            initial_state_dict={'a': 'X', 'c': 'X', 't': 'Z'},
            measure_state_dict={'a': 'Z', 'c': 'X', 't': 'X'},
            extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
            rounds=d, noise_params=_NC, noise_model='circuit_level',
        )
        return exp.build()
    bench(f'ls_cnot_d{d}', 'LS CNOT', build_ls_cnot, d_label=d)

# =============================================================================
# 5. Bell Teleportation — TG variant  — d=7, 15
# =============================================================================
print("\n=== 5. Bell Teleportation TG ===")
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.ir.operation import CSSLogicalOpSet
from lightstim.ir.tracker import SyndromeTracker

def _build_bell_tg_circuit(d, teleport_state='Z'):
    dx = 2 * (2 * d - 1) - 2
    system = QECSystem()
    p1 = system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch1')
    p2 = system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch2', offset=(dx, 0))
    p3 = system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch3', offset=(2 * dx, 0))

    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    INIT = {'patch1': teleport_state, 'patch2': 'X', 'patch3': 'Z'}
    builder.initialize(
        init_dict={q: INIT[system.index_to_owner_map[q]] for q in system.data_indices},
        n=system.num_qubits)
    se = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(se.circuit, rounds=d)

    executor = LogicalExecutor(builder=builder)
    executor.register_op_set(UnrotatedSurfaceCode, CSSLogicalOpSet())
    executor.apply_logical_operation('transversal_cnot', patches=[p2, p3])
    builder.apply_syndrome_extraction(UnrotatedSurfaceCodeExtractionBlock(system).circuit, rounds=1)
    executor.apply_logical_operation('transversal_cnot', patches=[p1, p2])
    builder.apply_syndrome_extraction(UnrotatedSurfaceCodeExtractionBlock(system).circuit, rounds=1)

    MEAS = {'patch1': 'X', 'patch2': 'Z', 'patch3': teleport_state}
    builder.apply_data_readout(
        final_measurements={q: MEAS[system.index_to_owner_map[q]] for q in system.data_indices})
    return builder.circuit

for d in [7, 15]:
    def build_bell_tg(d=d):
        return _build_bell_tg_circuit(d, 'Z')
    bench(f'bell_tg_d{d}', 'Bell Tele. (TG)', build_bell_tg, d_label=d)

# =============================================================================
# 6. Bell Teleportation — LS-ZZ variant  — d=7, 15
# =============================================================================
print("\n=== 6. Bell Teleportation LS-ZZ ===")
from lightstim.qec_code.surface_code.unrotated import UnrotatedTwoPatchCoupler

def _build_bell_ls_zz_circuit(d, teleport_state='Z'):
    step = (2 * d - 1) + 1
    system = QECSystem()
    system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch1')
    system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch2', offset=(0, step))
    system.add_patch(UnrotatedSurfaceCode(distance=d), name='patch3', offset=(0, 2 * step))

    coupler_proto = UnrotatedTwoPatchCoupler()
    system.register_coupler(coupler_proto, patch_names=['patch2', 'patch3'],
                            name='coupler_23', interaction_type='ZZ')
    system.register_coupler(coupler_proto, patch_names=['patch1', 'patch2'],
                            name='coupler_12', interaction_type='ZZ')

    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    init_zz = {'patch1': teleport_state, 'patch2': 'X', 'patch3': 'X'}
    builder.initialize(
        init_dict={q: init_zz[system.index_to_owner_map[q]]
                   for q in system.data_indices
                   if system.index_to_owner_map[q] in init_zz},
        n=system.num_qubits)
    builder.apply_syndrome_extraction(
        UnrotatedSurfaceCodeExtractionBlock(system).circuit, rounds=d)

    builder.activate_coupler('coupler_23')
    cp23_g = [system.local_to_global_map['coupler_23'][q]
              for q in system.coupler_patches['coupler_23'].data_indices]
    builder.initialize(init_dict={q: 'X' for q in cp23_g}, n=system.num_qubits)
    builder.apply_syndrome_extraction(
        UnrotatedSurfaceCodeExtractionBlock(system).circuit, rounds=d)
    builder.deactivate_coupler('coupler_23')
    builder.apply_data_readout(final_measurements={q: 'X' for q in cp23_g})

    builder.activate_coupler('coupler_12')
    cp12_g = [system.local_to_global_map['coupler_12'][q]
              for q in system.coupler_patches['coupler_12'].data_indices]
    builder.initialize(init_dict={q: 'X' for q in cp12_g}, n=system.num_qubits)
    builder.apply_syndrome_extraction(
        UnrotatedSurfaceCodeExtractionBlock(system).circuit, rounds=d)

    meas_zz = {'patch1': 'X', 'patch2': 'X', 'patch3': teleport_state}
    meas_dict = {q: meas_zz[system.index_to_owner_map[q]]
                 for q in system.data_indices
                 if system.index_to_owner_map[q] in meas_zz}
    meas_dict.update({q: 'X' for q in cp12_g})
    builder.apply_data_readout(final_measurements=meas_dict)
    return builder.circuit

for d in [7, 15]:
    def build_bell_ls_zz(d=d):
        return _build_bell_ls_zz_circuit(d, 'Z')
    bench(f'bell_ls_d{d}', 'Bell Tele. (LS)', build_bell_ls_zz, d_label=d)

# =============================================================================
# 7. Steane 7-to-1 Distillation — TG variant  — d=3 (fast), d=7 (may timeout)
# =============================================================================
print("\n=== 7. Steane 7-to-1 Distillation TG ===")
try:
    from lightstim.protocols.tg_distillation import build_distillation_circuit as _build_tg_dist

    for d in [3, 7]:
        def build_tg_7to1(d=d):
            circuit, _, _ = _build_tg_dist(d=d, rounds_init=d, rounds_gate=1)
            return circuit
        bench(f'tg_dist_d{d}', 'TG Distill.', build_tg_7to1, d_label=d,
              timeout=(d >= 7))
except Exception as e:
    print(f"  TG Distillation skipped: {e}")
    import traceback; traceback.print_exc()

# =============================================================================
# 8. Steane 7-to-1 Distillation — LS variant  — d=5, 9
# =============================================================================
print("\n=== 8. Steane 7-to-1 Distillation LS ===")
try:
    from lightstim.protocols.ls_distillation import build_distillation_circuit as _build_ls_dist

    for d in [5, 9]:
        def build_ls_7to1(d=d):
            circuit, _, _ = _build_ls_dist(d=d, rounds=d)
            return circuit
        bench(f'ls_dist_d{d}', 'LS Distill.', build_ls_7to1, d_label=d)
except Exception as e:
    print(f"  LS Distillation skipped: {e}")
    import traceback; traceback.print_exc()

# =============================================================================
# 9. CrossLS (Surface ↔ PQRM)
#    PQRM(1,2,4) d_surf=3  +  PQRM(1,4,6) d_surf=7
# =============================================================================
print("\n=== 9. CrossLS (Surface–PQRM) ===")
try:
    from lightstim.protocols.cross_ls.cross_ls_experiment import CrossLSExperiment

    CROSS_LS_CONFIGS = [
        ([1, 2, 4], 3),   # PQRM(1,2,4)  d_surf=3  — Table row 1
        ([1, 4, 6], 7),   # PQRM(1,4,6)  d_surf=7  — Table row 2
    ]

    for pqrm_para, d_surf in CROSS_LS_CONFIGS:
        pqrm_label = '-'.join(str(x) for x in pqrm_para)
        def build_cross_ls(pp=pqrm_para, ds=d_surf):
            exp = CrossLSExperiment(
                PQRM_para=pp, d_surf=ds, rounds=ds,
                PQRM_state='Z', surf_state='X',
                noise_params=_NC, noise_model='circuit_level',
            )
            return exp.build()
        bench(f'cross_ls_pqrm{pqrm_label}_dsurf{d_surf}',
              f'CrossLS PQRM({pqrm_label})', build_cross_ls, d_label=d_surf)
except Exception as e:
    print(f"  CrossLS skipped: {e}")
    import traceback; traceback.print_exc()

# =============================================================================
# Summary table
# =============================================================================
print(f"\nResults saved to {CKPT_PATH}")
print("\n" + "=" * 92)
print(f"{'Protocol':<28} {'d':>3} {'Qubits':>7} {'Det':>7} {'Obs':>5} {'AnnLoc':>8} {'ms':>10}")
print("=" * 92)
for k, v in results.items():
    if v.get('timeout'):
        print(f"  {v['label']:<26} {str(v['d']):>3} {'—':>7} {'—':>7} {'—':>5} {'—':>8} "
              f"{'TIMEOUT':>10}")
    elif v.get('num_qubits') is not None:
        print(f"  {v['label']:<26} {str(v['d']):>3} {v['num_qubits']:>7} "
              f"{v['num_detectors']:>7} {v['num_observables']:>5} "
              f"{v['annotation_loc']:>8} {v['compile_ms']:>10.1f}")
print("=" * 92)
