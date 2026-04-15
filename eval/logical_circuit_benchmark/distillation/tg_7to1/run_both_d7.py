"""
Quick TG d=7 'both' mode simulation using cached circuit.
Avoids rebuilding the large 15-patch circuit from scratch.
Appends results to TG_both_results.csv.
"""
import sys, os, csv, json, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import stim
from src.noise.config import NoiseConfig
from src.noise.injector import NoiseInjector
from src.simulation.decoder_backend.config import DecoderConfig
from src.simulation.decoder_backend.pipeline import SimulationPipeline
from eval.logical_circuit_benchmark.distillation.tg_7to1.TG_distillation_7_to_1 import (
    estimate_p_in, FlipAfterYResetFiltered,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STIM_PATH  = os.path.join(SCRIPT_DIR, 'circuits', 'TG_7to1_d7_r1.stim')
OBS_PATH   = os.path.join(SCRIPT_DIR, 'circuits', 'TG_7to1_d7_r1_obs.json')
CSV_PATH   = os.path.join(SCRIPT_DIR, 'TG_both_results.csv')

KEY_COLS  = ['d', 'rounds', 'r', 'p', 'p_injected']
DATA_COLS = ['p_in', 'ler_ps', 'post_selection_rate', 'shots', 'errors']
ALL_COLS  = KEY_COLS + DATA_COLS

D = 7
ROUNDS = 1
R = 1
P_CIRC = 5e-4
P_INJECTED_LIST = [1e-3, 5.6e-3, 3.2e-2, 1.78e-1]

MAX_SHOTS   = 50_000_000
MAX_ERRORS  = 100
BATCH_SIZE  = 5_000
NUM_WORKERS = 4

# Magic corner qubits (RY instructions in cached circuit)
MAGIC_CORNER_QUBITS = {1358, 1527, 1696, 1865, 2034, 2203, 2372}

print(f'Loading cached TG d={D} circuit...')
circuit = stim.Circuit.from_file(STIM_PATH)
with open(OBS_PATH) as f:
    meta = json.load(f)
ps  = meta['post_select_obs']   # [1, 2, 3]
tgt = meta['target_obs']        # [0]
print(f'  qubits={circuit.num_qubits}  det={circuit.num_detectors}  obs={circuit.num_observables}')
print(f'  ps={ps}  tgt={tgt}')

all_qubits = list(range(circuit.num_qubits))

# Load done keys to skip
done = set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            try:
                done.add((int(row['d']), int(row['rounds']), int(row['r']),
                          float(row['p']), float(row['p_injected'])))
            except (KeyError, ValueError):
                pass
print(f'Done keys loaded: {len(done)}')

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig('pymatching'),
    max_shots=MAX_SHOTS,
    max_errors=MAX_ERRORS,
    batch_size=BATCH_SIZE,
    post_select_corrected_observable_indices=ps,
    target_observable_indices=tgt,
    print_progress=True,
    progress_interval_sec=15.0,
    num_workers=NUM_WORKERS,
)

def make_noisy_both(p_circuit, p_injected):
    cfg = NoiseConfig(p_1q=p_circuit, p_2q=p_circuit,
                      p_meas=p_circuit, p_reset=p_circuit,
                      custom_params={'p_injected': p_injected})
    inj = NoiseInjector.from_circuit_level(cfg, all_qubits)
    inj.add_rule(FlipAfterYResetFiltered(MAGIC_CORNER_QUBITS, param_name='p_injected'))
    return inj.inject_noise(circuit)

print(f'\n-- Calibrating p_in (p_background={P_CIRC:.1e}) --')
p_in_map = {}
for p_inj in P_INJECTED_LIST:
    p_inj = float(p_inj)
    p_in = estimate_p_in(D, ROUNDS, p_inj, p_background=P_CIRC,
                         max_shots=MAX_SHOTS, max_errors=MAX_ERRORS,
                         batch_size=BATCH_SIZE)
    p_in_map[p_inj] = p_in
    print(f'    p_injected={p_inj:.2e}  →  p_in={p_in:.3e}')

print('\n-- Running both-mode simulation --')
rows = []
for p_inj in P_INJECTED_LIST:
    p_inj = float(p_inj)
    key = (D, ROUNDS, R, P_CIRC, p_inj)
    if key in done:
        print(f'  p_injected={p_inj:.2e} — SKIPPED (already in CSV)')
        continue

    print(f'\n  d={D} p={P_CIRC:.1e} p_inj={p_inj:.2e}')
    noisy = make_noisy_both(P_CIRC, p_inj)
    stats = pipeline.run(noisy)

    row = {
        'd': D, 'rounds': ROUNDS, 'r': R,
        'p': P_CIRC, 'p_injected': p_inj,
        'p_in': p_in_map[p_inj],
        'ler_ps': stats.logical_error_rate,
        'post_selection_rate': stats.post_selection_rate,
        'shots': stats.shots,
        'errors': stats.errors,
    }
    rows.append(row)
    print(f'  LER={stats.logical_error_rate:.3e}  accept={stats.post_selection_rate:.3f}'
          f'  shots={stats.shots:,}  errors={stats.errors}')

if rows:
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ALL_COLS)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f'\nAppended {len(rows)} row(s) → {CSV_PATH}')
else:
    print('\nNo new rows to append.')

print('\nDone.')
