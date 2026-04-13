"""
Save distillation simulation results to CSV.

Runs injection-only, full circuit-level, and/or both noise models for TG and LS
distillation protocols, then appends results to CSV files under their respective
eval/ subdirectories.

Noise modes
-----------
injection   p_injected on magic-patch resets only; everything else noiseless.
full        p applied uniformly to all noise channels.
both        p on all channels + p_injected extra on magic resets independently.

Usage (from repo root):
    python eval/run_distillation_simulations.py                        # all
    python eval/run_distillation_simulations.py --notebook tg          # TG only
    python eval/run_distillation_simulations.py --noise full           # full noise only
    python eval/run_distillation_simulations.py --noise both           # both modes
    python eval/run_distillation_simulations.py -d 3 --rounds 1 --max-errors 100

Output files:
    eval/logical_circuit_benchmark/distillation/tg_7to1/TG_injection_results.csv
    eval/logical_circuit_benchmark/distillation/tg_7to1/TG_full_noise_results.csv
    eval/logical_circuit_benchmark/distillation/tg_7to1/TG_both_results.csv
    eval/logical_circuit_benchmark/distillation/ls_7to1/LS_injection_results.csv
    eval/logical_circuit_benchmark/distillation/ls_7to1/LS_full_noise_results.csv
    eval/logical_circuit_benchmark/distillation/ls_7to1/LS_both_results.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import stim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.logical_circuit_benchmark.distillation.tg_7to1.TG_distillation_7_to_1 import (
    build_distillation_circuit as build_tg,
    inject_noise as inject_noise_tg,
    estimate_p_in as estimate_p_in_tg,
    _TG_MAGIC_NAMES,
)
from eval.logical_circuit_benchmark.distillation.ls_7to1.LS_distillation_7_to_1 import (
    build_distillation_circuit as build_ls,
    inject_noise as inject_noise_ls,
    estimate_p_in as estimate_p_in_ls,
    _LS_MAGIC_NAMES,
)
from src.simulation.decoder_backend.config import DecoderConfig
from src.simulation.decoder_backend.pipeline import SimulationPipeline
from src.simulation.observable_analysis import (
    build_obs_patch_matrix,
    identify_distillation_observables,
)

# ---------------------------------------------------------------------------
# Observable transform helper
# ---------------------------------------------------------------------------

def bake_observable_transform(circuit, T):
    """Rewrite OBSERVABLE_INCLUDE instructions to encode GF(2) transform T."""
    n_obs = T.shape[0]
    obs_targets = {i: [] for i in range(n_obs)}
    for inst in circuit:
        if isinstance(inst, stim.CircuitInstruction) and inst.name == "OBSERVABLE_INCLUDE":
            obs_targets[int(inst.gate_args_copy()[0])].extend(inst.targets_copy())
    new_obs = {}
    for i in range(n_obs):
        counts = {}
        for j in range(n_obs):
            if T[i, j]:
                for t in obs_targets[j]:
                    counts[t.value] = counts.get(t.value, 0) + 1
        new_obs[i] = [stim.target_rec(v) for v, c in counts.items() if c % 2 == 1]
    new_circuit = stim.Circuit()
    emitted = False
    for inst in circuit:
        if isinstance(inst, stim.CircuitInstruction) and inst.name == "OBSERVABLE_INCLUDE":
            if not emitted:
                for i in range(n_obs):
                    if new_obs[i]:
                        new_circuit.append("OBSERVABLE_INCLUDE", new_obs[i], [i])
                emitted = True
        else:
            new_circuit.append(inst)
    return new_circuit


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def make_pipeline(ps_idx, target_idx, max_shots, max_errors, batch_size, num_workers=1):
    return SimulationPipeline(
        decoder_config=DecoderConfig("pymatching"),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        post_select_corrected_observable_indices=ps_idx,
        target_observable_indices=[target_idx],
        print_progress=True,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# CSV writer (append with key-based dedup)
# ---------------------------------------------------------------------------

def write_csv(path, key_cols, data_cols, rows):
    """Append rows to CSV, skipping rows whose key_cols values already exist.

    If the existing file has a stale schema (any column added, removed, or
    renamed), it is archived to *_legacy.csv and a fresh file is started.
    """
    all_cols = key_cols + data_cols
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Load existing keys; detect stale schema (column set mismatch or missing key)
    existing_keys = set()
    has_stale = False
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = set(reader.fieldnames or [])
            if existing_fieldnames != set(all_cols):
                has_stale = True
            else:
                for row in reader:
                    try:
                        existing_keys.add(tuple(row[k] for k in key_cols))
                    except KeyError:
                        has_stale = True
                        break

    if has_stale:
        legacy = path.replace(".csv", "_legacy.csv")
        os.rename(path, legacy)
        print(f"  Archived stale schema → {legacy}")
        existing_keys = set()

    write_header = len(existing_keys) == 0

    new_rows = [r for r in rows
                if tuple(str(r[k]) for k in key_cols) not in existing_keys]

    if not new_rows:
        print(f"  No new rows (all keys already in {path})")
        return

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        if write_header:
            w.writeheader()
        w.writerows(new_rows)
    print(f"  Appended {len(new_rows)} row(s) → {path}")


# ---------------------------------------------------------------------------
# Sweep helper
# ---------------------------------------------------------------------------

def sweep(pipeline, noisy_fn, sweep_values, sweep_key, fixed_keys, label):
    """Run pipeline over sweep_values.

    Args:
        pipeline:     Configured SimulationPipeline.
        noisy_fn:     Callable(sweep_value) → noisy stim.Circuit.
        sweep_values: Iterable of values for the swept parameter.
        sweep_key:    Column name for the swept parameter (e.g. "p" or "p_injected").
        fixed_keys:   Dict of constant columns included in every row.
        label:        Label for progress output.

    Returns:
        List of result dicts.
    """
    rows = []
    for val in sweep_values:
        noisy = noisy_fn(val)
        stats = pipeline.run(noisy, json_metadata={sweep_key: val})
        rows.append({
            **fixed_keys,
            sweep_key: val,
            "ler_ps": stats.logical_error_rate,
            "post_selection_rate": stats.post_selection_rate,
            "shots": stats.shots,
            "errors": stats.errors,
        })
        print(f"    [{label}] {sweep_key}={val:.1e}  LER={stats.logical_error_rate:.3e}"
              f"  accept={stats.post_selection_rate:.3f}  shots={stats.shots:,}")
    return rows


# ---------------------------------------------------------------------------
# CSV schemas
# ---------------------------------------------------------------------------

TG_KEY_COLS  = ["d", "rounds", "r", "p", "p_injected"]
TG_DATA_COLS = ["p_in", "ler_ps", "post_selection_rate", "shots", "errors"]
LS_KEY_COLS  = ["d", "rounds", "p", "p_injected"]
LS_DATA_COLS = ["p_in", "ler_ps", "post_selection_rate", "shots", "errors"]


# ---------------------------------------------------------------------------
# TG simulations
# ---------------------------------------------------------------------------

def run_tg(d, rounds, r, p_values, p_injected_values, max_shots, max_errors, batch_size,
           noise_models, num_workers=1):
    print(f"\n=== TG 7-to-1 distillation  d={d}  rounds={rounds}  r={r} ===")
    key_prefix = {"d": d, "rounds": rounds, "r": r}

    # Build circuit once (noiseless); analyse observables
    circuit, _, system = build_tg(d=d, rounds=rounds, r=r)
    matrix, pn = build_obs_patch_matrix(circuit, system)
    T, tgt, ps = identify_distillation_observables(matrix, pn, ["W0"])
    circuit_T = bake_observable_transform(circuit, T)
    magic_qubits = {q for q, owner in system.index_to_owner_map.items()
                    if owner in _TG_MAGIC_NAMES}
    pipeline = make_pipeline(ps, tgt[0], max_shots, max_errors, batch_size, num_workers)

    # Pre-compute p_in calibrations.
    # injection mode: p_background=0 (corner qubit Z_ERROR only)
    # both mode: p_background=p (matches the circuit-level background in the distillation run)
    def _calibrate(p_background, label):
        print(f"\n-- Calibrating p_in  (p_background={p_background:.1e}) --")
        m = {}
        for p_inj in p_injected_values:
            p_inj = float(p_inj)
            p_in_est = estimate_p_in_tg(d, rounds, p_inj,
                                        p_background=p_background,
                                        max_shots=max_shots, max_errors=max_errors,
                                        batch_size=batch_size)
            m[p_inj] = p_in_est
            print(f"    p_injected={p_inj:.2e}  →  p_in={p_in_est:.3e}  [{label}]")
        return m

    def _add_p_in(rows, p_in_map, sweep_key="p_injected"):
        for row in rows:
            row["p_in"] = p_in_map.get(float(row.get(sweep_key, 0.0)), 0.0)
        return rows

    if "injection" in noise_models:
        p_in_map_inj = _calibrate(p_background=0.0, label="injection")
        print("\n-- Injection-only noise --")
        def noisy_inj(p_inj):
            return inject_noise_tg(circuit_T, magic_qubits, p=0.0,
                                   p_injected=p_inj, mode="injection")
        rows = _add_p_in(sweep(pipeline, noisy_inj, p_injected_values,
                               "p_injected", {**key_prefix, "p": 0.0}, "injection"),
                         p_in_map_inj)
        write_csv("eval/logical_circuit_benchmark/distillation/tg_7to1/TG_injection_results.csv",
                  TG_KEY_COLS, TG_DATA_COLS, rows)

    if "full" in noise_models:
        print("\n-- Full circuit-level noise --")
        def noisy_full(p):
            return inject_noise_tg(circuit_T, magic_qubits, p=p,
                                   p_injected=0.0, mode="full")
        rows = [dict(r, p_in=0.0) for r in sweep(pipeline, noisy_full, p_values,
                                                   "p", {**key_prefix, "p_injected": 0.0}, "full")]
        write_csv("eval/logical_circuit_benchmark/distillation/tg_7to1/TG_full_noise_results.csv",
                  TG_KEY_COLS, TG_DATA_COLS, rows)

    if "both" in noise_models:
        print("\n-- Both noise modes --")
        for p in p_values:
            p_in_map_both = _calibrate(p_background=float(p), label=f"both p={p:.1e}")
            def noisy_both(p_inj, _p=p):
                return inject_noise_tg(circuit_T, magic_qubits, p=_p,
                                       p_injected=p_inj, mode="both")
            rows = _add_p_in(sweep(pipeline, noisy_both, p_injected_values,
                                   "p_injected", {**key_prefix, "p": p}, "both"),
                             p_in_map_both)
            write_csv("eval/logical_circuit_benchmark/distillation/tg_7to1/TG_both_results.csv",
                      TG_KEY_COLS, TG_DATA_COLS, rows)


# ---------------------------------------------------------------------------
# LS simulations
# ---------------------------------------------------------------------------

def run_ls(d, rounds, p_values, p_injected_values, max_shots, max_errors, batch_size,
           noise_models, num_workers=1):
    print(f"\n=== LS 7-to-1 distillation  d={d}  rounds={rounds} ===")
    key_prefix = {"d": d, "rounds": rounds}

    circuit, _, system = build_ls(d=d, rounds=rounds)
    matrix, pn = build_obs_patch_matrix(circuit, system)
    _, tgt, ps = identify_distillation_observables(matrix, pn, ["W4"])
    magic_qubits = {q for q, owner in system.index_to_owner_map.items()
                    if owner in _LS_MAGIC_NAMES}
    # Data qubits only — injection noise targets these, matching the paper's model
    # (Z_ERROR on magic state data qubits; ancilla resets in SE rounds are excluded).
    magic_data_qubits = magic_qubits & system.data_indices
    pipeline = make_pipeline(ps, tgt[0], max_shots, max_errors, batch_size, num_workers)

    # Pre-compute p_in (logical input infidelity) for each p_injected value.
    # Pre-compute p_in calibrations.
    # injection mode: p_background=0 (data-qubit Z_ERROR only)
    # both mode: p_background=p (matches the circuit-level background in the distillation run)
    def _calibrate(p_background, label):
        print(f"\n-- Calibrating p_in  (p_background={p_background:.1e}) --")
        m = {}
        for p_inj in p_injected_values:
            p_inj = float(p_inj)
            p_in_est = estimate_p_in_ls(d, rounds, p_inj,
                                        p_background=p_background,
                                        max_shots=max_shots, max_errors=max_errors,
                                        batch_size=batch_size)
            m[p_inj] = p_in_est
            print(f"    p_injected={p_inj:.2e}  →  p_in={p_in_est:.3e}  [{label}]")
        return m

    def _add_p_in(rows, p_in_map, sweep_key="p_injected"):
        """Attach p_in to each row; 0.0 for full-noise rows with no injection."""
        for row in rows:
            row["p_in"] = p_in_map.get(float(row.get(sweep_key, 0.0)), 0.0)
        return rows

    if "injection" in noise_models:
        p_in_map_inj = _calibrate(p_background=0.0, label="injection")
        print("\n-- Injection-only noise --")
        def noisy_inj(p_inj):
            return inject_noise_ls(circuit, magic_qubits, p=0.0,
                                   p_injected=p_inj, mode="injection",
                                   data_indices=magic_data_qubits)
        rows = _add_p_in(sweep(pipeline, noisy_inj, p_injected_values,
                               "p_injected", {**key_prefix, "p": 0.0}, "injection"),
                         p_in_map_inj)
        write_csv("eval/logical_circuit_benchmark/distillation/ls_7to1/LS_injection_results.csv",
                  LS_KEY_COLS, LS_DATA_COLS, rows)

    if "full" in noise_models:
        print("\n-- Full circuit-level noise --")
        def noisy_full(p):
            return inject_noise_ls(circuit, magic_qubits, p=p,
                                   p_injected=0.0, mode="full")
        rows = [dict(r, p_in=0.0) for r in sweep(pipeline, noisy_full, p_values,
                                                   "p", {**key_prefix, "p_injected": 0.0}, "full")]
        write_csv("eval/logical_circuit_benchmark/distillation/ls_7to1/LS_full_noise_results.csv",
                  LS_KEY_COLS, LS_DATA_COLS, rows)

    if "both" in noise_models:
        print("\n-- Both noise modes --")
        for p in p_values:
            p_in_map_both = _calibrate(p_background=float(p), label=f"both p={p:.1e}")
            def noisy_both(p_inj, _p=p):
                return inject_noise_ls(circuit, magic_qubits, p=_p,
                                       p_injected=p_inj, mode="both",
                                       data_indices=magic_data_qubits)
            rows = _add_p_in(sweep(pipeline, noisy_both, p_injected_values,
                                   "p_injected", {**key_prefix, "p": p}, "both"),
                             p_in_map_both)
            write_csv("eval/logical_circuit_benchmark/distillation/ls_7to1/LS_both_results.csv",
                      LS_KEY_COLS, LS_DATA_COLS, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument("--type", choices=["tg", "ls", "all"], default="all",
    #                     help="Which distillation experiment to run (default: all)")
    parser.add_argument("--type", choices=["tg", "ls", "all"], default="tg",
                    help="Which distillation experiment to run (default: all)")
    parser.add_argument("--noise", choices=["injection", "full", "both", "all"], default="full",
                        help="Which noise model(s) to simulate (default: both)")

    parser.add_argument("-d", type=int, default=3, help="Code distance (default: 3)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="SE rounds per cycle (default: 1)")
    parser.add_argument("--r", type=int, default=1,
                        help="TG: SE rounds between transversal gate layers (default: 1)")
    parser.add_argument("--p-values", type=float, nargs="+",
                        default=np.logspace(-5, -1, 6),
                        help="Circuit-level error rates (default: 1e-3)")
    parser.add_argument("--p-injected", type=float, nargs="+",
                        default=np.logspace(-3, -1, 5),
                        help="Injection noise rates on magic resets "
                             "(default: logspace(-3,-1,5))")
    parser.add_argument("--max-shots", type=int, default=100_000_000)
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel decoder workers (default: 1)")
    args = parser.parse_args()

    noise_models = (["injection", "full", "both"] if args.noise == "all"
                    else [args.noise])
    p_values = np.array(args.p_values)
    p_injected_values = np.array(args.p_injected)

    if args.type in ("tg", "all"):
        run_tg(args.d, args.rounds, args.r, p_values, p_injected_values,
               args.max_shots, args.max_errors, args.batch_size, noise_models,
               args.num_workers)

    if args.type in ("ls", "all"):
        run_ls(args.d, args.rounds, p_values, p_injected_values,
               args.max_shots, args.max_errors, args.batch_size, noise_models,
               args.num_workers)

    print("\nDone.")


if __name__ == "__main__":
    main()
