"""
Save distillation simulation results to CSV.

Runs injection-only and full circuit-level noise models for both TG and LS
distillation protocols, then writes results to CSV files under their respective
eval/ subdirectories.

Usage (from repo root):
    python eval/run_distillation_simulations.py                  # all
    python eval/run_distillation_simulations.py --notebook tg    # TG only
    python eval/run_distillation_simulations.py --noise full     # full noise only
    python eval/run_distillation_simulations.py -d 3 --rounds 1 --max-errors 100

Output files:
    eval/TG_distillation/TG_injection_results.csv
    eval/TG_distillation/TG_full_noise_results.csv
    eval/LS_distillation/LS_injection_results.csv
    eval/LS_distillation/LS_full_noise_results.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import stim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.TG_distillation.TG_distillation_7_to_1 import build_distillation_circuit as build_tg
from eval.LS_distillation.LS_distillation_7_to_1 import build_distillation_circuit as build_ls
from src.noise.config import NoiseConfig
from src.noise.injector import NoiseInjector
from src.noise.rules import FlipAfterReset, NoiseRule
from src.simulation.decoder_backend.config import DecoderConfig
from src.simulation.decoder_backend.pipeline import SimulationPipeline
from src.simulation.observable_analysis import (
    build_obs_patch_matrix,
    identify_distillation_observables,
)

# ---------------------------------------------------------------------------
# Helpers
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


class FlipAfterResetXZ(FlipAfterReset):
    """FlipAfterReset that skips Y-basis resets (matches TG inject_injection_noise)."""
    def apply(self, instruction, config, active_qubits):
        if instruction.name in self.y_reset:
            return [], []
        return super().apply(instruction, config, active_qubits)


class FlipAfterResetMagic(NoiseRule):
    """FlipAfterReset filtered to a specific set of qubit indices (for LS injection noise)."""
    _X_RESET = {"RX", "MRX"}
    _Z_RESET = {"R", "RZ", "MR", "MRZ"}

    def __init__(self, magic_qubits, param_name="p_reset"):
        self.magic_qubits = set(magic_qubits)
        self.param_name = param_name

    def apply(self, instruction, config, active_qubits):
        if instruction.name not in self._X_RESET and instruction.name not in self._Z_RESET:
            return [], []
        targets = [t.value for t in instruction.targets_copy()
                   if t.is_qubit_target and t.value in self.magic_qubits]
        if not targets:
            return [], []
        p = config.get(self.param_name)
        if p <= 0:
            return [], []
        err_op = "Z_ERROR" if instruction.name in self._X_RESET else "X_ERROR"
        return [], [stim.CircuitInstruction(err_op, targets, [p])]


def make_pipeline(ps_idx, target_idx, max_shots, max_errors, batch_size):
    return SimulationPipeline(
        decoder_config=DecoderConfig("pymatching"),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        post_select_corrected_observable_indices=ps_idx,
        target_observable_indices=[target_idx],
        print_progress=False,
    )


def write_csv(path, key_cols, data_cols, rows):
    """Append rows to CSV, skipping any row whose key_cols values already exist.

    key_cols: list of column names that form the dedup key (e.g. ["d","rounds","r","p"])
    data_cols: remaining column names (e.g. ["ler_ps","post_selection_rate","shots","errors"])
    rows: list of dicts with keys = key_cols + data_cols
    """
    all_cols = key_cols + data_cols
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Load existing keys
    existing_keys = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    existing_keys.add(tuple(row[k] for k in key_cols))
                except KeyError:
                    pass  # stale schema — will be overwritten below

    # Write header only when creating a new file (no existing keys found)
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


def sweep(pipeline, noisy_fn, p_values, label, key_prefix):
    """Run pipeline over p_values. noisy_fn(p) -> noisy stim.Circuit.

    key_prefix: dict of fixed key columns to include in every row (e.g. {"d":3,"rounds":1,"r":1})
    Returns list of dicts.
    """
    rows = []
    for p in p_values:
        noisy = noisy_fn(p)
        stats = pipeline.run(noisy, json_metadata={"p": p})
        rows.append({**key_prefix, "p": p,
                     "ler_ps": stats.logical_error_rate,
                     "post_selection_rate": stats.post_selection_rate,
                     "shots": stats.shots,
                     "errors": stats.errors})
        print(f"    [{label}] p={p:.1e}  LER={stats.logical_error_rate:.3e}"
              f"  accept={stats.post_selection_rate:.3f}  shots={stats.shots:,}")
    return rows


# ---------------------------------------------------------------------------
# TG simulations
# ---------------------------------------------------------------------------

TG_KEY_COLS  = ["d", "rounds", "r", "p"]
TG_DATA_COLS = ["ler_ps", "post_selection_rate", "shots", "errors"]
LS_KEY_COLS  = ["d", "rounds", "p"]
LS_DATA_COLS = ["ler_ps", "post_selection_rate", "shots", "errors"]


def run_tg(d, rounds, r, p_values, max_shots, max_errors, batch_size, noise_models):
    print(f"\n=== TG 7-to-1 distillation  d={d}  rounds={rounds}  r={r} ===")
    key_prefix = {"d": d, "rounds": rounds, "r": r}

    # --- Injection-only ---
    if "injection" in noise_models:
        print("\n-- Injection-only noise --")
        circuit_inj, _, system = build_tg(d=d, rounds=rounds, r=r, injection_noise_only=True)
        matrix, pn = build_obs_patch_matrix(circuit_inj, system)
        T, tgt, ps = identify_distillation_observables(matrix, pn, ["W0"])
        circuit_inj_T = bake_observable_transform(circuit_inj, T)
        pipeline = make_pipeline(ps, tgt[0], max_shots, max_errors, batch_size)

        def noisy_inj(p):
            cfg = NoiseConfig(p_reset=p)
            inj = NoiseInjector(cfg)
            inj.add_rule(FlipAfterResetXZ(param_name="p_reset"))
            return inj.inject_noise(circuit_inj_T)

        rows = sweep(pipeline, noisy_inj, p_values, "injection", key_prefix)
        write_csv("eval/TG_distillation/TG_injection_results.csv",
                  TG_KEY_COLS, TG_DATA_COLS, rows)

    # --- Full circuit-level ---
    if "full" in noise_models:
        print("\n-- Full circuit-level noise --")
        circuit, _, system = build_tg(d=d, rounds=rounds, r=r)
        matrix, pn = build_obs_patch_matrix(circuit, system)
        T, tgt, ps = identify_distillation_observables(matrix, pn, ["W0"])
        circuit_T = bake_observable_transform(circuit, T)
        pipeline = make_pipeline(ps, tgt[0], max_shots, max_errors, batch_size)

        def noisy_full(p):
            cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p)
            return NoiseInjector.from_circuit_level(
                cfg, data_qubit_indices=list(system.data_indices)
            ).inject_noise(circuit_T)

        rows = sweep(pipeline, noisy_full, p_values, "full", key_prefix)
        write_csv("eval/TG_distillation/TG_full_noise_results.csv",
                  TG_KEY_COLS, TG_DATA_COLS, rows)


# ---------------------------------------------------------------------------
# LS simulations
# ---------------------------------------------------------------------------

def run_ls(d, rounds, p_values, max_shots, max_errors, batch_size, noise_models):
    print(f"\n=== LS 4-to-1 distillation  d={d}  rounds={rounds} ===")
    key_prefix = {"d": d, "rounds": rounds}

    # --- Injection-only ---
    if "injection" in noise_models:
        print("\n-- Injection-only noise --")
        circuit, _, system = build_ls(d=d, rounds=rounds)
        matrix, pn = build_obs_patch_matrix(circuit, system)
        _, tgt, ps = identify_distillation_observables(matrix, pn, ["W4"])
        pipeline = make_pipeline(ps, tgt[0], max_shots, max_errors, batch_size)

        magic_names = {"W1", "W2", "W3", "W5"}
        magic_qubits = {q for q, owner in system.index_to_owner_map.items()
                        if owner in magic_names}

        def noisy_inj(p):
            cfg = NoiseConfig(p_reset=p)
            inj = NoiseInjector(cfg)
            inj.add_rule(FlipAfterResetMagic(magic_qubits, param_name="p_reset"))
            return inj.inject_noise(circuit)

        rows = sweep(pipeline, noisy_inj, p_values, "injection", key_prefix)
        write_csv("eval/LS_distillation/LS_injection_results.csv",
                  LS_KEY_COLS, LS_DATA_COLS, rows)

    # --- Full circuit-level ---
    if "full" in noise_models:
        print("\n-- Full circuit-level noise --")
        circuit, _, system = build_ls(d=d, rounds=rounds)
        matrix, pn = build_obs_patch_matrix(circuit, system)
        _, tgt, ps = identify_distillation_observables(matrix, pn, ["W4"])
        pipeline = make_pipeline(ps, tgt[0], max_shots, max_errors, batch_size)

        def noisy_full(p):
            cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p)
            return NoiseInjector.from_circuit_level(
                cfg, data_qubit_indices=list(system.data_indices)
            ).inject_noise(circuit)

        rows = sweep(pipeline, noisy_full, p_values, "full", key_prefix)
        write_csv("eval/LS_distillation/LS_full_noise_results.csv",
                  LS_KEY_COLS, LS_DATA_COLS, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", choices=["tg", "ls", "all"], default="all",
                        help="Which distillation experiment to run (default: all)")
    parser.add_argument("--noise", choices=["injection", "full", "all"], default="all",
                        help="Which noise model(s) to simulate (default: all)")
    parser.add_argument("-d", type=int, default=3, help="Code distance (default: 3)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="SE rounds per cycle (default: 1)")
    parser.add_argument("--r", type=int, default=1,
                        help="TG: SE rounds between transversal gate layers (default: 1)")
    parser.add_argument("--p-values", type=float, nargs="+",
                        default=list(np.logspace(-3, -1, 7)),
                        help="Physical error rates to sweep (default: logspace(-3,-1,7))")
    parser.add_argument("--max-shots", type=int, default=50_000_000)
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5_000)
    args = parser.parse_args()

    noise_models = ["injection", "full"] if args.noise == "all" else [args.noise]
    p_values = np.array(args.p_values)

    if args.notebook in ("tg", "all"):
        run_tg(args.d, args.rounds, args.r, p_values,
               args.max_shots, args.max_errors, args.batch_size, noise_models)

    if args.notebook in ("ls", "all"):
        run_ls(args.d, args.rounds, p_values,
               args.max_shots, args.max_errors, args.batch_size, noise_models)

    print("\nDone.")


if __name__ == "__main__":
    main()
