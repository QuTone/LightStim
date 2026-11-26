"""
General CrossLS (Surface–PQRM Lattice Surgery) benchmark runner.

Experiments
-----------
    sweep   Full LER vs PER sweep:  PQRM × d_surf × state × p  (paper figures)
    rounds  LER vs rounds:          fixed PQRM(1,2,4), p=1e-3, Z state

CSV output  (per-task checkpointing, append-on-complete)
----------
    results/cross_ls_results.csv
    Columns: pqrm, d_surf, rounds, state, p_1q, p_2q, p_meas, p_reset,
             decoder, backend, n_det, n_ps, shots, kept, ps_rate,
             errors, ler, seconds

Usage
-----
    # Quick smoke test (d=3, one p value, 50k shots):
    PYTHONPATH=. venv/bin/python benchmarks/cross_ls/run_cross_ls.py --quick

    # Full paper sweep (all PQRM codes, d=3-7, p=5e-4/1e-3/2e-3):
    PYTHONPATH=. venv/bin/python benchmarks/cross_ls/run_cross_ls.py \\
        --experiment sweep --pqrm 1,2,4 1,3,5 1,4,6 \\
        --p-values 5e-4 1e-3 2e-3 --decoder bposd --backend gpu --num-workers 1

    # Rounds sweep:
    PYTHONPATH=. venv/bin/python benchmarks/cross_ls/run_cross_ls.py \\
        --experiment rounds --pqrm 1,2,4 --distances 3 5 7 \\
        --rounds-values 3 5 7 --p-values 1e-3
"""
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))  # repo root

from lightstim.protocols.cross_ls import CrossLSExperiment
from benchmarks._config import (
    add_run_options, benchmark_decoder, benchmark_noise, checkpoint_key, checkpoint_rows,
    decoder_metadata, validate_run_options,
)
from lightstim.config.results import append_result_row
from lightstim.config.runner import run_config_cli, sample_full_postselection
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig
from lightstim.simulation.decoder_backend.post_select import get_post_select_detector_indices

# ── Result schema ──────────────────────────────────────────────────────────────

def _ck_key(row: dict) -> tuple:
    fields = ("experiment", "pqrm", "d_surf", "rounds", "state", "p_1q", "p_2q",
              "ps_mode", "mode", "decoder", "backend", "decoder_params", "on_decode_failure",
              "noise_model", "noise_custom_params", "p_idle", "seed", "config_sha256")
    defaults = {"mode": "hybrid", "noise_model": "circuit_level", "noise_custom_params": "{}",
                "p_idle": 0, "seed": 0, "on_decode_failure": "error"}
    if row.get("decoder") not in (None, "none") and not isinstance(row.get("decoder_params"), str):
        defaults["decoder_params"] = decoder_metadata(
            benchmark_decoder(row["decoder"], row.get("backend"))
        )["decoder_params"]
    return checkpoint_key(row, fields, defaults)


def _load_done(path: Path) -> set:
    if not path.exists() or not path.stat().st_size:
        return set()
    return {_ck_key(r) for r in checkpoint_rows(path)}


def _run_experiment(args, out_path: Path, experiment: str) -> None:
    decoder = benchmark_decoder(args.decoder, args.backend, mle_time_limit=args.mle_time_limit,
                                on_decode_failure=args.on_decode_failure)
    run_options = dict(max_shots=args.max_shots, max_errors=args.max_errors,
                       batch_size=args.batch_size, seed=args.seed)
    done = _load_done(out_path)
    pipeline = None
    if args.mode != "full_postselection":
        pipeline = SimulationPipeline(
            decoder_config=decoder, num_workers=args.num_workers, print_progress=False,
            post_select_detector_indices=[] if args.mode == "decode" else None,
            **run_options,
        )
    states = args.states or (["Z", "X", "Y"] if experiment == "sweep" else ["Z"])
    for pqrm, d, state, p in product(args.pqrm_codes, args.distances, states, args.p_values):
        rounds_values = args.rounds_values or ([d] if experiment == "sweep" else [3, 5, 7])
        for rounds in rounds_values:
            idle_rate = args.p_idle
            if idle_rate is None and args.noise_model == "circuit_level":
                idle_rate = 0.0  # Historical CrossLS circuit-level baseline.
            noise = benchmark_noise(p, args.noise_model, args.noise_custom_params,
                                    p_idle=idle_rate, p_1q=args.p_1q)
            row = {
                "sweep_coordinates": {"experiment": experiment, "pqrm": pqrm, "distance": d,
                                      "basis": state, "p": p, "rounds": rounds},
                "experiment": experiment, "pqrm": "-".join(map(str, pqrm)), "d_surf": d,
                "rounds": rounds, "state": state, "p_1q": noise.p_1q, "p_2q": p,
                "p_idle": noise.p_idle, "p_meas": noise.p_meas, "p_reset": noise.p_reset,
                "noise_model": args.noise_model, "noise_custom_params": json.dumps(args.noise_custom_params, sort_keys=True),
                "seed": args.seed, "mode": args.mode, "ps_mode": args.ps_mode,
                "postselection_region": args.ps_mode if args.mode == "hybrid" else None,
                **decoder_metadata(decoder), "decoder": decoder.name,
            }
            if pipeline is None:
                row.update(decoder="none", decoder_name="none", backend=None, decoder_params="{}")
            if _ck_key(row) in done:
                print(f"  SKIP PQRM{tuple(pqrm)} d={d} rounds={rounds} {state} p={p:.2g}")
                continue
            exp = CrossLSExperiment(
                PQRM_para=pqrm, d_surf=d, rounds=rounds, PQRM_state=state, surf_state="X",
                noise_params=noise, noise_model=args.noise_model, if_detector=True,
                post_select_hybrid=args.mode == "hybrid" and args.ps_mode == "hybrid",
            )
            circuit = exp.build()
            row["n_det"] = circuit.num_detectors
            if pipeline is None:
                row["n_ps"] = circuit.num_detectors
            elif args.mode == "hybrid":
                row["n_ps"] = len(get_post_select_detector_indices(circuit))
            else:
                row["n_ps"] = 0
            start = time.perf_counter()
            if pipeline is None:
                result = sample_full_postselection(circuit, num_workers=args.num_workers, **run_options)
                row.update(shots=result["shots"], kept=result["accepted"], errors=result["errors"],
                           ps_rate=result["acceptance"], ler=result["logical_error_rate"])
            else:
                stats = pipeline.run(circuit)
                row.update(shots=stats.shots, kept=stats.post_selected_shots, errors=stats.errors,
                           ps_rate=stats.post_selection_rate, ler=stats.logical_error_rate)
            row["seconds"] = time.perf_counter() - start
            append_result_row(out_path, row)
            done.add(_ck_key(row))
            print(f"  PQRM{tuple(pqrm)} d={d} rounds={rounds} {state} p={p:.2g} "
                  f"LER={row['ler']:.2e} ({row['errors']}/{row['kept']:,})")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_pqrm(s: str):
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--experiment", nargs="+",
                    choices=["sweep", "rounds", "all"],
                    default=["all"])
    ap.add_argument("--pqrm", nargs="+", default=["1,2,4"],
                    help="PQRM codes, e.g. 1,2,4 1,3,5 1,4,6 (default: 1,2,4)")
    ap.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    ap.add_argument("--p-values", nargs="+", type=float,
                    default=[5e-4, 1e-3, 2e-3])
    ap.add_argument("--states", "--basis", nargs="+", choices=["Z", "X", "Y"], default=None,
                    help="States to test (default: Z X Y for sweep, Z for rounds)")
    ap.add_argument("--rounds-values", "--rounds", nargs="+", type=int, default=None,
                    help="Rounds list for rounds experiment (default: 3 5 7)")
    ap.add_argument("--mode", choices=["decode", "full_postselection", "hybrid"], default="hybrid")
    ap.add_argument("--ps-mode", "--postselection-region", choices=["hybrid", "pqrm_only"], default=None)
    ap.add_argument("--decoder", default="mwpf",
                    choices=["mwpf", "bposd", "cpu_bposd", "gpu_bposd", "nv-qldpc-decoder", "mle-ilp", "ionq-beam-search"],
                    help="Decoder (default: mwpf). pymatching cannot handle CrossLS hypergraph errors.")
    ap.add_argument("--p-1q", type=float, default=1e-6, help="Historical CrossLS one-qubit error rate")
    ap.add_argument("--p-idle", type=float, default=None,
                    help="Override idle noise (default: 0 for circuit_level, swept p for other models)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: d=[3], 1 p-value, 50k shots, 20 errors")
    add_run_options(ap)
    if run_config_cli(ap):
        return
    args = ap.parse_args()
    validate_run_options(ap, args)
    if args.ps_mode and args.mode != "hybrid":
        ap.error("--postselection-region/--ps-mode requires --mode hybrid")
    args.ps_mode = args.ps_mode or "hybrid"
    if args.decoder == "ionq-beam-search" and args.backend not in (None, "cpu"):
        ap.error("ionq-beam-search requires --backend cpu")

    # Parse PQRM codes
    args.pqrm_codes = [_parse_pqrm(s) for s in args.pqrm]

    if args.quick:
        args.distances  = [3]
        args.p_values   = [1e-3]
        args.max_shots  = 50_000
        args.max_errors = 20
        args.batch_size = 5_000

    experiments = (["sweep", "rounds"] if "all" in args.experiment
                   else args.experiment)

    out_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_ls_results.csv"

    print("=" * 60)
    print("CrossLS Benchmark — Surface Code × PQRM Lattice Surgery")
    pqrm_strs = ["-".join(str(x) for x in p) for p in args.pqrm_codes]
    print(f"PQRM codes  : {pqrm_strs}")
    print(f"Distances   : {args.distances}")
    print(f"p values    : {args.p_values}")
    print(f"PS mode     : {args.ps_mode}")
    print(f"Decoder     : {args.decoder}" + (f" ({args.backend})" if args.backend else ""))
    print(f"max_shots   : {args.max_shots:.0e}")
    print(f"max_errors  : {args.max_errors}")
    print(f"num_workers : {args.num_workers}")
    print(f"Output      : {out_path}")
    print("=" * 60)

    for exp in experiments:
        print(f"\n{'─'*50}")
        print(f"Experiment: {exp}")
        _run_experiment(args, out_path, exp)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    if out_path.exists():
        import pandas as pd
        df = pd.read_csv(out_path)
        print(f"  {out_path}  ({len(df)} rows)")
    print("=" * 60)


if __name__ == "__main__":
    main()
