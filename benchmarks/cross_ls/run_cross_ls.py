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
import csv
import sys
import time
from itertools import product
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))  # repo root

from lightstim.protocols.cross_ls import CrossLSExperiment
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig
from lightstim.simulation.decoder_backend.post_select import get_post_select_detector_indices

# ── Result schema ──────────────────────────────────────────────────────────────

_COLS = [
    "experiment", "pqrm", "d_surf", "rounds", "state",
    "p_1q", "p_2q", "ps_mode", "decoder", "backend",
    "n_det", "n_ps",
    "shots", "kept", "ps_rate", "errors", "ler", "seconds",
]
_KEY_COLS = frozenset({"shots", "kept", "ps_rate", "errors", "ler", "seconds"})


# ── Checkpointing ──────────────────────────────────────────────────────────────

def _ck_key(row: dict) -> tuple:
    return tuple(
        f"{v:.6e}" if isinstance(v, float) else str(v)
        for k, v in sorted(row.items()) if k not in _KEY_COLS
    )


def _load_done(path: Path) -> set:
    if not path.exists():
        return set()
    import pandas as pd
    df = pd.read_csv(path)
    return {_ck_key(r) for r in df.to_dict("records")}


def _append_row(path: Path, row: dict) -> None:
    header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        if header:
            w.writeheader()
        w.writerow(row)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _decoder_config(name: str, backend: str) -> DecoderConfig:
    if name == "pymatching":
        return DecoderConfig("pymatching", backend="cpu")
    if name == "mwpf":
        return DecoderConfig("mwpf", backend="cpu",
                             params={"cluster_node_limit": 50})
    if name == "bposd":
        return DecoderConfig("bposd", backend=backend, params={
            "max_iterations": 1000, "osd_order": 10,
            "bp_method": "min_sum", "ms_scaling_factor": 0,
            "osd_method": "osd_cs",
        })
    raise ValueError(f"Unknown decoder: {name!r}")


def _build_and_run(pqrm_para, d_surf, rounds, state, p, p_1q,
                   ps_mode, pipeline, experiment_tag, decoder_name, backend):
    ps_kwargs = {"post_select_hybrid": True} if ps_mode == "hybrid" else {}
    noise = NoiseConfig(p_1q=p_1q, p_2q=p, p_meas=p, p_reset=p)
    exp = CrossLSExperiment(
        PQRM_para=pqrm_para, d_surf=d_surf, rounds=rounds,
        PQRM_state=state, surf_state="X",
        noise_params=noise, if_detector=True,
        **ps_kwargs,
    )

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        circuit = exp.build()

    n_det = circuit.num_detectors
    n_ps  = len(get_post_select_detector_indices(circuit))

    t0    = time.perf_counter()
    stats = pipeline.run(circuit)
    elapsed = time.perf_counter() - t0

    pqrm_str = "-".join(str(x) for x in pqrm_para)
    return {
        "experiment": experiment_tag,
        "pqrm": pqrm_str,
        "d_surf": d_surf,
        "rounds": rounds,
        "state": state,
        "p_1q": p_1q,
        "p_2q": p,
        "ps_mode": ps_mode,
        "decoder": decoder_name,
        "backend": backend,
        "n_det": n_det,
        "n_ps": n_ps,
        "shots": stats.shots,
        "kept": stats.post_selected_shots,
        "ps_rate": round(stats.post_selection_rate, 4),
        "errors": stats.errors,
        "ler": stats.logical_error_rate,
        "seconds": round(elapsed, 2),
    }


# ── Experiment: sweep ──────────────────────────────────────────────────────────

def _run_sweep(args, out_path: Path) -> None:
    done     = _load_done(out_path)
    pipeline = SimulationPipeline(
        decoder_config=_decoder_config(args.decoder, args.backend),
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        print_progress=False,
    )

    states = args.states or ["Z", "X", "Y"]

    for pqrm_para, d, state, p in product(
        args.pqrm_codes, args.distances, states, args.p_values
    ):
        pqrm_str = "-".join(str(x) for x in pqrm_para)
        proto_row = {
            "experiment": "sweep", "pqrm": pqrm_str, "d_surf": d,
            "rounds": d, "state": state, "p_1q": args.p_1q, "p_2q": p,
            "ps_mode": args.ps_mode, "decoder": args.decoder, "backend": args.backend,
        }
        if _ck_key(proto_row) in done:
            print(f"  SKIP PQRM{tuple(pqrm_para)} d={d} state={state} p={p:.0e}")
            continue

        print(f"  PQRM{tuple(pqrm_para)} d={d} state={state} p={p:.0e} ...", end=" ", flush=True)
        row = _build_and_run(pqrm_para, d, d, state, p, args.p_1q,
                             args.ps_mode, pipeline, "sweep", args.decoder, args.backend)
        _append_row(out_path, row)
        print(f"LER={row['ler']:.2e}  PS={row['ps_rate']:.2f}  "
              f"({row['errors']}/{row['kept']:,})  {row['seconds']:.1f}s")


# ── Experiment: rounds sweep ───────────────────────────────────────────────────

def _run_rounds(args, out_path: Path) -> None:
    done     = _load_done(out_path)
    pipeline = SimulationPipeline(
        decoder_config=_decoder_config(args.decoder, args.backend),
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        print_progress=False,
    )

    states       = args.states or ["Z"]
    rounds_list  = args.rounds_values or [3, 5, 7]

    for pqrm_para, d, state, r, p in product(
        args.pqrm_codes, args.distances, states, rounds_list, args.p_values
    ):
        pqrm_str = "-".join(str(x) for x in pqrm_para)
        proto_row = {
            "experiment": "rounds", "pqrm": pqrm_str, "d_surf": d,
            "rounds": r, "state": state, "p_1q": args.p_1q, "p_2q": p,
            "ps_mode": args.ps_mode, "decoder": args.decoder, "backend": args.backend,
        }
        if _ck_key(proto_row) in done:
            print(f"  SKIP PQRM{tuple(pqrm_para)} d={d} r={r} state={state} p={p:.0e}")
            continue

        print(f"  PQRM{tuple(pqrm_para)} d={d} r={r} state={state} p={p:.0e} ...", end=" ", flush=True)
        row = _build_and_run(pqrm_para, d, r, state, p, args.p_1q,
                             args.ps_mode, pipeline, "rounds", args.decoder, args.backend)
        _append_row(out_path, row)
        print(f"LER={row['ler']:.2e}  PS={row['ps_rate']:.2f}  "
              f"({row['errors']}/{row['kept']:,})  {row['seconds']:.1f}s")


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
    ap.add_argument("--states", nargs="+", choices=["Z", "X", "Y"], default=None,
                    help="States to test (default: Z X Y for sweep, Z for rounds)")
    ap.add_argument("--rounds-values", nargs="+", type=int, default=None,
                    help="Rounds list for rounds experiment (default: 3 5 7)")
    ap.add_argument("--ps-mode", choices=["hybrid", "pqrm_only"], default="hybrid")
    ap.add_argument("--decoder", default="mwpf",
                    choices=["mwpf", "bposd"],
                    help="Decoder (default: mwpf). pymatching cannot handle CrossLS hypergraph errors.")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--p-1q", type=float, default=1e-6)
    ap.add_argument("--max-shots", type=int, default=2_000_000)
    ap.add_argument("--max-errors", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: d=[3], 1 p-value, 50k shots, 20 errors")
    args = ap.parse_args()

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
    print(f"Decoder     : {args.decoder} ({args.backend})")
    print(f"max_shots   : {args.max_shots:.0e}")
    print(f"max_errors  : {args.max_errors}")
    print(f"num_workers : {args.num_workers}")
    print(f"Output      : {out_path}")
    print("=" * 60)

    for exp in experiments:
        print(f"\n{'─'*50}")
        print(f"Experiment: {exp}")
        if exp == "sweep":
            _run_sweep(args, out_path)
        elif exp == "rounds":
            _run_rounds(args, out_path)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    if out_path.exists():
        import pandas as pd
        df = pd.read_csv(out_path)
        print(f"  {out_path}  ({len(df)} rows)")
    print("=" * 60)


if __name__ == "__main__":
    main()
