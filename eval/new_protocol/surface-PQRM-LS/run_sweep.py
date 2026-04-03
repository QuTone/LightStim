#!/usr/bin/env python3
"""
CrossLS sweep: PQRM × d_surf × state × p × decoder.

Usage examples:
  python eval/new_protocol/surface-PQRM-LS/run_sweep.py \\
    --pqrm 1,2,4 --decoder mwpf --workers 32 \\
    --p 2e-3 --max-shots 2000000 --max-errors 200 \\
    --output eval/new_protocol/surface-PQRM-LS/results/sweep_mwpf_p2e-3_pqrm124.csv

  python eval/new_protocol/surface-PQRM-LS/run_sweep.py \\
    --pqrm 1,2,4 1,3,5 --decoder bposd --backend gpu --gpu-id 0 \\
    --p 2e-3 --max-shots 2000000 --max-errors 200 \\
    --output eval/new_protocol/surface-PQRM-LS/results/sweep_gpu_p2e-3.csv

Supports resume: existing rows in the output CSV are skipped.
"""
import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_ls import CrossLSExperiment
from src.noise.config import NoiseConfig
from src.simulation.decoder_backend import SimulationPipeline, DecoderConfig
from src.simulation.decoder_backend.post_select import get_post_select_detector_indices

D_SURF_LIST = [3, 4, 5, 6, 7]
STATE_LIST = ["Z", "X", "Y"]
CSV_HEADER = [
    "timestamp", "pqrm", "d_surf", "rounds", "state",
    "p_1q", "p_2q", "p_meas", "p_reset",
    "decoder", "backend", "workers",
    "n_det", "n_ps", "max_shots", "max_errors",
    "shots", "kept", "ps_rate", "errors", "ler", "total_seconds",
]


def parse_args():
    parser = argparse.ArgumentParser(description="CrossLS full sweep")
    parser.add_argument("--pqrm", nargs="+", required=True,
                        help="PQRM params, e.g. 1,2,4 1,3,5")
    parser.add_argument("--decoder", required=True, choices=["mwpf", "bposd"])
    parser.add_argument("--backend", default=None, help="cpu or gpu (default: cpu for mwpf, gpu for bposd)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="GPU device ID (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--p", type=float, required=True, help="Physical error rate (p_2q=p_meas=p_reset)")
    parser.add_argument("--p-1q", type=float, default=1e-6, help="Single-qubit error rate")
    parser.add_argument("--max-shots", type=int, default=2_000_000)
    parser.add_argument("--max-errors", type=int, default=200)
    parser.add_argument("--ps-mode", default="hybrid", choices=["hybrid", "pqrm_only"],
                        help="Post-selection mode (default: hybrid)")
    parser.add_argument("--canonical", action="store_true", help="Use canonical PQRM logical")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()


def load_completed(csv_path):
    """Return set of (pqrm, d_surf, state) already in the CSV."""
    done = set()
    if not Path(csv_path).exists():
        return done
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["pqrm"], int(row["d_surf"]), row["state"]))
    return done


def run_sweep(args):
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    backend = args.backend
    if backend is None:
        backend = "gpu" if args.decoder == "bposd" else "cpu"

    pqrm_codes = []
    for s in args.pqrm:
        pqrm_codes.append([int(x) for x in s.split(",")])

    noise = NoiseConfig(p_1q=args.p_1q, p_2q=args.p, p_meas=args.p, p_reset=args.p)

    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig(args.decoder, backend=backend),
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        num_workers=args.workers,
        print_progress=False,
    )

    done = load_completed(args.output)
    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    pqrm_strs = ["-".join(str(x) for x in p) for p in pqrm_codes]
    total_points = len(pqrm_codes) * len(D_SURF_LIST) * len(STATE_LIST)
    skipped = sum(1 for p in pqrm_codes for d in D_SURF_LIST for s in STATE_LIST
                  if ("-".join(str(x) for x in p), d, s) in done)
    print(f"CrossLS sweep: PQRM={pqrm_strs}, p={args.p}, decoder={args.decoder}({backend}), "
          f"workers={args.workers}")
    print(f"max_shots={args.max_shots:,}, max_errors={args.max_errors}")
    print(f"Total: {total_points} points, skipping {skipped} already done, "
          f"running {total_points - skipped}")
    print(f"{'PQRM':<10} {'d':>3} {'r':>3} {'State':<6} {'#det':>5} {'#PS':>4} "
          f"{'Shots':>10} {'Kept':>10} {'PS%':>6} {'Err':>5} {'LER':>12} {'Time':>8}")
    print("-" * 95)
    sys.stdout.flush()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
            f.flush()

        for pqrm_para in pqrm_codes:
            pqrm_str = "-".join(str(x) for x in pqrm_para)
            for d in D_SURF_LIST:
                for state in STATE_LIST:
                    if (pqrm_str, d, state) in done:
                        continue

                    try:
                        ps_kwargs = {}
                        if args.ps_mode == "hybrid":
                            ps_kwargs["post_select_hybrid"] = True

                        exp = CrossLSExperiment(
                            PQRM_para=pqrm_para, d_surf=d, rounds=d,
                            PQRM_state=state, surf_state="X",
                            noise_params=noise, if_detector=True,
                            canonical_pqrm_logical=args.canonical,
                            **ps_kwargs,
                        )
                        circ = exp.build()
                        n_det = circ.num_detectors
                        n_ps = len(get_post_select_detector_indices(circ))

                        stats = pipeline.run(circuit=circ)
                        ler = stats.logical_error_rate
                        ps_rate = stats.post_selection_rate

                        row = [
                            datetime.now().isoformat(timespec="seconds"),
                            pqrm_str, d, d, state,
                            f"{args.p_1q:.0e}", f"{args.p:.0e}", f"{args.p:.0e}", f"{args.p:.0e}",
                            args.decoder, backend, args.workers,
                            n_det, n_ps, args.max_shots, args.max_errors,
                            stats.shots, stats.post_selected_shots,
                            f"{ps_rate:.4f}", stats.errors, f"{ler:.2e}", f"{stats.seconds:.1f}",
                        ]
                        writer.writerow(row)
                        f.flush()

                        print(f"{pqrm_str:<10} {d:>3} {d:>3} {state:<6} {n_det:>5} {n_ps:>4} "
                              f"{stats.shots:>10,} {stats.post_selected_shots:>10,} "
                              f"{ps_rate:>5.1%} {stats.errors:>5} {ler:>12.2e} {stats.seconds:>7.1f}s")
                        sys.stdout.flush()

                    except Exception as e:
                        print(f"{pqrm_str:<10} {d:>3} {d:>3} {state:<6} ERROR: {e}")
                        sys.stdout.flush()

    print("\nDone.")


if __name__ == "__main__":
    run_sweep(parse_args())
