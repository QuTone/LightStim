"""
SE Scheduling Comparison: perpendicular vs swapped.

Sweeps:
    scheduling : perpendicular, swapped
    basis      : Z, X
    distance   : 3, 5, 7, 9
    p          : 7e-3, 5e-3, 2e-3, 1e-3, 7e-4, 5e-4

Decoder : PyMatching (CPU)
Output  : eval/memory_benchmark/results/se_scheduling.csv

Usage:
    # small distances (d=3,5)
    venv/bin/python eval/memory_benchmark/run_se_scheduling.py --distances 3,5 --num-workers 30

    # large distances (d=7,9)
    venv/bin/python eval/memory_benchmark/run_se_scheduling.py --distances 7,9 --num-workers 16
"""

import argparse
import contextlib
import io
from itertools import product
from pathlib import Path

import pandas as pd

from src.ir.qec_system import QECSystem
from src.noise.config import NoiseConfig
from src.qec_code.surface_code.rotated import (
    RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock,
)
from experiments.memory import MemoryExperiment
from src.simulation.decoder_backend import (
    SimulationPipeline, ExperimentTask, DecoderConfig,
)

OUTPUT_PATH = Path(__file__).resolve().parent / "results" / "se_scheduling.csv"

SCHEDULINGS = ["perpendicular", "swapped"]
BASES       = ["Z", "X"]
DISTANCES   = [3, 5, 7, 9]
P_VALUES    = [7e-3, 5e-3, 2e-3, 1e-3, 7e-4, 5e-4]


# ── checkpoint ────────────────────────────────────────────────────────

CHECKPOINT_KEYS = ["scheduling", "basis", "distance", "p"]

def _ck_key(d: dict) -> tuple:
    return tuple(
        f"{d[k]:.6e}" if isinstance(d[k], float) else str(d[k])
        for k in CHECKPOINT_KEYS
    )

def load_done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return {_ck_key(r) for r in df.to_dict("records")}

def append_row(path: Path, row: dict):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


# ── circuit builder ───────────────────────────────────────────────────

def build_circuit(scheduling: str, basis: str, d: int, p: float):
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=d), name="patch")
    noise = NoiseConfig(p_idle=p, p_1q=p, p_2q=p, p_meas=p, p_reset=p)
    with contextlib.redirect_stdout(io.StringIO()):
        exp = MemoryExperiment(
            qec_system=system,
            extraction_block_class=RotatedSurfaceCodeExtractionBlock,
            rounds=d,
            noise_params=noise,
            noise_model="circuit_level",
            basis=basis,
            se_block_kwargs={"scheduling": scheduling},
        )
        circuit = exp.build()
    return circuit


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", type=str, default="3,5,7,9",
                        help="Comma-separated distances to run (default: 3,5,7,9)")
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--max-shots", type=int, default=1_000_000_000)
    args = parser.parse_args()

    distances = [int(x) for x in args.distances.split(",")]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done_keys = load_done_keys(OUTPUT_PATH)

    decoder = DecoderConfig("pymatching", backend="cpu")

    # Build all tasks for requested distances
    all_tasks = list(product(SCHEDULINGS, BASES, distances, P_VALUES))
    pending = [t for t in all_tasks if _ck_key({"scheduling": t[0], "basis": t[1], "distance": t[2], "p": t[3]}) not in done_keys]

    print(f"{'='*60}")
    print(f"SE Scheduling Benchmark")
    print(f"distances  : {distances}")
    print(f"num_workers: {args.num_workers}")
    print(f"max_errors : {args.max_errors}")
    print(f"max_shots  : {args.max_shots:.0e}")
    print(f"total tasks: {len(all_tasks)}  pending: {len(pending)}")
    print(f"output     : {OUTPUT_PATH}")
    print(f"{'='*60}\n")

    pipeline = SimulationPipeline(
        decoder_config=decoder,
        max_errors=args.max_errors,
        max_shots=args.max_shots,
        num_workers=args.num_workers,
        print_progress=True,
    )

    for i, (scheduling, basis, d, p) in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] scheduling={scheduling}  basis={basis}  d={d}  p={p:.1e}", flush=True)
        circuit = build_circuit(scheduling, basis, d, p)
        meta = {"scheduling": scheduling, "basis": basis, "distance": d, "p": p}
        stats = pipeline.run(circuit, meta)

        row = {
            **meta,
            "shots": stats.shots,
            "errors": stats.errors,
            "logical_error_rate": stats.logical_error_rate,
            "seconds": stats.seconds,
            "decoder": stats.decoder,
        }
        append_row(OUTPUT_PATH, row)
        print(f"  → LER={stats.logical_error_rate:.3e}  errors={stats.errors}  shots={stats.shots:,}  t={stats.seconds:.1f}s", flush=True)

    print(f"\n{'='*60}")
    print(f"DONE — results in {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
