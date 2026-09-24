"""
Verify an exported rotated-surface-code memory experiment against Microsoft's
real DEQ compiler and runtime (requires the optional `deq_runtime` package,
in addition to `deq` itself: pip install -e '.[deq]').

This is the real end-to-end tier: it shells out to `python -m deq transpile`
to compile the exported PROGRAM, then `python -m deq sample --noiseless` to
run it through DEQ's own simulator. A correctly-exported, noiseless memory
experiment must give an all-zero syndrome and an all-zero logical readout on
every shot; this script asserts exactly that.

Usage
-----
    venv/bin/python benchmarks/deq/run_deq_verify.py --distance 3 --rounds 3 --basis Z --shots 200

Noisy logical-error-rate comparison against LightStim's own
`lightstim.simulation.decoder_backend.pipeline.SimulationPipeline` (the
natural next step, mirroring the reference surface-code-deq library's own
`surface-code-deq-ler`) needs DEQ's noise-injection wiring worked out first
and is intentionally left as follow-up work, not attempted here.
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from lightstim.deq.export import export_rotated_surface_code_memory


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--distance", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--basis", choices=["X", "Z"], default="Z")
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--keep", help="Directory to write the .deq/.deq.jit files into (default: a temp dir).")
    args = ap.parse_args()

    if importlib.util.find_spec("deq_runtime") is None:
        print("deq_runtime is not installed (pip install deq_runtime, or pip install -e '.[deq]'). "
              "Only the CODE-validation tier (run_deq_export.py --validate) is available without it.",
              file=sys.stderr)
        sys.exit(1)

    import tempfile

    work_dir = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="lightstim_deq_verify_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    deq_text = export_rotated_surface_code_memory(distance=args.distance, rounds=args.rounds, basis=args.basis)
    program_name = f"RotatedSurfaceCodeD{args.distance}MemoryExperiment{args.basis}{args.rounds}"
    deq_path = work_dir / "device.deq"
    jit_path = work_dir / "device.deq.jit"
    deq_path.write_text(deq_text)

    print(f"Transpiling {program_name}...")
    subprocess.run(
        [sys.executable, "-m", "deq", "transpile", str(deq_path), "--program", program_name, "--out", str(jit_path)],
        check=True,
    )

    print(f"Sampling {args.shots} noiseless shots...")
    result = subprocess.run(
        [sys.executable, "-m", "deq", "sample", str(deq_path), "--program", program_name,
         "--shots", str(args.shots), "--noiseless", "--interpret"],
        check=True, capture_output=True, text=True,
    )
    syndromes = [line for line in result.stdout.splitlines() if line.startswith("Syndrome:")]
    readouts = [line for line in result.stdout.splitlines() if line.startswith("Readout:")]

    nonzero_syndromes = sum(1 for line in syndromes if int(line.split()[-1], 2) != 0)
    nonzero_readouts = sum(1 for line in readouts if int(line.split()[-1], 2) != 0)

    print(f"{len(syndromes)} shots sampled: {nonzero_syndromes} nonzero syndromes, {nonzero_readouts} nonzero readouts.")
    if nonzero_syndromes or nonzero_readouts:
        print("FAIL: a noiseless memory experiment must never trigger a syndrome or a logical flip.", file=sys.stderr)
        sys.exit(1)
    print(f"PASS: DEQ's own compiler and runtime reproduce a working distance-{args.distance} "
          f"{args.basis}-basis memory experiment, {args.rounds} rounds, {work_dir}")


if __name__ == "__main__":
    main()
