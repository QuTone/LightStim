"""
Export a LightStim rotated-surface-code memory experiment to Microsoft's DEQ
device DSL.

Usage
-----
    venv/bin/python benchmarks/deq/run_deq_export.py \\
        --distance 3 --rounds 3 --basis Z --out generated/rotated_surface_code_d3.deq

    # Also parse the output with the real `deq` package and validate its
    # CODE block(s) (requires the optional `deq` dependency: pip install
    # -e '.[deq]', or plain `pip install deq`):
    venv/bin/python benchmarks/deq/run_deq_export.py \\
        --distance 5 --rounds 5 --basis X --out generated/d5.deq --validate
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from lightstim.deq.export import export_rotated_surface_code_memory
from lightstim.deq.validate import deq_available, parse_and_validate_codes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--distance", type=int, required=True, help="Code distance (odd, >= 3).")
    ap.add_argument("--rounds", type=int, required=True,
                     help="Total syndrome-extraction rounds, matching MemoryExperiment(rounds=...).")
    ap.add_argument("--basis", choices=["X", "Z"], default="Z", help="Memory basis.")
    ap.add_argument("--out", required=True, help="Output .deq file path.")
    ap.add_argument("--validate", action="store_true",
                     help="Parse the output with the real `deq` package and validate its CODE block(s).")
    args = ap.parse_args()

    deq_text = export_rotated_surface_code_memory(distance=args.distance, rounds=args.rounds, basis=args.basis)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(deq_text)
    print(f"Wrote {out_path} ({len(deq_text.splitlines())} lines).")

    if args.validate:
        if not deq_available():
            print("--validate requested but the `deq` package is not installed "
                  "(pip install deq, or pip install -e '.[deq]').", file=sys.stderr)
            sys.exit(1)
        parse_and_validate_codes(deq_text)
        print("Parsed and validated OK.")


if __name__ == "__main__":
    main()
