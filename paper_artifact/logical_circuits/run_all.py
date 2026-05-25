"""
Reproduce all three logical-circuits paper figures from precomputed data.

Figures:
    results/bell_tele.png   — Bell-state teleportation LER vs PER (TG / LS-ZZ / LS-XX)
    results/routing.png     — LER vs. Routing distance (ZZ-LS and XX-LS, d=7, p=1e-3)
    results/distill.png     — Distillation P_out vs P_in (TG + LS 7-to-1, injection-only)

Usage (from repo root):
    venv/bin/python paper_artifact/logical_circuits/run_all.py

To regenerate the precomputed data, run the benchmark scripts in
benchmarks/logical_circuits/. See README.md for full instructions.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = [
    HERE / "plot_bell_tele.py",
    HERE / "plot_routing.py",
    HERE / "plot_distill.py",
]


def main():
    print("=" * 60)
    print("Logical Circuits — Paper Figure Reproduction")
    print("=" * 60)

    all_ok = True
    for script in SCRIPTS:
        print(f"\n--- {script.name} ---")
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"ERROR: {script.name} exited with code {result.returncode}")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("All figures reproduced successfully.")
        print(f"Results in: {HERE / 'results'}/")
        for f in sorted((HERE / "results").glob("*.png")):
            print(f"  {f.name}")
    else:
        print("Some scripts failed. Check output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
