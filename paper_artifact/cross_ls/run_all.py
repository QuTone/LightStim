"""
Reproduce CrossLS paper figures from precomputed data.

Figures:
    results/fig1_ler_vs_p.png  — LER vs PER (|Z⟩, d=3/5/7, 3 PQRM codes)
    results/fig2_ler_vs_d.png  — LER vs d (p=5e-4, all states × 3 PQRM codes)

Usage (from repo root):
    venv/bin/python paper_artifact/cross_ls/run_all.py

To regenerate precomputed data, run:
    PYTHONPATH=. venv/bin/python benchmarks/cross_ls/run_cross_ls.py --experiment sweep
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = [HERE / "plot_cross_ls.py"]


def main():
    print("=" * 60)
    print("CrossLS (Surface–PQRM Lattice Surgery) — Paper Figures")
    print("=" * 60)

    all_ok = True
    for script in SCRIPTS:
        print(f"\n--- {script.name} ---")
        result = subprocess.run([sys.executable, str(script)], capture_output=False)
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
