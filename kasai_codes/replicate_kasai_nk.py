#!/usr/bin/env python3
"""Replicate the published Kasai-code n,k values.

Run from the repository root:

    python kasai_codes/replicate_kasai_nk.py
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightstim.qec_code.kasai_code import KASAI_CODE_PRESETS, KasaiCode  # noqa: E402


def main() -> int:
    names = ["chen_p96", "chen_p192", "chen_p384", "kasai_p768"]

    header = (
        f"{'preset':<12} {'P':>5} {'n':>7} {'rank_x':>7} {'rank_z':>7} "
        f"{'k':>7} {'expected':>9} {'required commute':>16} {'status':>8}"
    )
    print(header)
    print("-" * len(header))

    ok = True
    for name in names:
        code = KasaiCode.from_preset(name)
        preset = KASAI_CODE_PRESETS[name]

        n = code.n_data
        k = code.num_logicals
        expected_n = preset["expected_n"]
        expected_k = preset["expected_k"]
        commute_ok = code.validate_required_commutativity()
        row_ok = n == expected_n and k == expected_k and commute_ok
        ok = ok and row_ok

        print(
            f"{name:<12} {code.P:>5} {n:>7} {code.rank_x:>7} {code.rank_z:>7} "
            f"{k:>7} {expected_n},{expected_k:>4} {str(commute_ok):>16} "
            f"{'ok' if row_ok else 'FAIL':>8}"
        )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
