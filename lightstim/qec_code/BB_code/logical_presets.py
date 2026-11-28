"""Precomputed (f, g, h, alpha, beta) for scalable BB code logical operators.

Keys: (l, m, tuple(A), tuple(B)) with A,B as tuples of tuples for hashability.

The named ``tour_de_gross`` preset selects the paper's logical basis and
cannot be combined with explicit polynomial logicals. Without a named preset,
BBCode._build_logical_operators() uses this order:
  1. Explicit f,g,h,alpha,beta passed to BBCode() constructor
  2. Preset lookup here (polynomial method, O(k), minimum-weight operators)
  3. Numerical fallback for l*m <= 200 (GF(2) RREF, O((lm)^3), higher-weight)
  4. ValueError

Sources:
  [[144,12,12]]: Bravyi et al. arXiv:2308.07915 Table 7 (minimum-weight choice).
                 Verified: polynomial gives all weight-12 operators vs
                 numerical which gives mixed weights up to 38.
                 Values extracted from processing/memoryexperi.ipynb.
"""

from typing import Optional, Dict, Any, List


def _key(l: int, m: int, A: List[List[int]], B: List[List[int]]) -> tuple:
    """Build hashable key for preset lookup."""
    return (l, m, tuple(tuple(row) for row in A), tuple(tuple(row) for row in B))


_A_STANDARD = [[3, 0], [0, 1], [0, 2]]
_B_STANDARD = [[0, 3], [1, 0], [2, 0]]

BB_LOGICAL_PRESETS: Dict[tuple, Dict[str, Any]] = {

    # [[144, 12, 12]]  l=12, m=6
    # A = x^3 + y + y^2,  B = y^3 + x + x^2
    # f, g, h from Table 7 of Bravyi et al. 2024 (arXiv:2308.07915)
    # All 24 logical operators have weight 12 (minimum weight).
    _key(12, 6, _A_STANDARD, _B_STANDARD): {
        "f": [[0,0],[1,0],[2,0],[3,0],[6,0],[7,0],[8,0],[9,0],[1,3],[5,3],[7,3],[11,3]],
        "g": [[1,0],[2,1],[0,2],[1,2],[2,3],[0,4]],
        "h": [[0,0],[0,1],[1,1],[0,2],[0,3],[1,3]],
        "alpha": [[0,0],[0,1],[2,1],[2,5],[3,2],[4,0]],
        "beta":  [[0,1],[0,5],[1,1],[0,0],[4,0],[5,2]],
    },

}


def get_preset(
    l: int,
    m: int,
    A: List[List[int]],
    B: List[List[int]],
) -> Optional[Dict[str, Any]]:
    """Return preset dict for (l,m,A,B) if available, else None."""
    key = _key(l, m, A, B)
    return BB_LOGICAL_PRESETS.get(key)


# Tour de gross, arXiv:2506.03094, Eqs. (27), (32), (34), (35).
# This basis is intentionally named: it is not the older memory-code basis
# above, and the paper's logical labels 1 and 7 refer to this ordering.
TOUR_DE_GROSS_CONFIG = {
    "l": 12, "m": 6, "d": 12,
    "A": [[0, 0], [0, 1], [3, -1]],
    "B": [[0, 0], [1, 0], [-1, -3]],
    "logical_preset": "tour_de_gross",
}
TOUR_DE_GROSS_LOGICALS = {
    "p": [[4, 0], [5, 0], [6, 1], [4, 2], [5, 4], [6, 5]],
    "q": [[3, 0], [4, 0], [3, 1], [3, 2], [4, 2], [3, 5]],
    "r": [[0, 0], [8, 0], [1, 1], [9, 1], [3, 4], [11, 4]],
    "s": [[1, 0], [9, 0], [4, 4], [8, 4], [0, 5], [8, 5]],
    "alpha": [[0, 0], [3, 5], [11, 5], [10, 1], [5, 4], [4, 2]],
    "beta": [[0, 0], [2, 4], [1, 2], [2, 5], [1, 1], [3, 1]],
    "mu": [1, 1], "nu": [1, 1],
}
