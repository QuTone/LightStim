"""Precomputed (f, g, h, alpha, beta) for scalable BB code logical operators.

Keys: (l, m, tuple(A), tuple(B)) with A,B as tuples of tuples for hashability.

For l*m <= 72, the numerical fallback works. Add presets here for larger codes
to avoid O((lm)^3) kernel computation. Run scripts/derive_bb_logicals.py to
attempt automatic derivation from numerical output.
"""

from typing import Optional, Dict, Any, List


def _key(l: int, m: int, A: List[List[int]], B: List[List[int]]) -> tuple:
    """Build hashable key for preset lookup."""
    return (l, m, tuple(tuple(row) for row in A), tuple(tuple(row) for row in B))


# Preset table: key -> {f, g, h, alpha, beta}
# [[72,12,6]] uses numerical fallback (l*m=36 <= 72).
# Add entries for larger codes when f,g,h,alpha,beta are known.
BB_LOGICAL_PRESETS: Dict[tuple, Dict[str, Any]] = {}


def get_preset(
    l: int,
    m: int,
    A: List[List[int]],
    B: List[List[int]],
) -> Optional[Dict[str, Any]]:
    """Return preset dict for (l,m,A,B) if available, else None."""
    key = _key(l, m, A, B)
    return BB_LOGICAL_PRESETS.get(key)
