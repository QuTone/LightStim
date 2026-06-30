"""Published Kasai-code affine permutation parameters.

Each affine permutation is represented as ``(a, b)`` for
``x -> a*x + b (mod P)``.
"""

from typing import Any, Dict, Optional


KASAI_CODE_PRESETS: Dict[str, Dict[str, Any]] = {
    # Kasai, "Breaking the Orthogonality Barrier in Quantum LDPC Codes",
    # arXiv:2601.08824, Table 1.
    "kasai_p768": {
        "P": 768,
        "J": 3,
        "L": 12,
        "expected_n": 9216,
        "expected_k": 4612,
        "f": [
            (763, 435),
            (679, 69),
            (397, 330),
            (61, 18),
            (697, 612),
            (373, 246),
        ],
        "g": [
            (289, 496),
            (257, 640),
            (625, 200),
            (41, 524),
            (193, 672),
            (449, 672),
        ],
    },

    # Zhao et al., "Towards Ultra-High-Rate Quantum Error Correction with
    # Reconfigurable Atom Arrays", arXiv:2604.16209, Table A1.
    "chen_p96": {
        "P": 96,
        "J": 3,
        "L": 12,
        "expected_n": 1152,
        "expected_k": 580,
        "f": [
            (5, 41),
            (85, 77),
            (73, 66),
            (1, 0),
            (1, 72),
            (37, 9),
        ],
        "g": [
            (61, 15),
            (1, 24),
            (89, 62),
            (25, 22),
            (85, 93),
            (25, 78),
        ],
    },
    "chen_p192": {
        "P": 192,
        "J": 3,
        "L": 12,
        "expected_n": 2304,
        "expected_k": 1156,
        "f": [
            (71, 127),
            (97, 80),
            (67, 117),
            (163, 165),
            (25, 60),
            (187, 33),
        ],
        "g": [
            (163, 165),
            (55, 183),
            (167, 79),
            (139, 41),
            (109, 78),
            (31, 27),
        ],
    },
    "chen_p384": {
        "P": 384,
        "J": 3,
        "L": 12,
        "expected_n": 4608,
        "expected_k": 2308,
        "f": [
            (233, 28),
            (373, 174),
            (25, 164),
            (145, 152),
            (73, 364),
            (1, 64),
        ],
        "g": [
            (187, 183),
            (325, 198),
            (121, 228),
            (151, 65),
            (103, 153),
            (61, 186),
        ],
    },
}


def get_preset(name: str) -> Optional[Dict[str, Any]]:
    """Return a copy of a preset by name."""
    preset = KASAI_CODE_PRESETS.get(name)
    if preset is None:
        return None
    return dict(preset)
