"""Published GALA code instances (arXiv:2608.07431).

Transcribed from the generator tables of the supplement. Each preset gives the
block count ``L``, the number of active block rows ``J``, the lift group
``H_k x C_m`` (``degree`` = k, ``cyclic`` = the orders of the factors of C_m),
and the lifts ``F``/``G`` — one entry per ``i in [L/2]``, each a list of
monomials ``(h_label, shifts)`` summed over ``F_2[G]``.

``h_label`` is ``"e"`` for a purely abelian lift, a sigma/tau label for ``S_3``
(Sec. S2), or a word in ``a, b, c`` for ``S_4``. ``shifts`` has one component
per cyclic factor, so the paper's ``x^(1,0)`` becomes ``(1, 0)``.

The paper writes a group-ring element with an identity ``h`` omitted, so a bare
``x^s`` is ``("e", (s,))``, and ``x^a + x^b`` is a two-monomial entry.
"""

from typing import Any, Dict, Optional


def _cyc(*shifts: int):
    """Identity element of H_k with the given cyclic shifts."""
    return ("e", tuple(shifts))


GALA_CODE_PRESETS: Dict[str, Dict[str, Any]] = {
    # ---- Table S4: ZX-dual codes, purely cyclic lifts ---------------------
    # The headline compact self-dual instance: girth-6-ish (t4 = 660),
    # LER < 1e-8 for memory at p = 1e-3 with a 3.1 ms SE cycle.
    "gala_132_30_12": {
        "L": 12, "J": 5, "degree": 1, "cyclic": (11,),
        "F": [[_cyc(2)], [_cyc(4)], [_cyc(3)], [_cyc(6)], [_cyc(3)], [_cyc(9)]],
        "G": [[_cyc(9)], [_cyc(2)], [_cyc(8)], [_cyc(5)], [_cyc(8)], [_cyc(7)]],
        "expected_n": 132, "expected_k": 30, "expected_d": 12,
    },
    "gala_192_40_12": {
        "L": 12, "J": 5, "degree": 1, "cyclic": (16,),
        "F": [[_cyc(10)], [_cyc(3)], [_cyc(2)], [_cyc(15)], [_cyc(12)], [_cyc(6)]],
        "G": [[_cyc(6)], [_cyc(10)], [_cyc(4)], [_cyc(1)], [_cyc(14)], [_cyc(13)]],
        "expected_n": 192, "expected_k": 40, "expected_d": 12,
    },
    "gala_228_46_12": {
        "L": 12, "J": 5, "degree": 1, "cyclic": (19,),
        "F": [[_cyc(10)], [_cyc(15)], [_cyc(6)], [_cyc(4)], [_cyc(5)], [_cyc(17)]],
        "G": [[_cyc(9)], [_cyc(2)], [_cyc(14)], [_cyc(15)], [_cyc(13)], [_cyc(4)]],
        "expected_n": 228, "expected_k": 46, "expected_d": 12,
    },
    # Polynomial lifts: entries that are sums of two group elements.
    "gala_136_36_8": {
        "L": 8, "J": 3, "degree": 1, "cyclic": (17,),
        "F": [[_cyc(7)], [_cyc(0), _cyc(5)], [_cyc(6)], [_cyc(1), _cyc(2)]],
        "G": [[_cyc(10)], [_cyc(16), _cyc(15)], [_cyc(11)], [_cyc(0), _cyc(12)]],
        "expected_n": 136, "expected_k": 36, "expected_d": 8,
    },
    "gala_136_34_12": {
        "L": 8, "J": 3, "degree": 1, "cyclic": (17,),
        "F": [[_cyc(2)], [_cyc(1)], [_cyc(3), _cyc(16)], [_cyc(13), _cyc(12)]],
        "G": [[_cyc(15)], [_cyc(4), _cyc(5)], [_cyc(14), _cyc(1)], [_cyc(16)]],
        "expected_n": 136, "expected_k": 34, "expected_d": 12,
    },
    # Two cyclic factors: C_3 x C_7.
    "gala_168_42_12": {
        "L": 8, "J": 3, "degree": 1, "cyclic": (3, 7),
        "F": [[_cyc(0, 1)], [_cyc(0, 3), _cyc(0, 2)],
              [_cyc(2, 4), _cyc(0, 4)], [_cyc(1, 3)]],
        "G": [[_cyc(0, 6)], [_cyc(2, 4)],
              [_cyc(1, 3), _cyc(0, 3)], [_cyc(0, 4), _cyc(0, 5)]],
        "expected_n": 168, "expected_k": 42, "expected_d": 12,
    },
    # ---- Table S4 with a non-abelian factor: S_3 x C_16 -------------------
    "gala_576_104_12": {
        "L": 12, "J": 5, "degree": 3, "cyclic": (16,),
        "F": [[("s1", (8,))], [_cyc(10)], [_cyc(8)],
              [("s1", (9,))], [("s0", (8,))], [("s1", (7,))]],
        "G": [[("s0", (8,))], [_cyc(6)], [_cyc(8)],
              [("s0", (7,))], [("s1", (8,))], [("s0", (9,))]],
        "expected_n": 576, "expected_k": 104, "expected_d": 12,
    },
    # ---- Table S3: rate-1/2 frontier codes ---------------------------------
    # Table S3 lists no J; rate 1/2 forces J = L/4 (Table S5: J > L/4 => r < 1/2),
    # since a full-rank lift gives k = k*m*(L - 2J).
    "gala_576_292_8": {
        "L": 12, "J": 3, "degree": 3, "cyclic": (2, 8),
        "F": [[_cyc(1, 0)], [("t2", (0, 1))], [("s1", (0, 5))],
              [_cyc(1, 7)], [_cyc(1, 5)], [("s0", (1, 2))]],
        "G": [[_cyc(1, 1)], [_cyc(1, 3)], [("s1", (1, 6))],
              [_cyc(1, 0)], [("t2", (0, 7))], [("s0", (0, 3))]],
        "expected_n": 576, "expected_k": 292, "expected_d": 8,
    },
    "gala_576_294_8": {
        "L": 12, "J": 3, "degree": 3, "cyclic": (2, 8),
        "F": [[_cyc(0, 4)], [("s1", (0, 7))], [("t2", (1, 6))],
              [_cyc(1, 4)], [("s0", (0, 6))], [("s1", (1, 6))]],
        "G": [[_cyc(0, 4)], [("s0", (0, 1))], [("t2", (1, 2))],
              [_cyc(1, 4)], [("s1", (0, 2))], [("s0", (1, 2))]],
        "expected_n": 576, "expected_k": 294, "expected_d": 8,
    },
    "gala_720_364_10": {
        "L": 12, "J": 3, "degree": 3, "cyclic": (4, 5),
        "F": [[_cyc(2, 4)], [("s1", (0, 2))], [_cyc(0, 2)],
              [_cyc(3, 3)], [("s0", (0, 1))], [("t0", (2, 2))]],
        "G": [[_cyc(2, 1)], [("s0", (0, 3))], [_cyc(0, 3)],
              [_cyc(1, 2)], [("s1", (0, 4))], [("t0", (2, 3))]],
        "expected_n": 720, "expected_k": 364, "expected_d": 10,
    },
    "gala_864_436_10": {
        "L": 12, "J": 3, "degree": 3, "cyclic": (2, 3, 4),
        "F": [[_cyc(1, 0, 2)], [("s1", (0, 0, 3))], [("t0", (0, 0, 1))],
              [_cyc(1, 1, 3)], [("s0", (0, 2, 0))], [_cyc(0, 0, 1)]],
        "G": [[_cyc(1, 2, 1)], [("s1", (0, 1, 0))], [_cyc(0, 0, 3)],
              [_cyc(1, 0, 2)], [("s0", (0, 0, 1))], [("t0", (0, 0, 3))]],
        "expected_n": 864, "expected_k": 436, "expected_d": 10,
    },
    "gala_1008_510_10": {
        "L": 12, "J": 3, "degree": 3, "cyclic": (2, 2, 7),
        "F": [[_cyc(0, 0, 4)], [_cyc(0, 0, 2)], [("s1", (1, 0, 5))],
              [_cyc(1, 0, 3)], [("t0", (0, 0, 5))], [("s0", (0, 1, 0))]],
        "G": [[_cyc(0, 0, 3)], [_cyc(0, 0, 5)], [("s0", (1, 0, 2))],
              [_cyc(1, 0, 4)], [("t0", (0, 0, 2))], [("s1", (0, 1, 0))]],
        "expected_n": 1008, "expected_k": 510, "expected_d": 10,
    },
    # S_4 lift, words in the paper's <a, b, c> presentation.
    "gala_1056_532_12": {
        "L": 12, "J": 3, "degree": 4, "cyclic": (2, 11),
        "F": [[_cyc(0, 10)], [("aca", (1, 2))], [("bacb", (0, 3))],
              [_cyc(0, 4)], [("c2", (0, 2))], [("ba", (0, 3))]],
        "G": [[_cyc(0, 1)], [("ba", (1, 9))], [("bacb", (0, 8))],
              [_cyc(0, 7)], [("c2", (0, 9))], [("aca", (0, 8))]],
        "expected_n": 1056, "expected_k": 532, "expected_d": 12,
    },
}


def get_preset(name: str) -> Optional[Dict[str, Any]]:
    """Return a copy of the named preset, or ``None`` if unknown."""
    preset = GALA_CODE_PRESETS.get(name)
    return dict(preset) if preset is not None else None
