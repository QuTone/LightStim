"""Finite-group lift elements for GALA codes (arXiv:2608.07431).

A GALA code lifts over ``G = H_k x C_m`` (or the semidirect ``H_k |x C_m^k``):
a small non-abelian permutation group ``H_k`` on ``[k]`` supplying the *active
orthogonality* pattern, and an abelian ``C_m`` — a product of cyclic factors —
supplying the shift symmetries that become AOD move schedules.

This module represents one *monomial* ``(h, shifts)`` of that group and its
action on the ``k * m`` lift points, indexed as ``a * m + b`` with ``a`` in
``[k]`` and ``b`` the mixed-radix encoding of the cyclic factors. A lift entry
in the general (*polynomial*) case is a sum of such monomials over ``F_2[G]``;
the *monomial* case of Definition 15/16 is a sum with a single term.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

# A monomial is (permutation of [k] as a tuple, shift per cyclic factor).
Perm = Tuple[int, ...]
Shifts = Tuple[int, ...]
Monomial = Tuple[Perm, Shifts]

# S_3 in the sigma/tau labels of arXiv:2608.07431 Sec. S2:
#   e = id, sigma_0 = (0 1 2), sigma_1 = (0 2 1),
#   tau_0 = (1 2), tau_1 = (0 2), tau_2 = (0 1).
# Stored as images: perm[i] is the image of i.
S3_ELEMENTS: Dict[str, Perm] = {
    "e": (0, 1, 2),
    "s0": (1, 2, 0),
    "s1": (2, 0, 1),
    "t0": (0, 2, 1),
    "t1": (2, 1, 0),
    "t2": (1, 0, 2),
}

# S_4 generators of the paper's presentation <a,b,c | a^2=b^3=c^4=abc=e>
# realized as a = (2 3), b = (0 1 2), c = (0 2 3 1).
S4_GENERATORS: Dict[str, Perm] = {
    "e": (0, 1, 2, 3),
    "a": (0, 1, 3, 2),
    "b": (1, 2, 0, 3),
    "c": (2, 0, 3, 1),
}


def compose(p: Perm, q: Perm) -> Perm:
    """Composition ``(p . q)(i) = p(q(i))``."""
    return tuple(p[q[i]] for i in range(len(q)))


def invert(p: Perm) -> Perm:
    """Inverse permutation."""
    out = [0] * len(p)
    for i, image in enumerate(p):
        out[image] = i
    return tuple(out)


def perm_from_word(word: str, generators: Dict[str, Perm], degree: int) -> Perm:
    """Evaluate a generator word such as ``"aca"`` or ``"c2"`` left-to-right.

    A digit repeats the preceding generator (``"c2"`` is ``c . c``), matching
    the exponent notation used in the paper's generator tables.
    """
    result: Perm = tuple(range(degree))
    previous: Perm | None = None
    for ch in word:
        if ch.isdigit():
            if previous is None:
                raise ValueError(f"Exponent with no preceding generator in {word!r}.")
            for _ in range(int(ch) - 1):
                result = compose(result, previous)
            continue
        if ch not in generators:
            raise ValueError(f"Unknown generator {ch!r} in word {word!r}.")
        previous = generators[ch]
        result = compose(result, previous)
    return result


def s3(label: str) -> Perm:
    """Look up an S_3 element by its sigma/tau label."""
    if label not in S3_ELEMENTS:
        raise ValueError(
            f"Unknown S_3 label {label!r}; expected one of {sorted(S3_ELEMENTS)}."
        )
    return S3_ELEMENTS[label]


def s4(word: str) -> Perm:
    """Build an S_4 element from a word in the paper's ``a, b, c`` generators."""
    return perm_from_word(word, S4_GENERATORS, 4)


def commutes(p: Perm, q: Perm) -> bool:
    """Whether two permutations commute."""
    return compose(p, q) == compose(q, p)


class LiftAlphabet:
    """Action of ``H_k x C_m`` on the ``k * m`` lift points.

    ``cyclic`` lists the orders of the cyclic factors of ``C_m``, so
    ``m = prod(cyclic)`` and a shift tuple has one entry per factor. ``degree``
    is ``k``, the degree of the ``H_k`` permutation action (``k = 1`` for a
    purely abelian lift).
    """

    def __init__(self, degree: int, cyclic: Sequence[int]):
        if degree < 1:
            raise ValueError("H_k degree must be >= 1.")
        if any(order < 1 for order in cyclic):
            raise ValueError("Cyclic factor orders must be >= 1.")
        self.degree = int(degree)
        self.cyclic: Tuple[int, ...] = tuple(int(o) for o in cyclic)
        self.m = 1
        for order in self.cyclic:
            self.m *= order
        self.size = self.degree * self.m

    # -- mixed-radix encoding of the abelian coordinates -------------------

    def _shift_point(self, b: int, shifts: Shifts) -> int:
        """Apply a per-factor shift to the mixed-radix abelian index ``b``.

        ``b`` encodes ``(b_1, ..., b_n)`` big-endian over ``cyclic``; each
        component is shifted independently modulo its own factor order.
        """
        if len(shifts) != len(self.cyclic):
            raise ValueError(
                f"Expected {len(self.cyclic)} shift components, got {len(shifts)}."
            )
        digits = []
        for order in reversed(self.cyclic):
            b, digit = divmod(b, order)
            digits.append(digit)
        digits.reverse()
        value = 0
        for order, digit, shift in zip(self.cyclic, digits, shifts):
            value = value * order + (digit + shift) % order
        return value

    def apply(self, monomial: Monomial, point: int) -> int:
        """Image of ``point`` under a monomial ``(h, shifts)``."""
        perm, shifts = monomial
        a, b = divmod(point, self.m)
        return perm[a] * self.m + self._shift_point(b, shifts)

    def invert_monomial(self, monomial: Monomial) -> Monomial:
        """Inverse of a monomial — the transpose of its permutation matrix."""
        perm, shifts = monomial
        inv_shifts = tuple(
            (-shift) % order for order, shift in zip(self.cyclic, shifts)
        )
        return invert(perm), inv_shifts
