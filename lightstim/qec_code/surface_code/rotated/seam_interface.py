"""Per-seam interface classification for the routed-corridor pipeline
(stage C of the user's two-step architecture).

THE ARCHITECTURE (the user's, verbatim ordering):
  step 2 builds stabilizers ON the routed bus in this order:
  ① fix the bus body colour — a single binary choice, arbitrary but FIXED
    (never negotiated per seam);
  ② each target patch EXTENDS toward the bus: the extension construct (a
    KF stretched wall or ordinary stabilizers) is decided by the eight-row
    seam table from the (bus body, patch) relation alone;
  ③ the patch + its extension attach to the bus as a whole;
  ④ the bus BOUNDARY stabilizers are built last, per the paper's §10.4
    rules, anchored on the fixed interfaces.
  A target patch's own stabilizers NEVER change in step 2.

THE EIGHT-ROW TABLE (docs/specs/paper-material.md §2.6, per attach seam,
three comparisons — letter / colour / type):
  #1 same-same-same  -> plain stitch (checkerboard continues, seam is DATA)
  #4 same-diff-diff  -> pure KF wall (uniform dominoes)
  #6 diff-same-diff  -> mixed KF wall (K&F Fig 4a spatial-Hadamard family)
  #7 diff-diff-same  -> recoloured mixed column (single-cell M checks)
  odd rows are illegal seams.  THEOREM (spec): on a parallel-law-legal
  seam the type difference equals colour⊕letter (patch attributes cancel),
  so legal seams always land on an even row.

THE BUS CONVENTION PAIR (the user's ruling, 2026-07-30): before the bus
body is built, TWO binary conventions are fixed together — the bus
COLOUR (checkerboard phase reference) and the bus LETTER.  Neither is
determined by the other, by the geometry, or by the measurement itself
(measured: the same mixed-letter joint builds at full distance under
either letter convention, with the #7 side swapping accordingly).  Both
are free, any fixed pick is valid; the canonical deterministic instance
used here is (majority target phase, majority measured letter) — it
also minimises the interface hardware, but correctness does not depend
on it.

THE BUS GAUGE ``g`` ("bus colouring", spec): the DIAGONAL move in that 2-bit
convention space — flipping letter and colour comparisons TOGETHER maps
#1↔#7 and #4↔#6 wholesale.  Consequently:
  * wall-ness (letter⊕colour, i.e. rows {4, 6} vs {1, 7}) is g-INVARIANT
    — classified before g is chosen;
  * g itself is fixed once per corridor by the bus-construction rule
    (spec'd production behaviour, _plan_snake's static assignment): a
    wall on a VERTICAL seam needs the mixed family (the pure vertical
    family has no compact-7 schedule) -> g = 1; otherwise g = 0.
"""
from dataclasses import dataclass

from .litinski_layouts import ptype

_FLIP = {"X": "Z", "Z": "X"}


def choose_bus_conventions(target, phis, bus=None):
    """① of the architecture: fix the (colour, letter) convention PAIR of
    one corridor — both free, both fixed here and never renegotiated.
    ``bus`` overrides the letter (a PLANNED basis); default = majority
    measured letter, tie -> 'X' (the historical canonical rule)."""
    letters = [P for _, P in target]
    n_x = sum(1 for P in letters if P == "X")
    bus_letter = bus if bus is not None else \
        ("X" if n_x >= len(letters) - n_x else "Z")
    return bus_letter, choose_colour_reference(phis)


def choose_colour_reference(phis):
    """The g = 0 colour reference of one corridor: the MAJORITY phase of
    its target patches (tie -> phase 0).  A single per-corridor bit, fixed
    before any seam is classified (the user's ruling: arbitrary but fixed;
    the majority choice is the canonical deterministic pick — standard
    same-phase merges stay plain, minority-phase targets land on the
    table's interface rows)."""
    ones = sum(1 for f in phis if f == 1)
    return 1 if ones * 2 > len(list(phis)) else 0


@dataclass(frozen=True)
class SeamInterface:
    """One target patch's classified attach seam."""
    name: str                 # target patch
    patch_cell: tuple         # coarse cell of the patch
    bus_cell: tuple           # adjacent corridor cell it attaches through
    axis: str                 # seam LINE direction: 'vertical'|'horizontal'
    letter_diff: bool         # measured P != bus letter   (at g = 0)
    colour_diff: bool         # patch phase != BUS_PHASE   (at g = 0)

    @property
    def is_wall(self):
        """g-invariant: rows {4, 6} carry a stretched KF wall."""
        return self.letter_diff != self.colour_diff

    def row(self, g):
        """The final table row under bus gauge ``g``."""
        ld = self.letter_diff ^ bool(g)
        cd = self.colour_diff ^ bool(g)
        return {(False, False): 1, (False, True): 4,
                (True, False): 6, (True, True): 7}[(ld, cd)]


def patch_phase(checks):
    """A patch's live checkerboard phase: the phi for which its bulk
    letters equal ptype(x, y, phi).  Reads ONE pure bulk weight-4."""
    for c in checks:
        feet = c.get("pauli")
        if not isinstance(feet, dict) or len(feet) != 4:
            continue
        if c["type"] not in ("X", "Z"):
            continue
        sx, sy = c["syn"]
        return 0 if ptype(sx, sy, 0) == c["type"] else 1
    raise ValueError("patch has no pure bulk weight-4 check to read")


def classify_attach_seam(name, patch_cell, bus_cell, measured, bus_letter,
                         phi_patch, phi_ref):
    """The g = 0 comparisons for one attach seam (letter / colour; the
    type bit follows by the spec theorem and is self-checked downstream
    against live facing-lobe geometry)."""
    pa, pb = tuple(patch_cell), tuple(bus_cell)
    if abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) != 1:
        raise ValueError(f"{name}: {pa} and {pb} are not cell-adjacent")
    axis = 'vertical' if pa[1] == pb[1] else 'horizontal'
    return SeamInterface(name=name, patch_cell=pa, bus_cell=pb, axis=axis,
                         letter_diff=(measured != bus_letter),
                         colour_diff=(phi_patch != phi_ref))


def choose_bus_gauge(interfaces):
    """① of the architecture: ONE corridor convention bit, fixed by the
    bus-construction rule — the mixed set (g = 1) iff some wall sits on a
    vertical seam (spec'd schedulability rule); the pure set otherwise."""
    return int(any(i.is_wall and i.axis == 'vertical' for i in interfaces))
