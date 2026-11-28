"""Logical lowering for Tour de gross in-module Pauli measurements.

Equations (14)--(17) of https://arxiv.org/abs/2506.03094 supply the
construction: shifted native measurements, measurement-based pi/2 rotations,
and conjugation of a native measurement.  The bidirectional graph search here
is an implementation choice, not an algorithm specified in that paper.

All public logical indices are one-based. Strings use ordinary left-to-right
qubit order (character 0 is logical qubit 1). A shift tableau means W P W^dagger.
Operations are returned in TIME ORDER. Frame updates are essential operations:
a consumer must apply them or propagate their effects into later outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Union

import stim


@dataclass(frozen=True)
class RecordParity:
    """XOR of already frame-corrected measurement keys, plus a constant bit."""

    keys: tuple[str, ...] = ()
    constant: bool = False

    def __post_init__(self):
        odd = set()
        for key in self.keys:
            if key in odd:
                odd.remove(key)
            else:
                odd.add(key)
        object.__setattr__(self, "keys", tuple(sorted(odd)))
        object.__setattr__(self, "constant", bool(self.constant))

    def __xor__(self, other: "RecordParity") -> "RecordParity":
        return RecordParity(self.keys + other.keys, self.constant ^ other.constant)

    def evaluate(self, records: Mapping[str, bool]) -> bool:
        value = self.constant
        for key in self.keys:
            value ^= bool(records[key])
        return value


@dataclass(frozen=True)
class Shift:
    x: int
    y: int


@dataclass(frozen=True)
class Measure:
    p1: str
    p7: str
    key: str


@dataclass(frozen=True)
class PauliFrameUpdate:
    """Multiply the effective logical Pauli frame by pauli^parity."""

    pauli: str
    parity: RecordParity


Operation = Union[Shift, Measure, PauliFrameUpdate]


@dataclass(frozen=True)
class NativeMeasurement:
    pauli: str
    shift: tuple[int, int]
    p1: str
    p7: str
    sign: bool = False


@dataclass(frozen=True)
class PauliPathStep:
    """One signed conjugation along the selected target-to-base path.

    ``before`` and ``after`` include an explicit sign. ``axis`` is the
    positive Pauli axis and ``direction`` specifies exp(i direction pi A/4).
    This records the final path, not the search's explored vertices.
    """

    before: str
    axis: str
    after: str
    direction: int = 1


@dataclass(frozen=True)
class SearchRound:
    """Observed search state at initialization or after one complete layer.

    Counts are unsigned, deduplicated Pauli vertices on each side. A frontier
    contains the current layer awaiting expansion. Round zero's ``new`` counts
    describe the starting known sets, including a reused native-side cache;
    later rounds count only vertices first discovered on that side in the round.
    The first meeting vertex is retained even though the layer is finished.
    """

    round: int
    expanded_side: str
    target_frontier: int
    native_frontier: int
    target_new: int
    native_new: int
    target_total: int
    native_total: int
    connected: bool
    meeting_pauli: str | None


@dataclass(frozen=True)
class BBPPMPlan:
    target: str
    operations: tuple[Operation, ...]
    output: RecordParity
    rotations: tuple[str, ...] = ()
    base_measurement: str = ""
    consumes_pivot: bool = False
    path: tuple[PauliPathStep, ...] = ()
    search_trace: tuple[SearchRound, ...] = ()
    search_status: str = "not_recorded"


class UnsupportedBBPPMError(ValueError):
    """A requested operation cannot be synthesized with the supplied resources."""


def _letters(pauli: stim.PauliString) -> str:
    return "".join("IXYZ"[pauli[i]] for i in range(len(pauli)))


def _signature(tableau: stim.Tableau) -> tuple[str, ...]:
    return tuple(str(tableau.x_output(i)) for i in range(len(tableau))) + tuple(
        str(tableau.z_output(i)) for i in range(len(tableau))
    )


def _encode(pauli: str) -> int:
    n = len(pauli)
    x = sum(1 << i for i, p in enumerate(pauli) if p in "XY")
    z = sum(1 << i for i, p in enumerate(pauli) if p in "ZY")
    return x | (z << n)


def _decode(value: int, n: int) -> str:
    return "".join("IXZY"[((value >> i) & 1) | (((value >> (i + n)) & 1) << 1)]
                   for i in range(n))


def _rotate(pauli: stim.PauliString, axis: stim.PauliString,
            direction: int = 1) -> stim.PauliString:
    """Conjugate by exp(+i direction pi axis / 4), including its sign."""
    if pauli.commutes(axis):
        return pauli.copy()
    return (axis * pauli) * (1j * direction)


class BBPPMCompiler:
    """Compile pure PPMs using a verified physical-to-logical shift action.

    Args:
        shift_maps: Map (x,y) shift labels to W P W^dagger tableaus. It must
            include identity and inverses. No assumption about the physical
            period of x/y is made; equivalent logical shifts are deduplicated.
        pivot: Logical ancilla, one-based (paper qubit 1).
        partner: Other directly accessible logical qubit (paper qubit 7).

    Native targets can include the pivot without consuming its input state.
    General synthesis consumes the pivot as an ancilla and therefore accepts
    targets on the other logical qubits only. This does not provide a generic
    twelve-data-qubit measurement using a twelve-qubit module.
    """

    def __init__(self, shift_maps: Mapping[tuple[int, int], stim.Tableau],
                 *, pivot: int = 1, partner: int = 7):
        if not shift_maps:
            raise ValueError("shift_maps must contain at least the identity")
        self.num_logicals = len(next(iter(shift_maps.values())))
        n = self.num_logicals
        if not 1 <= pivot <= n or not 1 <= partner <= n or pivot == partner:
            raise ValueError("pivot and partner must be distinct valid one-based indices")
        self.pivot = pivot
        self.partner = partner
        # Sorting makes both native representatives and the search reproducible.
        self.shift_maps = {}
        signature_keys = {}
        for key, tableau in sorted(shift_maps.items()):
            if len(tableau) != n or len(key) != 2:
                raise ValueError("all shift maps must have the same size and two-index keys")
            key = (int(key[0]), int(key[1]))
            sig = _signature(tableau)
            if sig not in signature_keys:
                signature_keys[sig] = key
                self.shift_maps[key] = tableau.copy()
        identity_sig = _signature(stim.Tableau(n))
        if identity_sig not in signature_keys:
            raise ValueError("shift_maps must contain the identity")
        self.identity_shift = signature_keys[identity_sig]
        self.inverse_shifts = {}
        for key, tableau in self.shift_maps.items():
            inverse = signature_keys.get(_signature(tableau.inverse()))
            if inverse is None:
                raise ValueError(f"shift {key} has no inverse in shift_maps")
            self.inverse_shifts[key] = inverse

        native = {}
        ordered_keys = sorted(self.shift_maps,
                              key=lambda key: (key != self.identity_shift, key))
        for key in ordered_keys:
            tableau = self.shift_maps[key]
            for p1 in "IXYZ":
                for p7 in "IXYZ":
                    if p1 == p7 == "I":
                        continue
                    base = stim.PauliString(n)
                    base[pivot - 1], base[partner - 1] = p1, p7
                    image = tableau(base)
                    name = _letters(image)
                    native.setdefault(name, NativeMeasurement(
                        name, key, p1, p7, image.sign.real < 0))
        self.native_measurements: dict[str, NativeMeasurement] = native
        # Eq. (16) needs a NONIDENTITY pivot factor Q. Simply erasing the
        # pivot from every native Pauli would also admit Q=I, for which the
        # three-measurement gadget is not a unitary rotation. The authors'
        # decomposition.rs likewise filters with has_pivot_support() before
        # constructing its search generators. Counts of all stripped native
        # strings must not be used as a substitute for this condition.
        rotations = {}
        for name, measurement in sorted(native.items()):
            if name[pivot - 1] == "I":
                continue
            axis = name[:pivot - 1] + "I" + name[pivot:]
            if any(p != "I" for p in axis):
                rotations.setdefault(axis, measurement)
        self.native_rotations: dict[str, NativeMeasurement] = rotations
        self._axis_words = tuple(sorted(_encode(p) for p in rotations))
        self._seed_parents = {word: None for word in self._axis_words}
        self._seed_frontier = list(self._axis_words)
        self._search_cache = {}
        self._search_traces: dict[int, tuple[SearchRound, ...]] = {}

    def _parse(self, targets) -> stim.PauliString:
        if isinstance(targets, (str, stim.PauliString)):
            result = stim.PauliString(targets)
            if len(result) != self.num_logicals:
                raise ValueError(f"expected {self.num_logicals} Pauli letters")
        else:
            result = stim.PauliString(self.num_logicals)
            seen = set()
            for index, letter in targets:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise ValueError("logical indices must be one-based integers")
                if not 1 <= index <= self.num_logicals or index in seen:
                    raise ValueError("logical indices must be distinct and in range")
                if letter not in "IXYZ" or len(letter) != 1:
                    raise ValueError("Pauli letters must be I, X, Y, or Z")
                seen.add(index)
                result[index - 1] = letter
        if result.sign not in (1, -1):
            raise ValueError("measurement target must be Hermitian")
        return result

    def _native_ops(self, native: NativeMeasurement, key: str):
        before = self.inverse_shifts[native.shift]
        ops = []
        if before != self.identity_shift:
            ops.append(Shift(*before))
        ops.append(Measure(native.p1, native.p7, key))
        if native.shift != self.identity_shift:
            ops.append(Shift(*native.shift))
        return ops, RecordParity((key,), native.sign)

    def _pivot_measure(self, letter: str, key: str):
        p = ["I"] * self.num_logicals
        p[self.pivot - 1] = letter
        return self._native_ops(self.native_measurements["".join(p)], key)

    def rotation_operations(self, axis, *, direction: int = 1,
                            prefix: str = "rotation") -> tuple[Operation, ...]:
        """Implement exp(+i direction pi R/4) by Eq. (16), consuming pivot.

        For Q=X use Z_a, X_a R, Y_a measurements (cyclically for Q=Y/Z).
        Their bit results a,m,s produce exp(-i (-1)^(a+m+s) pi R/4).
        Apply/track R if a XOR m XOR s XOR (direction==+1) is one.
        This formula retains all three measurement outcomes, not just the last.
        """
        if direction not in (-1, 1):
            raise ValueError("rotation direction must be +1 or -1")
        pauli = self._parse(axis)
        if pauli.sign != 1:
            direction *= -1
        name = _letters(pauli)
        if name not in self.native_rotations:
            raise UnsupportedBBPPMError("rotation axis is not in Eq. (15)'s native set")
        measurement = self.native_rotations[name]
        q = measurement.pauli[self.pivot - 1]
        a, b = {"X": ("Z", "Y"), "Y": ("X", "Z"), "Z": ("Y", "X")}[q]
        first, ra = self._pivot_measure(a, prefix + ".prepare")
        middle, rm = self._native_ops(measurement, prefix + ".joint")
        last, rs = self._pivot_measure(b, prefix + ".readout")
        correction = ra ^ rm ^ rs ^ RecordParity(constant=direction == 1)
        return tuple(first + middle + last + [PauliFrameUpdate(name, correction)])

    def _neighbors(self, word: int):
        n = self.num_logicals
        mask = (1 << n) - 1
        x, z = word & mask, word >> n
        for axis in self._axis_words:
            if ((x & (axis >> n)).bit_count() +
                    (z & (axis & mask)).bit_count()) & 1:
                yield word ^ axis, axis

    def _find_path(self, target: int):
        """Exact bidirectional BFS in the Pauli/transvection graph.

        A rotation axis R sends an anticommuting P to the unsigned RP.
        Right-side layers are cached across requests. The target-side path is
        retained in chronological order; dagger changes signs but not edges.
        A completed search keeps its original trace in ``_search_traces``;
        returning a cached path does not pretend to perform new expansions.
        """
        if target in self._search_cache:
            return self._search_cache[target]
        left = {target: None}
        frontier = [target]
        right = self._seed_parents
        meet = target if target in right else None
        trace = [SearchRound(
            round=0, expanded_side="initial", target_frontier=len(frontier),
            native_frontier=len(self._seed_frontier), target_new=len(left),
            native_new=len(right), target_total=len(left), native_total=len(right),
            connected=meet is not None,
            meeting_pauli=_decode(meet, self.num_logicals) if meet is not None else None,
        )]
        if target in right:
            meet = target
        else:
            meet = None
            while frontier and self._seed_frontier and meet is None:
                expand_left = len(frontier) <= len(self._seed_frontier)
                own = left if expand_left else right
                other = right if expand_left else left
                current = frontier if expand_left else self._seed_frontier
                following = []
                for word in current:
                    for nxt, axis in self._neighbors(word):
                        if nxt in own:
                            continue
                        own[nxt] = (word, axis)
                        following.append(nxt)
                        if nxt in other and meet is None:
                            meet = nxt
                if expand_left:
                    frontier = following
                else:
                    self._seed_frontier = following
                trace.append(SearchRound(
                    round=len(trace), expanded_side="target" if expand_left else "native",
                    target_frontier=len(frontier), native_frontier=len(self._seed_frontier),
                    target_new=len(following) if expand_left else 0,
                    native_new=0 if expand_left else len(following),
                    target_total=len(left), native_total=len(right),
                    connected=meet is not None,
                    meeting_pauli=_decode(meet, self.num_logicals) if meet is not None else None,
                ))
            if meet is None:
                self._search_traces[target] = tuple(trace)
                raise UnsupportedBBPPMError(
                    "the supplied native rotations do not connect this target to a native measurement")
        before, word = [], meet
        while left[word] is not None:
            word, axis = left[word]
            before.append(axis)
        path = list(reversed(before))
        word = meet
        while right[word] is not None:
            word, axis = right[word]
            path.append(axis)
        result = (tuple(path), word)
        self._search_cache[target] = result
        self._search_traces[target] = tuple(trace)
        return result

    def compile(self, targets, *, prefix: str = "ppm") -> BBPPMPlan:
        """Return a complete frame-aware measurement plan in time order.

        General synthesis maps the requested data P to a native data axis R.
        If R itself is a native measurement, measure it directly. Otherwise
        prepare the pivot in Q and measure native Q R; the XOR of those two
        outcomes is the data measurement result. Then reverse the rotations.
        Rotation gadgets still consume the pivot, even when the central
        measurement is direct. Pivot-only preparation outcomes are random,
        not failures. The final pivot may be discarded/reused as an ancilla.
        """
        target = self._parse(targets)
        name = _letters(target)
        sign = target.sign.real < 0
        signed_name = ("-" if sign else "") + name
        if all(p == "I" for p in name):
            return BBPPMPlan(signed_name, (), RecordParity(constant=sign),
                             search_status="identity")
        if name in self.native_measurements:
            ops, output = self._native_ops(self.native_measurements[name], prefix + ".result")
            return BBPPMPlan(signed_name, tuple(ops), output ^ RecordParity(constant=sign),
                             base_measurement=name, search_status="native")
        if name[self.pivot - 1] != "I":
            raise UnsupportedBBPPMError(
                "a non-native target involving the pivot cannot preserve its unknown input; "
                "general synthesis reserves the pivot as an ancilla")
        target_word = _encode(name)
        search_status = "cached" if target_word in self._search_cache else "searched"
        path, seed_word = self._find_path(target_word)
        axes = tuple(_decode(word, self.num_logicals) for word in path)
        seed = _decode(seed_word, self.num_logicals)
        operations = []
        transformed = target.copy()
        signed_path = []
        for i, axis in enumerate(axes):
            operations.extend(self.rotation_operations(axis, prefix=f"{prefix}.pre{i}"))
            before = ("-" if transformed.sign.real < 0 else "+") + _letters(transformed)
            transformed = _rotate(transformed, stim.PauliString(axis))
            after = ("-" if transformed.sign.real < 0 else "+") + _letters(transformed)
            signed_path.append(PauliPathStep(before, axis, after))
        if _letters(transformed) != seed:
            raise AssertionError("rotation search and signed conjugation disagree")
        if seed in self.native_measurements:
            native = self.native_measurements[seed]
            middle, r_middle = self._native_ops(native, prefix + ".result")
            operations.extend(middle)
            output = r_middle ^ RecordParity(constant=transformed.sign.real < 0)
        else:
            native = self.native_rotations[seed]
            q = native.pauli[self.pivot - 1]
            prepare, r_prepare = self._pivot_measure(q, prefix + ".prepare")
            middle, r_middle = self._native_ops(native, prefix + ".result")
            operations.extend(prepare)
            operations.extend(middle)
            output = r_prepare ^ r_middle ^ RecordParity(constant=transformed.sign.real < 0)
        # Undo in reverse TIME order. This is not the matrix-product order.
        for i, axis in reversed(list(enumerate(axes))):
            operations.extend(self.rotation_operations(
                axis, direction=-1, prefix=f"{prefix}.post{i}"))
        return BBPPMPlan(signed_name, tuple(operations), output, axes,
                         native.pauli, True, tuple(signed_path),
                         search_trace=self._search_traces[target_word],
                         search_status=search_status)


__all__ = ["BBPPMCompiler", "BBPPMPlan", "Measure", "NativeMeasurement",
           "PauliFrameUpdate", "PauliPathStep", "RecordParity", "SearchRound", "Shift",
           "UnsupportedBBPPMError"]
