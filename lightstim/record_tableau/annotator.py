"""Streaming detector / observable annotation on a record-augmented tableau.

``RecordTableauAnnotator`` replays a physical Stim circuit instruction by
instruction on a record tableau (the C++ engine in ``cpp/record_tableau.cc``)
and emits DETECTOR and
OBSERVABLE_INCLUDE instructions directly after the measurement layer that
determines them.  It replaces LightStim's back-propagation, write-back and
GF(2) RREF with tableau queries.

Symbolic logical values: at a classification boundary the initialised
constraints that no measurement has re-anchored are labelled as logicals and
given a symbolic value lambda_j.  A deterministic parity that involves some
lambda is a logical relation.  As in the tracker, a joint logical value
measured mid-circuit (a merge / PPM outcome) is absorbed: it enters the
elimination basis but is not an observable; a single-logical mid-circuit
relation is a logical-frame update and is not absorbed.  A new lambda pattern
at the data readout becomes an observable, and relations whose pattern is
already spanned reduce to plain detectors.

Output mirrors the tracker path: detectors carry the measured qubit's
coordinates plus a relative time 0 (the builder emits SHIFT_COORDS per round,
and detectors follow an immediately succeeding SHIFT_COORDS), and a physical
REPEAT block is re-emitted as REPEAT wherever its annotated iterations are
identical.
"""
from dataclasses import dataclass
from itertools import groupby
from typing import Dict, List, Optional, Tuple

import stim
import numpy as np

LAMBDA_BASE = -(1 << 40)            # symbolic logical values LAMBDA_BASE - j (as in the C++ engine)

MEAS_BASIS = {"M": "Z", "MZ": "Z", "MX": "X", "MY": "Y",
              "MR": "Z", "MRZ": "Z", "MRX": "X", "MRY": "Y"}
RESET_BASIS = {"R": "Z", "RZ": "Z", "RX": "X", "RY": "Y",
               "MR": "Z", "MRZ": "Z", "MRX": "X", "MRY": "Y"}
DROP = {"DETECTOR", "OBSERVABLE_INCLUDE"}
PASS = {"TICK", "QUBIT_COORDS", "SHIFT_COORDS"}
SIGN_ONLY = {"I", "X", "Y", "Z", "II", "I_ERROR", "II_ERROR"}


@dataclass(frozen=True, slots=True)
class _Annotation:
    """An emitted DETECTOR / OBSERVABLE_INCLUDE: record lookbacks (sorted,
    negative) and arguments.  Much cheaper to build than a
    stim.CircuitInstruction; it becomes Stim text when the circuit is built."""
    name: str
    offsets: tuple
    args: tuple

    def __str__(self) -> str:
        args = f"({', '.join(map(str, self.args))})" if self.args else ""
        return f"{self.name}{args} " + " ".join(f"rec[{o}]" for o in self.offsets)


def _offsets(records, total: int = 0) -> tuple:
    return tuple(r - total for r in sorted(records))


Instruction = stim.CircuitInstruction | stim.CircuitRepeatBlock | _Annotation
Instructions = List[Instruction]


def _text(instruction: Instruction) -> str:
    if isinstance(instruction, stim.CircuitRepeatBlock):   # str() gives its repr
        circuit = stim.Circuit()
        circuit.append(instruction)
        return str(circuit)
    return str(instruction)


def _circuit(instructions: Instructions) -> stim.Circuit:
    # One parse of the joined text is several times faster than appending
    # instruction objects one by one (stim fuses adjacent gates either way).
    return stim.Circuit("\n".join(map(_text, instructions)))


PAULI_CODE = {"X": 1, "Y": 2, "Z": 3}


def _block_unitary(block: stim.Circuit, num_qubits: int) -> stim.Tableau:
    """The Clifford part of a measurement block, on ``num_qubits`` qubits."""
    unitary = stim.Tableau.from_circuit(
        block, ignore_noise=True, ignore_measurement=True, ignore_reset=True)
    return unitary + stim.Tableau(num_qubits - len(unitary))


def _measured_input(unitary: stim.Tableau, basis: str, qubit: int) -> stim.PauliString:
    """The block-input Pauli that a ``basis`` measurement of ``qubit`` at the
    block output measures."""
    query = {"X": unitary.inverse_x_output, "Y": unitary.inverse_y_output,
             "Z": unitary.inverse_z_output}[basis]
    return query(qubit, unsigned=True)


def _row(pauli: stim.PauliString) -> np.ndarray:
    return np.concatenate(pauli.to_numpy()).astype(np.uint8)


def _make_engine(num_qubits: int):
    try:
        from .cpp import _record_tableau_cpp as native
    except ImportError as err:
        raise ImportError(
            "The record_tableau detector backend needs its C++ engine, which `pip install` builds "
            "from the third_party/stim submodule: run `git submodule update --init` and reinstall "
            "(`pip install -e .`) with a C++20 compiler (set CXX, e.g. CXX=g++, if the default "
            "compiler is older).") from err
    return native.Engine(num_qubits)


@dataclass(slots=True)
class _Op:
    """A pre-parsed circuit instruction (reused across REPEAT iterations)."""
    kind: str
    instruction: stim.CircuitInstruction
    targets: tuple | list = ()

    @property
    def name(self) -> str:
        return self.instruction.name


class RecordTableauAnnotator:
    def __init__(self, num_qubits: int = 0, *, num_logicals: Optional[int] = None):
        self.engine = _make_engine(num_qubits)
        self.readouts: List[Tuple[int, int]] = []          # [start, stop) record ranges
        self.round_ends: List[tuple] = []   # (record count, expected logicals, code stabilizers, code logicals)
        self.default_num_logicals = num_logicals
        self.num_lambdas = 0
        self.lambda_basis: Dict[int, Tuple[frozenset, frozenset]] = {}
        self.num_observables = 0
        self.allocate_observable = None     # observable-id source (e.g. the tracker's), else 0, 1, ...
        self.absorbed_patterns: List[frozenset] = []   # mid-circuit joint logical relations
        self.num_detectors = 0
        self.coords: Dict[int, List[float]] = {}
        self.meas_qubit: List[int] = []
        self.stateful_code_frames: Dict[int, tuple] = {}             # block-start record count -> code frame
        self.dependency_regions: List[tuple] = []           # (start, stop, redundant checks allowed)
        self.suppressed_detectors: set[int] = set()
        self.retained_regions: List[Tuple[int, int]] = []
        self.retained_rows: Dict[int, List[np.ndarray]] = {}
        self.output_frames: Dict[int, tuple] = {}
        self._unitaries = 0         # unitary instructions since the last measurement layer
        self._frame_checked_at = 0
        self._pending: Instructions = []   # annotations waiting for a following SHIFT_COORDS
        self._layer_bases: List[str] = []   # contiguous mid-circuit measurements not yet processed
        self._layer_targets: List[int] = []
        self._recent: Instructions = []    # instructions of the round before a REPEAT
        self._out: Instructions = []
        self._cursor = 0
        self._synced = 0            # top-level instructions of the builder circuit consumed
        self._last_targets = 0      # targets of the last consumed instruction

    # ---- hints from the builder ---------------------------------------------
    def mark_disposable_block(self, block, starts, num_qubits, stabilizers, logicals) -> None:
        """Anchor coupled ancilla readouts to their data-only output checks.

        Removing the input reset factors and then the measured output factors
        gives both the surviving check and its output record parity. Bell
        readouts can involve a neighbouring ancilla's result in this parity.
        Ordinary one-ancilla checks need no additional canonicalization.
        """
        unitary = _block_unitary(block, num_qubits)
        codes = PAULI_CODE
        resets, measurements = {}, []
        for inst in block.flattened():
            if inst.name in RESET_BASIS:
                resets.update((t.value, codes[RESET_BASIS[inst.name]])
                              for t in inst.targets_copy())
            if inst.name in MEAS_BASIS:
                measurements.extend((t.value, MEAS_BASIS[inst.name])
                                    for t in inst.targets_copy())
        rows, records = [], []
        coupled = False
        measured = {q: (i, basis) for i, (q, basis) in enumerate(measurements)}
        for i, (q, basis) in enumerate(measurements):
            input_pauli = _measured_input(unitary, basis, q)
            pauli = stim.PauliString(num_qubits)
            pauli[q] = codes[basis]
            for a in input_pauli.pauli_indices():
                if a not in resets:
                    continue
                code = resets[a]
                if input_pauli[a] not in (0, code):
                    return
                if input_pauli[a]:
                    output = {1: unitary.x_output, 2: unitary.y_output, 3: unitary.z_output}[code]
                    pauli *= output(a)
            value = {i}
            for a in pauli.pauli_indices():
                if a not in measured:
                    continue
                j, measured_basis = measured[a]
                if pauli[a] not in (0, codes[measured_basis]):
                    return
                if pauli[a]:
                    value ^= {j}
                    pauli[a] = 0
            coupled |= bool(pauli.weight and value - {i})
            rows.append(pauli)
            records.append(value)
        if not coupled:
            return
        rows = [_row(p) for p in rows]

        # A fixed representative prevents the logical generator from drifting
        # among equivalent checks. Reduce only the static code definitions;
        # measurement evolution continues to use the record tableau.
        pivots = {}
        for row in stabilizers():
            row = row.copy()
            while np.any(row):
                p = int(np.flatnonzero(row)[-1])
                if p not in pivots:
                    pivots[p] = row
                    break
                row ^= pivots[p]
        canonical = np.array(logicals(), copy=True)
        for row in canonical:
            for p, check in sorted(pivots.items(), reverse=True):
                if row[p]:
                    row ^= check
        frame = (rows, records, canonical)
        for start in starts:
            self.output_frames[start] = frame

    def mark_retained_block(self, block: stim.Circuit, starts: List[int], num_qubits: int) -> None:
        """Keep the measured checks with known input-reset factors removed.

        A retained-data measurement can include a propagated reset constraint.
        Re-anchoring that whole product loses the reset's independent, empty
        record value. Use the Clifford tableau to choose the same clean check
        representatives at each block boundary, without a tracker or RREF.
        """
        unitary = _block_unitary(block, num_qubits)
        resets = {}
        clean = []
        for inst in block.flattened():
            if inst.name in MEAS_BASIS:
                for target in inst.targets_copy():
                    pauli = _measured_input(unitary, MEAS_BASIS[inst.name], target.value)
                    for q, code in resets.items():
                        if pauli[q] not in (0, code):
                            raise ValueError("Measurement factor is not stabilized by its input reset.")
                        pauli[q] = 0
                    clean.append(_row(unitary(pauli)))
            elif inst.name in RESET_BASIS:
                for target in inst.targets_copy():
                    resets[target.value] = PAULI_CODE[RESET_BASIS[inst.name]]
        for start in starts:
            self.retained_regions.append((start, start + block.num_measurements))
            self.retained_rows[start] = clean

    def mark_readout(self, start: int, count: int) -> None:
        self.readouts.append((start, start + count))

    def mark_round_end(self, num_measurements: int, expected_logicals: int,
                       stabilizer_rows=None, logical_rows=None, aux_rows=None) -> None:
        """Classification boundary once ``num_measurements`` records exist.

        ``stabilizer_rows`` ([X|Z] rows) are the code stabilizers active at the
        boundary; fresh constraints they explain are stabilizers, the fresh
        constraints left over are logicals.  If that leaves the wrong count
        (a constraint's value already moved into records), the declared
        ``logical_rows`` classify the code frame directly.
        """
        self.round_ends.append((num_measurements, expected_logicals, stabilizer_rows, logical_rows,
                                aux_rows))

    def mark_stateful_code_frames(self, block_starts, stabilizer_rows, logical_rows, expected_logicals: int,
                          aux_rows=None) -> None:
        """Retained-data blocks starting at these record counts: at each TICK
        inside them (after new unitaries) try the stateful-code-frame
        classification against the code stabilizers and declared logicals."""
        frame = (stabilizer_rows, logical_rows, expected_logicals, aux_rows)
        for m in block_starts:
            self.stateful_code_frames.setdefault(m, frame)

    def mark_dependency_region(self, start: int, stop: int, allowed: bool) -> None:
        """Records [start, stop) come from checks that may (BB) or may not
        (color codes) be linearly dependent; only the former get dependent rows."""
        self.dependency_regions.append((start, stop, allowed))

    def _deps_allowed(self, m: int) -> bool:
        for a, b, allowed in reversed(self.dependency_regions):
            if a <= m < b:
                return allowed
        return True

    def _try_canonicalize_stateful_code_frame(self) -> None:
        """Builder's _try_canonicalize_stateful_code_frame, at a TICK inside a
        retained-data block once new unitaries have been applied."""
        frame = self.stateful_code_frames.get(self.engine.num_measurements)
        if frame is None or self._unitaries == self._frame_checked_at:
            return
        self._frame_checked_at = self._unitaries
        self._canonicalize_code_frame(*frame)

    def _canonicalize_code_frame(self, stabilizer_rows, logical_rows, expected, aux_rows=None) -> None:
        """Label the still-missing logicals from the declared code frame
        (stabilizers, logical operators, auxiliary constraints)."""
        if expected is not None and expected > self.num_lambdas:
            self.num_lambdas += self.engine.canonicalize_stateful_code_frame(
                stabilizer_rows, logical_rows, expected - self.num_lambdas, self.num_lambdas, aux_rows)

    def stabilizer_canonicalization(self, stabilizer_rows, expected_logicals: int) -> None:
        """Tracker's stabilizer_canonicalization: re-anchor the code stabilizers
        ([X|Z] rows) with UNMEASURED records, then promote the fresh
        constraints left over to logicals.  Call after ``sync``."""
        self.engine.rebase_onto_code_basis(stabilizer_rows, True)
        self._promote_logicals(expected_logicals)

    def _readout_end(self, m: int) -> Optional[int]:
        """End of the readout group containing record ``m`` (None if mid-circuit)."""
        for a, b in self.readouts:
            if a <= m < b:
                return b
        return None

    # ---- logical classification ---------------------------------------------
    def _promote_logicals(self, expected: Optional[int]) -> None:
        """Tracker's promote_stabilizer_rows_to_logicals: when exactly the
        missing number of initialised constraints were never re-anchored by
        a measurement, they are the logicals (each gets its symbol lambda_j)."""
        if expected is None or expected <= self.num_lambdas:
            return
        cands = self.engine.promotable_stabilizers()
        if len(cands) == expected - self.num_lambdas:
            self.engine.promote_stabilizers_to_logicals(list(cands), self.num_lambdas)
            self.num_lambdas = expected

    def _finish_measurement_block_group(self, readout: bool) -> None:
        """Builder's _finish_measurement_block_group: classification at the
        SE-round boundaries registered with ``mark_round_end`` (rebase onto
        the code basis, promote, else the stateful-code-frame fallback)."""
        m = self.engine.num_measurements
        while self.round_ends and self.round_ends[0][0] <= m:
            _, expected, rows, logical_rows, aux_rows = self.round_ends.pop(0)
            if rows is not None and expected is not None and expected > self.num_lambdas:
                self.engine.rebase_onto_code_basis(rows, False)
            self._promote_logicals(expected)
            if logical_rows is not None and rows is not None:
                self._canonicalize_code_frame(rows, logical_rows, expected, aux_rows)
        if not readout and self.default_num_logicals is not None:
            self._promote_logicals(self.default_num_logicals)

    # ---- relations -> DETECTOR / OBSERVABLE_INCLUDE ---------------------------
    def _emit(self, relations, out: Instructions, readout: bool = False) -> None:
        """Case B relations (records plus lambdas) to annotations, as the
        tracker decides: no logical component -> DETECTOR; a new logical
        pattern at the data readout -> OBSERVABLE_INCLUDE (its logical rows'
        observable); a joint pattern mid-circuit -> absorbed (absorbed_ops)."""
        total = self.engine.num_measurements
        for m, val in relations:
            lam = {r for r in val if r <= LAMBDA_BASE}
            rec = {r for r in val if r >= 0}
            own = rec.copy()
            while lam:
                piv = max(lam)
                if piv not in self.lambda_basis:
                    break
                bl, br = self.lambda_basis[piv]
                lam ^= bl
                rec ^= br
            if lam and not readout and len(lam) == 1:
                # A single logical inside a mid-circuit relation is a logical-
                # frame update (the tracker folds it into the logical row), not
                # a measured logical DOF.
                continue
            if lam:                                     # new logical outcome
                self.lambda_basis[max(lam)] = (frozenset(lam), frozenset(rec))
                if readout:
                    idx = (self.allocate_observable() if self.allocate_observable is not None
                           else self.num_observables)
                    out.append(_Annotation("OBSERVABLE_INCLUDE", _offsets(own, total), (idx,)))
                    self.num_observables += 1
                else:
                    self.absorbed_patterns.append(frozenset(lam))
            else:
                if m in self.suppressed_detectors:
                    continue
                coords = (*self.coords.get(self.meas_qubit[m], ()), 0)
                out.append(_Annotation("DETECTOR", _offsets(rec, total), coords))
                self.num_detectors += 1

    # ---- instruction dispatch -------------------------------------------------
    def _parse(self, inst: stim.CircuitInstruction) -> _Op:
        name = inst.name
        if name in DROP:
            return _Op("drop", inst)
        if name in PASS:
            if name == "QUBIT_COORDS":
                args = inst.gate_args_copy()
                for t in inst.targets_copy():
                    self.coords[t.value] = args
            return _Op("pass", inst)
        gd = stim.gate_data(name)
        if name in MEAS_BASIS:
            return _Op("meas", inst, [t.value for t in inst.targets_copy()])
        if name in RESET_BASIS:
            return _Op("reset", inst, [t.value for t in inst.targets_copy()])
        if gd.is_noisy_gate or name in SIGN_ONLY:
            if gd.produces_measurements:
                raise NotImplementedError(f"record tableau: unsupported instruction {name}")
            return _Op("pass", inst)
        if gd.is_unitary:
            ts = inst.targets_copy()
            if gd.is_two_qubit_gate:
                qs = []
                for a, b in zip(ts[::2], ts[1::2]):
                    if a.is_qubit_target and b.is_qubit_target:
                        qs += [a.value, b.value]      # classical control = Pauli feedback
                return _Op("unitary", inst, qs)
            return _Op("unitary", inst, [t.value for t in ts])
        raise NotImplementedError(f"record tableau: unsupported instruction {name}")

    def _flush(self, out: Instructions) -> None:
        self._close_layer()
        if self._pending:
            out.extend(self._pending)
            self._pending = []

    def _keep_logical(self, row) -> bool:
        """Re-anchor a logical representative that carries one logical symbol,
        keeping that symbol; False if the row is not such a representative."""
        value = self.engine.logical_value(row)
        if value is None:
            return False
        self.engine.anchor_logical(row, sorted(value))
        return True

    def adopt_subsystem_logicals(self, rows) -> None:
        """Commit the tracker's subsystem classification (its bare logical
        rows) to the logical bank: a row already carrying one logical symbol
        keeps it, a new logical gets the next symbol."""
        self.flush_state()
        rows = np.asarray(rows, dtype=np.uint8)
        for row in (rows if rows.size else []):
            if self._keep_logical(row):
                continue
            before = self.num_lambdas
            self._canonicalize_code_frame(np.zeros((0, row.size), dtype=np.uint8), row[None, :],
                                          self.num_lambdas + 1)
            if self.num_lambdas == before:
                raise NotImplementedError(
                    "record tableau: a subsystem logical constraint mixes several tracked logicals")

    def flush_state(self) -> None:
        """Process a buffered measurement layer now, so the engine state
        reflects every synced instruction. Its annotations still come out in
        order with the next instruction."""
        self._close_layer()

    def _close_layer(self) -> None:
        """Process a contiguous mid-circuit measurement layer as one block,
        like the tracker's terminal readout layer of a measurement block."""
        if not self._layer_targets:
            return
        bases, targets = "".join(self._layer_bases), self._layer_targets
        self._layer_bases, self._layer_targets = [], []
        start = self.engine.num_measurements
        if self.dependency_regions:
            self.engine.allow_deps = self._deps_allowed(self.engine.num_measurements)
        self.engine.clean_rows = self.retained_rows.get(self.engine.num_measurements, [])
        self._emit(self.engine.measure(bases, targets), self._pending)
        self._unitaries = self._frame_checked_at = 0
        self._finish_measurement_block_group(False)
        frame = self.output_frames.get(start)
        if frame is not None:
            rows, records, logicals = frame
            self.engine.anchor_checks(rows, [sorted(start + r for r in v) for v in records])
            for row in logicals:
                self._keep_logical(row)

    def _run(self, op: _Op, out: Instructions, emit_instruction: bool = True) -> None:
        if op.kind == "drop":
            return
        if op.kind != "meas" or op.name in RESET_BASIS:
            self._close_layer()
        self.engine.retained = any(a <= self.engine.num_measurements < b
                                   for a, b in self.retained_regions)
        if op.name != "SHIFT_COORDS" and op.kind != "meas":
            self._flush(out)
        if emit_instruction:
            out.append(op.instruction)
        if op.kind == "unitary":
            if op.targets:
                self.engine.unitary(op.name, op.targets)
                self._unitaries += 1
        elif op.name == "TICK":
            if self.stateful_code_frames:
                self._try_canonicalize_stateful_code_frame()
        elif op.kind == "reset" and op.name not in MEAS_BASIS:
            self.engine.reset(RESET_BASIS[op.name], op.targets)
        elif op.kind == "meas":
            basis = MEAS_BASIS[op.name]
            start = self.engine.num_measurements + len(self._layer_targets)
            group_end = self._readout_end(start)
            readout = group_end is not None
            self.meas_qubit.extend(op.targets)
            if not readout and op.name not in RESET_BASIS:
                self._layer_bases.append(basis * len(op.targets))
                self._layer_targets.extend(op.targets)
                return
            self._close_layer()
            if readout:
                rel = self.engine.readout(basis, op.targets)
                if self.engine.num_measurements >= group_end:
                    rel = rel + self.engine.finish_readout()   # close the readout group
            else:
                rel = self.engine.measure(basis, op.targets)
            self._emit(rel, self._pending, readout)
            self._unitaries = self._frame_checked_at = 0
            self._finish_measurement_block_group(readout)
            if op.name in RESET_BASIS:                        # MR / MRX / MRY
                self.engine.reset(RESET_BASIS[op.name], op.targets)

    def _walk(self, block: stim.Circuit, out: Instructions) -> None:
        for item in block:
            if isinstance(item, stim.CircuitRepeatBlock):
                self._repeat(item, out)
            else:
                self._run(self._parse(item), out)

    @staticmethod
    def _is_annotation(instruction: Instruction) -> bool:
        return instruction.name in DROP

    @classmethod
    def _xor_previous(cls, cur: Instructions, prev: Optional[Instructions], per_round: int) -> Optional[Instructions]:
        """Tracker's _make_periodic_detector_body: replace each detector by its
        XOR with the previous round's corresponding detector."""
        if prev is None or len(prev) != len(cur):
            return None
        out = []
        for a, b in zip(cur, prev):
            if a.name != "DETECTOR":
                if a != b:
                    return None
                out.append(a)
                continue
            if b.name != "DETECTOR" or a.args != b.args:
                return None
            offs = set(a.offsets) ^ {o - per_round for o in b.offsets}
            out.append(_Annotation("DETECTOR", _offsets(offs), a.args))
        return out

    def _previous_round(self, body: Instructions) -> Optional[Instructions]:
        """The annotated round emitted just before the loop, if its physical
        instructions equal the loop body's."""
        need = [inst for inst in body if not self._is_annotation(inst)]
        count, i = 0, len(self._recent)
        while i > 0 and count < len(need):
            i -= 1
            if not self._is_annotation(self._recent[i]):
                count += 1
        got = self._recent[i:]
        if [inst for inst in got if not self._is_annotation(inst)] != need:
            return None
        return got

    def _compress_logical_frame(self, frame, iteration: Instructions, per_round: int) -> bool:
        """Move a round-local logical parity into the loop's observable.

        Reduce against this round's detectors, oldest record first, to replace
        old anchors with current results. Decline if the delta is not local.
        """
        row, baseline = frame
        value = self.engine.logical_value(row)
        if value is None:
            return False
        total = self.engine.num_measurements
        delta = set(value) ^ baseline
        pivots = {}
        for inst in iteration:
            if inst.name != "DETECTOR":
                continue
            parity = {total + o for o in inst.offsets}
            while parity:
                p = min(parity)
                if p not in pivots:
                    pivots[p] = parity
                    break
                parity ^= pivots[p]
        for p, parity in sorted(pivots.items()):
            if p in delta:
                delta ^= parity
        if not all(total - per_round <= r < total for r in delta):
            return False
        if delta:
            iteration.append(_Annotation("OBSERVABLE_INCLUDE", _offsets(delta, total), (0,)))
            symbol = next(r for r in baseline if r <= LAMBDA_BASE)
            self.engine.shift_logical_records(symbol, sorted(delta))
        self.engine.anchor_logical(row, sorted(baseline))
        return True

    def _repeat(self, item: stim.CircuitRepeatBlock, out: Instructions) -> None:
        """Run every iteration on the tableau and re-emit the loop the way the
        tracker compresses steady rounds: REPEAT of the annotated body when all
        iterations are identical, else REPEAT of the body whose detectors are
        XORed with the previous round's (fixed old anchors), else runs of
        identical iterations."""
        body = item.body_copy()
        nested = any(isinstance(x, stim.CircuitRepeatBlock) for x in body)
        ops = None if nested else [self._parse(x) for x in body]
        self._flush(out)
        top = out is self._out
        if top:
            self._remember(out)
        per_round = body.num_measurements
        logical_frame = None
        frame = self.output_frames.get(self.engine.num_measurements)
        if (not nested and frame is not None and len(frame[1]) == per_round
                and self.num_lambdas == 1 and not self.lambda_basis
                and self.num_observables == 0):
            for row in frame[2]:
                value = self.engine.logical_value(row)
                if value is not None:
                    logical_frame = (row, frozenset(value))
                    break
        iterations: List[Instructions] = []
        for _ in range(item.repeat_count):
            it: Instructions = []
            if nested:
                self._walk(body, it)
            else:
                for op in ops:
                    self._run(op, it)
            self._flush(it)
            if logical_frame is not None and not self._compress_logical_frame(logical_frame, it, per_round):
                logical_frame = None
            iterations.append(it)
        count = len(iterations)
        emitted = None
        if count and iterations[0] and all(x == iterations[0] for x in iterations):
            emitted = iterations[0]
        elif count and iterations[0] and not nested:
            prev = self._previous_round(iterations[0]) if top else None
            transformed = [self._xor_previous(iterations[0], prev, per_round)] + [
                self._xor_previous(iterations[i], iterations[i - 1], per_round) for i in range(1, count)]
            if transformed[0] is not None and all(t == transformed[0] for t in transformed):
                emitted = transformed[0]
        if emitted is not None:
            out.append(stim.CircuitRepeatBlock(count, _circuit(emitted)))
            if top:
                self._recent = []               # the last round now lives inside the loop
                self._cursor = len(out)
            return
        last_count = 0
        for instructions, run in groupby(iterations):
            last_count = sum(1 for _ in run)
            if last_count == 1:
                out.extend(instructions)
            elif instructions:
                out.append(stim.CircuitRepeatBlock(last_count, _circuit(instructions)))
        if top and last_count > 1:
            self._recent = []
            self._cursor = len(out)

    def _remember(self, out: Instructions, keep: int = 1 << 16) -> None:
        """Remember instructions emitted since the last call."""
        self._recent.extend(out[self._cursor:])
        self._cursor = len(out)
        if len(self._recent) > keep:
            del self._recent[:len(self._recent) - keep]

    def _process(self, segment: stim.Circuit, fused: Optional[stim.CircuitInstruction] = None) -> stim.Circuit:
        """Annotate a circuit segment using Stim instructions throughout.

        ``fused`` carries targets that Stim merged into the last instruction
        of the previously annotated prefix; they are processed first and
        the instruction is not re-emitted.
        """
        self.engine.ensure_qubits(segment.num_qubits)
        out: Instructions = []
        self._out, self._cursor = out, 0
        if fused is not None:
            self.engine.ensure_qubits(max(t.value for t in fused.targets_copy()) + 1)
            self._run(self._parse(fused), out, emit_instruction=False)
        self._walk(segment, out)
        self._flush(out)
        self._remember(out)
        return _circuit(out)

    def annotate(self, circuit: stim.Circuit) -> stim.Circuit:
        """One-shot annotation of a complete physical circuit."""
        return self._process(circuit)

    # ---- builder integration ----------------------------------------------------
    def sync(self, circuit: stim.Circuit) -> stim.Circuit:
        """Annotate the instructions appended since the previous sync.

        Stim fuses an appended instruction into the previous one when the gate
        matches (e.g. a data ``R`` followed by an SE block's ancilla ``R``), so
        the last consumed instruction may have grown; its new targets are
        processed as their own instruction.
        """
        fused = None
        if self._synced:
            last = circuit[self._synced - 1]
            if isinstance(last, stim.CircuitInstruction):
                ts = last.targets_copy()
                if len(ts) > self._last_targets:
                    fused = stim.CircuitInstruction(
                        last.name, ts[self._last_targets:], last.gate_args_copy(), tag=last.tag)
        if fused is None and self._synced >= len(circuit):
            return circuit
        annotated = self._process(circuit[self._synced:], fused)
        new = circuit[:self._synced] + annotated if annotated else circuit
        self._synced = len(new)
        last = new[-1] if len(new) else None
        self._last_targets = len(last.targets_copy()) if isinstance(last, stim.CircuitInstruction) else 0
        return new

    def shift_synced(self, inserted_before: int, count: int) -> None:
        """Keep the sync position valid when instructions are inserted."""
        if inserted_before < self._synced:
            self._synced += count


def annotate_circuit(circuit: stim.Circuit, *, num_logicals: int,
                     readout_hints: bool = True) -> stim.Circuit:
    """Detector/observable annotation of a physical circuit without builder hints.

    Data readouts are taken to be measurement layers of never-before-measured
    qubits that are never used afterwards.
    """
    ann = RecordTableauAnnotator(circuit.num_qubits, num_logicals=num_logicals)
    if readout_hints:
        flat = circuit.flattened()
        items = [i for i in flat if i.name not in DROP and i.name not in PASS]
        used_later, measured_before = set(), set()
        terminal = set()
        for idx in range(len(items) - 1, -1, -1):
            qs = {t.value for t in items[idx].targets_copy() if t.is_qubit_target}
            if items[idx].name in MEAS_BASIS and items[idx].name not in RESET_BASIS \
                    and not (qs & used_later):
                terminal.add(idx)
            used_later |= qs
        m = 0
        for idx, inst in enumerate(items):
            if inst.name in MEAS_BASIS:
                qs = {t.value for t in inst.targets_copy()}
                k = len(inst.targets_copy())
                if idx in terminal and not (qs & measured_before):
                    ann.mark_readout(m, k)
                measured_before |= qs
                m += k
    return ann.annotate(circuit)
