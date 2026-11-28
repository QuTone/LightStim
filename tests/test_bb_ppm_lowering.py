"""Semantic checks of BB measurement lowering (not resource-count snapshots)."""
import itertools
import random
from collections import Counter

import numpy as np
import pytest
import stim

from lightstim.qec_code.BB_code.ppm.lowering import (
    BBPPMCompiler, Measure, PauliFrameUpdate, RecordParity, Shift,
    UnsupportedBBPPMError,
)
from lightstim.qec_code.BB_code.ppm.shift import logical_shift_tableaus


@pytest.fixture(scope="module")
def compiler():
    return BBPPMCompiler(logical_shift_tableaus())


def _letters(p):
    return "".join("IXYZ"[p[i]] for i in range(len(p)))


def _choi(n, *, consume_pivot=False, seed=0):
    """Bell references expose changes to every unmeasured data observable."""
    sim = stim.TableauSimulator(seed=seed)
    sim.set_num_qubits(2 * n)
    for q in range(n):
        if consume_pivot and q == 0:
            continue
        sim.h(q)
        sim.cx(q, n + q)
    return sim


def _execute(compiler, sim, operations, forced=None):
    records = {}
    for op in operations:
        if isinstance(op, Shift):
            sim.do_tableau(compiler.shift_maps[(op.x, op.y)],
                           list(range(compiler.num_logicals)))
        elif isinstance(op, Measure):
            p = stim.PauliString(compiler.num_logicals)
            p[compiler.pivot - 1] = op.p1
            p[compiler.partner - 1] = op.p7
            if forced is None or op.key not in forced:
                records[op.key] = sim.measure_observable(p)
            else:
                records[op.key] = forced[op.key]
                sim.postselect_observable(p, desired_value=forced[op.key])
        elif isinstance(op, PauliFrameUpdate):
            if op.parity.evaluate(records):
                sim.do_pauli_string(stim.PauliString(op.pauli))
        else:
            raise AssertionError(f"unhandled operation {op}")
    return records


def _assert_same_data_state(actual, expected, *, consumed_pivot):
    for stabilizer in expected.canonical_stabilizers():
        if consumed_pivot and stabilizer[0] != 0:
            continue
        assert actual.peek_observable_expectation(stabilizer) == 1, stabilizer


def _check_plan(compiler, target, seed=0):
    plan = compiler.compile(target, prefix=f"sample{seed}")
    actual = _choi(compiler.num_logicals, consume_pivot=plan.consumes_pivot, seed=seed)
    expected = actual.copy()
    records = _execute(compiler, actual, plan.operations)
    result = plan.output.evaluate(records)
    expected.postselect_observable(stim.PauliString(target), desired_value=result)
    _assert_same_data_state(actual, expected, consumed_pivot=plan.consumes_pivot)
    return plan


def test_record_parity_is_xor_not_set_union():
    parity = RecordParity(("a", "b", "a"), True)
    assert parity == RecordParity(("b",), True)
    assert parity.evaluate({"b": False})
    assert not (parity ^ RecordParity(("b",), True)).evaluate({})


def test_shift_direction_and_y_signs_for_every_native(compiler):
    assert len(compiler.native_measurements) == 540
    assert len(compiler.shift_maps) == 36
    for i, native in enumerate(compiler.native_measurements.values()):
        plan = _check_plan(compiler, native.pauli, seed=i)
        assert not plan.consumes_pivot
        assert sum(isinstance(op, Measure) for op in plan.operations) == 1
        assert not any(isinstance(op, PauliFrameUpdate) for op in plan.operations)


@pytest.mark.parametrize("q", "XYZ")
@pytest.mark.parametrize("direction", (-1, 1))
def test_rotation_projector_identity_all_eight_branches(q, direction):
    """Check the Kraus operator, including signs, on a three-data-qubit R."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.diag([1, -1]).astype(complex)
    matrices = dict(I=I, X=X, Y=Y, Z=Z)
    a, b = {"X": ("Z", "Y"), "Y": ("X", "Z"), "Z": ("Y", "X")}[q]
    R = np.kron(np.kron(X, Y), Z)
    eye = np.eye(8)
    U = (eye + 1j * direction * R) / np.sqrt(2)
    for aa, mm, ss in itertools.product((0, 1), repeat=3):
        eig_a, vec_a = np.linalg.eigh(matrices[a])
        eig_b, vec_b = np.linalg.eigh(matrices[b])
        ket_a = vec_a[:, np.flatnonzero(np.isclose(eig_a, (-1) ** aa))[0]]
        ket_b = vec_b[:, np.flatnonzero(np.isclose(eig_b, (-1) ** ss))[0]]
        K = ((ket_b.conj() @ ket_a) * eye +
             (-1) ** mm * (ket_b.conj() @ matrices[q] @ ket_a) * R) / 2
        correction = aa ^ mm ^ ss ^ (direction == 1)
        if correction:
            K = R @ K
        phase = np.trace(U.conj().T @ K) / 8
        assert abs(phase) > 0
        np.testing.assert_allclose(K, phase * U, atol=1e-12)


@pytest.mark.parametrize("direction", (-1, 1))
def test_actual_rotation_operations_match_clifford_on_entangled_data(compiler, direction):
    # Exercise all three pivot bases chosen by the real native-table provider.
    examples = {}
    for axis, measurement in compiler.native_rotations.items():
        examples.setdefault(measurement.pauli[0], axis)
    assert set(examples) == set("XYZ")
    n = compiler.num_logicals
    for q, axis in examples.items():
        R = stim.PauliString(axis)
        ops = compiler.rotation_operations(axis, direction=direction)
        for seed in range(8):
            actual = _choi(n, consume_pivot=True, seed=seed)
            original = actual.canonical_stabilizers()
            _execute(compiler, actual, ops)
            long_axis = R + stim.PauliString(n)
            for p in original:
                if p[0] != 0:
                    continue
                transformed = p if p.commutes(long_axis) else (long_axis * p) * (1j * direction)
                assert actual.peek_observable_expectation(transformed) == 1


def test_example_uses_multiple_rotations_and_uncomputes_in_reverse_time_order(compiler):
    target = "IXIIII IXXIII".replace(" ", "")
    plan = _check_plan(compiler, target, seed=42)
    assert len(plan.rotations) >= 2
    assert plan.consumes_pivot
    assert all(axis[0] == "I" for axis in plan.rotations)
    # Different outcomes on a Choi state test both measurement eigenspaces.
    for seed in range(8):
        _check_plan(compiler, target, seed=seed)


@pytest.mark.parametrize("sign", ("", "-"))
def test_mixed_pauli_example_records_one_signed_rotation_and_correct_projector(sign):
    # A single shared plan connects the notebook's derivation to its physical
    # experiment. Check the actual projector on entangled reference data too.
    # The notebook starts with a fresh compiler; cached seed frontiers can
    # select a different equally valid path for this target.
    compiler = BBPPMCompiler(logical_shift_tableaus())
    target = sign + "IXIIIIZIIIIX"  # +/- X2 Z7 X12
    plan = _check_plan(compiler, target, seed=29)
    assert len(plan.path) == 1
    step = plan.path[0]
    assert (step.before, step.axis, step.after, step.direction) == (
        (sign or "+") + "IXIIIIZIIIIX", "IXIIIIXIIIIX",
        (sign or "+") + "IIIIIIYIIIII", 1)
    assert plan.rotations == (step.axis,)
    assert plan.base_measurement == "IIIIIIYIIIII"  # Direct Y7 endpoint.
    assert Counter(type(op).__name__ for op in plan.operations) == {
        "Measure": 7, "Shift": 4, "PauliFrameUpdate": 2}
    assert plan.output == RecordParity(("sample29.result",), bool(sign))
    assert Measure("I", "Y", "sample29.result") in plan.operations
    assert not any(isinstance(op, Measure) and op.key == "sample29.prepare"
                   for op in plan.operations)


@pytest.mark.parametrize("sign", ("", "-"))
def test_synthesized_shifted_native_endpoint_preserves_projector_and_sign(compiler, sign):
    fresh = BBPPMCompiler(compiler.shift_maps)
    # This non-native target reaches +/- X4 X6, whose native measurement
    # requires an inverse shift, one basic X1 readout, and a restoring shift.
    target = sign + "IXXXIXZXZYXY"
    assert target.lstrip("-") not in fresh.native_measurements
    plan = _check_plan(fresh, target, seed=41)
    assert plan.path and plan.consumes_pivot
    assert _letters(stim.PauliString(plan.path[-1].after)) == "IIIXIXIIIIII"
    assert plan.base_measurement == "IIIXIXIIIIII"
    native = fresh.native_measurements[plan.base_measurement]
    assert native.shift != fresh.identity_shift
    measurement = Measure(native.p1, native.p7, "sample41.result")
    index = plan.operations.index(measurement)
    assert plan.operations[index - 1:index + 2] == (
        Shift(*fresh.inverse_shifts[native.shift]), measurement, Shift(*native.shift))
    assert plan.output.keys == ("sample41.result",)
    assert not any(isinstance(op, Measure) and op.key == "sample41.prepare"
                   for op in plan.operations)


@pytest.mark.parametrize("sign", ("", "-"))
def test_non_native_endpoint_keeps_prepared_pivot_measurement(compiler, sign):
    fresh = BBPPMCompiler(compiler.shift_maps)
    endpoint = "IIIXIXXXIXII"  # X4 X6 X7 X8 X10, available only as Q1 R.
    assert endpoint in fresh.native_rotations
    assert endpoint not in fresh.native_measurements
    plan = _check_plan(fresh, sign + endpoint, seed=57)
    assert plan.path == () and plan.consumes_pivot
    assert plan.base_measurement == "XIIXIXXXIXII"
    assert plan.output == RecordParity(("sample57.prepare", "sample57.result"), bool(sign))
    assert [op.key for op in plan.operations if isinstance(op, Measure)] == [
        "sample57.prepare", "sample57.result"]


def test_mixed_pauli_projector_remains_correct_with_a_warmed_search_cache(compiler):
    compiler.compile("IXIIIIIXXIII")
    plan = _check_plan(compiler, "IXIIIIZIIIIX", seed=73)
    assert len(plan.path) == len(plan.rotations)
    # Cache history may choose another axis/base, so only assert the actual
    # signed path agrees with the selected native data measurement.
    final = stim.PauliString(plan.path[-1].after)
    native_data = stim.PauliString(plan.base_measurement)
    native_data[compiler.pivot - 1] = 'I'
    assert _letters(final) == _letters(native_data)


@pytest.mark.parametrize("sign", ("", "-"))
def test_signed_path_matches_the_conjugations_and_preserves_requested_sign(compiler, sign):
    target = sign + "IXIIIIIXXIII"  # X2 X8 X9 has a negative final seed.
    plan = compiler.compile(target)
    assert stim.PauliString(plan.target) == stim.PauliString(target)
    before = stim.PauliString(target)
    assert tuple(step.axis for step in plan.path) == plan.rotations
    for step in plan.path:
        assert stim.PauliString(step.before) == before
        axis = stim.PauliString(step.axis)
        assert not axis.commutes(before)
        after = (axis * before) * (1j * step.direction)
        assert stim.PauliString(step.after) == after
        before = after
    assert before == stim.PauliString(("+" if sign else "-") + "IXXXIIIIIIII")
    assert plan.output.constant == (sign == "")


def test_direct_native_plan_has_no_rotation_path_and_keeps_negative_target(compiler):
    plan = compiler.compile("-XIIIIIIIIIII")
    assert plan.target == "-XIIIIIIIIIII"
    assert plan.path == plan.rotations == ()
    assert plan.output.constant


def test_random_data_ppms_preserve_full_projector_instrument(compiler):
    rng = random.Random(25603094)
    for i in range(32):
        target = "I" + "".join(rng.choice("IXYZ") for _ in range(11))
        if i % 3 == 0:
            target = "-" + target
        _check_plan(compiler, target, seed=i)


def test_noncommuting_sequence_retains_correct_measurement_backaction(compiler):
    # These anticommute. They must not be combined as multiplication of outcomes.
    targets = ("IXIIII IXXIII".replace(" ", ""), "IZIIII IXXIII".replace(" ", ""))
    assert not stim.PauliString(targets[0]).commutes(stim.PauliString(targets[1]))
    actual = _choi(compiler.num_logicals, consume_pivot=True, seed=17)
    expected = actual.copy()
    seen_keys = set()
    for i, target in enumerate(targets):
        plan = compiler.compile(target, prefix=f"step{i}")
        keys = {op.key for op in plan.operations if isinstance(op, Measure)}
        assert not (keys & seen_keys)
        seen_keys.update(keys)
        records = _execute(compiler, actual, plan.operations)
        expected.postselect_observable(stim.PauliString(target),
                                       desired_value=plan.output.evaluate(records))
        _assert_same_data_state(actual, expected, consumed_pivot=True)


def test_unknown_pivot_input_is_rejected_only_when_general_synthesis_is_needed(compiler):
    # Direct native measurements preserve the usual projection semantics on all12.
    assert not compiler.compile(((1, "Y"), (7, "X"))).consumes_pivot
    candidate = next("X" + "".join(t) for t in itertools.product("IXYZ", repeat=11)
                     if "X" + "".join(t) not in compiler.native_measurements)
    with pytest.raises(UnsupportedBBPPMError, match="unknown input"):
        compiler.compile(candidate)


def test_input_validation_and_identity(compiler):
    with pytest.raises(ValueError, match="distinct"):
        compiler.compile(((2, "X"), (2, "Z")))
    with pytest.raises(ValueError, match="in range"):
        compiler.compile(((0, "X"),))
    with pytest.raises(ValueError, match="Hermitian"):
        compiler.compile("i" + "I" * 12)
    identity = compiler.compile("-" + "I" * 12)
    assert not identity.operations
    assert identity.output.evaluate({})


def test_incomplete_native_set_reports_unreachable_target():
    compiler = BBPPMCompiler({(0, 0): stim.Tableau(3)}, partner=2)
    with pytest.raises(UnsupportedBBPPMError, match="do not connect"):
        compiler.compile("IIX")

def test_available_axes_generate_full_data_clifford_group(compiler):
    # Constructive sufficient generating set: X_i and Z_i quarter-turns give
    # all local Cliffords; nearest-neighbor Z_i Z_(i+1) quarter-turns, with
    # local Z corrections, give a connected chain of CZs. Their conjugation
    # syntheses below therefore prove universality, not just random coverage.
    from lightstim.qec_code.BB_code.ppm.lowering import _decode, _encode
    n = compiler.num_logicals
    required = []
    for q in range(1, n):
        for letter in "XZ":
            p = ["I"] * n
            p[q] = letter
            required.append("".join(p))
    for q in range(1, n - 1):
        p = ["I"] * n
        p[q] = p[q + 1] = "Z"
        required.append("".join(p))
    for target in required:
        path, seed = compiler._find_path(_encode(target))
        transformed = stim.PauliString(target)
        for word in path:
            axis_name = _decode(word, n)
            assert axis_name in compiler.native_rotations
            axis = stim.PauliString(axis_name)
            assert not transformed.commutes(axis)
            transformed = (axis * transformed) * 1j
        assert _letters(transformed) == _decode(seed, n)
        assert _letters(transformed) in compiler.native_rotations


def _unsigned_rotation_neighbors(vertices, axes):
    """Independent Pauli-algebra oracle for reported graph cardinalities."""
    neighbors = set()
    for name in vertices:
        pauli = stim.PauliString(name)
        for axis in axes:
            if not axis.commutes(pauli):
                neighbors.add(_letters(axis * pauli))
    return neighbors


def test_search_trace_counts_complete_first_layer_not_only_selected_path(compiler):
    fresh = BBPPMCompiler(compiler.shift_maps)
    target = "IXIIIIZIIIIX"
    plan = fresh.compile(target)
    assert plan.search_status == "searched"
    initial, expanded = plan.search_trace
    assert (initial.round, initial.expanded_side) == (0, "initial")
    assert (initial.target_frontier, initial.target_new, initial.target_total) == (1, 1, 1)
    assert (initial.native_frontier, initial.native_new, initial.native_total) == (245, 245, 245)
    assert not initial.connected and initial.meeting_pauli is None

    axes = [stim.PauliString(name) for name in fresh.native_rotations]
    neighbors = _unsigned_rotation_neighbors({target}, axes) - {target}
    assert len(neighbors) > len(plan.path) == 1
    assert (expanded.round, expanded.expanded_side) == (1, "target")
    assert expanded.target_frontier == expanded.target_new == len(neighbors)
    assert expanded.target_total == 1 + len(neighbors)
    assert expanded.native_frontier == expanded.native_total == 245
    assert expanded.native_new == 0
    assert expanded.connected
    assert expanded.meeting_pauli in neighbors & set(fresh.native_rotations)
    assert expanded.meeting_pauli == _letters(stim.PauliString(plan.path[-1].after))


def test_search_trace_both_sides_cache_and_warmed_initialization(compiler):
    fresh = BBPPMCompiler(compiler.shift_maps)
    target = "IXIIIIIXXIII"
    plan = fresh.compile(target, prefix="trace")
    assert plan.search_status == "searched"
    trace = plan.search_trace
    assert {row.expanded_side for row in trace[1:]} == {"target", "native"}
    assert [row.round for row in trace] == list(range(len(trace)))
    assert all(not row.connected for row in trace[:-1])
    assert trace[-1].connected

    # Use Stim multiplication, rather than the compiler's packed-bit neighbor
    # function, to count distinct Pauli nodes and their full layer boundaries.
    axes = [stim.PauliString(name) for name in fresh.native_rotations]
    seen = {"target": {target}, "native": set(fresh.native_rotations)}
    frontier = {side: set(nodes) for side, nodes in seen.items()}
    for row in trace[1:]:
        side = row.expanded_side
        other = "native" if side == "target" else "target"
        added = _unsigned_rotation_neighbors(frontier[side], axes) - seen[side]
        seen[side].update(added)
        frontier[side] = added
        assert getattr(row, f"{side}_new") == len(added)
        assert getattr(row, f"{other}_new") == 0
        for endpoint in ("target", "native"):
            assert getattr(row, f"{endpoint}_total") == len(seen[endpoint])
            assert getattr(row, f"{endpoint}_frontier") == len(frontier[endpoint])
        assert row.connected == bool(seen["target"] & seen["native"])
    assert trace[-1].meeting_pauli in seen["target"] & seen["native"]
    path_nodes = {target} | {_letters(stim.PauliString(edge.after)) for edge in plan.path}
    assert trace[-1].meeting_pauli in path_nodes

    repeated = fresh.compile(target, prefix="trace")
    assert repeated.search_status == "cached"
    assert repeated.search_trace == trace
    assert repeated.path == plan.path and repeated.operations == plan.operations
    assert repeated.output == plan.output

    native_total, native_frontier = len(fresh._seed_parents), len(fresh._seed_frontier)
    assert native_total > len(fresh.native_rotations)
    warmed = fresh.compile("IXIIIIZIIIIX")
    assert warmed.search_status == "searched"
    assert warmed.search_trace[0].native_total == native_total
    assert warmed.search_trace[0].native_frontier == native_frontier
    assert warmed.search_trace[0].target_total == 1


@pytest.mark.parametrize("target,status", [("-XIIIIIIIIIII", "native"), ("I" * 12, "identity")])
def test_direct_measurements_do_not_invent_search_expansions(compiler, target, status):
    plan = compiler.compile(target)
    assert plan.search_status == status
    assert plan.search_trace == ()
