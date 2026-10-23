"""Frozen #124 semantic oracles for replacement native role algebra.

The expected digests were acquired with b996a350d501a0e7d1a674c72a897dbc7d9f4137,
LIGHTSTIM_PAULI_BACKEND=reference, and C++ GF(2) RREF. They encode *all*
returned rows, source lineages, record parities, role indices, and PS indices.
Unlike a comparison between two new code paths, these constants cannot change
silently when both backends accidentally adopt a different semantic policy.
"""

import dataclasses
import copy
from contextlib import ExitStack
from functools import wraps
from unittest.mock import patch
import hashlib
import json

import numpy as np
import pytest
import stim

from lightstim.ir.tracker_algebra import Rows, classify_stabilizers, readout_closures
from lightstim.ir.tracker import SyndromeTracker


def _plain(value):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "entries": value.tolist()}
    if dataclasses.is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _case(seed, n):
    rng = np.random.default_rng(seed)
    count = 4 + seed % 4
    # S is a physical commuting space; K need not commute internally or with S.
    # Give K two expressly anticommuting rows, alongside random directions.
    z = rng.integers(0, 2, size=(count, n), dtype=np.uint8)
    s = np.hstack([np.zeros_like(z), z])
    s = np.vstack([s, s[0], s[0] ^ s[1], np.zeros(2*n, dtype=np.uint8)])
    records = [[2*i, 2*i+1, 900, 900] for i in range(len(s))]
    records[-1] = [700, 701]  # A Pauli-zero source is not an empty expression.
    k = rng.integers(0, 2, size=(3+seed % 3, 2*n), dtype=np.uint8)
    pair = np.zeros((2, 2*n), dtype=np.uint8)
    pair[0, 0], pair[1, n] = 1, 1
    k = np.vstack([k, pair, s[seed % len(s)]])
    selected = (0, len(s)-3, len(s)-1)
    classified = classify_stabilizers(Rows(s, records), k,
                                     postselected_rows=selected,
                                     include_dependencies=True)
    live = Rows(np.vstack([s[1], s[-1]]), [[300], [301, 302]])
    m = np.vstack([s[0], s[1], s[0] ^ s[2], s[-1], s[0]])
    measurement = Rows(m, [[400+i] for i in range(len(m))])
    closed = readout_closures(
        classified.bare, classified.dressed, live, measurement,
        postselected_rows=sorted(classified.safely_postselected_output_rows),
        include_dependencies=True)
    return classified, closed


_ORACLE = {
    (0, 1): '28a1304f78418bc76e098bed9e4090a987e22a62150e550f3a6d1edb081f406a',
    (1, 1): '6eb3b5e42d2448c505d46310548e0f5ab8bf7b5acb27e2cc9280adf2334e82e9',
    (2, 1): 'be22a6eb40bddde6484013a2da7c20ae741cc8cb3219389bd8decbf5aaeed687',
    (3, 1): 'a063e4067667eeda70240244ab64aeb800624dc6203ba3d3134215395508eece',
    (4, 1): 'c24c5fe31ce98b5f0c2e2b3f14d04ec6e6bdde267cc84ac03008f4eba9d5e26b',
    (5, 1): 'a8223e9ec7f344138f8c8d47db6b2e339fa9a329362ae8698d9b1d0bab9c8d31',
    (6, 1): '7df8559f96b6a2ac5c5f114622dbc4c55ab376cea639eb7aa5052ea095efc1ee',
    (7, 1): 'c8fc48f03de7db23cba6a7a7ecae04677daa3a8feaff69c061c55f63fcaf6242',
    (0, 2): '378d5791a7712a420ebca27d033e09245d2f7910b1caf53bee98413326791e92',
    (1, 2): 'eb6cb61ff2e9391ca941947aae5c033ef4a2d9d279a4031a0630e6c1483a4a1a',
    (2, 2): 'f9dab3305a242d9efb7a3ec3122c069db3e0360466d5b602c624e82dad2d1fdb',
    (3, 2): '41fce63aab40aad03e0087049bb54594e1b379f1cca3a06935a9cf331b533877',
    (4, 2): '980f7ef2ed6c4c4316b4fc6bb0b760fb3619c043408ef6e44a6c445f1b1549a1',
    (5, 2): '8d10ac9c413bf13a18324d06509c01bbdda1035c821367a713b70ce4453835f7',
    (6, 2): '5b22ebdc6e50e9f66eb544523107781249b54a8fdee913fad5e382813eef53d3',
    (7, 2): 'f28f825e7902f71b25b3d1d3019c660ebec74f0b28fb8d3bec21b359a2ffa5a6',
    (0, 3): '41f3526711eaf466f749da69ac2e2e7a85207a4f824d3e0151f5c7863df1611f',
    (1, 3): '8e12f0c9f40c0d73a7570d465ee7b6effc978703b69b2e417705dde4d0acc07f',
    (2, 3): '129ffbed02385ac794c9d58210f74518bfc042e9a692613577ceb2f7c830bf88',
    (3, 3): '6db4c3404c17c593ea55046d97be6f1cc0044cdedf280721849a1fa539c531a9',
    (4, 3): 'd407b7a7dbd931db5521e4ee00fcaa505056fc663b1e606587ed9965ce9ffc60',
    (5, 3): '77c240d85cb7da2ff19d949191b3199a60d16c36c3d28af556a4100e0e199976',
    (6, 3): '01b315a17c81671edc721a56ddaf90fb8ade88efb7fc9767d6bcf9b652f49e78',
    (7, 3): '60948dc05fbf4e5f271c45d3caf4712b1b9abd343927d57944fd8e4c69cb7d80',
    (0, 4): 'c84c8629e2253a23de46ed49b833acc16c099e6e721f67b909f288efebb7d02b',
    (1, 4): 'e4d04608382b4556b4f8fb0feeedb370da411b61c12790fb1d88267c969019d3',
    (2, 4): '02b719e118d5ddba4be104a33b85dfbd9094f09a09882137a4af04b8c6ecccd3',
    (3, 4): 'dbb7a890ac587828238b270d11a98525d9bbe65b21fc26ac85ebff7b48040917',
    (4, 4): 'd00596fc013efc2aedef8b72b22c27fef18832d3296f19a48dda93a24cbee415',
    (5, 4): '08fc0dc2b4cee5ab06ebeb3ab01dfe8316ded4b7bbad32d9471b1372aa9d57ee',
    (6, 4): '65e01786ce9bf6e70e3f68d835b4cea869907c2f3000bb625acc88619b433ee7',
    (7, 4): '8dc71e27d7b9973a05b1db6cbc7609e7b951eeba46d4b915f394770ec0a8fb9c',
    (0, 65): 'b488b0ea1613f2b01d9dc7c98b29808d5f2c02fc02d09de7d608485e69552510',
    (1, 65): 'ebfedc63ff1400d33d66c2f368919c6aa04977ccfba21cf875eea5894a134767',
    (2, 65): '7a1620c6aa194bc4551fc5fcbb3a29c16518f427ab207c109f42e5b5ec198bd5',
    (3, 65): '342652d7e22b46b1897375b3ac0fa10d852d75e1da827c0a0d51f4edc6801d07',
    (4, 65): 'ac2b41ee1ea836fd1feca29ce1760a80d0a2944d15c6271d609523b364432f67',
    (5, 65): '6e057cef60b7d0dc5f572d566a10d2ec488c4ad8372f13a3be05f8adcd8fa225',
    (6, 65): '49e0316b01cfe91bb114ddd054ba808776f336689cf5264d4491e67e9e715d79',
    (7, 65): '30fd32b8de9fe6713f7f354e74b79d566244a9e5e6223fcc9523b97c5ff500a5',
}


@pytest.mark.parametrize("mode", ["reference", "persistent"])
@pytest.mark.parametrize("seed,n", list(_ORACLE))
def test_role_and_closure_outputs_match_frozen_latest_tracker(monkeypatch, mode, seed, n):
    if mode == "persistent":
        extension = pytest.importorskip("lightstim.utils.cpp._pauli_backend_cpp")
        if not hasattr(extension, "PersistentRecordState"):
            pytest.skip("optional native tracking extension has not been built")
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", mode)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native" if mode == "persistent" else "reference")
    actual = _case(seed, n)
    assert _digest(actual) == _ORACLE[seed, n]


def test_fixture_explicitly_exercises_noncommuting_role_space():
    # The role declaration is an arbitrary binary subspace, not a stabilizer
    # state. It is invalid to put its independent generators into a single
    # commuting inverse-tableau basis and assume that completes classification.
    s = Rows(np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.uint8), [[0], [1]])
    k = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.uint8)
    result = classify_stabilizers(s, k)
    np.testing.assert_array_equal(result.bare.matrix, s.matrix[:1])
    np.testing.assert_array_equal(result.dressed.matrix, s.matrix[1:])
    assert result.bare.records == ((0,),)
    assert result.dressed.records == ((1,),)


def _state_snapshot(tracker):
    names = ("num_qubits", "expected_num_logicals", "total_measurements",
             "total_observables", "logical_history", "_logical_history_observable_ids",
             "_classified_stabilizer_rows", "_dressed_stabilizer_rows",
             "_v2_role_space", "post_select_row_indices",
             "stabilizer_with_logical_components", "_gauge_logical_vectors")
    output = {name: _plain(getattr(tracker, name)) for name in names}
    for name in ("stabilizers", "logicals", "absorbed_ops"):
        tableau = getattr(tracker, name)
        output[name] = _plain((tableau.matrix, tableau.records))
    return output


def _mutable_state_case(seed):
    tracker = SyndromeTracker(3, logical_tracking="slabs")
    s = np.zeros((6, 6), dtype=np.uint8)
    s[:3, 3:] = np.eye(3, dtype=np.uint8)
    s[3] = s[0] ^ s[1]
    s[4] = s[0]
    tracker.stabilizers.add_stabilizers(s, [[], [-1, -1], [2], [3], [4, 4, 5], [6, 7]])
    tracker.logicals.add_stabilizers(s[[0, 2]], [[8], [9, 9, 10]])
    tracker.post_select_row_indices = {seed % 5, 5}
    tracker._classified_stabilizer_rows = {0, 2, 3, 4, 5}
    active = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, seed % 2]], dtype=np.uint8)
    tracker._v2_classify_stabilizers(active)
    trace = [_state_snapshot(tracker)]
    tracker.process_unitary_block(stim.Circuit("H 2\nCX 0 1"))
    trace.append(_state_snapshot(tracker))
    # ndarray and nested-list mutation are both public compatibility surfaces.
    # Change a real row after the native state exists, then re-enter a boundary.
    tracker.stabilizers.matrix[0] ^= tracker.logicals.matrix[0]
    tracker.stabilizers.records[0].extend([18, 18, 19])
    tracker.logicals.records[0] = [8, 20, 20]
    tracker._v2_classify_stabilizers(active)
    trace.append(_state_snapshot(tracker))
    probe = copy.deepcopy(tracker)
    probe.process_unitary_block(stim.Circuit("S 2\nCX 1 0"))
    trace.extend([_state_snapshot(probe), _state_snapshot(tracker)])
    tracker.expand(1)
    tracker.process_unitary_block(stim.Circuit("H 3\nCX 2 3"))
    trace.append(_state_snapshot(tracker))
    assert trace[2] == trace[4], "a repeat/deepcopy probe must not mutate its source"
    return trace


_STATE_ORACLE = {
    0: ('689f797397ceec66b953073605c9fd4816ac21308053be1fb3819ba93dfb5218', 'f27df3b6654e529b7296798477614d95d99cf4509e7c369acca5a9a9759a056c', 'fa2b40472cba8e6cff23b45f07110e588fe10bf44786301911d7e2cfccb940c2', '696383b6e8985466fcc12b205eac72b971efb4db571b0f840f4e0bc99514d225', 'fa2b40472cba8e6cff23b45f07110e588fe10bf44786301911d7e2cfccb940c2', '5418d0b258e53007697fb24007f9524edd3b15fe7da48ceef501dc3bfcbea0b4'),
    1: ('0343a5be158d77af13f706c088ef4bcbb52b85919e1a1289e4b08f8942c2c253', '2898ace76cda753f8bc90608cd5020acdd137e5b06d7fee5345b7891b1ec9243', 'af74616c5c6f83568a7784456ba3b3e827bf68d45e1407c781896700fe175dc3', '979a3b9b32e517bf3b572140159105b2fc5b2dc3395052c2be9e9e6793eba56a', 'af74616c5c6f83568a7784456ba3b3e827bf68d45e1407c781896700fe175dc3', 'ba435fe8f568c065d9949e01693644a5c1a0695ff0596172260fb019ca920c9a'),
    2: ('8acf68f384f6b26be816b80a5a2696494f4b3ea965d624f19849c3689811cdde', '1e5e8ad04cf696f446d836a001ff27d42328e4661d2d88df1457e1b976de7bb6', '655c4a281117e84f272521c0b8b080bd605879c9138777dde618c0c5dbdd8939', 'a13532d89de5a6124aa2448d47d2042ddde9cf38ccc9a46fd3addcd1c1024f36', '655c4a281117e84f272521c0b8b080bd605879c9138777dde618c0c5dbdd8939', '666f3538ee8230b2e9b41344bb4f051740a0788f4c35d479d8e655fa9d625a1c'),
    3: ('4cbbe4ccf91d2f76ef1c432225e0f5c40617660928a7eb895c6d12902f06d4ca', '5a18ed6f366a1cedf813beae04b12e3b4cb430d27437b8487c6d85aa781aa5c0', '06e3156e296d82957931635b9ca33fad64b10937cc16102026af215139298cd0', '3ece29be98ad591e0bb28cee46d994a2401f94eb93b0d0b03f4837d8b8a09fce', '06e3156e296d82957931635b9ca33fad64b10937cc16102026af215139298cd0', '745b660daf5d0c99fd7c8629ce36e3889c8d1850a185dd86acc1cc321a55fe83'),
    4: ('be91cdde8b6b6d45f0512159aeeb7ebd8d9ac564b4f625516bf8faa712543138', 'aed0956094209d699e8f1a3eef590745e1181655bfbde932e301507d2dc6a23e', '07cf83dcad9b94004e28ad4ecb40f2ff1c3cf39cbcf5c63786413f499fd4b97e', '152804484e079c14a5a4d32c269633552206b46a35d395c13d29022287bc0899', '07cf83dcad9b94004e28ad4ecb40f2ff1c3cf39cbcf5c63786413f499fd4b97e', '116a7f816147ce1667991554da18c0575779061c5aca52e2fb966a130454a578'),
}


@pytest.mark.parametrize("mode", ["reference", "persistent"])
@pytest.mark.parametrize("seed", list(_STATE_ORACLE))
def test_mutable_state_and_copy_boundaries_match_frozen_latest_tracker(monkeypatch, mode, seed):
    if mode == "persistent":
        extension = pytest.importorskip("lightstim.utils.cpp._pauli_backend_cpp")
        if not hasattr(extension, "PersistentRecordState"):
            pytest.skip("optional native tracking extension has not been built")
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", mode)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native" if mode == "persistent" else "reference")
    actual = _mutable_state_case(seed)
    for boundary, (snapshot, expected) in enumerate(zip(actual, _STATE_ORACLE[seed])):
        assert _digest(snapshot) == expected, f"seed {seed}, semantic boundary {boundary}"
    assert len(actual) == len(_STATE_ORACLE[seed])


def _protocol(name):
    from lightstim.ir.builder import CircuitBuilder
    from lightstim.ir.qec_system import QECSystem
    from lightstim.qec_code.bacon_shor import BaconShorCode, BaconShorCodeExtractionBlock
    from lightstim.qec_code.surface_code.unrotated import (
        UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock,
    )
    from tests.test_tracker_v2_builder_state import _postselected_transport
    from tests.test_tracker_v2_native import _two_patch_circuit

    if name == "staged_readout":
        return _two_patch_circuit("ZZ", "ZZ", 1, 2, True)
    if name == "two_merges_history":
        return _two_patch_circuit("ZZ", "ZZ", 2, 1, False)
    if name == "teleportation":
        return _two_patch_circuit("ZX", "XZ", 1, 1, False)
    if name.startswith("postselection_"):
        return _postselected_transport(4, buffered=name.endswith("buffered"))
    system = QECSystem()
    code, extraction = ((BaconShorCode, BaconShorCodeExtractionBlock)
                        if name == "noncommuting_gauges" else
                        (UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock))
    system.add_patch(code(distance=3), name="memory")
    tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    builder.initialize({q: "Z" for q in system.data_indices}, system.num_qubits)
    block = extraction(system)
    builder.apply_syndrome_extraction(block.circuit, rounds=7,
                                     measurement_blocks=getattr(block, "measurement_blocks", None))
    builder.apply_data_readout({q: "Z" for q in system.data_indices})
    return builder.to_stim_circuit()


def _protocol_trace(name):
    from lightstim.ir.builder import CircuitBuilder
    trace = []
    # Observe user-visible Builder boundaries, not implementation-dependent
    # calls within role elimination. Each snapshot includes all four banks.
    methods = ("initialize", "apply_syndrome_extraction", "apply_unitary_block",
               "apply_data_readout", "apply_staged_data_readout",
               "activate_coupler", "deactivate_coupler")
    with ExitStack() as stack:
        for method in methods:
            original = getattr(CircuitBuilder, method)

            @wraps(original)
            def boundary(builder, *args, _name=method, _original=original, **kwargs):
                result = _original(builder, *args, **kwargs)
                trace.append((_name, _state_snapshot(builder.tracker)))
                return result

            stack.enter_context(patch.object(CircuitBuilder, method, boundary))
        circuit = _protocol(name)
    return circuit, trace


_PROTOCOL_ORACLE = {
    'staged_readout': {'boundaries': 7, 'trace_sha256': '317f6f7a576fa206fd3a1da64074a1408282ca4a8658072f8259b008d827b679', 'circuit_sha256': '9f6b7ab78722fe850157c1e2d9abf8b05ff17f6204b6d4e0286b539348a6414b'},
    'two_merges_history': {'boundaries': 13, 'trace_sha256': '600203df475126de55b79dc16df6190ce2772dbca11be07b176d0c389201edda', 'circuit_sha256': '75d65e0e58092cdbc90519c907f6b749d12f25d169fa2203312a0d8cc163c9ad'},
    'teleportation': {'boundaries': 8, 'trace_sha256': '83f1a51dc46377449c89bcac510ce6b01964a93f8c577cfc286d9163ed20139a', 'circuit_sha256': 'd5ea9f0cc2b5057a38598ec353791e921707d326f1e70097847a33f7593c12b0'},
    'postselection_buffered': {'boundaries': 2, 'trace_sha256': 'ba716858021e326b7592c7ce2470cbaeaab6bd978535ac525fa3e755c87dd269', 'circuit_sha256': '048ba045a6b60a7ce8a68cb9c2ed943d6b745cbb44eab6886c2c356040221352'},
    'postselection_explicit': {'boundaries': 5, 'trace_sha256': '2d12e2bbf6929e3bcd3cac983840ab823f8f86b1554effc740c2b014b4e81fc2', 'circuit_sha256': '241829fdefa7393ac6a71f37897675418eb36a154454121778666c642a215149'},
    'noncommuting_gauges': {'boundaries': 3, 'trace_sha256': 'b495ce2d8f3582431954769ac0eedb3111f0c41b29d5a57185a64be80e8af773', 'circuit_sha256': '82213ae37fb0a2d193b5799d80bbc28d36dcc3349f5f11ab67037a3658e32fc5'},
    'steady_repeat': {'boundaries': 3, 'trace_sha256': 'cdfc7e2cb4df6e7f651b06dc43966a587e30439727ce8b63361478a18c972537', 'circuit_sha256': '27f996445d285d47ea09445ea25121e95d1074fb1eab4ca09832cc0ffb00a224'},
}


@pytest.mark.integration
@pytest.mark.parametrize("name", list(_PROTOCOL_ORACLE))
def test_protocol_boundaries_and_repeat_match_frozen_latest_tracker(monkeypatch, name):
    extension = pytest.importorskip("lightstim.utils.cpp._pauli_backend_cpp")
    if not hasattr(extension, "PersistentRecordState"):
        pytest.skip("optional native tracking extension has not been built")
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", "persistent")
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native")
    circuit, trace = _protocol_trace(name)
    expected = _PROTOCOL_ORACLE[name]
    assert len(trace) == expected["boundaries"]
    assert _digest(trace) == expected["trace_sha256"]
    assert hashlib.sha256(str(circuit).encode()).hexdigest() == expected["circuit_sha256"]
    circuit.detector_error_model()
    assert not circuit.compile_detector_sampler(seed=91).sample(32, append_observables=True).any()


def _failed_readout_snapshot():
    tracker = SyndromeTracker(2, logical_tracking="slabs")
    tracker.stabilizers.add_stabilizers(np.array([[1, 0, 0, 0]], dtype=np.uint8), [[]])
    tracker.logicals.add_stabilizers(np.array([[0, 0, 0, 1]], dtype=np.uint8), [[-1]])
    tracker._v2_rebuild_absorbed()
    tracker._v2_sync_budget()
    tracker.process_unitary_block(stim.Circuit("H 0\nH 0"))
    circuit = stim.Circuit("RX 0\nR 1\nM 0 1")
    with pytest.raises(RuntimeError, match="UNMEASURED sentinel") as error:
        tracker.process_data_measurement(
            circuit, np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.uint8),
            {0: (0, 0), 1: (1, 0)})
    return {"state": _state_snapshot(tracker), "circuit": str(circuit), "error": str(error.value)}


_FAILED_READOUT_ORACLE = '05df40a65bc598207a273ce169a37bad628ee87621953e2fd219e9b666f14734'


@pytest.mark.parametrize("mode", ["reference", "persistent"])
def test_failed_readout_retains_frozen_latest_partial_state_contract(monkeypatch, mode):
    if mode == "persistent":
        extension = pytest.importorskip("lightstim.utils.cpp._pauli_backend_cpp")
        if not hasattr(extension, "PersistentRecordState"):
            pytest.skip("optional native tracking extension has not been built")
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", mode)
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", "native" if mode == "persistent" else "reference")
    assert _digest(_failed_readout_snapshot()) == _FAILED_READOUT_ORACLE
