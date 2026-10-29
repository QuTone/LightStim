"""Bounded correctness checks; no production Monte Carlo sweep in CI."""
import pytest
import stim

from examples.quantinuum_color_code_memory.circuit import (
    NOISE_MODELS, build_circuit, noise_config,
)
from tests.conftest import importorskip_safe


def _rank(rows):
    pivots = {}
    for row in rows:
        while row:
            pivot = row.bit_length() - 1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                break
    return len(pivots)


def _annotation_rows(circuit):
    measurement = 0
    rows, observable = [], 0
    for inst in circuit.flattened():
        if inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            row = 0
            for target in inst.targets_copy():
                row ^= 1 << (measurement + target.value)
            if inst.name == "DETECTOR":
                rows.append(row)
            else:
                observable ^= row
        else:
            one = stim.Circuit()
            one.append(inst)
            measurement += one.num_measurements
    return rows + [observable]


@pytest.mark.parametrize("basis", "XYZ")
@pytest.mark.parametrize("sign", [1, -1])
def test_six_states_have_complete_deterministic_annotations(basis, sign):
    circuit = build_circuit(basis=basis, sign=sign)
    assert (circuit.num_qubits, circuit.num_measurements,
            circuit.num_detectors, circuit.num_observables) == (13, 133, 126, 1)
    dets, obs = circuit.compile_detector_sampler(seed=42).sample(32, separate_observables=True)
    assert not dets.any() and not obs.any()
    flows = [sum(1 << m for m in flow.measurements_copy())
             for flow in circuit.flow_generators()
             if flow.input_copy().weight == flow.output_copy().weight == 0]
    annotations = _annotation_rows(circuit)
    assert _rank(flows) == _rank(annotations) == _rank(flows + annotations) == 127
    from lightstim.simulation.decoder_backend.post_select import get_post_select_detector_indices
    assert get_post_select_detector_indices(circuit) == []


@pytest.mark.parametrize("model", NOISE_MODELS)
def test_noise_does_not_turn_virtual_swaps_into_noisy_gates(model):
    clean = build_circuit()
    noisy = build_circuit(noise_model=model, noise=noise_config(model))
    physical_pairs = sum(len(i.targets_copy()) // 2 for i in clean
                         if i.name in ("CX", "CZ"))
    noisy_pairs = sum(len(i.targets_copy()) // 2 for i in noisy if i.name == "DEPOLARIZE2")
    assert noisy_pairs == physical_pairs == 18 + 20 * 26
    assert all(i.tag == "noiseless" for i in noisy if i.name == "SWAP")
    assert noisy.without_noise() == clean
    assert noisy.detector_error_model().num_observables == 1


def test_noise_models_have_distinct_idle_semantics():
    standard = build_circuit(noise=noise_config("circuit_level"))
    idle = build_circuit(noise_model="circuit_level_with_idling",
                         noise=noise_config("circuit_level_with_idling"))
    count = lambda c: sum(len(i.targets_copy()) for i in c if i.name == "DEPOLARIZE1")
    assert count(idle) > count(standard)
    zero_idle = build_circuit(noise_model="circuit_level_with_idling",
                             noise=noise_config("circuit_level"))
    assert zero_idle == standard


@pytest.mark.parametrize("basis,expected_distance", [("X", 3), ("Y", 2), ("Z", 3)])
@pytest.mark.parametrize("model", NOISE_MODELS)
def test_fault_distance_and_documented_y_boundary(basis, expected_distance, model):
    circuit = build_circuit(basis=basis, noise_model=model, noise=noise_config(model))
    signatures = {}
    ambiguous = False
    for inst in circuit.detector_error_model().flattened():
        if inst.type != "error" or not inst.args_copy()[0]:
            continue
        dets, logical = set(), set()
        for target in inst.targets_copy():
            if target.is_relative_detector_id():
                dets.symmetric_difference_update([target.val])
            elif target.is_logical_observable_id():
                logical.symmetric_difference_update([target.val])
        assert dets or not logical, "Undetected single DEM fault"
        dets, logical = frozenset(dets), frozenset(logical)
        ambiguous |= dets in signatures and signatures[dets] != logical
        signatures[dets] = logical
    assert ambiguous == (basis == "Y")
    witness = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=6,
        dont_explore_edges_increasing_symptom_degree=False,
    )
    assert len(witness) == expected_distance


def test_tesseract_pipeline_smoke():
    importorskip_safe("tesseract_decoder")
    from examples.quantinuum_color_code_memory.run import run_case
    result = run_case(rounds=3, shots=16, p=0)
    assert result["shots"] == result["accepted"] == 16
    assert result["errors"] == 0


def test_plot_requires_all_six_states():
    pytest.importorskip("matplotlib")
    from examples.quantinuum_color_code_memory.plot import summarize
    with pytest.raises(ValueError, match="six-state average"):
        summarize([])
