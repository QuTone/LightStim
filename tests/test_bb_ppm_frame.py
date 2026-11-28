"""Classical frame branches agree with independent physical Pauli corrections."""
import numpy as np
import pytest
import stim

from lightstim.qec_code.BB_code.ppm.frame import ClassicalPauliFrame, unpack


def _evaluate(expression, records):
    indices, constant = unpack(expression)
    return constant ^ bool(sum(records[i] for i in indices) % 2)


@pytest.mark.parametrize('seed', [13, 47, 89, 123])
def test_classical_frame_matches_corrected_state_through_cliffords_and_resets(seed):
    rng = np.random.default_rng(seed)
    physical = stim.TableauSimulator(seed=seed)
    corrected = stim.TableauSimulator(seed=seed + 1)
    physical.set_num_qubits(4)
    corrected.set_num_qubits(4)
    frame = ClassicalPauliFrame(4)
    records = []

    for _ in range(100):
        action = int(rng.integers(5))
        q = int(rng.integers(4))
        if action == 0:
            name = str(rng.choice(['H', 'S', 'S_DAG', 'H_YZ', 'SQRT_X', 'SQRT_Y']))
            gate = stim.Circuit()
            gate.append(name, [q])
            physical.do(gate)
            corrected.do(gate)
            frame.clifford(name, [q])
        elif action == 1:
            other = int(rng.choice([r for r in range(4) if r != q]))
            name = str(rng.choice(['CX', 'CY', 'CZ', 'SWAP', 'ISWAP']))
            gate = stim.Circuit()
            gate.append(name, [q, other])
            physical.do(gate)
            corrected.do(gate)
            frame.clifford(name, [q, other])
        elif action == 2:
            letter = str(rng.choice(['X', 'Y', 'Z']))
            observable = stim.PauliString(4)
            observable[q] = letter
            records.append(physical.measure_observable(observable))
            result = _evaluate(frame.measure({q: letter}), records)
            corrected.postselect_observable(observable, desired_value=result)
        elif action == 3:
            letter = str(rng.choice(['X', 'Y', 'Z']))
            observable = stim.PauliString(4)
            observable[q] = letter
            raw = physical.measure_observable(observable)
            result = raw ^ _evaluate(frame.anticommutes({q: letter}), records)
            corrected.postselect_observable(observable, desired_value=result)
            # Implement both resets independently by measurement and correction.
            correction = stim.Circuit()
            correction.append('Z' if letter == 'X' else 'X', [q])
            if raw:
                physical.do(correction)
            if result:
                corrected.do(correction)
            frame.reset([q])
        else:
            letter = str(rng.choice(['X', 'Y', 'Z']))
            controls = tuple(rng.choice(len(records), min(3, len(records)), replace=False))
            invert = bool(rng.integers(2))
            control = frame.effective_parity(controls, invert)
            if _evaluate(control, records):
                correction = stim.Circuit()
                correction.append(letter, [q])
                corrected.do(correction)
            frame.virtual_feedback({q: letter}, controls, invert)

        reconstructed = physical.copy()
        for qubit in range(4):
            for letter, expression in [('X', frame.x[qubit]), ('Z', frame.z[qubit])]:
                if _evaluate(expression, records):
                    correction = stim.Circuit()
                    correction.append(letter, [qubit])
                    reconstructed.do(correction)
        assert reconstructed.canonical_stabilizers() == corrected.canonical_stabilizers()


def test_feedback_parity_cancellation_and_reset_clear_frame():
    frame = ClassicalPauliFrame(2)
    frame.measure({0: 'Z'})
    frame.virtual_feedback({1: 'Y'}, [0, 0])
    assert frame.anticommutes({1: 'X'}) == 0
    frame.virtual_feedback({1: 'Y'}, [0], invert=True)
    assert unpack(frame.anticommutes({1: 'X'})) == ((0,), True)
    frame.reset([1])
    assert frame.anticommutes({1: 'X'}) == frame.anticommutes({1: 'Z'}) == 0
