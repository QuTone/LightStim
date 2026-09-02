from types import SimpleNamespace

import pytest
import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector


def test_xz_biased_model_uses_z_dominant_gate_and_idle_channels():
    cfg = NoiseInjector.compute_XZ_biased_params(
        p_1q=0.01,
        p_2q=0.02,
        p_meas=0.0,
        p_reset=0.0,
        eta=0.1,
        p_idle=0.03,
    )

    assert cfg.custom_params["p_1q_z"] > cfg.custom_params["p_1q_x"]
    assert cfg.custom_params["p_2q_z"] > cfg.custom_params["p_2q_x"]
    assert cfg.custom_params["p_idle_z"] > cfg.custom_params["p_idle_x"]

    circuit = stim.Circuit(
        """
        R 0
        TICK[SE_start]
        H 0
        """
    )
    noisy = NoiseInjector.from_XZ_biased(cfg, [0]).inject_noise(circuit)

    assert not any(inst.name == "DEPOLARIZE1" for inst in noisy)
    assert any(
        inst.name == "X_ERROR"
        and inst.gate_args_copy()[0] == pytest.approx(cfg.custom_params["p_idle_x"])
        for inst in noisy
    )
    assert any(
        inst.name == "Z_ERROR"
        and inst.gate_args_copy()[0] == pytest.approx(cfg.custom_params["p_idle_z"])
        for inst in noisy
    )


def test_uniform_circuit_level_adds_idle_noise_to_each_operated_moment():
    config = NoiseConfig(
        p_1q=0.01,
        p_2q=0.02,
        p_meas=0.03,
        p_reset=0.04,
        p_idle=0.05,
    )
    clean = stim.Circuit(
        """
        R 0 1 2
        TICK
        H 0
        TICK
        CX 0 1
        TICK
        M 0
        TICK
        H[noiseless] 1
        TICK
        X 0
        """
    )

    noisy = NoiseInjector.from_uniform_circuit_level(
        config,
        [0, 1, 2],
    ).inject_noise(clean)

    assert noisy == stim.Circuit(
        """
        R 0 1 2
        X_ERROR(0.04) 0 1 2
        TICK
        H 0
        DEPOLARIZE1(0.01) 0
        DEPOLARIZE1(0.05) 1 2
        TICK
        CX 0 1
        DEPOLARIZE2(0.02) 0 1
        DEPOLARIZE1(0.05) 2
        TICK
        X_ERROR(0.03) 0
        M 0
        DEPOLARIZE1(0.05) 1 2
        TICK
        H[noiseless] 1
        TICK
        X 0
        DEPOLARIZE1(0.01) 0
        DEPOLARIZE1(0.05) 1 2
        """
    )
    assert noisy.without_noise() == clean


def test_uniform_idling_uses_all_operations_in_a_moment():
    clean = stim.Circuit(
        """
        R 0 1 2
        TICK
        H 0
        X 1
        TICK
        M 0
        R 0
        TICK
        MPAD 0
        """
    )
    noisy = NoiseInjector.from_uniform_circuit_level(
        NoiseConfig(p_idle=0.125),
        [0, 1, 2],
    ).inject_noise(clean)

    idle_errors = [
        instruction
        for instruction in noisy
        if instruction.name == "DEPOLARIZE1"
    ]
    assert [
        [target.value for target in instruction.targets_copy()]
        for instruction in idle_errors
    ] == [[2], [1, 2]]
    assert noisy.without_noise() == clean


def test_uniform_circuit_level_recurses_into_repeat_blocks():
    clean = stim.Circuit(
        """
        R 0 1
        TICK
        REPEAT 2 {
            H 0
            TICK
        }
        """
    )
    noisy = NoiseInjector.from_uniform_circuit_level(
        NoiseConfig(p_idle=0.125),
        [0, 1],
    ).inject_noise(clean)

    assert noisy == stim.Circuit(
        """
        R 0 1
        TICK
        REPEAT 2 {
            H 0
            DEPOLARIZE1(0.125) 1
            TICK
        }
        """
    )
    assert noisy.without_noise() == clean


def test_uniform_repeat_boundaries_match_flattened_circuit():
    clean = stim.Circuit(
        """
        R 0 1 2
        TICK
        M 0
        REPEAT 2 {
            TICK
            R 0
        }
        TICK
        H 0
        """
    )
    config = NoiseConfig(p_idle=0.125)

    repeated = NoiseInjector.from_uniform_circuit_level(
        config,
        [0, 1, 2],
    ).inject_noise(clean)
    flattened = NoiseInjector.from_uniform_circuit_level(
        config,
        [0, 1, 2],
    ).inject_noise(clean.flattened())

    assert repeated.flattened() == flattened


def test_uniform_repeat_without_tick_falls_back_to_exact_flattened_semantics():
    clean = stim.Circuit(
        """
        R 0 1
        TICK
        REPEAT 2 {
            H 0
        }
        TICK
        """
    )
    config = NoiseConfig(p_idle=0.125)

    repeated = NoiseInjector.from_uniform_circuit_level(
        config,
        [0, 1],
    ).inject_noise(clean)
    flattened = NoiseInjector.from_uniform_circuit_level(
        config,
        [0, 1],
    ).inject_noise(clean.flattened())

    assert repeated == flattened
    assert repeated.without_noise() == clean.flattened()
    assert sum(
        instruction.name == "DEPOLARIZE1" for instruction in repeated
    ) == 1


def test_uniform_zero_idle_preserves_nested_repeat_blocks():
    clean = stim.Circuit(
        """
        REPEAT 2 {
            REPEAT 3 {
                H 0
                TICK
            }
        }
        """
    )

    noisy = NoiseInjector.from_uniform_circuit_level(
        NoiseConfig(p_1q=0.125, p_idle=0),
        [0],
    ).inject_noise(clean)

    assert isinstance(noisy[0], stim.CircuitRepeatBlock)
    outer_body = noisy[0].body_copy()
    assert isinstance(outer_body[0], stim.CircuitRepeatBlock)
    assert noisy.without_noise() == clean


def test_uniform_nested_repeat_falls_back_to_exact_flattened_semantics():
    clean = stim.Circuit(
        """
        R 0 1
        TICK
        REPEAT 2 {
            REPEAT 3 {
                H 0
                TICK
            }
        }
        """
    )
    config = NoiseConfig(p_idle=0.125)

    nested = NoiseInjector.from_uniform_circuit_level(
        config,
        [0, 1],
    ).inject_noise(clean)
    flattened = NoiseInjector.from_uniform_circuit_level(
        config,
        [0, 1],
    ).inject_noise(clean.flattened())

    assert nested == flattened
    assert nested.without_noise() == clean.flattened()


def test_uniform_idling_includes_qubits_measured_in_an_earlier_moment():
    clean = stim.Circuit(
        """
        R 0 1
        TICK
        M 0
        TICK
        H 1
        """
    )

    noisy = NoiseInjector.from_uniform_circuit_level(
        NoiseConfig(p_idle=0.125),
        [0, 1],
    ).inject_noise(clean)

    idle_errors = [
        instruction
        for instruction in noisy
        if instruction.name == "DEPOLARIZE1"
    ]
    assert [
        [target.value for target in instruction.targets_copy()]
        for instruction in idle_errors
    ] == [[1], [0]]


def test_uniform_circuit_level_noises_empty_tick_moments():
    clean = stim.Circuit(
        """
        R 0 1
        TICK
        TICK
        TICK[noiseless]
        """
    )
    noisy = NoiseInjector.from_uniform_circuit_level(
        NoiseConfig(p_idle=0.125),
        [0, 1],
    ).inject_noise(clean)

    assert noisy == stim.Circuit(
        """
        R 0 1
        TICK
        DEPOLARIZE1(0.125) 0 1
        TICK
        TICK[noiseless]
        """
    )


def test_builder_uniform_circuit_level_targets_non_data_qubits():
    # The builder's historical factory argument contains data qubits only.
    # Uniform idling instead needs every physical qubit in the circuit.
    system = SimpleNamespace(
        data_coords=[(0, 0)],
        index_map={(0, 0): 0},
    )
    builder = CircuitBuilder(tracker=None, system_config=system)
    builder.circuit = stim.Circuit(
        """
        R 0 1
        TICK
        H 0
        """
    )

    noisy = builder.build_noisy_circuit(
        NoiseConfig(p_idle=0.125),
        noise_model="uniform_circuit_level",
    )

    assert any(
        instruction.name == "DEPOLARIZE1"
        and [target.value for target in instruction.targets_copy()] == [1]
        for instruction in noisy
    )
