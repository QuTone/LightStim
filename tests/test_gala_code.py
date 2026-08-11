import numpy as np
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.gala_code import (
    GALA_CODE_PRESETS,
    GalaCode,
    GalaCodeExtractionBlock,
)
from lightstim.qec_code.gala_code.group import LiftAlphabet, commutes, compose, invert, s3, s4
from lightstim.simulation.decoder_backend import DecoderConfig, SimulationPipeline


def test_gala_presets_replicate_published_nk():
    """Every preset reproduces the (n, k) published in arXiv:2608.07431."""
    for name, preset in GALA_CODE_PRESETS.items():
        code = GalaCode.from_preset(name)

        assert code.n_data == preset["expected_n"], name
        assert code.num_logicals == preset["expected_k"], name
        assert code.n_data == code.L * code.lift_size, name


def test_gala_presets_are_valid_css_codes():
    """H_X H_Z^T = 0 for every preset — the authoritative orthogonality test."""
    for name in GALA_CODE_PRESETS:
        code = GalaCode.from_preset(name, compute_k=False)

        assert code.check_css_orthogonality(), name


def test_gala_zx_dual_presets_satisfy_strict_definition_11():
    """The Table S4 (ZX-dual) instances meet the per-pair Gamma_J criterion."""
    strict = ["gala_132_30_12", "gala_192_40_12", "gala_228_46_12",
              "gala_136_36_8", "gala_136_34_12", "gala_168_42_12",
              "gala_576_104_12"]
    for name in strict:
        code = GalaCode.from_preset(name, compute_k=False)

        assert code.validate_required_commutativity(), name


def test_gala_rate_half_presets_rely_on_psi_cancellation():
    """Rate-1/2 instances are valid CSS codes without the strict per-pair rule.

    Orthogonality needs only the sums Psi_r to vanish; in gala_576_292_8 the
    commutators [F_1, G_5] and [F_2, G_4] are individually non-zero and cancel
    each other inside Psi_0. Guards against "fixing" the construction to the
    stricter criterion, which would reject these published codes.
    """
    code = GalaCode.from_preset("gala_576_292_8", compute_k=False)

    assert not code.validate_required_commutativity()
    assert code.check_css_orthogonality()


def test_gala_132_is_the_compact_self_dual_instance():
    """The headline [[132,30,12]] code: L=12, J=5, purely cyclic C_11 lift."""
    code = GalaCode.from_preset("gala_132_30_12")

    assert (code.L, code.J, code.degree, code.cyclic) == (12, 5, 1, (11,))
    assert (code.n_data, code.num_logicals) == (132, 30)
    assert code.stabilizer_weight == 12
    hx, hz = code.get_css_matrices()
    assert hx.shape == (5 * 11, 132) and hz.shape == (5 * 11, 132)
    # Monomial lift: every check has exactly one term per block.
    assert np.all(hx.sum(axis=1) == 12) and np.all(hz.sum(axis=1) == 12)


def test_gala_polynomial_lift_has_summed_entries():
    """Polynomial lifts sum several group elements into one lift entry."""
    code = GalaCode.from_preset("gala_136_36_8", compute_k=False)

    assert any(len(entry) > 1 for entry in code.f)
    assert code.stabilizer_weight == 12       # sum_i (|F_i| + |G_i|), not L
    assert code.stabilizer_weight != code.L


def test_gala_rate_matches_j_over_l():
    """Rate-1/2 instances use J = L/4; ZX-dual instances use larger J."""
    rate_half = GalaCode.from_preset("gala_576_292_8", compute_k=False)
    assert rate_half.J * 4 == rate_half.L

    zx_dual = GalaCode.from_preset("gala_132_30_12", compute_k=False)
    assert zx_dual.J * 4 > zx_dual.L      # J > L/4 => rate < 1/2


# --------------------------------------------------------------------------- #
# Group lift arithmetic
# --------------------------------------------------------------------------- #


def test_s3_labels_match_paper_cycles():
    """sigma/tau labels of Sec. S2: sigma_0=(0 1 2), tau_0=(1 2), etc."""
    assert s3("e") == (0, 1, 2)
    assert s3("s0") == (1, 2, 0)     # (0 1 2)
    assert s3("s1") == (2, 0, 1)     # (0 2 1)
    assert s3("t0") == (0, 2, 1)     # (1 2)
    assert compose(s3("s0"), s3("s1")) == s3("e")
    assert not commutes(s3("s0"), s3("t0"))


def test_s4_generator_words_and_exponents():
    """S_4 words in <a,b,c>; a digit repeats the preceding generator."""
    assert s4("e") == (0, 1, 2, 3)
    assert s4("a") == (0, 1, 3, 2)          # (2 3)
    assert s4("c2") == compose(s4("c"), s4("c"))
    assert s4("aca") == compose(compose(s4("a"), s4("c")), s4("a"))
    with pytest.raises(ValueError, match="Unknown generator"):
        s4("z")


def test_lift_alphabet_acts_on_product_of_cyclic_factors():
    """C_3 x C_7 shifts each component independently; inverse undoes it."""
    alpha = LiftAlphabet(degree=3, cyclic=(3, 7))
    assert alpha.m == 21 and alpha.size == 63

    mono = (s3("s0"), (1, 2))
    moved = alpha.apply(mono, 0)
    assert moved == alpha.apply(mono, 0)                      # deterministic
    assert alpha.apply(alpha.invert_monomial(mono), moved) == 0
    # Acting on all points is a permutation (the lift matrix is a permutation).
    assert sorted(alpha.apply(mono, p) for p in range(alpha.size)) == list(
        range(alpha.size)
    )


def test_invert_is_a_two_sided_inverse():
    for perm in (s3("s0"), s3("t1"), s4("aca")):
        identity = tuple(range(len(perm)))
        assert compose(perm, invert(perm)) == identity
        assert compose(invert(perm), perm) == identity


# --------------------------------------------------------------------------- #
# Errors and end-to-end
# --------------------------------------------------------------------------- #


def test_gala_rejects_bad_parameters():
    base = dict(GALA_CODE_PRESETS["gala_132_30_12"])

    with pytest.raises(ValueError, match="Unknown GALA preset"):
        GalaCode.from_preset("not_a_preset")

    with pytest.raises(ValueError, match="positive even integer"):
        GalaCode(**{**base, "L": 11, "compute_k": False})

    with pytest.raises(ValueError, match=r"1 <= J <= L/2"):
        GalaCode(**{**base, "J": 7, "compute_k": False})

    with pytest.raises(ValueError, match="must contain L/2"):
        GalaCode(**{**base, "F": base["F"][:3], "compute_k": False})

    with pytest.raises(ValueError, match="cannot use H_k element"):
        GalaCode(**{**base, "F": [[("s0", (1,))]] * 6, "compute_k": False})


def test_gala_memory_experiment_end_to_end_smoke():
    """KasaiCode-style path: GalaCode -> MemoryExperiment -> DEM -> decode.

    Uses the smallest preset ([[132,30,12]]) so the build stays fast.
    """
    system = QECSystem()
    system.add_patch(GalaCode.from_preset("gala_132_30_12"), name="gala")
    circuit = MemoryExperiment(
        qec_system=system,
        extraction_block_class=GalaCodeExtractionBlock,
        rounds=2,
        noise_params=NoiseConfig(p_idle=0.0, p_1q=1e-3, p_2q=1e-3,
                                 p_meas=1e-3, p_reset=1e-3),
        noise_model="circuit_level",
        basis="Z",
        z_only=True,
    ).build()

    # z_only: J * lift_size Z detectors per round, plus the final layer.
    assert circuit.num_detectors == 3 * (5 * 11)
    assert circuit.num_observables == 30
    assert circuit.detector_error_model().num_errors > 0

    stats = SimulationPipeline(
        decoder_config=DecoderConfig("ldpc-bp", params={"max_iter": 30}),
        max_shots=100, max_errors=10_000, batch_size=50,
        num_workers=1, print_progress=False,
    ).run(circuit)
    assert stats.shots >= 100
