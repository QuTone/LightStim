"""Tests for the exact most-likely-error (mle-ilp) decoder."""

import importlib.util

import numpy as np
import pytest
import stim

from lightstim.simulation.decoder_backend.dem_matrices import dem_to_matrices
from lightstim.simulation.decoder_backend.registry import get_decoder, list_decoders

_HAS_HIGHSPY = importlib.util.find_spec("highspy") is not None

# Both backends are HiGHS and must agree exactly; "scipy" needs no extra install.
_SOLVERS = ["scipy"] + (["highs"] if _HAS_HIGHSPY else [])


def _surface_code_dem(distance=3, rounds=3, p=1e-3):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )
    # decompose_errors=False: the ILP handles hyperedges natively, and
    # decomposing would change the optimisation problem being solved.
    return circuit.detector_error_model(decompose_errors=False,
                                        flatten_loops=True)


def _decode(dem, detectors, **params):
    decoder = get_decoder("mle-ilp", backend="cpu", params=params)
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    out = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=np.packbits(
            detectors, axis=1, bitorder="little"))
    return np.unpackbits(out, axis=1, bitorder="little")


def test_registered():
    assert "mle-ilp" in list_decoders()


@pytest.mark.parametrize("solver", _SOLVERS)
def test_solution_satisfies_syndrome(solver):
    """The returned error must actually explain the observed syndrome."""
    dem = _surface_code_dem()
    H, _, priors = dem_to_matrices(dem, sparse=True, merge_duplicates=True)
    detectors, _, _ = dem.compile_sampler(seed=5).sample(shots=25)

    decoder = get_decoder("mle-ilp", backend="cpu", params={"solver": solver})
    decoder.setup(H=H, priors=priors)
    for syndrome in detectors.astype(np.uint8):
        correction, ok = decoder.decode_single(syndrome)
        assert ok
        assert np.array_equal((H @ correction) % 2, syndrome)


@pytest.mark.skipif(not _HAS_HIGHSPY, reason="highspy not installed")
def test_backends_agree_on_optimal_cost():
    """highspy and scipy are the same solver; they must find the same optimum.

    Compares cost rather than the error vector itself: degenerate instances can
    have several distinct minimisers of equal weight.
    """
    dem = _surface_code_dem()
    H, _, priors = dem_to_matrices(dem, sparse=True, merge_duplicates=True)
    weights = np.log((1 - np.clip(priors, 1e-15, 0.5 - 1e-15))
                     / np.clip(priors, 1e-15, 0.5 - 1e-15))
    detectors, _, _ = dem.compile_sampler(seed=5).sample(shots=15)

    decoders = {}
    for solver in ("scipy", "highs"):
        d = get_decoder("mle-ilp", backend="cpu", params={"solver": solver})
        d.setup(H=H, priors=priors)
        decoders[solver] = d

    for syndrome in detectors.astype(np.uint8):
        costs = {}
        for solver, d in decoders.items():
            correction, ok = d.decode_single(syndrome)
            assert ok
            costs[solver] = weights @ correction
        assert costs["scipy"] == pytest.approx(costs["highs"], abs=1e-6)


def test_auto_resolves_to_scipy():
    """'auto' must pick scipy even when highspy is installed.

    scipy vendors its own (older, measurably faster) HiGHS build rather than
    importing highspy, so presence of highspy is not a reason to prefer it.
    """
    from lightstim.simulation.decoder_backend.decoders import mle_ilp

    assert mle_ilp._resolve_solver("auto") == "scipy"
    assert mle_ilp._resolve_solver("scipy") == "scipy"
    assert mle_ilp._resolve_solver("highs") == "highs"


def test_rejects_unknown_solver():
    from lightstim.simulation.decoder_backend.decoders import mle_ilp

    with pytest.raises(ValueError, match="Unknown solver"):
        mle_ilp._resolve_solver("gurobi")


def _bb_dem(rounds=2, p=1e-3):
    from lightstim.ir.qec_system import QECSystem
    from lightstim.noise.config import NoiseConfig
    from lightstim.protocols.memory import MemoryExperiment
    from lightstim.qec_code.BB_code import BBCode, BBCodeExtractionBlock

    code = BBCode(l=6, m=6, A=[[3, 0], [0, 1], [0, 2]],
                  B=[[0, 3], [1, 0], [2, 0]])
    system = QECSystem()
    system.add_patch(code, name="bb")
    experiment = MemoryExperiment(
        qec_system=system, extraction_block_class=BBCodeExtractionBlock,
        rounds=rounds, noise_params=NoiseConfig(p_1q=p, p_2q=p, p_meas=p,
                                                p_reset=p, p_idle=p),
        noise_model="circuit_level", basis="Z")
    return experiment.build().detector_error_model(decompose_errors=False,
                                                   flatten_loops=True)


def test_solves_bb_code_dem():
    """QLDPC DEMs are far denser than surface-code ones — cover that shape.

    A BB [[72,12,6]] detector row touches ~200 error mechanisms and columns
    reach weight 13, versus a handful for the surface code. This is the regime
    where solver behaviour diverges, so it needs its own coverage.
    """
    dem = _bb_dem()
    H, _, priors = dem_to_matrices(dem, sparse=True, merge_duplicates=True)
    assert np.asarray(H.sum(axis=1)).ravel().mean() > 50, "expected a dense DEM"

    detectors, _, _ = dem.compile_sampler(seed=7).sample(shots=5)
    decoder = get_decoder("mle-ilp", backend="cpu")
    decoder.setup(H=H, priors=priors)
    for syndrome in detectors.astype(np.uint8):
        correction, ok = decoder.decode_single(syndrome)
        assert ok
        assert np.array_equal((H @ correction) % 2, syndrome)


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_HIGHSPY, reason="highspy not installed")
def test_beats_mwpm_at_high_noise():
    """Exact MLE should be at least as accurate as MWPM on the same circuit.

    At p=5e-3 the gap is large enough to see in a few hundred shots; MLE
    exploits hyperedge correlations that the decomposed matching graph loses.
    """
    import pymatching

    p, shots, d, rounds = 5e-3, 400, 5, 5
    kwargs = dict(distance=d, rounds=rounds,
                  after_clifford_depolarization=p,
                  before_measure_flip_probability=p,
                  after_reset_flip_probability=p,
                  before_round_data_depolarization=p)
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", **kwargs)

    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    detectors, observables, _ = dem.compile_sampler(seed=3).sample(shots=shots)

    predicted = _decode(dem, detectors, solver="highs")[:, :observables.shape[1]]
    mle_ler = (predicted != observables).any(axis=1).mean()

    matcher = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    mwpm_ler = (matcher.decode_batch(detectors) != observables).any(axis=1).mean()

    assert mle_ler <= mwpm_ler
