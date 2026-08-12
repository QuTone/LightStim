"""Tests for the exact most-likely-error (mle-ilp) decoder."""

import numpy as np
import pytest
import scipy.sparse as sp
import stim

from lightstim.simulation.decoder_backend.dem_matrices import dem_to_matrices
from lightstim.simulation.decoder_backend.registry import get_decoder, list_decoders


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


def test_solution_satisfies_syndrome():
    """The returned error must actually explain the observed syndrome."""
    dem = _surface_code_dem()
    H, _, priors = dem_to_matrices(dem, sparse=True, merge_duplicates=True)
    detectors, _, _ = dem.compile_sampler(seed=5).sample(shots=25)

    decoder = get_decoder("mle-ilp", backend="cpu")
    decoder.setup(H=H, priors=priors)
    for syndrome in detectors.astype(np.uint8):
        correction, ok = decoder.decode_single(syndrome)
        assert ok
        assert np.array_equal((H @ correction) % 2, syndrome)


def test_finds_true_optimum_on_tiny_instance():
    """The solution must be genuinely minimum-weight, not merely feasible.

    Checked against brute force over all 2^n errors on an instance small
    enough to enumerate. Compares cost rather than the error vector, since a
    degenerate instance can have several distinct minimisers of equal weight.
    """
    import itertools

    rng = np.random.default_rng(0)
    n_dets, n_errs = 6, 14
    H = (rng.random((n_dets, n_errs)) < 0.35).astype(np.uint8)
    H[:, H.sum(axis=0) == 0] = 0
    H[0, 0] = 1                       # guarantee at least one non-empty column
    priors = rng.uniform(1e-3, 0.2, size=n_errs)
    weights = np.log((1 - priors) / priors)

    decoder = get_decoder("mle-ilp", backend="cpu")
    decoder.setup(H=sp.csr_matrix(H), priors=priors)

    all_errors = np.array(list(itertools.product([0, 1], repeat=n_errs)),
                          dtype=np.uint8)
    all_syndromes = (all_errors @ H.T) % 2
    all_costs = all_errors @ weights

    for syndrome in np.unique(all_syndromes, axis=0):
        matches = (all_syndromes == syndrome).all(axis=1)
        best = all_costs[matches].min()
        correction, ok = decoder.decode_single(syndrome)
        assert ok
        assert np.array_equal((H @ correction) % 2, syndrome)
        assert weights @ correction == pytest.approx(best, abs=1e-9)


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
    where solver behaviour diverged in benchmarking, so it needs coverage.
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

    predicted = _decode(dem, detectors)[:, :observables.shape[1]]
    mle_ler = (predicted != observables).any(axis=1).mean()

    matcher = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    mwpm_ler = (matcher.decode_batch(detectors) != observables).any(axis=1).mean()

    assert mle_ler <= mwpm_ler
