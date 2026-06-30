import numpy as np

from lightstim.qec_code.kasai_code import KASAI_CODE_PRESETS, KasaiCode


def test_kasai_presets_replicate_published_nk():
    for name, preset in KASAI_CODE_PRESETS.items():
        code = KasaiCode.from_preset(name)

        assert code.n_data == preset["expected_n"]
        assert code.num_logicals == preset["expected_k"]
        assert code.rank_x == code.rank_z
        assert code.validate_required_commutativity()


def test_kasai_presets_have_intended_noncommuting_pairs():
    for name in KASAI_CODE_PRESETS:
        code = KasaiCode.from_preset(name, compute_k=False)

        assert code.noncommuting_pairs() == [(0, 3), (1, 2)]


def test_kasai_dense_css_matrices_for_small_preset():
    code = KasaiCode.from_preset("chen_p96", compute_k=False)
    hx, hz = code.get_css_matrices()

    assert hx.shape == (3 * 96, 12 * 96)
    assert hz.shape == (3 * 96, 12 * 96)
    assert np.all(hx.sum(axis=1) == 12)
    assert np.all(hz.sum(axis=1) == 12)
    assert np.count_nonzero((hx @ hz.T) % 2) == 0
