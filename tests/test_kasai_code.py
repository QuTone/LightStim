from collections import defaultdict

import numpy as np
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.kasai_code import (
    KASAI_CODE_PRESETS,
    KasaiChenExtractionBlock,
    KasaiCode,
    find_commuting_layout_reference,
)


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


def _chen_p96_system():
    system = QECSystem()
    system.add_patch(KasaiCode.from_preset("chen_p96", compute_k=False),
                     name="chen_p96")
    return system


def test_kasai_chen_se_layers_match_stabilizer_supports():
    """Every CNOT layer is conflict-free and the union of scheduled CNOTs per
    ancilla equals its registered stabilizer support, for both bases."""
    system = _chen_p96_system()
    blk = KasaiChenExtractionBlock(system)

    assert blk.cnot_depth == 24
    # chen_p96's published layout: a reference APM with 3 length-32 orbits.
    assert blk.layout_reference is not None
    assert blk.layout_reference[1] == 32

    xs = {s["syn_idx"]: set(s["data_indices"])
          for s in system.active_stabilizers_x}
    zs = {s["syn_idx"]: set(s["data_indices"])
          for s in system.active_stabilizers_z}
    sched = defaultdict(set)
    n_layers = 0
    for inst in blk.circuit:
        if inst.name in ("CX", "CNOT"):
            n_layers += 1
            ts = [t.value for t in inst.targets_copy()]
            used = set()
            for a, b in zip(ts[0::2], ts[1::2]):
                assert a not in used and b not in used
                used.update((a, b))
                if a in xs:
                    sched[a].add(b)
                elif b in zs:
                    sched[b].add(a)
    assert n_layers == 24
    assert all(sched[k] == v for k, v in xs.items())
    assert all(sched[k] == v for k, v in zs.items())


def test_kasai_chen_se_layout_condition():
    """All arXiv:2604.16209v2 instances satisfy the co-design condition with
    orbit length >= 32; kasai_p768 (not co-designed) is rejected."""
    for name in KASAI_CODE_PRESETS:
        pr = KASAI_CODE_PRESETS[name]
        ref = find_commuting_layout_reference(
            [tuple(t) for t in pr["f"]], [tuple(t) for t in pr["g"]],
            pr["P"], min_orbit_length=pr["P"] // (2 * pr["J"]),
        )
        if name.startswith("chen_"):
            assert ref is not None and ref[1] >= 32, name
        else:
            assert ref is None, name


def test_kasai_chen_se_rejects_non_codesigned_code():
    system = QECSystem()
    system.add_patch(KasaiCode.from_preset("kasai_p768", compute_k=False),
                     name="kasai_p768")
    with pytest.raises(ValueError, match="co-design condition"):
        KasaiChenExtractionBlock(system)
    # escape hatch still builds a valid (if hardware-inefficient) circuit
    blk = KasaiChenExtractionBlock(system, enforce_layout_condition=False)
    assert blk.cnot_depth == 24


def test_kasai_dense_css_matrices_for_small_preset():
    code = KasaiCode.from_preset("chen_p96", compute_k=False)
    hx, hz = code.get_css_matrices()

    assert hx.shape == (3 * 96, 12 * 96)
    assert hz.shape == (3 * 96, 12 * 96)
    assert np.all(hx.sum(axis=1) == 12)
    assert np.all(hz.sum(axis=1) == 12)
    assert np.count_nonzero((hx @ hz.T) % 2) == 0
