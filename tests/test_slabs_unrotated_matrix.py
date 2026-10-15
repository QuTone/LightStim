"""Native unrotated AB/BC held merges for every X/Z preparation and readout."""
from itertools import product

import pytest

from tests.slabs_unrotated_helpers import build_case


BASES = ["".join(bases) for bases in product("XZ", repeat=3)]


@pytest.mark.parametrize("initials", BASES)
@pytest.mark.parametrize("finals", BASES)
def test_native_unrotated_held_merges_match_ideal_record_space(initials, finals):
    result = build_case(initials, finals)
    assert result["all_classification_checks_pass"], result["checks"]
    assert result["complete_signed_flows"]
    assert result["DEM_valid"]
