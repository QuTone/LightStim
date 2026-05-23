"""Named configurations for 4D geometric codes.

Source: Table 1 of the 4D Hadamard code paper.
L matrix is 4x4 Hermite Normal Form (upper triangular, a_ij < a_jj).
d is exact for Det2–Det45; Det68 and Det152 have only upper bounds (excluded here).

Usage:
    from lightstim.qec_code.four_d_geo_code.configs import FOUR_D_CONFIGS
    cfg = FOUR_D_CONFIGS["det3"]
    code = FourDGeoCode(L=cfg["L"], d=cfg["d"])
    # [[n=18, k=6, d=3]]
"""

from typing import Dict, Any

FOUR_D_CONFIGS: Dict[str, Dict[str, Any]] = {
    "det2": {
        "L": [[1,0,0,1],[0,1,0,1],[0,0,1,0],[0,0,0,2]],
        "n": 12, "k": 6, "d": 2,
    },
    "det3": {
        "L": [[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,3]],
        "n": 18, "k": 6, "d": 3,
    },
    "det5": {
        "L": [[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,5]],
        "n": 30, "k": 6, "d": 4,
    },
    "det9": {
        "L": [[1,0,0,5],[0,1,0,6],[0,0,1,7],[0,0,0,9]],
        "n": 54, "k": 6, "d": 6,
    },
    "hadamard": {
        "L": [[1,1,1,1],[0,2,0,2],[0,0,2,2],[0,0,0,4]],
        "n": 96, "k": 6, "d": 8,
    },
    "det16": {
        "L": [[1,0,0,3],[0,1,0,5],[0,0,1,7],[0,0,0,16]],
        "n": 96, "k": 6, "d": 8,
    },
    "det18": {
        "L": [[1,0,0,3],[0,1,0,5],[0,0,1,7],[0,0,0,18]],
        "n": 108, "k": 6, "d": 9,
    },
    "det45": {
        "L": [[1,0,1,6],[0,1,0,11],[0,0,3,9],[0,0,0,15]],
        "n": 270, "k": 6, "d": 15,
    },
}
