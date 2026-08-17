import os
import subprocess
import sys
import difflib

import pytest


_BUILD_SCRIPT = r"""
import contextlib
import io
import os

from lightstim.protocols.ppm.spec import PatchSpec, PPMStep, origin_of
from lightstim.protocols.ppm.sequential import SequentialPPMExperiment

d = 3
case = os.environ["LIGHTSTIM_PPM_CASE"]
if case == "straight":
    patches = [
        PatchSpec("A", origin_of(0, 0, d, seam=True), d, "X_horizontal"),
        PatchSpec("B", origin_of(2, 0, d, seam=True), d, "X_horizontal"),
    ]
    targets = [("A", "Z"), ("B", "Z")]
    route = [(1, 0)]
elif case == "bent":
    patches = [
        PatchSpec("A", origin_of(0, 0, d, seam=True), d, "X_horizontal"),
        PatchSpec("B", origin_of(2, 1, d, seam=True), d, "X_vertical"),
    ]
    targets = [("A", "Z"), ("B", "Z")]
    route = [(1, 0), (2, 0)]
elif case == "branched":
    patches = [
        PatchSpec("A", origin_of(0, 0, d, seam=True), d, "X_horizontal"),
        PatchSpec("B", origin_of(4, 0, d, seam=True), d, "X_horizontal"),
        PatchSpec("C", origin_of(2, 1, d, seam=True), d, "X_vertical"),
    ]
    targets = [("A", "Z"), ("B", "Z"), ("C", "Z")]
    route = [(1, 0), (2, 0), (3, 0)]
else:
    raise ValueError(case)

step = PPMStep(targets, route=route)
states = {patch.name: "Z" for patch in patches}
experiment = SequentialPPMExperiment(
    patches,
    [step],
    initial_states=states,
    final_measure_states=states,
    rounds=d,
    rounds_init=1,
)
with contextlib.redirect_stdout(io.StringIO()):
    circuit = experiment.build()
print(circuit)
"""


def _circuit_text(hash_seed, case):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(hash_seed)
    env["LIGHTSTIM_PPM_CASE"] = case
    result = subprocess.run(
        [sys.executable, "-c", _BUILD_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize("case", ["straight", "bent", "branched"])
def test_ppm_circuit_is_deterministic_across_hash_seeds(case):
    seed_0 = _circuit_text(0, case)
    seed_1 = _circuit_text(1, case)
    diff = "".join(difflib.unified_diff(
        seed_0.splitlines(keepends=True),
        seed_1.splitlines(keepends=True),
        fromfile="PYTHONHASHSEED=0",
        tofile="PYTHONHASHSEED=1",
    ))
    assert seed_0 == seed_1, diff
