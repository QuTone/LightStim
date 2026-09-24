"""Structural composition of a rotated-surface-code memory experiment.

Repetition is expressed the way Microsoft's own generated libraries express
it: ``REPEAT N { SyndromeExtraction 0 }`` calls the gadget by name N times,
rather than duplicating an inlined instruction body.
"""

from __future__ import annotations

from . import text
from .code import DeqExportError


def memory_register_compose(code_name: str, rounds: int, name: str) -> text.ComposeBlock:
    """Reusable REPEAT-only building block: keep an already-prepared patch alive.

    Matches the shape of the reference surface-code-deq library's own
    ``COMPOSE ...Memory { INPUT ...; REPEAT rounds { SyndromeExtraction 0 };
    OUTPUT ... }`` — it takes an already-prepared code-typed port and extends
    it by ``rounds`` more syndrome-extraction rounds.
    """
    if rounds < 0:
        raise DeqExportError("rounds must be >= 0")
    lines = [f"INPUT {code_name} 0"]
    if rounds > 0:
        lines += text.repeat(rounds, ["SyndromeExtraction 0"])
    lines.append(f"OUTPUT {code_name} 0")
    return text.ComposeBlock(name, lines)


def memory_experiment_program(name: str, rounds: int, basis: str) -> text.ProgramBlock:
    """Full, runnable memory experiment: Prepare -> (rounds-1) x SE -> Measure.

    ``PrepareZ``/``PrepareX`` already fuses data reset with one SE round, so
    the REPEAT count is ``rounds - 1`` to give ``rounds`` total SE rounds —
    matching LightStim's own ``MemoryExperiment(rounds=rounds)`` convention
    exactly, so the two can be compared 1:1. Emitted as a PROGRAM (not a
    COMPOSE) because it has no ports — it's a closed, top-level, transpilable
    unit, not a reusable library building block.
    """
    if basis not in ("X", "Z"):
        raise DeqExportError(f"basis must be 'X' or 'Z'; got {basis!r}")
    if rounds < 1:
        raise DeqExportError("rounds must be >= 1")

    prepare_name = "PrepareZ" if basis == "Z" else "PrepareX"
    measure_name = "MeasureZ" if basis == "Z" else "MeasureX"

    lines = [f"{prepare_name} 0"]
    extra_rounds = rounds - 1
    if extra_rounds > 0:
        lines += text.repeat(extra_rounds, ["SyndromeExtraction 0"])
    lines.append(f"{measure_name} 0")
    return text.ProgramBlock(name, lines)
