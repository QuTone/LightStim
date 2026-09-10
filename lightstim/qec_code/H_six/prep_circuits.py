"""``[[6, 2, 2]]`` encoder circuits, ported from CQCL/Magic-H6 ``Code614.py``.

``get_dist_circ`` is the encoder block of Magic-H6's ``get_dist_circ`` (the
non-fault-tolerant path used for the level-1 distillation demo). It prepares the
``|++>_L`` codeword.

``get_ft_init_circ`` is the fault-tolerant, flag-verified state prep for
``|00>_L`` shown in Fig. 5 of arXiv:2506.14688. Qubits 0 and 2 are the only
controls; two flag ancillas catch hook errors on them. The circuit ends with a
measurement of the flags that the caller must **post-select on 0**.

Both helpers take *global* qubit indices and return a noiseless ``stim.Circuit``.
The gate order is verbatim from ``Code614.py`` (noise ops omitted -- inject
separately).
"""

from typing import Sequence

import stim

# get_dist_circ's encoder block, as (gate, [data-label operands]) with labels
# 0..5 the six data qubits in Magic-H6 order. Verbatim from Code614.py.
_DISTILLATION_ENCODER = (
    ("H", (0, 1)),
    ("H", (2, 4)),

    ("CX", (2, 3)),
    ("CX", (4, 5)),

    ("CX", (2, 0)),
    ("CX", (3, 1)),

    ("CX", (0, 4)),
    ("CX", (1, 5)),
    
    ("CX", (4, 2)),
    ("CX", (5, 3)),
)


def get_dist_circ(data: Sequence[int]) -> stim.Circuit:
    """Encoder block of ``get_dist_circ`` -> ``|++>_L``. ``data``: 6 global indices."""
    if len(data) != 6:
        raise ValueError(f"Need 6 data-qubit indices, got {len(data)}.")
    circuit = stim.Circuit()
    for gate, labels in _DISTILLATION_ENCODER:
        circuit.append(gate, [data[i] for i in labels])
    return circuit


def get_ft_init_circ(data: Sequence[int], flags: Sequence[int]) -> stim.Circuit:
    """Flag-verified FT prep of ``|00>_L`` (ported from ``Code614.py``).

    Args:
        data:  6 global data-qubit indices (labels 0..5).
        flags: 2 global flag-ancilla indices.

    The returned circuit resets the flags, runs the flagged encoder, and
    measures the flags last; the caller must discard shots where either flag
    reads 1.
    """
    if len(data) != 6:
        raise ValueError(f"Need 6 data-qubit indices, got {len(data)}.")
    if len(flags) != 2:
        raise ValueError(f"Need 2 flag-ancilla indices, got {len(flags)}.")
    s = list(data)
    a = list(flags)
    c = stim.Circuit()
    c.append("R", a)
    c.append("H", [s[0], s[2]])
    c.append("TICK", tag="SE_start")
    c.append("CX", [s[0], a[0]])
    c.append("CX", [s[2], a[1]])
    c.append("CX", [s[0], s[1]])
    c.append("CX", [s[2], s[3]])
    c.append("CX", [s[0], s[4]])
    c.append("CX", [s[2], s[5]])
    c.append("CX", [s[0], s[5]])
    c.append("CX", [s[2], s[4]])
    c.append("CX", [s[0], a[0]])
    c.append("CX", [s[2], a[1]])
    c.append("TICK")
    c.append("M", a)
    return c


__all__ = [
    "get_dist_circ",
    "get_ft_init_circ",
]
