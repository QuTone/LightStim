"""Record-augmented stabilizer tableau for automatic detector construction.

State: stim's inverse tableau of a Clifford frame C, with generators
S_k = C Z_k C^dag and destabilizers D_k = C X_k C^dag.  Each generator
carries the set of measurement records whose parity is its eigenvalue (in
place of stim's sign bit), or UNKNOWN; logical generators carry a symbolic
value lambda_j.  A physical Pauli P is located by q = C^dag P C: its X part
lists the generators P anticommutes with, and its Z part is P's decomposition
into generators when it commutes with all of them, so no GF(2) elimination is
needed.

``cpp/record_tableau.cc`` is the engine, ``annotator.py`` the streaming
DETECTOR / OBSERVABLE_INCLUDE construction, and ``tracker_view.py`` a
SyndromeTracker-compatible read-only view of the state.
"""
from .annotator import RecordTableauAnnotator, annotate_circuit


def native_available() -> bool:
    try:
        from .cpp import _record_tableau_cpp  # noqa: F401
        return True
    except ImportError:
        return False


__all__ = ["RecordTableauAnnotator", "annotate_circuit", "native_available"]
