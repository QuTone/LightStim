"""An editable install must reject stale native tracking implementations."""
import sys
from types import ModuleType

import pytest

from lightstim.utils.record_backend import record_state_class


def test_persistent_rejects_old_extension_with_actionable_error(monkeypatch):
    old = ModuleType("lightstim.utils.cpp._pauli_backend_cpp")
    old.PersistentRecordState = type("PersistentRecordState", (), {})
    old.PackedRows = type("PackedRows", (), {})
    monkeypatch.setitem(sys.modules, old.__name__, old)
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", "persistent")
    with pytest.raises(ImportError, match="matching native Tracker extension; rebuild"):
        record_state_class()


def test_reference_does_not_require_native_extension(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightstim.utils.cpp._pauli_backend_cpp", None)
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", "reference")
    assert record_state_class() is None
