"""An editable install must reject stale native tracking implementations."""
import sys
from types import ModuleType

import pytest

from lightstim.utils.record_backend import record_state_class


@pytest.mark.parametrize("previous_methods", [(), (
    "commuting_relation", "coordinate_rows", "fold_rows", "trace_qubits",
    "begin_measurement_batch", "measurement_anticommuting", "measurement_relation",
), (
    "commuting_relation", "coordinate_rows", "fold_rows", "trace_qubits",
    "begin_measurement_batch", "measurement_anticommuting", "measurement_relation",
    "process_measurement_batch",
)], ids=["before-native-roles", "before-native-batch", "before-native-readout-batch"])
def test_persistent_rejects_old_extension_with_actionable_error(monkeypatch, previous_methods):
    old = ModuleType("lightstim.utils.cpp._pauli_backend_cpp")
    old.PersistentRecordState = type("PersistentRecordState", (), {
        name: lambda *args: None for name in previous_methods})
    old.PackedRows = type("PackedRows", (), {})
    monkeypatch.setitem(sys.modules, old.__name__, old)
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", "persistent")
    with pytest.raises(ImportError, match="matching native Tracker extension; rebuild"):
        record_state_class()


def test_reference_does_not_require_native_extension(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightstim.utils.cpp._pauli_backend_cpp", None)
    monkeypatch.setenv("LIGHTSTIM_TRACKER_BACKEND", "reference")
    assert record_state_class() is None
