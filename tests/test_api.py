"""
API Endpoint Tests — FastAPI TestClient smoke tests.

Verifies that each endpoint:
  1. Returns HTTP 200
  2. Returns a payload with dem/timeline/detslice keys
  3. DEM has detectors > 0

These tests do NOT start a real server — they use FastAPI's TestClient
(ASGI in-process), so no port binding needed.

Run:  pytest tests/test_api.py -m smoke -q
"""
import pytest

try:
    from fastapi.testclient import TestClient
    from api.main import app
    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _API_AVAILABLE,
    reason="fastapi or api.main not available",
)

SMOKE = pytest.mark.smoke


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _assert_payload(resp):
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:200]}"
    body = resp.json()
    assert "dem" in body and "timeline" in body and "detslice" in body
    assert len(body["dem"]["detectors"]) > 0
    assert len(body["dem"]["error_mechanisms"]) > 0
    return body


@SMOKE
def test_api_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@SMOKE
def test_api_memory_rotated(client):
    body = _assert_payload(client.post("/api/circuit/memory", json={
        "code": "rotated", "distance": 3, "rounds": 3, "p": 0.001,
    }))
    assert body["dem"]["metadata"]["source"] == "rotated_memory_z"


@SMOKE
def test_api_memory_color(client):
    _assert_payload(client.post("/api/circuit/memory", json={
        "code": "color", "distance": 3, "rounds": 3, "p": 0.001,
    }))


@SMOKE
def test_api_two_patch_ls(client):
    body = _assert_payload(client.post("/api/circuit/two-patch-ls", json={
        "distance1": 3, "distance2": 3, "interaction_type": "ZZ", "rounds": 2, "p": 0.001,
    }))
    # Two patches → y-span > 0
    ys = [d["coords"]["y"] for d in body["dem"]["detectors"]]
    assert max(ys) > min(ys)


@SMOKE
def test_api_logical_h(client):
    _assert_payload(client.post("/api/circuit/logical-h", json={
        "distance": 3, "rounds": 2, "p": 0.001,
    }))


@SMOKE
def test_api_state_injection(client):
    _assert_payload(client.post("/api/circuit/state-injection", json={
        "distance": 3, "rounds": 2, "inject_state": "Y", "protocol": "corner", "p": 0.001,
    }))


@SMOKE
def test_api_tg_distillation(client):
    body = _assert_payload(client.post("/api/circuit/tg-distillation", json={
        "distance": 3, "rounds": 3, "p": 0.001,
    }))
    # TG distillation should have many detectors (7 working + 7 magic patches)
    assert len(body["dem"]["detectors"]) > 100


@SMOKE
def test_api_invalid_code_returns_500(client):
    """Bad input should return 500 with a detail message, not crash."""
    resp = client.post("/api/circuit/memory", json={
        "code": "rotated", "distance": 99, "rounds": 3, "p": 0.001,
    })
    assert resp.status_code == 500
    assert "detail" in resp.json()
