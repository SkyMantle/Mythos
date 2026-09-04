"""HTTP command surface: a missed route means the console cannot lock or record."""
from __future__ import annotations

from queue import Queue

from fastapi.testclient import TestClient

from fpvscan.web.server import create_app


class FakeEngine:
    def __init__(self):
        self.events = Queue()
        self.cmds = []
        self._snap = {
            "mode": "SWEEP",
            "lock_target": None,
            "detections": [],
            "source": "mock",
        }

    def command(self, name, **kw):
        self.cmds.append((name, kw))

    def snapshot(self):
        return self._snap


def _client():
    eng = FakeEngine()
    return TestClient(create_app(eng)), eng


def test_state_and_index():
    c, eng = _client()
    r = c.get("/api/state")
    assert r.status_code == 200
    assert r.json()["mode"] == "SWEEP"
    html = c.get("/")
    assert html.status_code == 200
    assert b"html" in html.content.lower() or html.headers["content-type"].startswith("text/html")


def test_lock_sweep_clear_snapshot_bias_record():
    c, eng = _client()
    assert c.post("/api/lock/5800000000").json() == {"ok": True}
    assert eng.cmds[-1] == ("lock", {"freq_hz": 5800000000.0})
    assert c.post("/api/sweep").json() == {"ok": True}
    assert eng.cmds[-1] == ("sweep", {})
    assert c.post("/api/clear").json() == {"ok": True}
    assert eng.cmds[-1] == ("clear", {})
    assert c.post("/api/snapshot").json() == {"ok": True}
    assert eng.cmds[-1] == ("snapshot", {})
    assert c.post("/api/bias_tee/on").json() == {"ok": True}
    assert eng.cmds[-1] == ("bias_tee", {"on": True})
    assert c.post("/api/bias_tee/off").json() == {"ok": True}
    assert eng.cmds[-1] == ("bias_tee", {"on": False})
    assert c.post("/api/record/start").json() == {"ok": True}
    assert eng.cmds[-1] == ("rec_start", {})
    assert c.post("/api/record/stop").json() == {"ok": True}
    assert eng.cmds[-1] == ("rec_stop", {})
