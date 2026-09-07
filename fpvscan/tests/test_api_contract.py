from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fpvscan.app.bootstrap import attach_test_api
from fpvscan.app.repositories.sqlite_store import SqliteTestStore
from tests.conftest import FakeEngine


def _client() -> TestClient:
    app = FastAPI()
    attach_test_api(app, FakeEngine(), store=SqliteTestStore())
    return TestClient(app)


def test_parameter_and_session_http_contract() -> None:
    client = _client()
    health = client.get("/api/test/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    catalog = client.get("/api/test/parameters")
    assert catalog.status_code == 200
    keys = {item["key"] for item in catalog.json()["items"]}
    assert "scan.threshold_db" in keys
    assert "video.afc" in keys
    tagged = {item["key"]: item for item in catalog.json()["items"]}
    assert tagged["scan.start_hz"]["task"] == "scan_width"
    assert tagged["scan.start_hz"]["modes"] == ["sweep"]
    assert tagged["scan.start_hz"]["affects"] == "grid"
    assert tagged["scan.threshold_db"]["task"] is None

    sweep = client.get("/api/test/parameters", params={"mode": "sweep"})
    assert sweep.status_code == 200
    sweep_keys = {item["key"] for item in sweep.json()["items"]}
    assert sweep_keys == {
        "scan.start_hz", "scan.stop_hz", "scan.channel_bw_hz",
        "scan.cluster_step_mhz", "scan.hit_filter",
    }
    tagged_s = {item["key"]: item for item in sweep.json()["items"]}
    assert tagged_s["scan.cluster_step_mhz"]["task"] == "scan_grid"
    assert tagged_s["scan.cluster_step_mhz"]["enum_values"] == ["off", "12", "8", "4"]
    assert tagged_s["scan.hit_filter"]["task"] == "hit_filter"
    assert tagged_s["scan.hit_filter"]["enum_labels"]["hide_near_dup"] == "ховати сусідів"
    assert "scan.threshold_db" not in sweep_keys
    assert "scan.fft_size" not in sweep_keys

    lock = client.get("/api/test/parameters", params={"mode": "lock"})
    assert lock.status_code == 200
    lock_items = lock.json()["items"]
    lock_keys = {item["key"] for item in lock_items}
    assert "video.afc" in lock_keys
    assert "video.track_window_margin" in lock_keys
    assert "video.h_phase_frac" in lock_keys
    assert "video.average" in lock_keys
    assert "video.sample_rate" not in lock_keys
    assert "video.crop_left_frac" not in lock_keys
    assert "video.sharpen" not in lock_keys
    assert "scan.threshold_db" not in lock_keys
    tasks = {item["key"]: item["task"] for item in lock_items}
    assert tasks["sdr.gain_db"] == "picture_jump"
    assert tasks["video.track_window_margin"] == "phase_tear_h"
    assert tasks["video.motion_thresh"] == "phase_tear_v"

    bad = client.get("/api/test/parameters", params={"mode": "waterfall"})
    assert bad.status_code == 422

    current = client.get("/api/test/parameters/current")
    assert current.status_code == 200
    assert "scan.threshold_db" in current.json()["values"]

    applied = client.put("/api/test/parameters", json={
        "idempotency_key": str(uuid4()),
        "values": {"scan.threshold_db": 6.5},
    })
    assert applied.status_code == 200
    assert applied.json()["values"]["scan.threshold_db"] == 6.5

    created = client.post("/api/test/sessions", json={
        "idempotency_key": str(uuid4()),
        "mode": "lock",
        "title": "bench",
    })
    assert created.status_code == 201
    session_id = created.json()["id"]

    live = client.get("/api/test/live")
    assert live.status_code == 200
    body = live.json()
    assert body["mode"] == "LOCK"
    assert body["lock_state"] is True

    obs = client.post(f"/api/test/sessions/{session_id}/observations", json={
        "idempotency_key": str(uuid4()),
        "fields": {"signal_quality": 4, "picture_lock": True},
    })
    assert obs.status_code == 201
    assert obs.json()["fields"]["signal_quality"] == 4
    assert obs.json()["engine_status"]["mode"] == "LOCK"

    listed = client.get(f"/api/test/sessions/{session_id}/observations")
    assert listed.status_code == 200
    assert len(listed.json()["items"]) == 1


def test_observation_console_payload_is_201() -> None:
    """Operator «Записати»: empty optionals + extra symptom bools must not 422."""
    client = _client()
    old = [
        {"key": "signal_quality", "label": "Signal quality", "type": "rating", "min": 1, "max": 5},
        {"key": "picture_lock", "label": "Picture lock", "type": "bool"},
        {"key": "analog_noise", "label": "Analog noise", "type": "rating", "min": 1, "max": 5},
        {"key": "osd_present", "label": "OSD present", "type": "bool"},
        {"key": "frequency_mhz", "label": "Frequency", "type": "number", "min": 100, "max": 6200},
        {"key": "bandwidth_note", "label": "Bandwidth note", "type": "text"},
        {"key": "aircraft_type", "label": "Aircraft type", "type": "text"},
        {"key": "notes", "label": "Notes", "type": "text"},
        {"key": "tags", "label": "Tags", "type": "tag_list"},
    ]
    schema = client.put("/api/test/observation-schema", json={
        "idempotency_key": str(uuid4()),
        "fields": old,
    })
    assert schema.status_code == 200
    created = client.post("/api/test/sessions", json={
        "idempotency_key": str(uuid4()),
        "mode": "lock",
        "title": "ui-smoke",
    })
    assert created.status_code == 201
    session_id = created.json()["id"]
    payload = {
        "idempotency_key": str(uuid4()),
        "timestamp_ms": 1757060000000,
        "frequency_hz": 4988e6,
        "rssi": None,
        "snr": 22.0,
        "video_metrics": {"locked": True, "fps": 8.1, "decode": "ok"},
        "parameter_snapshot": {"video.spectrum_every": 1},
        "picture_jump": False,
        "line_tearing": False,
        "spectrum_stutter": True,
        "fields": {
            "signal_quality": 4,
            "picture_lock": True,
            "picture_jump": False,
            "line_tearing": False,
            "spectrum_stutter": True,
            "osd_present": False,
            "frequency_mhz": "",
            "bandwidth_note": "",
            "aircraft_type": "",
            "notes": "",
            "tags": [],
        },
    }
    obs = client.post(
        f"/api/test/sessions/{session_id}/observations", json=payload,
    )
    assert obs.status_code == 201, obs.text
    body = obs.json()
    assert body["frequency_hz"] == 4988e6
    assert body["fields"]["signal_quality"] == 4
    assert body["fields"]["picture_lock"] is True
    assert body["fields"]["picture_jump"] is False
    assert body["fields"]["line_tearing"] is False
    assert body["fields"]["spectrum_stutter"] is True
    assert body["fields"]["frequency_mhz"] == 4988.0
    assert body["fields"]["bandwidth_note"] == "10 MHz"
    assert "aircraft_type" not in body["fields"]
    assert "notes" not in body["fields"]
