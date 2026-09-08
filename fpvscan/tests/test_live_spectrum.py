from __future__ import annotations

import asyncio
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fpvscan.app.bootstrap import attach_test_api
from fpvscan.app.repositories.sqlite_store import SqliteTestStore
from fpvscan.app.services.live_service import LiveService
from fpvscan.app.services.parameter_service import ParameterService
from tests.conftest import FakeEngine


def test_live_includes_spectrum_and_grid(engine) -> None:
    async def run() -> None:
        live = await LiveService(engine).live_context()
        spec = live["spectrum"]
        grid = live["grid"]
        assert spec["bins"] == [-40.0, -20.0, -38.0]
        assert spec["center_hz"] == 4988e6
        assert spec["cursor_hz"] == 4988e6
        assert spec["bw_hz"] == 10e6
        assert spec["span_hz"] == 20e6
        assert spec["nfft"] == 2048
        assert spec["t_mono_ms"]
        assert grid["start_hz"] == 400e6
        assert grid["t_mono_ms"]
        assert grid["stop_hz"] == 6000e6
        assert grid["current_hz"] == 4988e6
        assert grid["pass_count"] > 0
        assert live["frequency_hz"] == 4988e6
        assert live["snr"] == 22.0
        assert live["afc_pegged"] is False
        assert live["hunt_span_hz"] == 0.25e6
        assert live["video_metrics"]["afc_pegged"] is False
        assert live["video_metrics"]["timings_ms"] == {"decode": 50.0}
        assert live["video_metrics"]["overflows"] == 0
        assert live["video_metrics"]["fps"] == 6.2

    asyncio.run(run())


def test_live_reports_afc_pegged(engine) -> None:
    engine._snap["afc_pegged"] = True
    engine._snap["afc_hz"] = -1.5e6
    engine._snap["freq_err_hz"] = -100e3
    engine._snap["hunt_span_hz"] = 0.25e6

    async def run() -> None:
        live = await LiveService(engine).live_context()
        assert live["afc_pegged"] is True
        assert live["afc_hz"] == -1.5e6
        assert live["freq_err_hz"] == -100e3
        assert live["hunt_span_hz"] == 0.25e6
        assert live["spectrum"]["afc_pegged"] is True
        assert live["video_metrics"]["afc_pegged"] is True

    asyncio.run(run())


def test_put_params_update_live_spectrum_metadata(engine, store) -> None:
    params = ParameterService(engine, store)
    live_svc = LiveService(engine)

    async def run() -> None:
        applied = await params.apply_parameters(uuid4(), {
            "video.channel_bw_hz": 8e6,
            "video.sample_rate": 25e6,
            "scan.start_hz": 500e6,
            "sdr.gain_db": 40,
        })
        assert "video.channel_bw_hz" in applied.applied_keys
        assert "video.sample_rate" not in applied.pending_keys
        assert applied.affects["video.sample_rate"] == "picture"
        assert applied.affects["scan.start_hz"] == "grid"
        assert "sdr.gain_db" not in applied.pending_keys
        assert ("refresh_lock", {}) in engine.commands
        live = await live_svc.live_context()
        assert live["spectrum"]["bw_hz"] == 8e6
        assert live["spectrum"]["span_hz"] == 25e6
        assert live["grid"]["start_hz"] == 500e6
        assert live["parameters"]["sdr.gain_db"] == 40
        assert live["frequency_hz"] == 4988e6

    asyncio.run(run())


def test_lock_target_recenters_spectrum(engine) -> None:
    engine._snap["lock_target"] = 5018e6
    engine._snap["tuned_hz"] = 5018e6

    async def run() -> None:
        live = await LiveService(engine).live_context()
        assert live["frequency_hz"] == 5018e6
        assert live["spectrum"]["cursor_hz"] == 5018e6
        assert live["spectrum"]["center_hz"] == 5018e6

    asyncio.run(run())


def test_live_http_spectrum_contract() -> None:
    app = FastAPI()
    attach_test_api(app, FakeEngine(), store=SqliteTestStore())
    client = TestClient(app)
    body = client.get("/api/test/live").json()
    assert "bins" in body["spectrum"]
    assert body["spectrum"]["bw_hz"] == 10e6
    assert body["grid"]["start_hz"] == 400e6
    put = client.put("/api/test/parameters", json={
        "idempotency_key": str(uuid4()),
        "values": {"video.channel_bw_hz": 14e6, "scan.sample_rate": 40e6},
    })
    assert put.status_code == 200
    assert "scan.sample_rate" in put.json()["pending_keys"]
    live = client.get("/api/test/live").json()
    assert live["spectrum"]["bw_hz"] == 14e6
    assert live["grid"]["step_hz"] > 0
