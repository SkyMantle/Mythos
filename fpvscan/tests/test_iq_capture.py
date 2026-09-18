from __future__ import annotations

import asyncio
import errno
import json
import logging
import threading
import time
from pathlib import Path
from queue import Queue

import httpx
import numpy as np
from fastapi.testclient import TestClient

import fpvscan.engine as engine_module
from fpvscan import paths
from fpvscan.engine import (
    Engine,
    IQ_CAPTURE_DEFAULT_SECONDS,
    IQ_CAPTURE_MAX_BYTES,
    IQCaptureBusy,
)
from fpvscan.iqbuffer import IQRingBuffer
from fpvscan.web.server import create_app


CFG = {
    "scan": {
        "start_hz": 400e6,
        "stop_hz": 6000e6,
        "sample_rate": 20e6,
        "channel_bw_hz": 16e6,
        "step_hz": 0,
        "fft_size": 2048,
        "priority_bands": False,
    },
    "video": {
        "sample_rate": 10.0,
        "channel_bw_hz": 8e6,
        "lo_offset_hz": 0.0,
    },
    "sdr": {
        "gain_db": 31,
        "auto_gain": True,
        "agc": False,
        "bias_tee": True,
        "device": "test-device",
        "rx_channel": 0,
    },
    "rotator": {"enable": False},
}


class _Source:
    name = "stub"
    sample_rate = 10.0
    gain_db = 31
    agc = False
    bias_tee = True
    device = "test-device"
    overflows = 0
    clip_frac = 0.0
    adc_rms = 0.0

    def open(self):
        raise AssertionError("IQ capture must not open a second SDR handle")


def _engine_with_ring(tmp_path: Path, monkeypatch, data: np.ndarray) -> Engine:
    monkeypatch.setattr(paths, "CAPS", tmp_path / "caps")
    eng = Engine(_Source(), CFG, Queue())
    eng.state.mode = "LOCK"
    eng.state.lock_target = 1.1e9
    eng.state.tuned_hz = 1.1e9
    eng._lock_tuned = 1.1e9
    ring = IQRingBuffer(5)
    ring.capture_context = {
        "center_hz": 1.1e9,
        "tuned_hz": 1.1e9,
        "sample_rate": 10.0,
        "bandwidth_hz": 8e6,
        "gain_db": 31,
        "agc": False,
        "bias_tee": True,
    }
    ring.write(data)
    eng._ring = ring
    return eng


def test_capture_saves_bounded_newest_cf32_and_metadata(
        tmp_path: Path, monkeypatch) -> None:
    samples = np.arange(8, dtype=np.float32).astype(np.complex64) * (1 + 2j)
    eng = _engine_with_ring(tmp_path, monkeypatch, samples)

    result = eng.capture_iq(seconds=0.3, note="operator note")

    capture = Path(result["path"])
    sidecar = Path(result["metadata_path"])
    assert capture.parent.resolve() == (tmp_path / "caps").resolve()
    assert capture.suffix == ".cf32"
    assert sidecar == capture.with_suffix(".json")
    assert result["sample_count"] == 3
    assert result["duration_s"] == 0.3
    assert result["byte_count"] == 3 * 8
    np.testing.assert_array_equal(
        np.fromfile(capture, dtype=np.complex64), samples[-3:])
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["center_hz"] == 1.1e9
    assert meta["tuned_hz"] == 1.1e9
    assert meta["actual_samples"] == 3
    assert meta["actual_duration_s"] == 0.3
    assert meta["requested_seconds"] == 0.3
    assert meta["format"] == "complex64"
    assert meta["note"] == "operator note"
    assert meta["device"] == "test-device"
    assert meta["rx_channel"] == 0
    assert not list(capture.parent.glob(".*.tmp"))


def test_capture_empty_ring_is_503(tmp_path: Path, monkeypatch) -> None:
    eng = _engine_with_ring(
        tmp_path, monkeypatch, np.zeros(0, dtype=np.complex64))
    response = TestClient(create_app(eng)).post(
        "/api/iq/capture", json={"seconds": 0.2})
    assert response.status_code == 503


def test_capture_endpoint_uses_real_engine_and_atomic_writer(
        tmp_path: Path, monkeypatch) -> None:
    samples = np.arange(8, dtype=np.float32).astype(np.complex64) * (1 + 2j)
    eng = _engine_with_ring(tmp_path, monkeypatch, samples)

    response = TestClient(create_app(eng)).post(
        "/api/iq/capture",
        json={"seconds": 0.5, "note": "operator capture"},
    )

    assert response.status_code == 200, response.text
    result = response.json()
    capture = Path(result["path"])
    sidecar = Path(result["metadata_path"])
    assert capture.read_bytes() == samples[-5:].tobytes()
    assert json.loads(sidecar.read_text(encoding="utf-8"))["note"] == \
        "operator capture"
    assert not list(capture.parent.glob(".*.tmp"))


def test_capture_writer_runs_without_holding_ring_lock(
        tmp_path: Path, monkeypatch) -> None:
    eng = _engine_with_ring(
        tmp_path, monkeypatch, np.ones(4, dtype=np.complex64))
    real_write = engine_module.write_capture
    appended = threading.Event()

    def checked_write(*args, **kwargs):
        def append() -> None:
            eng._ring.write(np.array([9 + 2j], dtype=np.complex64))
            appended.set()

        thread = threading.Thread(target=append)
        thread.start()
        thread.join(timeout=0.5)
        assert appended.is_set(), "ring mutex was held during disk writer"
        return real_write(*args, **kwargs)

    monkeypatch.setattr(engine_module, "write_capture", checked_write)
    result = eng.capture_iq(seconds=0.2)

    assert result["sample_count"] == 2
    assert appended.is_set()


def test_concurrent_capture_is_busy_while_writer_runs(
        tmp_path: Path, monkeypatch) -> None:
    eng = _engine_with_ring(
        tmp_path, monkeypatch, np.ones(4, dtype=np.complex64))
    real_write = engine_module.write_capture
    writing = threading.Event()
    release = threading.Event()
    errors: list[Exception] = []

    def blocking_write(*args, **kwargs):
        writing.set()
        assert release.wait(2)
        return real_write(*args, **kwargs)

    def first_capture() -> None:
        try:
            eng.capture_iq(seconds=0.2)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    monkeypatch.setattr(engine_module, "write_capture", blocking_write)
    thread = threading.Thread(target=first_capture)
    thread.start()
    assert writing.wait(1)
    try:
        with np.testing.assert_raises(IQCaptureBusy):
            eng.capture_iq(seconds=0.2)
        response = TestClient(create_app(eng)).post(
            "/api/iq/capture", json={"seconds": 0.2})
        assert response.status_code == 409
    finally:
        release.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert errors == []


class _TwentyMspsSpyRing:
    capacity = 10_000_000
    last_write_age_s = 0.0

    def __init__(self) -> None:
        self.capture_context = {
            "center_hz": 1.1e9,
            "tuned_hz": 1.1e9,
            "sample_rate": 20e6,
            "bandwidth_hz": 16e6,
            "gain_db": 31,
        }
        self.copied_samples = 0
        self.copied_bytes = 0
        self.snapshot_calls = 0

    def snapshot_into(self, out: np.ndarray):
        self.snapshot_calls += 1
        self.copied_samples = int(out.size)
        self.copied_bytes = int(out.nbytes)
        return out, 0


def _engine_with_spy_ring(tmp_path: Path, monkeypatch
                          ) -> tuple[Engine, _TwentyMspsSpyRing]:
    monkeypatch.setattr(paths, "CAPS", tmp_path / "caps")
    eng = Engine(_Source(), CFG, Queue())
    eng.state.mode = "LOCK"
    eng.state.lock_target = 1.1e9
    ring = _TwentyMspsSpyRing()
    eng._ring = ring
    return eng, ring


def test_default_capture_at_20_msps_is_bounded_to_eight_mb(
        tmp_path: Path, monkeypatch) -> None:
    eng, ring = _engine_with_spy_ring(tmp_path, monkeypatch)

    def fake_write(path, *_args, **_kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path

    monkeypatch.setattr(engine_module, "write_capture", fake_write)
    result = eng.capture_iq()

    assert result["requested_duration_s"] == IQ_CAPTURE_DEFAULT_SECONDS
    assert ring.snapshot_calls == 1
    assert ring.copied_samples == 1_000_000
    assert ring.copied_bytes == 8_000_000
    assert ring.copied_bytes <= IQ_CAPTURE_MAX_BYTES


def test_half_second_at_20_msps_is_rejected_before_copy(
        tmp_path: Path, monkeypatch) -> None:
    eng, ring = _engine_with_spy_ring(tmp_path, monkeypatch)

    response = TestClient(create_app(eng)).post(
        "/api/iq/capture", json={"seconds": 0.5})

    assert response.status_code == 413
    assert "80.0 МБ" in response.json()["detail"]
    assert ring.snapshot_calls == 0
    assert ring.copied_bytes == 0


def test_stale_lock_ring_is_503_before_copy(
        tmp_path: Path, monkeypatch) -> None:
    eng, ring = _engine_with_spy_ring(tmp_path, monkeypatch)
    ring.last_write_age_s = 2.0

    response = TestClient(create_app(eng)).post(
        "/api/iq/capture", json={"seconds": 0.05})

    assert response.status_code == 503
    assert "не оновлювалось" in response.json()["detail"]
    assert ring.snapshot_calls == 0


def test_capture_route_runs_disk_work_off_event_loop() -> None:
    called = threading.Event()

    class _FakeEngine:
        cfg = {"rotator": {"enable": False}}
        events = Queue()

        def capture_iq(self, **_kwargs):
            called.set()
            time.sleep(0.12)
            return {"success": True}

        def snapshot(self):
            return {"mode": "LOCK"}

    async def run() -> None:
        app = create_app(_FakeEngine())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
                transport=transport, base_url="http://test") as client:
            request = asyncio.create_task(client.post(
                "/api/iq/capture",
                json={"seconds": 0.1, "note": "async"},
            ))
            await asyncio.to_thread(called.wait, 1)
            ticked = False

            async def ticker():
                nonlocal ticked
                await asyncio.sleep(0.02)
                ticked = True

            await ticker()
            assert ticked is True
            response = await request
            assert response.status_code == 200
            assert response.json()["success"] is True

    asyncio.run(run())


def test_capture_failure_logs_exception_without_exposing_it(caplog) -> None:
    class _FailingEngine:
        cfg = {"rotator": {"enable": False}}
        events = Queue()

        def capture_iq(self, **_kwargs):
            raise PermissionError("secret filesystem detail")

        def snapshot(self):
            return {"mode": "LOCK"}

    with caplog.at_level(logging.ERROR, logger="fpvscan.web.server"):
        response = TestClient(create_app(_FailingEngine())).post(
            "/api/iq/capture",
            json={"seconds": 0.5, "note": "operator capture"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "не вдалося записати IQ capture"}
    assert "IQ capture failed" in caplog.text
    assert "PermissionError: secret filesystem detail" in caplog.text


def test_numpy_short_write_counts_as_disk_full() -> None:
    from fpvscan.sdr.file import is_disk_full_error

    assert is_disk_full_error(OSError("1340000 requested and 786432 written"))
    assert is_disk_full_error(OSError(errno.ENOSPC, "No space left on device"))
    assert not is_disk_full_error(PermissionError("secret filesystem detail"))


def test_capture_disk_full_is_507_not_500(
        tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    import shutil

    samples = np.ones(4, dtype=np.complex64)
    eng = _engine_with_ring(tmp_path, monkeypatch, samples)
    monkeypatch.setattr(
        "fpvscan.sdr.file.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=100, used=99, free=64),
    )
    response = TestClient(create_app(eng)).post(
        "/api/iq/capture", json={"seconds": 0.2, "note": "diag"})
    assert response.status_code == 507
    assert "немає місця" in response.json()["detail"]
