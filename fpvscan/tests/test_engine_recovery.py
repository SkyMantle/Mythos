from __future__ import annotations

import threading
import time
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fpvscan.app.core.logging import sanitize_log_text
from fpvscan.engine import Engine
from fpvscan.sdr.bladerf import BladeRFError
from fpvscan.web.server import create_app
from run import _watch_engine


CFG = {
    "scan": {
        "start_hz": 1.0e9,
        "stop_hz": 1.1e9,
        "sample_rate": 2e6,
        "channel_bw_hz": 1e6,
        "fft_size": 64,
        "averages": 1,
        "priority_bands": False,
    },
    "video": {
        "sample_rate": 2e6,
        "channel_bw_hz": 1e6,
        "capture_ms": 10,
    },
    "sdr": {"gain_db": 20, "quick_tune": False},
    "rotator": {"enable": False},
}


class _NoDelayEvent:
    def __init__(self) -> None:
        self._set = False
        self._lock = threading.Lock()

    def set(self) -> None:
        with self._lock:
            self._set = True

    def clear(self) -> None:
        with self._lock:
            self._set = False

    def is_set(self) -> bool:
        with self._lock:
            return self._set

    def wait(self, _timeout=None) -> bool:
        return self.is_set()


class _LifecycleSource:
    name = "fake"
    fixed_freq = False
    sample_rate = 2e6
    center_freq = 1.0e9
    timeouts = 0
    stream_restarts = 0
    overflows = 0
    clip_frac = 0.0
    adc_rms = 0.0
    bias_tee = False

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def set_sample_rate(self, hz: float) -> None:
        self.sample_rate = hz

    def set_gain(self, _db: float) -> None:
        pass


class _OpenRecoverySource(_LifecycleSource):
    def __init__(self, failures: int | None) -> None:
        self.failures = failures
        self.open_calls = 0
        self.close_calls = 0

    def open(self) -> None:
        self.open_calls += 1
        if self.failures is None or self.open_calls <= self.failures:
            raise BladeRFError("bladerf_open: No devices available (-7)", code=-7)

    def close(self) -> None:
        self.close_calls += 1


class _ImmediateWait:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, event, delay: float) -> bool:
        self.delays.append(delay)
        return event.is_set()


def _recovery_engine(failures: int) -> tuple[Engine, threading.Event]:
    engine = Engine(_LifecycleSource(), CFG, Queue())
    engine._io_cancel = _NoDelayEvent()
    recovered = threading.Event()
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        if calls <= failures:
            raise BladeRFError(
                "sync_rx: Operation timed out (-6)", code=-6)
        if calls == failures + 1:
            recovered.set()
            return
        engine._stop.wait(1)

    engine._do_sweep = operation
    return engine, recovered


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return bool(predicate())


def test_first_timeout_then_success_keeps_engine_alive() -> None:
    engine, recovered = _recovery_engine(1)
    engine.start()
    try:
        assert recovered.wait(1)
        assert _wait_until(lambda: engine._sdr_consecutive_errors == 0)
        assert engine.worker_alive()
        assert engine._worker_failure is None
        assert engine._sdr_recovery_attempts == 1
        assert engine._sdr_consecutive_errors == 0
        assert engine._sdr_state == "OK"
    finally:
        engine.stop()


def test_eight_timeouts_do_not_silently_kill_engine() -> None:
    engine, recovered = _recovery_engine(8)
    engine.start()
    try:
        assert recovered.wait(1)
        assert _wait_until(lambda: engine._sdr_consecutive_errors == 0)
        assert engine.worker_alive()
        assert engine._worker_failure is None
        assert engine._sdr_recovery_attempts == 8
        assert engine._sdr_consecutive_errors == 0
    finally:
        engine.stop()


def test_open_no_device_retries_then_resumes_lock_on_same_worker() -> None:
    source = _OpenRecoverySource(failures=3)
    wait = _ImmediateWait()
    engine = Engine(source, CFG, Queue(), recovery_wait=wait)
    engine.state.mode = "LOCK"
    engine.state.lock_target = 1.1e9
    resumed = threading.Event()

    def lock_once() -> None:
        resumed.set()
        engine._stop.wait(1)

    engine._do_lock = lock_once
    engine.start()
    worker = engine._thread
    try:
        assert resumed.wait(1)
        assert engine._thread is worker
        assert engine.worker_alive()
        assert source.open_calls == 4
        assert engine._sdr_open_attempts == 4
        assert engine._sdr_open_failures == 3
        assert wait.delays[:3] == pytest.approx(
            [1.0, 2.0, 5.0], abs=0.001)
        assert engine._sdr_state == "OK"
        assert engine.state.mode == "LOCK"
        assert engine.state.lock_target == 1.1e9
    finally:
        engine.stop()


def test_permanent_no_device_keeps_health_and_worker_online() -> None:
    source = _OpenRecoverySource(failures=None)
    engine = Engine(source, CFG, Queue())
    watchdog_stop = threading.Event()
    server = SimpleNamespace(should_exit=False)
    watchdog = threading.Thread(
        target=_watch_engine,
        args=(engine, server, watchdog_stop),
        kwargs={"interval_s": 0.001},
    )
    engine.start()
    watchdog.start()
    try:
        assert _wait_until(lambda: engine._sdr_state == "WAITING_DEVICE")
        response = TestClient(create_app(engine)).get("/api/health")
        time.sleep(0.02)

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "degraded"
        assert body["engine_alive"] is True
        assert body["sdr_state"] == "WAITING_DEVICE"
        assert body["sdr_open_attempts"] >= 1
        assert body["sdr_open_failures"] >= 1
        assert "retry" in body["message"]
        assert engine._worker_failure is None
        assert server.should_exit is False
    finally:
        watchdog_stop.set()
        watchdog.join(timeout=1)
        engine.stop()


def test_shutdown_interrupts_open_backoff_promptly() -> None:
    source = _OpenRecoverySource(failures=None)
    waiting = threading.Event()

    def blocking_wait(event, delay: float) -> bool:
        assert delay > 0.9
        waiting.set()
        return event.wait(delay)

    engine = Engine(source, CFG, Queue(), recovery_wait=blocking_wait)
    engine.start()
    assert waiting.wait(1)

    started = time.monotonic()
    engine.stop()

    assert time.monotonic() - started < 0.5
    assert not engine.worker_alive()


def test_mode_command_during_open_backoff_resumes_manual_sweep_hold() -> None:
    source = _OpenRecoverySource(failures=1)
    waiting = threading.Event()
    wait_calls = 0

    def interruptible_wait(event, delay: float) -> bool:
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            waiting.set()
            return event.wait(delay)
        return False

    engine = Engine(source, CFG, Queue(), recovery_wait=interruptible_wait)
    engine.state.mode = "LOCK"
    engine.state.lock_target = 5.1e9
    resumed = threading.Event()

    def sweep_once() -> None:
        resumed.set()
        engine._stop.wait(1)

    engine._do_sweep = sweep_once
    engine.start()
    try:
        assert waiting.wait(1)
        engine.command("sweep", auto_lock=False)
        assert resumed.wait(1)
        assert engine.state.mode == "SWEEP"
        assert engine.state.lock_target is None
        assert engine._sweep_auto_lock is False
        assert source.open_calls == 2
        assert engine.worker_alive()
    finally:
        engine.stop()


def test_repeated_sync_timeout_never_reopens_and_recovers() -> None:
    source = _OpenRecoverySource(failures=0)
    wait = _ImmediateWait()
    engine = Engine(source, CFG, Queue(), recovery_wait=wait)
    recovered = threading.Event()
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise BladeRFError(
                "sync_rx: Operation timed out after stream restart (-6)",
                code=-6,
            )
        recovered.set()
        engine._stop.wait(1)

    engine._do_sweep = operation
    engine.start()
    try:
        assert recovered.wait(1)
        assert _wait_until(lambda: engine._sdr_state == "OK", timeout=2.0)
        assert source.open_calls == 1
        assert engine._sdr_open_attempts == 1
        assert engine._sdr_recovery_attempts == 3
        assert engine.worker_alive()
        assert engine._sdr_state == "OK"
        assert engine._worker_failure is None
    finally:
        engine.stop()


def test_successful_sweep_retune_read_clears_reconfig_and_health_is_ok() -> None:
    class SweepSource(_LifecycleSource):
        def set_read_cancel_event(self, event) -> None:
            self.cancel = event

        def retune_and_read(self, hz: float, n: int) -> np.ndarray:
            self.center_freq = hz
            return np.zeros(n, dtype=np.complex64)

    engine = Engine(SweepSource(), CFG, Queue())
    engine.worker_alive = lambda: True
    engine._set_sdr_state("RECONFIGURING")

    iq = engine._grab(1.05e9, 64)
    health = TestClient(create_app(engine)).get("/api/health")

    assert iq.size == 64
    assert engine._sdr_state == "OK"
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["sdr_state"] == "OK"


def test_no_device_gain_records_gain_as_root_operation() -> None:
    class LostGainSource(_LifecycleSource):
        def set_gain(self, _db: float) -> None:
            raise BladeRFError(
                "set_gain: No devices available (-7)", code=-7)

    engine = Engine(LostGainSource(), CFG, Queue())
    engine._set_sdr_state("OK")

    with pytest.raises(BladeRFError) as caught:
        engine._apply_bias_tee_gain(False)
    engine._record_sdr_error(
        caught.value, "close device and enter open recovery loop")

    assert engine._sdr_operation == "idle"
    assert engine._sdr_last_operation == "gain"
    assert engine._sdr_last_error_operation == "gain"
    assert engine._sdr_root_error_operation == "gain"


def test_failed_sweep_retune_enters_waiting_device_and_health_is_503() -> None:
    class LostRetuneSource(_LifecycleSource):
        def __init__(self) -> None:
            self.open_calls = 0

        def open(self) -> None:
            self.open_calls += 1
            if self.open_calls > 1:
                raise BladeRFError(
                    "bladerf_open: No devices available (-7)", code=-7)

        def set_read_cancel_event(self, event) -> None:
            self.cancel = event

        def retune_and_read(self, _hz: float, _n: int) -> np.ndarray:
            raise BladeRFError(
                "set_frequency: No devices available (-7)", code=-7)

    engine = Engine(LostRetuneSource(), CFG, Queue())
    engine.start()
    try:
        assert _wait_until(lambda: engine._sdr_state == "WAITING_DEVICE")
        health = TestClient(create_app(engine)).get("/api/health")
        assert health.status_code == 503
        assert health.json()["sdr_state"] == "WAITING_DEVICE"
        assert health.json()["last_error_operation"] == "frequency"
        assert engine.worker_alive()
        assert engine._worker_failure is None
    finally:
        engine.stop()


def test_device_lost_quiesces_then_reopens_through_waiting_device() -> None:
    source = _OpenRecoverySource(failures=0)
    wait = _ImmediateWait()
    engine = Engine(source, CFG, Queue(), recovery_wait=wait)
    recovered = threading.Event()
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise BladeRFError(
                "sync_rx: No devices available (-7)", code=-7)
        recovered.set()
        engine._stop.wait(1)

    engine._do_sweep = operation
    engine.start()
    try:
        assert recovered.wait(1)
        assert source.open_calls == 2
        assert engine._sdr_open_attempts == 2
        assert source.close_calls >= 3
        assert engine.worker_alive()
        assert engine._sdr_state == "OK"
        assert engine._worker_failure is None
        health = TestClient(create_app(engine)).get("/api/health")
        assert health.status_code == 200
        assert health.json()["sdr_state"] == "OK"
    finally:
        engine.stop()


def test_device_lost_root_code_survives_failed_restore_without_watchdog() -> None:
    class RestoreFailsSource(_LifecycleSource):
        def __init__(self) -> None:
            self.open_calls = 0
            self.close_calls = 0

        def open(self) -> None:
            self.open_calls += 1

        def close(self) -> None:
            self.close_calls += 1

        def set_sample_rate(self, hz: float) -> None:
            if self.open_calls >= 2:
                raise BladeRFError(
                    "set_sample_rate: unexpected error (-1)", code=-1)
            self.sample_rate = hz

    source = RestoreFailsSource()
    engine = Engine(source, CFG, Queue())
    first = True

    def operation() -> None:
        nonlocal first
        if first:
            first = False
            raise BladeRFError(
                "sync_rx: No devices available (-5)", code=-5)
        engine._stop.wait(1)

    engine._do_sweep = operation
    watchdog_stop = threading.Event()
    server = SimpleNamespace(should_exit=False)
    watchdog = threading.Thread(
        target=_watch_engine,
        args=(engine, server, watchdog_stop),
        kwargs={"interval_s": 0.001},
    )
    engine.start()
    watchdog.start()
    try:
        assert _wait_until(
            lambda: source.open_calls >= 2
            and engine._sdr_state == "WAITING_DEVICE",
            timeout=1.0,
        )
        assert engine._sdr_last_error_code == -5
        assert "No devices available" in (engine._sdr_last_error or "")
        assert engine.worker_alive()
        assert engine._worker_failure is None
        assert server.should_exit is False
    finally:
        watchdog_stop.set()
        watchdog.join(timeout=1)
        engine.stop()


def test_unexpected_programming_exception_remains_fatal_for_watchdog() -> None:
    class BrokenSource(_LifecycleSource):
        def open(self) -> None:
            raise TypeError("bad source implementation")

    engine = Engine(BrokenSource(), CFG, Queue())
    engine._worker_started = True
    with pytest.raises(TypeError, match="bad source implementation"):
        engine._worker_entry()

    server = SimpleNamespace(should_exit=False)
    _watch_engine(engine, server, threading.Event(), interval_s=0.001)

    assert server.should_exit
    assert engine._sdr_state == "ERROR"
    assert "bad source implementation" in (engine._worker_failure or "")


class _BlockedSweepSource(_LifecycleSource):
    def __init__(self) -> None:
        self.cancel = None
        self.entered = threading.Event()
        self.active = 0
        self.max_active = 0

    def set_read_cancel_event(self, event) -> None:
        self.cancel = event

    def retune_and_read(self, _hz: float, n: int) -> np.ndarray:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            while not self.cancel.is_set():
                time.sleep(0.001)
            raise InterruptedError("cancelled blocked SWEEP read")
        finally:
            self.active -= 1


def test_sweep_to_lock_cancels_stale_generation_without_overlap() -> None:
    source = _BlockedSweepSource()
    engine = Engine(source, CFG, Queue())
    stale_processed = 0
    locked = threading.Event()

    def sweep_once() -> None:
        nonlocal stale_processed
        engine._grab(1.05e9, 64)
        stale_processed += 1

    def lock_once() -> None:
        locked.set()
        engine._stop.wait(1)

    engine._do_sweep = sweep_once
    engine._do_lock = lock_once
    engine.start()
    try:
        assert source.entered.wait(1)
        generation = engine._sweep_generation
        engine.command("lock", freq_hz=5.1e9, force=True)
        assert locked.wait(1)
        assert engine.state.mode == "LOCK"
        assert engine.state.lock_target == 5.1e9
        assert engine._sweep_generation == generation + 1
        assert stale_processed == 0
        assert source.max_active == 1
    finally:
        engine.stop()


class _ReaderTransitionSource(_LifecycleSource):
    timeout_ms = 50

    def __init__(self) -> None:
        self.cancel = None
        self.reader_entered = threading.Event()
        self.active = 0
        self.max_active = 0
        self.direct_after_reader = False

    def set_read_cancel_event(self, event) -> None:
        self.cancel = event

    def retune_and_read(self, _hz: float, n: int) -> np.ndarray:
        assert self.active == 0
        self.direct_after_reader = self.reader_entered.is_set()
        return np.zeros(n, dtype=np.complex64)

    def read(self, n: int) -> np.ndarray:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.reader_entered.set()
        try:
            while not self.cancel.is_set():
                time.sleep(0.001)
            raise InterruptedError("reader stopped")
        finally:
            self.active -= 1


def test_lock_to_sweep_joins_reader_before_direct_grab() -> None:
    source = _ReaderTransitionSource()
    engine = Engine(source, CFG, Queue())
    engine.state.mode = "LOCK"
    engine.state.lock_target = 1.1e9
    engine._start_reader(1.1e9, 2e6, 0.01)
    assert source.reader_entered.wait(1)

    engine._handle_command("sweep", {"auto_lock": False})
    iq = engine._grab(1.05e9, 64)

    assert iq.size == 64
    assert engine._reader_thread is None
    assert source.active == 0
    assert source.max_active == 1
    assert source.direct_after_reader


class _HealthEngine:
    cfg = {"rotator": {"enable": False}}
    events = Queue()

    def __init__(self, snapshot: dict) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> dict:
        return dict(self._snapshot)


def test_health_reflects_dead_and_degraded_worker() -> None:
    dead = TestClient(create_app(_HealthEngine({
        "engine_alive": False,
        "sdr_state": "ERROR",
        "engine_error": "worker died",
    }))).get("/api/health")
    degraded = TestClient(create_app(_HealthEngine({
        "engine_alive": True,
        "sdr_state": "DEGRADED",
        "sdr_consecutive_errors": 8,
        "sdr_last_error": "sync_rx timeout",
    }))).get("/api/health")
    waiting = TestClient(create_app(_HealthEngine({
        "engine_alive": True,
        "sdr_state": "WAITING_DEVICE",
        "sdr_open_attempts": 4,
        "sdr_open_failures": 4,
        "sdr_next_retry_ms": 5000,
        "sdr_last_error": "No devices available",
        "sdr_last_error_code": -7,
        "sdr_last_recovery_action": "waiting for device",
    }))).get("/api/health")
    short_reconfigure = TestClient(create_app(_HealthEngine({
        "engine_alive": True,
        "sdr_state": "RECONFIGURING",
        "sdr_state_age_ms": 250,
    }))).get("/api/health")
    stuck_reconfigure = TestClient(create_app(_HealthEngine({
        "engine_alive": True,
        "sdr_state": "RECONFIGURING",
        "sdr_state_age_ms": 5_001,
    }))).get("/api/health")

    assert dead.status_code == 503
    assert dead.json()["status"] == "unhealthy"
    assert degraded.status_code == 503
    assert degraded.json()["status"] == "degraded"
    assert degraded.json()["sdr_consecutive_errors"] == 8
    assert waiting.status_code == 503
    assert waiting.json()["engine_alive"] is True
    assert waiting.json()["sdr_state"] == "WAITING_DEVICE"
    assert waiting.json()["sdr_open_attempts"] == 4
    assert waiting.json()["last_error_code"] == -7
    assert short_reconfigure.status_code == 200
    assert short_reconfigure.json()["status"] == "ok"
    assert stuck_reconfigure.status_code == 503
    assert stuck_reconfigure.json()["status"] == "degraded"


def test_watchdog_requests_server_exit_after_worker_death() -> None:
    engine = SimpleNamespace(
        _worker_started=True,
        _stop=threading.Event(),
        worker_alive=lambda: False,
    )
    server = SimpleNamespace(should_exit=False)
    _watch_engine(engine, server, threading.Event(), interval_s=0.001)
    assert server.should_exit


def test_timeout_log_text_has_no_raw_control_characters() -> None:
    text = sanitize_log_text(
        "sync_rx:\x00 Operation\r timed out (-6)\x1b[31m")
    assert "\x00" not in text
    assert "\r" not in text
    assert "\x1b" not in text
    assert "\\x00" in text
    assert "\\r" in text
