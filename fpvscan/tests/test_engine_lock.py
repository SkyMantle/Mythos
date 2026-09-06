#!/usr/bin/env python3
"""Регресія: LOCK не повинен рвати IQ-нитку через дрібний роз'їзд fs."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from queue import Queue

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan.engine import Engine


class _Source:
    """Імітує плату, яка віддає сусідню частоту дискретизації."""

    name = "mock"

    def __init__(self, requested_fs: float, actual_fs: float):
        self.requested_fs = requested_fs
        self.actual_fs = actual_fs
        self._fs = actual_fs
        self.set_sample_rate_calls = 0
        self.retune_calls = 0

    @property
    def sample_rate(self):
        return self._fs

    @property
    def center_freq(self):
        return 5800e6

    def open(self):
        pass

    def close(self):
        pass

    def set_gain(self, db):
        pass

    def set_sample_rate(self, hz):
        self.set_sample_rate_calls += 1
        self.requested_fs = float(hz)
        self._fs = self.actual_fs

    def set_center_freq(self, hz):
        pass

    def read(self, n: int) -> np.ndarray:
        return np.zeros(int(n), dtype=np.complex64)

    def retune_and_read(self, hz: float, n: int) -> np.ndarray:
        self.retune_calls += 1
        return self.read(n)


def _cfg(fs: float) -> dict:
    return {
        "sdr": {"gain_db": 30},
        "scan": {
            "sample_rate": fs,
            "start_hz": 400e6,
            "stop_hz": 6000e6,
            "confirm_hits": 2,
        },
        "video": {
            "sample_rate": fs,
            "channel_bw_hz": 8e6,
            "lo_offset_hz": 0.0,
            "capture_ms": 10,
            "idle_ms": 0,
            "ring_seconds": 0.05,
            "afc": False,
            "width": 64,
            "spectrum_every": 10_000,
            "min_lines": 250,
        },
    }


def test_lock_keeps_reader_when_fs_off_by_fraction():
    """0.4 Гц різниці: точне `!=` спрацювало б щокадру, допуск 1 Гц — ні."""
    want = 2_000_000.0
    src = _Source(want, want + 0.4)
    eng = Engine(src, _cfg(want), Queue())
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        for _ in range(3):
            eng._do_lock()
        assert src.retune_calls == 1, (
            f"читач перезапускався {src.retune_calls} разів "
            f"(має бути один старт на весь LOCK)"
        )
    finally:
        eng._stop_reader()


def test_lock_retunes_when_fs_really_changes():
    want = 2_000_000.0
    src = _Source(want, want)
    eng = Engine(src, _cfg(want), Queue())
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        eng._do_lock()
        assert src.retune_calls == 1
        src.actual_fs = want + 50_000.0
        src._fs = src.actual_fs
        eng._do_lock()
        assert src.retune_calls == 2, (
            f"справжня зміна fs мала перезапустити читач, "
            f"було retune_calls={src.retune_calls}"
        )
        assert src.set_sample_rate_calls >= 1
    finally:
        eng._stop_reader()


def test_run_writes_actual_rate_into_cfg():
    want = 35_000_000.0
    actual = 34_999_872.0
    src = _Source(want, actual)
    cfg = _cfg(want)
    eng = Engine(src, cfg, Queue())
    eng._stop.set()
    t0 = time.perf_counter()
    eng._run()
    assert time.perf_counter() - t0 < 2.0
    assert cfg["scan"]["sample_rate"] == actual
    assert cfg["video"]["sample_rate"] == actual


class _SlowReadSource(_Source):
    """read() hangs past the old 2 s join, like bladeRF sync_rx / _recover."""

    def __init__(self, requested_fs: float, actual_fs: float, block_s: float):
        super().__init__(requested_fs, actual_fs)
        self.block_s = block_s
        self._gate = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.entered = threading.Event()

    def _enter(self):
        with self._gate:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)

    def _leave(self):
        with self._gate:
            self.in_flight -= 1

    def read(self, n: int) -> np.ndarray:
        self._enter()
        self.entered.set()
        try:
            time.sleep(self.block_s)
            return np.zeros(int(n), dtype=np.complex64)
        finally:
            self._leave()

    def retune_and_read(self, hz: float, n: int) -> np.ndarray:
        # Counts as using the device: must not overlap an in-flight read().
        self._enter()
        try:
            self.retune_calls += 1
            return np.zeros(int(n), dtype=np.complex64)
        finally:
            self._leave()


def test_start_reader_waits_out_in_flight_read():
    """AFC recenter / channel change must not touch src during sync_rx.

    BladeRF.read() can block for timeout_ms (3.5 s) plus _recover() sleeps
    (~8 s). _stop_reader used to join(2) and then _start_reader retuned
    the same handle — two owner threads, which the SDR contract forbids
    and which hangs NIOS if sync_config runs mid-transfer.
    """
    want = 2_000_000.0
    src = _SlowReadSource(want, want, block_s=2.3)
    eng = Engine(src, _cfg(want), Queue())
    try:
        eng._start_reader(5800e6, want, 0.05)
        assert src.entered.wait(1.0), "reader never entered src.read"
        eng._start_reader(5801e6, want, 0.05)
        assert src.max_in_flight == 1, (
            f"src was used from two threads at once "
            f"(max in-flight={src.max_in_flight})"
        )
    finally:
        eng._stop_reader()


def test_start_reader_refuses_if_join_times_out():
    """A wedged read must not be abandoned so a second thread can use src."""
    want = 2_000_000.0
    src = _SlowReadSource(want, want, block_s=0.5)
    eng = Engine(src, _cfg(want), Queue())
    eng._reader_join_s = lambda: 0.05  # type: ignore[method-assign]
    try:
        eng._start_reader(5800e6, want, 0.05)
        assert src.entered.wait(1.0)
        raised = False
        try:
            eng._start_reader(5801e6, want, 0.05)
        except RuntimeError as e:
            raised = True
            assert "IQ" in str(e) or "читання" in str(e)
        assert raised, "second start must refuse while the old read owns src"
        assert src.max_in_flight == 1
    finally:
        eng._reader_join_s = lambda: 2.0  # type: ignore[method-assign]
        eng._stop_reader()


def test_manual_lock_listener_is_registered_once():
    """Heartbeat applyState() раніше вішав новий keydown щодва секунди."""
    html = (Path(__file__).resolve().parents[1]
            / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    start = html.index("function applyState")
    end = html.index("async function lock")
    apply_body = html[start:end]
    assert "addEventListener" not in apply_body
    assert html.count("manual-freq').addEventListener") == 1


if __name__ == "__main__":
    test_lock_keeps_reader_when_fs_off_by_fraction()
    test_lock_retunes_when_fs_really_changes()
    test_run_writes_actual_rate_into_cfg()
    test_start_reader_waits_out_in_flight_read()
    test_start_reader_refuses_if_join_times_out()
    test_manual_lock_listener_is_registered_once()
    print("OK")
