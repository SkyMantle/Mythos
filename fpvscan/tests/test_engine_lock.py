#!/usr/bin/env python3
"""Регресія: LOCK не повинен рвати IQ-нитку через дрібний роз'їзд fs."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from queue import Queue, Empty

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


def test_manual_lock_listener_is_registered_once():
    """Heartbeat applyState() раніше вішав новий keydown щодва секунди."""
    html = (Path(__file__).resolve().parents[1]
            / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    start = html.index("function applyState")
    end = html.index("async function lock")
    apply_body = html[start:end]
    assert "addEventListener" not in apply_body
    assert html.count("manual-freq').addEventListener") == 1


class _RangeSource(_Source):
    """Імітує підлогу/стелю bladeRF 2.0: 47 МГц … 6 ГГц."""

    def __init__(self, fs: float = 2_000_000.0):
        super().__init__(fs, fs)
        self.last_tune = None

    def set_center_freq(self, hz: float):
        if hz < 47e6 or hz > 6e9:
            raise RuntimeError(f"out of range {hz}")
        self.last_tune = float(hz)

    def retune_and_read(self, hz: float, n: int) -> np.ndarray:
        self.retune_calls += 1
        self.set_center_freq(hz)
        return self.read(n)


def test_lock_below_board_floor_returns_to_sweep():
    """«5.8» в полі МГц (мали на увазі ГГц) не повинно вбити нитку."""
    src = _RangeSource()
    eng = Engine(src, _cfg(src.actual_fs), Queue())
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5.8e6
    try:
        for _ in range(10):
            eng._do_lock()
        assert eng.state.mode == "SWEEP", eng.state.mode
        assert eng.state.lock_target is None
        assert eng._ring is None
        assert eng._reader_thread is None
    finally:
        eng._stop_reader()


def test_lock_above_board_ceiling_returns_to_sweep():
    src = _RangeSource()
    eng = Engine(src, _cfg(src.actual_fs), Queue())
    eng.state.mode = "LOCK"
    eng.state.lock_target = 99999e6
    try:
        eng._do_lock()
        assert eng.state.mode == "SWEEP"
        assert eng.state.lock_target is None
    finally:
        eng._stop_reader()


def test_lock_in_band_still_starts_reader():
    src = _RangeSource()
    eng = Engine(src, _cfg(src.actual_fs), Queue())
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        eng._do_lock()
        assert eng.state.mode == "LOCK"
        assert src.last_tune == 5800e6
        assert eng._ring is not None
    finally:
        eng._stop_reader()


def test_run_survives_rejected_lock():
    """Бюджет USB (8 підряд) раніше вбивав _run після «5.8» МГц."""
    src = _RangeSource()
    cfg = _cfg(src.actual_fs)
    cfg["scan"]["start_hz"] = 5700e6
    cfg["scan"]["stop_hz"] = 5900e6
    cfg["scan"]["step_hz"] = 50e6
    eng = Engine(src, cfg, Queue())
    eng.start()
    try:
        t0 = time.perf_counter()
        while not (eng._thread and eng._thread.is_alive()) and time.perf_counter() - t0 < 2:
            time.sleep(0.01)
        assert eng._thread is not None and eng._thread.is_alive()
        eng.command("lock", freq_hz=5.8e6)
        t0 = time.perf_counter()
        rejected = False
        while time.perf_counter() - t0 < 3:
            try:
                ev = eng.events.get(timeout=0.05)
            except Empty:
                continue
            if ev.get("type") == "notice" and "5.800" in str(ev.get("data", {}).get("text", "")):
                rejected = True
                break
        assert rejected, "не було повідомлення про відмову стати на 5.8 МГц"
        assert eng._thread.is_alive(), "робоча нитка померла після відхиленого LOCK"
        assert eng.state.mode == "SWEEP"
        assert eng.state.lock_target is None
        assert src.retune_calls >= 1
    finally:
        eng.stop()


if __name__ == "__main__":
    test_lock_keeps_reader_when_fs_off_by_fraction()
    test_lock_retunes_when_fs_really_changes()
    test_run_writes_actual_rate_into_cfg()
    test_manual_lock_listener_is_registered_once()
    test_lock_below_board_floor_returns_to_sweep()
    test_lock_above_board_ceiling_returns_to_sweep()
    test_lock_in_band_still_starts_reader()
    test_run_survives_rejected_lock()
    print("OK")
