#!/usr/bin/env python3
"""Регресія: LOCK не повинен рвати IQ-нитку через дрібний роз'їзд fs."""
from __future__ import annotations

import sys
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
    test_manual_lock_listener_is_registered_once()
    print("OK")
