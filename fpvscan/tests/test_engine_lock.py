#!/usr/bin/env python3
"""Регресія: LOCK не повинен рвати IQ-нитку через дрібний роз'їзд fs."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from queue import Queue

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan.engine import Detection, Engine


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


def test_lock_bw_does_not_open_full_nyquist_for_typical_vtx():
    """10 МГц зайнятості + 2·MERGE_TOL (12 МГц) = 22 МГц вікно.

    При типових 35 Мвідл/с це int(fs/ch_bw)=1: каналайзер не децимує,
    сусідній Raceband (19 МГц) аліаситься в ЧМ-дискримінатор і LOCK
    показує суміш двох бортів.
    """
    fs = 35e6
    src = _Source(fs, fs)
    eng = Engine(src, _cfg(fs), Queue())
    occupied = 10e6
    neighbor = 19e6  # крок Raceband (R5=5806, R6=5843)
    eng.state.detections[5806] = Detection(
        freq_hz=5806e6, bandwidth_hz=occupied, snr_db=18.0, hits=2,
    )
    ch_bw = eng._lock_bw(5806e6, 10e6)
    dec = max(1, int(fs / ch_bw))
    assert ch_bw <= occupied + 2e6, (
        f"канал утримання {ch_bw/1e6:.1f} МГц — заширокий для "
        f"зайнятості {occupied/1e6:.0f} МГц (не можна додавати 2·MERGE_TOL)"
    )
    assert ch_bw >= occupied, "вікно не має бути вужчим за зміряну зайнятість"
    assert dec >= 2, (
        f"dec={dec} при ch_bw={ch_bw/1e6:.1f} МГц: каналайзер вимкнув "
        f"фільтр і пропустить сусіда на 19 МГц"
    )
    # Без децимації сусід на 19 МГц при 35 Мвідл/с аліаситься в −16 МГц
    # (всередині Найквіста ±17.5 МГц) і потрапляє в дискримінатор.
    nyq = fs / 2
    alias = neighbor - fs
    assert abs(alias) < nyq, "передумова: аліас сусіда лежить у смузі ADC"
    assert ch_bw < fs / 2, (
        f"ch_bw={ch_bw/1e6:.1f} МГц ≥ fs/2 — dec=1, аліас {alias/1e6:.1f} МГц "
        f"не відфільтровано"
    )


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
    test_lock_bw_does_not_open_full_nyquist_for_typical_vtx()
    test_manual_lock_listener_is_registered_once()
    print("OK")
