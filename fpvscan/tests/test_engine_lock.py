#!/usr/bin/env python3
"""Регресія: LOCK не повинен рвати IQ-нитку через дрібний роз'їзд fs."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from queue import Queue

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan.dsp import demod, spectrum
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


def test_inspect_bw_does_not_open_full_nyquist_for_typical_vtx():
    """10 МГц зайнятості + 2·MERGE_TOL (12 МГц) = 22 МГц вікно.

    При типових 35 Мвідл/с це int(fs/out_bw)=1: каналайзер не децимує,
    сусідній Raceband (~19 МГц) аліаситься в ЧМ-дискримінатор, і
    INSPECT підтверджує шпору/спідницю як окреме відео.
    """
    fs = 35e6
    occupied = 10e6
    neighbor = 19e6
    src = _Source(fs, fs)
    eng = Engine(src, _cfg(fs), Queue())
    out_bw = eng._inspect_bw(occupied)
    dec = max(1, int(fs / out_bw))
    assert out_bw <= occupied + 2e6, (
        f"вікно INSPECT {out_bw/1e6:.1f} МГц — зашироке для "
        f"зайнятості {occupied/1e6:.0f} МГц (не можна додавати 2·MERGE_TOL)"
    )
    assert out_bw >= occupied, "вікно не має бути вужчим за зміряну зайнятість"
    assert dec >= 2, (
        f"dec={dec} при out_bw={out_bw/1e6:.1f} МГц: каналайзер вимкнув "
        f"фільтр і пропустить сусіда на 19 МГц"
    )
    nyq = fs / 2
    alias = neighbor - fs
    assert abs(alias) < nyq, "передумова: аліас сусіда лежить у смузі ADC"
    assert out_bw < fs / 2, (
        f"out_bw={out_bw/1e6:.1f} МГц ≥ fs/2 — dec=1, аліас {alias/1e6:.1f} МГц "
        f"потрапляє в дискримінатор"
    )
    old = max(occupied + 2 * Engine.MERGE_TOL_HZ, 8e6)
    assert max(1, int(fs / old)) == 1, "старий 2·MERGE_TOL має лишатись зламаним"


def test_inspect_bw_hides_neighbor_line_rate_from_spur():
    """Шпора на DC + живий борт на 16 МГц (всередині Найквіста 35 Мвідл/с).

    Старе вікно 22 МГц (dec=1) не фільтрує сусіда — INSPECT підтверджує
    шпору як PAL/NTSC. Вузьке вікно FIR-децимує, рядкова сусіда зникає.
    """
    fs = 35e6
    n = int(fs * 0.04)
    t = np.arange(n, dtype=np.float64) / fs
    line = demod.LINE_NTSC
    sync = (np.mod(t * line, 1.0) < 0.08).astype(np.float64)

    def fm_at(if_hz: float, dev_hz: float = 2e6) -> np.ndarray:
        inst = if_hz + np.where(sync > 0.5, -dev_hz * 0.5, dev_hz * 0.5)
        ph = 2 * np.pi * np.cumsum(inst) / fs
        return np.exp(1j * ph).astype(np.complex64)

    spur = 0.15 * np.exp(2j * np.pi * 0.3e6 * t).astype(np.complex64)
    iq = spur + fm_at(16e6)
    eng = Engine(_Source(fs, fs), _cfg(fs), Queue())
    occupied = 10e6
    new_bw = eng._inspect_bw(occupied)
    old_bw = max(occupied + 2 * Engine.MERGE_TOL_HZ, 8e6)

    def classify(bw: float):
        ch, fs2 = demod.channelize(iq, fs, 0.0, out_bw_hz=bw, fast=False)
        base = demod.fm_demod(ch, fs2, deviation_hz=max(bw, 8e6) / 5)
        # FIR startup is not the leak — drop a tenth so we score steady state.
        return demod.classify_video(base[len(base) // 10:], fs2)

    old = classify(old_bw)
    new = classify(new_bw)
    assert old.is_video, (
        f"передумова: широке вікно має бачити рядкову сусіда, got {old}"
    )
    assert not new.is_video, (
        f"після звуження INSPECT шпора не має проходити як відео, got {new}"
    )

    ch, fs2 = demod.channelize(fm_at(0.0, 4e6), fs, 0.0,
                               out_bw_hz=new_bw, fast=False)
    base = demod.fm_demod(ch, fs2, deviation_hz=max(new_bw, 8e6) / 5)
    on_ch = demod.classify_video(base[len(base) // 10:], fs2)
    assert on_ch.is_video, f"свій канал не має відсіюватись: {on_ch}"


def test_inspect_passes_helper_bw_to_channelize():
    """_inspect має різати смугу через _inspect_bw, а не 2·MERGE_TOL."""
    fs = 35e6
    occupied = 10e6
    src = _Source(fs, fs)
    cfg = _cfg(fs)
    cfg["scan"]["inspect_ms"] = 5
    eng = Engine(src, cfg, Queue())
    seen: list[float] = []
    orig = demod.channelize

    def wrap(iq, rate, offset_hz, out_bw_hz, fast=True):
        seen.append(float(out_bw_hz))
        return orig(iq, rate, offset_hz, out_bw_hz, fast=fast)

    demod.channelize = wrap
    try:
        occ = spectrum.Occupancy(
            center_hz=5806e6, bandwidth_hz=occupied,
            peak_db=-20.0, snr_db=15.0,
        )
        eng._inspect(None, 5800e6, fs, occ)
    finally:
        demod.channelize = orig
    assert seen, "channelize не викликався"
    assert seen[0] == eng._inspect_bw(occupied)
    assert seen[0] != occupied + 2 * Engine.MERGE_TOL_HZ


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
    test_inspect_bw_does_not_open_full_nyquist_for_typical_vtx()
    test_inspect_bw_hides_neighbor_line_rate_from_spur()
    test_inspect_passes_helper_bw_to_channelize()
    test_manual_lock_listener_is_registered_once()
    print("OK")
