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


class _ToneSource:
    """Несуча на фіксованій RF: ІЧ = RF − LO. Імітує розстроєний VTx."""

    name = "tone"

    def __init__(self, fs: float, rf_hz: float):
        self._fs = float(fs)
        self._fc = 0.0
        self.rf_hz = float(rf_hz)
        self.tune_history: list[float] = []
        self._t = 0.0

    @property
    def sample_rate(self):
        return self._fs

    @property
    def center_freq(self):
        return self._fc

    def open(self):
        pass

    def close(self):
        pass

    def set_gain(self, db):
        pass

    def set_sample_rate(self, hz):
        self._fs = float(hz)

    def set_center_freq(self, hz):
        self._fc = float(hz)
        self.tune_history.append(self._fc)

    def read(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.complex64)
        t = self._t + np.arange(n, dtype=np.float64) / self._fs
        ifc = self.rf_hz - self._fc
        out = np.exp(2j * np.pi * ifc * t).astype(np.complex64)
        self._t += n / self._fs
        return out

    def retune_and_read(self, hz: float, n: int) -> np.ndarray:
        self.set_center_freq(hz)
        return self.read(n)


def _afc_cfg(fs: float, **video_kw) -> dict:
    cfg = _cfg(fs)
    cfg["video"].update({
        "afc": True,
        "afc_gain": 0.5,
        "afc_deadband_hz": 80e3,
        "afc_limit_hz": 8e6,
        "lo_offset_hz": 0.0,
        "channel_bw_hz": 8e6,
        "capture_ms": 10,
        "idle_ms": 0,
        "ring_seconds": 0.08,
        "spectrum_every": 10_000,
    }, **video_kw)
    return cfg


def _run_lock(eng: Engine, src: _ToneSource, n: int = 12) -> None:
    try:
        for _ in range(n):
            eng._do_lock()
    finally:
        eng._stop_reader()


def test_afc_still_corrects_when_channel_is_wider_than_nyquist_pad():
    """Типовий 5.8 ГГц: зайнята смуга + 12 МГц запас обнуляли safe_lim.

    Раніше clamp зануляв _afc *до* перевірки lim<=0, тож ані цифровий
    зсув, ані апаратна перебудова не рухались — LOCK лишався на
    розстроєній несучій.
    """
    fs = 20e6
    target = 5800e6
    offset = 1.2e6
    src = _ToneSource(fs, target + offset)
    eng = Engine(src, _afc_cfg(fs), Queue())
    eng.state.mode = "LOCK"
    eng.state.lock_target = target
    # 30 МГц зайнято + 2*MERGE_TOL → ch_bw=42 МГц > 0.9*fs → старий lim=0
    eng.state.detections[5800] = Detection(
        freq_hz=target, bandwidth_hz=30e6, snr_db=20.0, hits=2,
    )
    _run_lock(eng, src)
    residual = abs((target + offset) - src.center_freq - eng._afc)
    assert residual < 250e3, (
        f"AFC не злазила з розстройки: LO={src.center_freq:.0f} "
        f"afc={eng._afc:.0f} afc_hw={eng._afc_hw:.0f} residual={residual:.0f}"
    )


def test_afc_recenters_accumulate_instead_of_rewinding():
    """Друга апаратна перебудова має додаватись до першої, не цілитись
    знову від f+off (інакше LO застрягає на lim, а DecodeState скидається)."""
    fs = 20e6
    target = 5800e6
    offset = 2.4e6
    src = _ToneSource(fs, target + offset)
    eng = Engine(src, _afc_cfg(fs, afc_limit_hz=800e3, channel_bw_hz=8e6), Queue())
    eng.state.mode = "LOCK"
    eng.state.lock_target = target
    _run_lock(eng, src, n=16)
    # Стара формула завжди вертала LO до target+lim ≈ target+800 кГц.
    # З накопиченням _afc_hw підходимо до справжньої несучої.
    assert src.center_freq != 0.0
    missed = abs((target + offset) - src.center_freq)
    assert missed < 1.0e6, (
        f"перебудови не накопичились: LO={src.center_freq:.0f} "
        f"(ціль {target + offset:.0f}), history={src.tune_history[-6:]}"
    )
    assert src.center_freq > target + 1.0e6, (
        f"LO лишився біля першого lim: {src.center_freq:.0f}"
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
    test_afc_still_corrects_when_channel_is_wider_than_nyquist_pad()
    test_afc_recenters_accumulate_instead_of_rewinding()
    test_manual_lock_listener_is_registered_once()
    print("OK")
