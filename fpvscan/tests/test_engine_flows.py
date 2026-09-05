"""Engine paths that decide what gets inspected, locked, or dropped."""
from __future__ import annotations

import time
from queue import Queue
from unittest.mock import patch

import numpy as np
import pytest

from fpvscan.dsp.cvbs import DecodeState, FIELD_LINES
from fpvscan.dsp.spectrum import Occupancy
from fpvscan.engine import Detection, Engine
from fpvscan.recorder import FfmpegMissing
from fpvscan.sdr.sim import C_LINE_PAL, Emitter, SimSource
from tests.helpers import MockSource, engine_cfg, make_engine


def _drain(q: Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_command_exception_emits_error_notice():
    events = Queue()
    eng = make_engine(events=events)
    eng.command("lock")  # missing freq_hz
    eng._drain_commands()
    notices = [e for e in _drain(events) if e["type"] == "notice"]
    assert notices
    assert notices[0]["data"]["level"] == "error"
    assert "lock" in notices[0]["data"]["text"]
    assert eng.state.mode == "SWEEP"


def test_bias_tee_unsupported_source_emits_error():
    class NoTee:
        """Delegates to MockSource except set_bias_tee (hasattr must be false)."""

        def __init__(self):
            self._inner = MockSource()

        def __getattr__(self, name):
            if name == "set_bias_tee":
                raise AttributeError(name)
            return getattr(self._inner, name)

    src = NoTee()
    events = Queue()
    eng = make_engine(src=src, events=events)
    eng.command("bias_tee", on=True)
    eng._drain_commands()
    notices = [e for e in _drain(events) if e["type"] == "notice"]
    assert notices and notices[0]["data"]["level"] == "error"
    assert src.gain is None


def test_rec_start_missing_ffmpeg_stays_idle():
    events = Queue()
    eng = make_engine(events=events)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.cfg["video"]["ffmpeg_path"] = "/no/such/ffmpeg"
    eng.command("rec_start")
    eng._drain_commands()
    assert eng._rec is None
    notices = [e for e in _drain(events) if e["type"] == "notice"]
    assert notices and notices[0]["data"]["level"] == "error"
    assert isinstance(FfmpegMissing("x"), RuntimeError)


def test_auto_peek_expires_back_to_sweep():
    src = MockSource()
    eng = make_engine(src=src)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng.state.auto_until = time.time() - 1.0
    try:
        eng._do_lock()
        assert eng.state.mode == "SWEEP"
        assert eng.state.lock_target is None
        assert eng.state.auto is False
        assert eng._ring is None
        assert eng._reader_thread is None
    finally:
        eng._stop_reader()


def test_auto_peek_skipped_when_already_locked():
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 1280e6
    eng._maybe_peek(Detection(freq_hz=5800e6, bandwidth_hz=12e6, snr_db=20,
                              confidence=0.99))
    assert eng.state.lock_target == 1280e6


def test_sweep_skips_occupancy_outside_analog_bw():
    inspected = []
    eng = make_engine()
    eng.cfg["scan"]["start_hz"] = 5700e6
    eng.cfg["scan"]["stop_hz"] = 5900e6
    eng.cfg["scan"]["step_hz"] = 200e6
    narrow = Occupancy(5800e6, 1.0e6, -20.0, 15.0)
    analog = Occupancy(5800e6, 10.0e6, -20.0, 15.0)
    wide = Occupancy(5800e6, 40.0e6, -20.0, 15.0)

    def fake_inspect(_iq, _center, _fs, occ):
        inspected.append(occ)

    with patch.object(eng, "_inspect", side_effect=fake_inspect), \
         patch("fpvscan.engine.spectrum.find_occupied",
               return_value=[narrow, analog, wide]):
        eng._do_sweep()
    assert inspected == [analog]


def test_sweep_aborts_when_lock_command_arrives():
    src = MockSource()
    eng = make_engine(src=src)
    eng.cfg["scan"]["start_hz"] = 400e6
    eng.cfg["scan"]["stop_hz"] = 800e6
    eng.cfg["scan"]["step_hz"] = 50e6
    eng.command("lock", freq_hz=5800e6)
    eng._do_sweep()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert src.retune_calls == 0


def test_grab_prefers_fast_retune():
    src = MockSource()
    src.fast_calls = 0

    def fast(hz, n):
        src.fast_calls += 1
        return src.retune_and_read(hz, n)

    src.retune_and_read_fast = fast
    eng = make_engine(src=src)
    iq = eng._grab(5800e6, 16)
    assert src.fast_calls == 1
    assert len(iq) == 16


def test_inspect_debug_emits_reason_on_source_error():
    class Boom(MockSource):
        def retune_and_read(self, hz, n):
            raise RuntimeError("usb glitch")

    events = Queue()
    eng = make_engine(src=Boom(), events=events)
    eng.cfg["scan"]["debug_candidates"] = True
    occ = Occupancy(5800e6, 10e6, -20.0, 12.0)
    eng._inspect(np.zeros(8, dtype=np.complex64), 5800e6, 20e6, occ)
    cands = [e for e in _drain(events) if e["type"] == "candidate"]
    assert cands
    assert "usb glitch" in cands[0]["data"]["reason"]
    assert eng.state.detections == {}


def test_inspect_accepts_pal_sim_and_merges():
    src = SimSource(
        emitters=[Emitter(1280e6, -18.0, "fpv", 8e6, C_LINE_PAL, "t")],
        noise_db=-85, seed=2,
    )
    src.open()
    src.set_sample_rate(20e6)
    events = Queue()
    eng = Engine(src, engine_cfg(), events)
    eng.cfg["scan"]["inspect_ms"] = 80
    eng.cfg["scan"]["confirm_hits"] = 1
    eng.cfg["scan"]["auto_peek"] = False
    occ = Occupancy(1280e6, 10e6, -20.0, 18.0)
    eng._inspect(np.zeros(8, dtype=np.complex64), 1280e6, 20e6, occ)
    dets = [e for e in _drain(events) if e["type"] == "detection"]
    assert dets, "PAL sim emitter must pass classify_video and emit a detection"
    det = dets[0]["data"]
    assert det["standard"] == "PAL"
    assert abs(det["freq_hz"] - 1280e6) < 1.0
    assert det["hits"] == 1


def test_reader_error_is_reraised_on_next_lock():
    src = MockSource()
    eng = make_engine(src=src)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        eng._do_lock()
        eng._reader_err = RuntimeError("overflow")
        with pytest.raises(RuntimeError, match="overflow"):
            eng._do_lock()
        assert eng._reader_err is None
        assert eng._ring is None
    finally:
        eng._stop_reader()


def test_afc_zero_headroom_retunes_reader():
    """When fs cannot hold channel_bw + digital AFC, leftover _afc is retuned.

    freq_error_hz is stubbed to 0 so the clamp-to-lim branch does not wipe
    `_afc` before the lim<=0 recenter path runs (zero IQ + a digital shift
    yields a bogus ±fs/2 signed-zero error).
    """
    src = MockSource()
    eng = make_engine(src=src, afc=True, channel_bw_hz=20e6, sample_rate=20e6)
    eng.cfg["video"]["sample_rate"] = 20e6
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._afc = 1.5e6
    try:
        with patch("fpvscan.engine.demod.freq_error_hz", return_value=0.0):
            eng._do_lock()
        assert src.retune_calls >= 2
        assert eng._afc == 0.0
    finally:
        eng._stop_reader()


class _RecRing:
    def __init__(self):
        self.n = None

    def snapshot(self, n):
        self.n = int(n)
        return np.zeros(int(n), dtype=np.complex64), 0


def test_lock_snapshot_length_follows_field_lines():
    """Mislabeled NTSC-as-PAL asks for a 312.5-line window; NTSC is 262.5."""
    fs = 20e6
    period = fs / 15625.0
    margin = 1.7

    def _n(standard: str) -> int:
        src = MockSource(fs=fs)
        eng = make_engine(src=src, capture_ms=200, idle_ms=0, afc=False)
        eng.cfg["video"]["sample_rate"] = fs
        eng.cfg["video"]["track_window_margin"] = margin
        eng.state.mode = "LOCK"
        eng.state.lock_target = 5800e6
        ring = _RecRing()
        eng._ring = ring
        eng._lock_tuned = 5800e6
        eng._lock_dec = 1
        eng._lock_state = DecodeState(period=period, standard=standard, lost=0)
        try:
            eng._do_lock()
        finally:
            eng._stop_reader()
        return ring.n

    n_ntsc = _n("NTSC")
    n_pal = _n("PAL")
    expect_ntsc = int(period * FIELD_LINES["NTSC"] * margin)
    expect_pal = int(period * FIELD_LINES["PAL"] * margin)
    assert n_ntsc == expect_ntsc
    assert n_pal == expect_pal
    assert n_pal > n_ntsc


def test_decimation_change_resets_decode_state():
    src = MockSource()
    eng = make_engine(src=src, afc=False)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._lock_dec = 99
    eng._lock_state = DecodeState(period=1234.0, standard="PAL", lost=0)
    try:
        eng._do_lock()
        assert eng._lock_state.period is None
        assert eng._lock_dec != 99
    finally:
        eng._stop_reader()
