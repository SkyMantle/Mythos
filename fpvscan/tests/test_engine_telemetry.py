"""Console snapshot + LOCK frame payload — missing keys blank the web UI."""
from __future__ import annotations

import time
from queue import Queue
from unittest.mock import patch

import numpy as np

from fpvscan.dsp.cvbs import Frame
from fpvscan.engine import Detection
from fpvscan.iqbuffer import IQRingBuffer
from tests.helpers import MockSource, make_engine


def _drain(q: Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def _still_frame(lines: int = 220, width: int = 32) -> Frame:
    return Frame(
        luma=np.full((lines, width), 128, dtype=np.uint8),
        line_rate=15625.0,
        lines=lines,
        standard="PAL",
        locked=True,
    )


def test_snapshot_exposes_clip_overflow_recording_and_snr_order():
    """applyState() reads clip_frac, rec_seconds, overflows, and detections."""
    src = MockSource()
    src.overflows = 4
    src.clip_frac = 0.0025
    src.bias_tee = True
    eng = make_engine(src=src)
    eng._fps_ema = 5.25

    class DummyRec:
        started_at = time.time() - 3.2

    eng._rec = DummyRec()
    eng.state.detections[100] = Detection(
        freq_hz=1000e6, bandwidth_hz=10e6, snr_db=8.0, hits=2)
    eng.state.detections[200] = Detection(
        freq_hz=5800e6, bandwidth_hz=20e6, snr_db=22.0, hits=3)
    eng.state.detections[150] = Detection(
        freq_hz=1280e6, bandwidth_hz=12e6, snr_db=18.0, hits=1)

    snap = eng.snapshot()
    assert snap["overflows"] == 4
    assert snap["clip_frac"] == 0.0025
    assert snap["bias_tee"] is True
    assert snap["recording"] is True
    assert snap["rec_seconds"] >= 3.0
    assert snap["fps"] == 5.25
    freqs = [d["freq_hz"] for d in snap["detections"]]
    assert freqs == [5800e6, 1000e6]
    assert all(d["hits"] >= 2 for d in snap["detections"])


def test_lock_frame_payload_has_keys_the_console_reads():
    """showFrame() uses img/freq/standard/line_rate/lines/locked/afc_hz."""
    events = Queue()
    eng = make_engine(events=events, min_lines=1, afc=False)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        with patch("fpvscan.engine.cvbs.decode", return_value=_still_frame()):
            eng._do_lock()
        frames = [e for e in _drain(events) if e["type"] == "frame"]
        assert frames
        data = frames[-1]["data"]
        for key in ("img", "freq_hz", "standard", "line_rate",
                    "lines", "locked", "afc_hz"):
            assert key in data, key
        assert data["freq_hz"] == 5800e6
        assert data["standard"] == "PAL"
        assert data["locked"] is True
        assert data["lines"] == 220
        assert isinstance(data["img"], (bytes, bytearray))
        assert data["img"][:4] == b"RIFF"
    finally:
        eng._stop_reader()


def test_fps_ema_updates_after_two_emitted_frames():
    events = Queue()
    eng = make_engine(events=events, min_lines=1, afc=False)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        with patch("fpvscan.engine.cvbs.decode",
                   side_effect=[_still_frame(), _still_frame()]):
            eng._do_lock()
            eng._do_lock()
        assert eng._fps_ema > 0
        assert eng.snapshot()["fps"] > 0
    finally:
        eng._stop_reader()


def test_bias_tee_false_does_not_cut_gain():
    """Board present but set_bias_tee failed: do not apply the LNA offset."""
    src = MockSource()

    def refuse(on):
        src.bias = bool(on)
        return False

    src.set_bias_tee = refuse
    eng = make_engine(src=src)
    eng.command("bias_tee", on=True)
    eng._drain_commands()
    assert src.gain is None
    notices = [e for e in _drain(eng.events) if e["type"] == "notice"]
    assert notices and notices[-1]["data"]["level"] == "error"


def test_afc_clamps_to_limit_without_recentering():
    """A 400 kHz offset must stop at afc_limit, not walk past Nyquist."""
    src = MockSource(tone_hz=400e3)
    eng = make_engine(src=src, afc=True, afc_limit_hz=200e3,
                      afc_deadband_hz=1.0, afc_gain=1.0,
                      afc_recenter_frac=1.1)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        eng._do_lock()
        assert abs(eng._afc) == 200e3
        assert src.retune_calls == 1
    finally:
        eng._stop_reader()


def test_lock_without_target_falls_through_to_sweep():
    """Stuck LOCK with a cleared target must keep scanning, not spin in _do_lock."""
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = None
    hits = []
    eng._do_lock = lambda: hits.append("lock")
    eng._do_sweep = lambda: hits.append("sweep") or eng._stop.set()
    eng._run()
    assert hits == ["sweep"]
    assert "lock" not in hits


def test_reader_loop_writes_then_exits_when_ring_is_gone():
    src = MockSource()
    eng = make_engine(src=src)
    eng._ring = IQRingBuffer(4096)
    eng._reader_stop.clear()
    src.read = lambda n, _orig=src.read: (
        eng._reader_stop.set() or _orig(n))
    eng._reader_loop(20e6)
    assert eng._ring.filled > 0

    eng._ring = None
    eng._reader_stop.clear()
    eng._reader_loop(20e6)  # must return, not raise
