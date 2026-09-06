"""Engine paths that drop USB errors, reject junk, or stall LOCK — not in #5/#7."""
from __future__ import annotations

import time
from queue import Queue
from unittest.mock import patch

import numpy as np
import pytest

from fpvscan import paths
from fpvscan.dsp.cvbs import Frame
from fpvscan.dsp.spectrum import Occupancy
from fpvscan.engine import Detection
from tests.helpers import MockSource, make_engine


def _drain(q: Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


class _BoomSource(MockSource):
    def retune_and_read(self, hz, n):
        raise RuntimeError("usb")


def test_run_raises_after_eight_receiver_failures():
    """A single USB glitch must not kill the server; eight in a row must."""
    src = _BoomSource()
    eng = make_engine(src=src)
    sleeps = []

    def fake_sleep(dt):
        sleeps.append(dt)

    with patch("fpvscan.engine.time.sleep", side_effect=fake_sleep):
        with pytest.raises(RuntimeError, match="usb"):
            eng._run()
    assert sleeps == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    notices = [e for e in _drain(eng.events) if e["type"] == "notice"]
    assert len(notices) == 8
    assert src.opened is False


def test_run_survives_seven_receiver_failures_then_stop():
    src = _BoomSource()
    eng = make_engine(src=src)
    sleeps = []

    def fake_sleep(dt):
        sleeps.append(dt)
        if len(sleeps) >= 7:
            eng._stop.set()

    with patch("fpvscan.engine.time.sleep", side_effect=fake_sleep):
        eng._run()
    assert len(sleeps) == 7
    assert not any(e["type"] == "notice" and "8/8" in e["data"]["text"]
                   for e in _drain(eng.events))


def test_run_applies_bias_tee_and_gain_offset_at_start():
    src = MockSource()
    eng = make_engine(src=src)
    eng.cfg["sdr"]["bias_tee"] = True
    eng._stop.set()
    eng._run()
    assert src.bias is True
    assert src.gain == 15  # 30 - 15


def test_run_primes_quick_tune_when_configured():
    src = MockSource()
    src.primed = None

    def prime(pts):
        src.primed = list(pts)
        return 2

    src.prime_quick_tune = prime
    eng = make_engine(src=src)
    eng.cfg["sdr"]["quick_tune"] = True
    eng.cfg["scan"]["start_hz"] = 5700e6
    eng.cfg["scan"]["stop_hz"] = 5900e6
    eng.cfg["scan"]["step_hz"] = 200e6
    eng._stop.set()
    eng._run()
    assert src.primed is not None
    assert src.primed == sorted(set(int(p) for p in src.primed))
    assert all(5700e6 <= p <= 5900e6 for p in src.primed)


def test_lock_returns_without_decoding_when_ring_is_short():
    class ShortRing:
        def snapshot(self, n):
            return np.zeros(8, dtype=np.complex64), 0

    events = Queue()
    eng = make_engine(events=events)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = ShortRing()
    eng._lock_tuned = 5800e6
    eng._lock_n = 0
    eng._do_lock()
    assert eng._lock_n == 0
    assert not any(e["type"] == "frame" for e in _drain(events))


def test_first_lock_emits_spectrum_next_does_not():
    """`_lock_n % spectrum_every == 1` — first frame is the spectrum tick."""
    events = Queue()
    eng = make_engine(events=events, spectrum_every=8, afc=False)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        eng._do_lock()
        assert any(e["type"] == "spectrum" for e in _drain(events))
        eng._do_lock()
        assert not any(e["type"] == "spectrum" for e in _drain(events))
    finally:
        eng._stop_reader()


def test_sweep_wraps_plan_and_counts_a_completed_pass():
    eng = make_engine()
    eng.cfg["scan"]["start_hz"] = 5700e6
    eng.cfg["scan"]["stop_hz"] = 5900e6
    eng.cfg["scan"]["step_hz"] = 200e6
    plan = eng._sweep_plan()
    assert plan
    eng._sweep_i = len(plan)
    with patch("fpvscan.engine.spectrum.find_occupied", return_value=[]):
        eng._do_sweep()
    assert eng.state.sweeps_done == 1
    assert eng._sweep_i == 1


def test_sweep_retunes_sample_rate_when_more_than_one_hz_off():
    src = MockSource()
    src._fs = 20e6 + 50.0
    eng = make_engine(src=src)
    eng.cfg["scan"]["start_hz"] = 5700e6
    eng.cfg["scan"]["stop_hz"] = 5900e6
    eng.cfg["scan"]["step_hz"] = 200e6
    with patch("fpvscan.engine.spectrum.find_occupied", return_value=[]):
        eng._do_sweep()
    assert src.set_sample_rate_calls >= 1
    assert src.sample_rate == 20e6


def test_inspect_rejects_cw_tone_and_does_not_detect():
    """A Wi-Fi-like CW occupying analog BW must not become a detection."""
    src = MockSource(tone_hz=1e6)
    events = Queue()
    eng = make_engine(src=src, events=events)
    eng.cfg["scan"]["confirm_hits"] = 1
    eng.cfg["scan"]["auto_peek"] = False
    eng.cfg["scan"]["inspect_ms"] = 25
    occ = Occupancy(5800e6, 10e6, -20.0, 18.0)
    eng._inspect(np.zeros(8, dtype=np.complex64), 5800e6, 20e6, occ)
    assert not any(e["type"] == "detection" for e in _drain(events))
    assert eng.state.detections == {}


def test_inspect_debug_emits_rejected_candidate():
    events = Queue()
    eng = make_engine(events=events)
    eng.cfg["scan"]["debug_candidates"] = True
    eng.cfg["scan"]["confirm_hits"] = 1
    eng.cfg["scan"]["auto_peek"] = False
    occ = Occupancy(5800e6, 10e6, -20.0, 12.0)
    eng._inspect(np.zeros(8, dtype=np.complex64), 5800e6, 20e6, occ)
    cands = [e for e in _drain(events) if e["type"] == "candidate"]
    assert cands
    assert cands[0]["data"]["accepted"] is False
    assert cands[0]["data"]["freq_hz"] == 5800e6
    assert eng.state.detections == {}


def test_merge_keeps_the_stronger_second_hit():
    eng = make_engine()
    eng.cfg["scan"]["auto_peek"] = False
    eng.cfg["scan"]["confirm_hits"] = 2
    eng._merge(Detection(freq_hz=5800e6, bandwidth_hz=10e6, snr_db=8,
                         confidence=0.9))
    eng._merge(Detection(freq_hz=5803e6, bandwidth_hz=12e6, snr_db=20,
                         confidence=0.9))
    assert len(eng.state.detections) == 1
    kept = next(iter(eng.state.detections.values()))
    assert kept.freq_hz == 5803e6
    assert kept.snr_db == 20
    assert kept.hits == 2
    assert kept.bandwidth_hz == 12e6


def test_lock_bw_floors_at_eight_mhz():
    eng = make_engine()
    assert eng._lock_bw(100e6, 3e6) == 8e6


def test_afc_deadband_leaves_correction_untouched():
    src = MockSource()
    eng = make_engine(src=src, afc=True, afc_deadband_hz=50e3)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        with patch("fpvscan.engine.demod.freq_error_hz", return_value=10e3):
            eng._do_lock()
        assert eng._afc == 0.0
    finally:
        eng._stop_reader()


def test_snapshot_command_writes_webp_photo(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PHOTOS", tmp_path)
    frame = Frame(
        luma=np.full((220, 32), 128, dtype=np.uint8),
        line_rate=15625.0,
        lines=220,
        standard="PAL",
        locked=True,
    )
    events = Queue()
    eng = make_engine(events=events, min_lines=200, afc=False)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.command("snapshot")
    eng._drain_commands()
    assert eng._snap is True
    try:
        with patch("fpvscan.engine.cvbs.decode", return_value=frame):
            eng._do_lock()
        assert eng._snap is False
        shots = list(tmp_path.glob("shot_*.webp"))
        assert len(shots) == 1
        assert shots[0].read_bytes()[:4] == b"RIFF"
        notices = [e for e in _drain(events) if e["type"] == "notice"]
        assert notices and "знімок" in notices[-1]["data"]["text"]
    finally:
        eng._stop_reader()


def test_rec_start_while_already_recording_is_noop():
    class DummyRec:
        started_at = time.time()

        def stop(self):
            raise AssertionError("must not stop the existing recorder")

    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    rec = DummyRec()
    eng._rec = rec
    eng.command("rec_start")
    eng._drain_commands()
    assert eng._rec is rec
