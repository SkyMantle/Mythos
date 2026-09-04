"""Engine command, merge, peek, sweep-plan, and digital-AFC regressions."""
from __future__ import annotations

import time
from queue import Queue

import numpy as np

from fpvscan.engine import Detection, Engine
from tests.helpers import MockSource, engine_cfg, make_engine


def _drain(q: Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_lock_command_enters_lock_and_resets_afc():
    eng = make_engine()
    eng._afc = 1.5e6
    eng._acc = np.zeros((10, 10))
    eng._lock_tuned = 123.0
    eng.command("lock", freq_hz=5800e6)
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng.state.auto is False
    assert eng._afc == 0.0
    assert eng._acc is None
    assert eng._lock_tuned is None


def test_sweep_command_clears_lock_and_stops_recording():
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6

    class DummyRec:
        started_at = time.time()

        def stop(self):
            return {"path": "/tmp/rec.mp4", "bytes": 1_000_000,
                    "seconds": 2.0, "kbps": 400}

    rec = DummyRec()
    eng._rec = rec
    eng.command("sweep")
    eng._drain_commands()
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None
    assert eng._rec is None
    notices = [e for e in _drain(eng.events) if e["type"] == "notice"]
    assert notices and "rec.mp4" in notices[-1]["data"]["text"]


def test_clear_drops_detections_and_peek_memory():
    eng = make_engine()
    eng.state.detections[5800] = Detection(freq_hz=5800e6, bandwidth_hz=10e6, snr_db=12)
    eng._peeked[5800] = time.time()
    eng._sweep_i = 9
    eng.command("clear")
    eng._drain_commands()
    assert eng.state.detections == {}
    assert eng._peeked == {}
    assert eng._sweep_i == 0


def test_bias_tee_on_cuts_receiver_gain_by_offset():
    src = MockSource()
    eng = make_engine(src=src)
    eng.command("bias_tee", on=True)
    eng._drain_commands()
    assert src.bias is True
    assert src.gain == 15  # 30 - 15
    eng.command("bias_tee", on=False)
    eng._drain_commands()
    assert src.bias is False
    assert src.gain == 30


def test_rec_start_without_lock_emits_error():
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng.command("rec_start")
    eng._drain_commands()
    assert eng._rec is None
    notices = [e for e in _drain(eng.events) if e["type"] == "notice"]
    assert notices and notices[0]["data"]["level"] == "error"


def test_merge_requires_confirm_hits_before_emit():
    events = Queue()
    eng = make_engine(events=events)
    d1 = Detection(freq_hz=5800e6, bandwidth_hz=12e6, snr_db=10,
                   confidence=0.9, first_seen=1.0, last_seen=1.0)
    # disable auto-peek so merge is observable in isolation
    eng.cfg["scan"]["auto_peek"] = False
    eng._merge(d1)
    assert not any(e["type"] == "detection" for e in _drain(events))
    snap = eng.snapshot()
    assert snap["detections"] == []

    d2 = Detection(freq_hz=5802e6, bandwidth_hz=11e6, snr_db=8,
                   confidence=0.9, first_seen=2.0, last_seen=2.0)
    eng._merge(d2)
    emitted = [e for e in _drain(events) if e["type"] == "detection"]
    assert len(emitted) == 1
    # better SNR estimate (first hit) is kept
    assert emitted[0]["data"]["freq_hz"] == 5800e6
    assert emitted[0]["data"]["hits"] == 2
    assert len(eng.snapshot()["detections"]) == 1


def test_merge_does_not_collapse_far_apart_signals():
    eng = make_engine()
    eng.cfg["scan"]["auto_peek"] = False
    eng.cfg["scan"]["confirm_hits"] = 1
    eng._merge(Detection(freq_hz=1280e6, bandwidth_hz=8e6, snr_db=9, confidence=0.8))
    eng._merge(Detection(freq_hz=5800e6, bandwidth_hz=12e6, snr_db=11, confidence=0.8))
    assert len(eng.state.detections) == 2


def test_auto_peek_locks_once_then_respects_cooldown():
    eng = make_engine()
    det = Detection(freq_hz=5800e6, bandwidth_hz=12e6, snr_db=15,
                    confidence=0.9, first_seen=time.time(), last_seen=time.time())
    eng.cfg["scan"]["confirm_hits"] = 1
    eng._merge(det)
    assert eng.state.mode == "LOCK"
    assert eng.state.auto is True
    assert eng.state.lock_target == 5800e6
    until = eng.state.auto_until

    eng.state.mode = "SWEEP"
    eng.state.lock_target = None
    eng.state.auto = False
    eng._merge(Detection(freq_hz=5800e6, bandwidth_hz=12e6, snr_db=16,
                         confidence=0.95, first_seen=time.time(), last_seen=time.time()))
    # cooldown: must not lock again
    assert eng.state.mode == "SWEEP"
    assert eng.state.auto_until == until


def test_auto_peek_skips_low_confidence():
    eng = make_engine()
    eng.cfg["scan"]["confirm_hits"] = 1
    eng._merge(Detection(freq_hz=5800e6, bandwidth_hz=12e6, snr_db=10,
                         confidence=0.2))
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None


def test_sweep_plan_step_overlaps_channel():
    eng = make_engine()
    plan = eng._sweep_plan()
    assert plan[0] == 400e6 + 20e6 / 2
    step = plan[1] - plan[0]
    # auto step = max(fs*0.25, fs*0.9 - ch_bw/2) = max(5e6, 18e6-10e6) = 8e6
    assert step == 8e6
    assert plan[-1] < 6000e6


def test_sweep_plan_interleaves_priority_bands():
    eng = make_engine()
    plain = eng._sweep_plan()
    eng.cfg["scan"]["priority_bands"] = True
    prio = eng._sweep_plan()
    assert len(prio) == 2 * len(plain)
    assert prio[0] == plain[0]
    assert prio[1] == 420e6 + 20e6 / 2  # first priority-band 433 point
    assert prio[2] == plain[1]


def test_sweep_plan_fixed_freq_is_single_point():
    src = MockSource()
    src.fixed_freq = True
    src._fc = 1280e6
    eng = Engine(src, engine_cfg(), Queue())
    assert eng._sweep_plan() == [1280e6]


def test_lock_bw_uses_measured_detection():
    eng = make_engine()
    eng.state.detections[5800] = Detection(freq_hz=5800e6, bandwidth_hz=12e6, snr_db=10)
    assert eng._lock_bw(5801e6, 20e6) == 12e6 + 2 * Engine.MERGE_TOL_HZ
    assert eng._lock_bw(2400e6, 20e6) == 20e6  # max(8e6, default)


def test_emit_drops_old_events_to_keep_a_frame():
    events = Queue(maxsize=2)
    eng = Engine(MockSource(), engine_cfg(), events)
    events.put_nowait({"type": "spectrum", "data": 1})
    events.put_nowait({"type": "spectrum", "data": 2})
    eng._emit("frame", {"img": b"x"})
    kinds = []
    while not events.empty():
        kinds.append(events.get_nowait()["type"])
    assert "frame" in kinds


def test_emit_non_frame_is_dropped_when_queue_full():
    events = Queue(maxsize=1)
    eng = Engine(MockSource(), engine_cfg(), events)
    events.put_nowait({"type": "spectrum", "data": 1})
    eng._emit("notice", {"text": "nope"})
    assert events.qsize() == 1
    assert events.get_nowait()["type"] == "spectrum"


def test_digital_afc_does_not_restart_iq_reader():
    src = MockSource(tone_hz=200e3)
    eng = make_engine(src=src, afc=True)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        for _ in range(4):
            eng._do_lock()
        assert src.retune_calls == 1
        assert eng._afc != 0.0
        assert abs(eng._afc) < 8e6
        assert eng._lock_tuned == 5800e6
    finally:
        eng._stop_reader()


def test_lo_offset_disabled_when_it_does_not_fit_in_passband():
    src = MockSource()
    eng = make_engine(src=src, lo_offset_hz=12e6, channel_bw_hz=20e6,
                      sample_rate=20e6)
    eng.cfg["video"]["sample_rate"] = 20e6
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        eng._do_lock()
        # 12e6 + 10e6 > 0.45*20e6 → offset forced to 0, tune stays on channel
        assert eng._lock_tuned == 5800e6
    finally:
        eng._stop_reader()


def test_unknown_command_is_ignored():
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng.command("nope")
    eng._drain_commands()
    assert eng.state.mode == "SWEEP"
    assert not any(e["type"] == "notice" for e in _drain(eng.events))


def test_snapshot_reports_source_and_empty_lock():
    src = MockSource()
    eng = make_engine(src=src)
    snap = eng.snapshot()
    assert snap["mode"] == "SWEEP"
    assert snap["source"] == "mock"
    assert snap["recording"] is False
    assert snap["lock_target"] is None
    assert snap["fps"] == 0
