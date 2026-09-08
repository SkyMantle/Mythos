"""Commands that can corrupt a recording or hide a receiver fault."""
from __future__ import annotations

import time
from queue import Queue

from fpvscan.iqbuffer import IQRingBuffer
from tests.helpers import MockSource, make_engine


def _drain(q: Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


class DummyRec:
    started_at = 0.0

    def __init__(self):
        self.stopped = False
        self.started_at = time.time()

    def stop(self):
        self.stopped = True
        return {"path": "/tmp/rec.mp4", "bytes": 2_000_000,
                "seconds": 4.0, "kbps": 500}


def test_lock_command_stops_recording():
    """Switching channels while recording used to mux two VTx into one file."""
    eng = make_engine()
    rec = DummyRec()
    eng._rec = rec
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.command("lock", freq_hz=5843e6)
    eng._drain_commands()
    assert rec.stopped
    assert eng._rec is None
    assert eng.state.lock_target == 5843e6
    notices = [e for e in _drain(eng.events) if e["type"] == "notice"]
    assert notices and "rec.mp4" in notices[-1]["data"]["text"]


def test_rec_stop_command_closes_active_recorder():
    eng = make_engine()
    rec = DummyRec()
    eng._rec = rec
    eng.command("rec_stop")
    eng._drain_commands()
    assert rec.stopped
    assert eng._rec is None


def test_rec_stop_when_idle_is_noop():
    eng = make_engine()
    eng.command("rec_stop")
    eng._drain_commands()
    assert eng._rec is None
    assert not any(e["type"] == "notice" for e in _drain(eng.events))


def test_bias_tee_gain_failure_emits_error_notice():
    """LNA offset must not fail silently: ADC clip then looks like a dead VTx."""
    src = MockSource()

    def boom(_db):
        raise RuntimeError("usb stall")

    src.set_gain = boom
    eng = make_engine(src=src)
    eng.command("bias_tee", on=True)
    eng._drain_commands()
    notices = [e for e in _drain(eng.events) if e["type"] == "notice"]
    texts = " ".join(n["data"]["text"] for n in notices)
    assert "gain" in texts
    assert any(n["data"]["level"] == "error" for n in notices)


def test_snapshot_exposes_stage_timings_after_mark():
    eng = make_engine()
    t0 = time.perf_counter() - 0.012
    eng._mark("channelize", t0)
    snap = eng.snapshot()
    assert "channelize" in snap["timings_ms"]
    assert snap["timings_ms"]["channelize"] >= 10.0


def test_reader_loop_chunk_is_five_milliseconds():
    """A 50 ms chunk would add a field of latency before the first LOCK frame."""
    src = MockSource()
    eng = make_engine(src=src)
    eng._ring = IQRingBuffer(1 << 16)
    sizes = []

    def rec(n, _orig=src.read):
        sizes.append(int(n))
        eng._reader_stop.set()
        return _orig(n)

    src.read = rec
    eng._reader_loop(20e6)
    assert sizes == [max(1024, int(20e6 * 0.005))]
