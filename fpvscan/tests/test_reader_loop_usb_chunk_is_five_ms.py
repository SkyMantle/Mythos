"""LOCK USB reader must grab 5 ms chunks, not a field or the 10 ms dump."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_reader_loop_usb_chunk_is_five_ms_not_dump_or_field():
    """#103 `_start_reader` dump is 10 ms (`max(2048, int(fs*0.01))`).
    #18 / helpers `capture_ms` is a field (shipped 80). The live
    USB loop is a third size: `chunk = max(1024, int(fs * 0.005))`
    so each `sync_rx` stays well under `timeout_ms=3500` and the
    ring keeps filling while decode runs. A leftover dump-sized
    read (10 ms) is 2× the budget and still shorter than one PAL
    field, so tracking never shrinks. A leftover `capture_ms=80`
    chunk blocks USB ~16× a 5 ms transfer — overruns on Pi, the
    glance's `auto_peek_secs` burns on a stall. A leftover
    `int(fs * 0.005)` without the 1024 floor at a slow replay
    rate issues tiny `sync_rx` and starves the ring. Direct
    `_reader_loop`; do not start a thread and do not pin file
    pacing (#102). Occupancy/decode stay out. `src.sample_rate`
    is deliberately not `fs` so a leftover `self.src.sample_rate`
    clock fails.
    """
    src = MockSource(fs=10e6)
    ns: list[int] = []
    orig = src.read
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)

    def _read(n):
        ns.append(int(n))
        eng._reader_stop.set()
        return orig(n)

    src.read = _read
    fs = 20e6
    eng._reader_loop(fs)
    five_ms = max(1024, int(fs * 0.005))
    dump = max(2048, int(fs * 0.01))
    field = int(fs * 0.080)
    assert ns == [five_ms]
    assert ns[0] == 100_000
    assert ns[0] != dump
    assert ns[0] != 200_000
    assert ns[0] != field
    assert ns[0] != int(src.sample_rate * 0.005)
    assert ns[0] != 1024

    ns.clear()
    eng._reader_stop.clear()
    slow = 100e3
    eng._reader_loop(slow)
    assert ns == [max(1024, int(slow * 0.005))]
    assert ns[0] == 1024
    assert ns[0] != int(slow * 0.005)
