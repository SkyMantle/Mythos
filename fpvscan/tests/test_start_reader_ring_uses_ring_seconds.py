"""LOCK ring capacity must be ring_seconds, not the 10 ms PLL dump."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_start_reader_ring_capacity_is_ring_seconds_not_first_chunk():
    """#101 `_start_reader` fills the ring with a 10 ms settled dump.
    `_do_lock` snapshots `capture_ms` (shipped 80) from that ring
    while `_reader_loop` keeps writing. Capacity is
    `max(len(first), int(fs * ring_seconds))` — shipped
    `ring_seconds=0.5`, engine fallback `max(0.3, 3·capture)`.
    A leftover `capacity=len(first)` / `int(fs * 0.01)` is 10 ms
    at 35 Msps. The next field's 80 ms snapshot wraps the ring
    twice while DecodeState thinks `abs_t0` is continuous —
    verticals jump, the tracking window never shrinks, every
    field grabs full `capture_ms`. A leftover
    `int(fs * capture_ms/1000)` (80 ms) is still shorter than
    one 0.5 s hold plus decode stall, so `_reader_loop` overwrites
    the samples TBC has not snapshotted yet. Direct `_start_reader`
    with helper `ring_seconds=0.05` (1 M samples at 20 Msps) vs
    the 200 k first dump / 100 k 5 ms capture. Occupancy/decode
    stay out of the path.
    """
    fs = 20e6
    ring_s = 0.05
    src = MockSource(fs=fs)
    eng = make_engine(src)
    first_n = max(2048, int(fs * 0.01))
    capture_n = int(fs * 0.005)
    want_cap = max(first_n, int(fs * ring_s))
    try:
        eng._start_reader(5800e6, fs, ring_s)
        assert eng._ring is not None
        assert eng._ring.capacity == want_cap
        assert eng._ring.capacity == 1_000_000
        assert eng._ring.capacity != first_n
        assert eng._ring.capacity != capture_n
        assert eng._ring.capacity != 2048
        assert src.retune_calls == 1
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
