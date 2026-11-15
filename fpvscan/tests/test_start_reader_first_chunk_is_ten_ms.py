"""LOCK must prime the IQ ring with a 10 ms PLL dump, not a field grab."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


class _LenSource(MockSource):
    """Records how many samples the settled dump asked for."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.retune_n: list[int] = []

    def retune_and_read(self, hz: float, n: int):
        self.retune_n.append(int(n))
        return super().retune_and_read(hz, n)


def test_start_reader_first_chunk_is_ten_ms_not_capture_or_ring():
    """#21 INSPECT dumps `inspect_ms` (shipped 25) so the 15.6 kHz
    line is measured on settled IQ. #101 `_start_reader` uses that
    same settled `retune_and_read`, not `_grab` / fast. The missing
    half: the dump length is `max(2048, int(fs * 0.01))` — 10 ms —
    not shipped `capture_ms=80`, not `_reader_loop`'s 5 ms USB
    chunk, and not `ring_seconds` (shipped 0.5). `_do_lock` later
    snapshots `capture_ms` from the ring; the first dump only has
    to cover PLL settle plus a few lines so DecodeState can lock.
    A leftover `int(fs * capture_ms/1000)` blocks USB ~8× longer
    before the first field (80 ms at 35 Msps ≈ 2.8 M samples) and
    the glance's `auto_peek_secs=5` burns on a stall. A leftover
    `int(fs * 0.005)` (copy-paste from `_reader_loop`) puts 5 ms
    into the ring — less than one PAL field — so tracking never
    shrinks and every field grabs full `capture_ms`. A leftover
    `int(fs * ring_seconds)` dumps half a second in one blocking
    retune. Direct `_start_reader`; occupancy/decode stay out.
    Helper `capture_ms=5` / `ring_seconds=0.05` so those leftovers
    are 100 k / 1 M samples at 20 Msps, distinct from 200 k.
    """
    fs = 20e6
    src = _LenSource(fs=fs)
    eng = make_engine(src)
    ten_ms = max(2048, int(fs * 0.01))
    five_ms = max(1024, int(fs * 0.005))
    ring_n = int(fs * 0.05)
    capture_n = int(fs * 0.005)
    try:
        eng._start_reader(5800e6, fs, 0.05)
        assert src.retune_n == [ten_ms]
        assert src.retune_n[0] == 200_000
        assert src.retune_n[0] != five_ms
        assert src.retune_n[0] != capture_n
        assert src.retune_n[0] != ring_n
        assert src.retune_n[0] != 2048
        assert src.retune_calls == 1
        assert src.center_freq == 5800e6
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
        assert eng._ring is not None
    finally:
        eng._stop_reader()
