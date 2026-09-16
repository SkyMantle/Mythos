"""LOCK reader prime fill — an empty ring looks like a frozen picture."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_lock_reader_primes_ten_ms_or_at_least_2048():
    """_start_reader grabs max(2048, 10 ms) before the snapshot loop.
    A 0-sample prime leaves the ring empty, _do_lock sleeps idle_ms and
    returns, and the console shows LOCK with no frames until the reader
    happens to catch up — on a stalled USB that is never.
    """
    ns = []

    src = MockSource(fs=20e6)
    orig = src.retune_and_read

    def rec(hz, n):
        ns.append(int(n))
        return orig(hz, n)

    src.retune_and_read = rec
    eng = make_engine(src=src)
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert ns[0] == max(2048, int(20e6 * 0.01))
        assert ns[0] == 200_000
        assert src.center_freq == 5800e6
        assert eng._ring is not None
        # reader thread may have appended more after the prime
        assert eng._ring.filled >= ns[0]
    finally:
        eng._stop_reader()

    ns.clear()
    slow = MockSource(fs=100e3)
    orig2 = slow.retune_and_read

    def rec2(hz, n):
        ns.append(int(n))
        return orig2(hz, n)

    slow.retune_and_read = rec2
    eng2 = make_engine(src=slow)
    try:
        eng2._start_reader(1280e6, 100e3, 0.05)
        assert ns[0] == 2048
    finally:
        eng2._stop_reader()
