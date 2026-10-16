"""A short ring snapshot is the written prefix, not a zero-padded grab."""
from __future__ import annotations

import numpy as np

from fpvscan.iqbuffer import IQRingBuffer


def test_short_snapshot_returns_what_was_written():
    """#62 covered the unwrapped tail (filled > n, no wrap). #52 the
    wrap concatenate. The first LOCK grab is the other branch:
    `_start_reader` primes 10 ms, then snapshot(capture_ms) asks for
    80 ms. `avail = min(n, filled, capacity)` must return the 10 ms
    that exist with abs_start=0 — not zero-pad, not the unallocated
    head. `_do_lock` skips decode when len < n; a padded snapshot
    would look full and decode PLL garbage as a field.
    """
    ring = IQRingBuffer(capacity=16)
    ring.write(np.arange(5, dtype=np.complex64))
    out, abs_start = ring.snapshot(12)
    np.testing.assert_array_equal(out.real, np.arange(5, dtype=np.float32))
    assert abs_start == 0
    assert len(out) == 5
    out[0] = 99
    again, _ = ring.snapshot(12)
    assert again[0].real == 0.0


def test_empty_ring_snapshot_is_empty():
    """Before the reader writes, snapshot must be length 0 with
    abs_start=0. A leftover `np.zeros(n)` would pass the
    `len(iq) < n` gate and decode silence as a locked field.
    """
    ring = IQRingBuffer(capacity=16)
    out, abs_start = ring.snapshot(8)
    assert len(out) == 0
    assert abs_start == 0
    assert ring.filled == 0
