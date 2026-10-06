"""LOCK ring wrap must keep stream order or genlock tears once per ring."""
from __future__ import annotations

import numpy as np
import pytest

from fpvscan.iqbuffer import IQRingBuffer


def test_wrap_snapshot_is_stream_order_with_abs_start():
    """Once the reader has lapped the ring, snapshot concatenates
    `buf[start:] + buf[:end]`. Swapping those pieces (or returning a
    view of the wrapped buffer) puts a discontinuity in the middle of
    every field — CVBS tracking then loses V-sync once per
    `ring_seconds`. abs_start must stay `filled - n` so decode can
    genlock across non-adjacent snapshots.
    """
    ring = IQRingBuffer(capacity=8)
    ring.write(np.arange(6, dtype=np.complex64))
    ring.write(np.arange(6, 10, dtype=np.complex64))
    out, abs_start = ring.snapshot(8)
    assert abs_start == 2
    np.testing.assert_array_equal(out.real, np.arange(2, 10, dtype=np.float32))
    out[0] = 99
    again, _ = ring.snapshot(8)
    assert again[0].real == 2.0


def test_oversized_write_keeps_the_tail():
    """A chunk larger than the ring keeps only the last `capacity`
    samples. The reader never does this (5 ms vs 0.5 s), but a
    capture_ms/ring_seconds inversion would, and the decoder would
    then see the *head* of the grab — the PLL dump — instead of the
    settled IQ.
    """
    ring = IQRingBuffer(capacity=4)
    ring.write(np.arange(10, dtype=np.complex64))
    out, abs_start = ring.snapshot(4)
    np.testing.assert_array_equal(out.real, np.arange(6, 10, dtype=np.float32))
    assert abs_start == 0
    assert ring.filled == 4


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        IQRingBuffer(capacity=0)
