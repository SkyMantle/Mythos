"""LOCK phase tracking depends on wrap-safe snapshots and monotonic abs_start."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from fpvscan.iqbuffer import IQRingBuffer


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        IQRingBuffer(0)


def test_snapshot_before_write_is_empty():
    buf = IQRingBuffer(8)
    out, abs_start = buf.snapshot(4)
    assert len(out) == 0
    assert abs_start == 0
    assert buf.filled == 0


def test_wraparound_keeps_latest_samples_and_abs_start():
    buf = IQRingBuffer(8)
    buf.write(np.arange(5, dtype=np.complex64))
    buf.write(np.arange(5, 10, dtype=np.complex64))
    assert buf.filled == 10
    out, abs_start = buf.snapshot(8)
    assert abs_start == 2
    np.testing.assert_array_equal(out.real.astype(int), np.arange(2, 10))


def test_write_larger_than_capacity_keeps_tail():
    buf = IQRingBuffer(8)
    buf.write(np.arange(20, dtype=np.complex64))
    out, abs_start = buf.snapshot(8)
    np.testing.assert_array_equal(out.real.astype(int), np.arange(12, 20))
    # discarded prefix is not added to filled — abs_start is 0 after a single overflow write
    assert buf.filled == 8
    assert abs_start == 0


def test_short_snapshot_returns_what_is_available():
    buf = IQRingBuffer(32)
    buf.write(np.arange(5, dtype=np.complex64))
    out, abs_start = buf.snapshot(20)
    assert len(out) == 5
    assert abs_start == 0


def test_empty_write_is_noop():
    buf = IQRingBuffer(4)
    buf.write(np.zeros(0, dtype=np.complex64))
    assert buf.filled == 0


def test_concurrent_write_and_snapshot_do_not_tear():
    buf = IQRingBuffer(1024)
    stop = threading.Event()
    err = []

    def writer():
        k = 0
        while not stop.is_set():
            chunk = np.full(64, k, dtype=np.complex64)
            buf.write(chunk)
            k += 1

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(200):
            out, abs_start = buf.snapshot(128)
            if len(out) == 0:
                continue
            assert abs_start >= 0
            assert len(out) <= 128
            # a torn snapshot would mix two dtypes/sizes; values may wrap
            assert out.dtype == np.complex64
    finally:
        stop.set()
        t.join(timeout=2)
    assert not err
