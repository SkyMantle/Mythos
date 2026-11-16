#!/usr/bin/env python3
"""Регресія: довгий IQ-запис не повинен перевищувати USB-таймаут sync_rx."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan.sdr.bladerf import (
    BladeRF,
    iter_sync_rx_chunks,
    max_sync_rx_samples,
)


def test_documented_4s_capture_exceeds_single_sync_rx():
    """scripts/record.py рекомендує 2–4 с; один sync_rx з timeout 3.5 с
    на 4 с ефіру гарантовано не встигає."""
    fs = 40e6
    timeout_ms = 3500
    n = int(fs * 4.0)
    assert n / fs * 1000 > timeout_ms
    cap = max_sync_rx_samples(fs, timeout_ms)
    assert cap / fs * 1000 < timeout_ms
    chunks = list(iter_sync_rx_chunks(n, fs, timeout_ms))
    assert len(chunks) >= 2
    assert sum(chunks) == n
    assert all(c / fs * 1000 < timeout_ms for c in chunks)


def test_default_3s_capture_is_also_split():
    """Навіть документаний -d 3 при 40 Мвідл/с майже з'їдає 3.5 с таймауту."""
    fs = 40e6
    n = int(fs * 3.0)
    chunks = list(iter_sync_rx_chunks(n, fs, 3500))
    assert sum(chunks) == n
    assert max(c / fs * 1000 for c in chunks) <= 3500 * 0.4 + 1e-6


def test_lock_and_inspect_reads_stay_one_usb_call():
    """Короткі читання рушія (80 мс LOCK, 25 мс INSPECT) не дробляться."""
    fs = 35e6
    timeout_ms = 3500
    for seconds in (0.005, 0.025, 0.08, 0.12):
        n = int(fs * seconds)
        chunks = list(iter_sync_rx_chunks(n, fs, timeout_ms))
        assert chunks == [n], (seconds, chunks)


def test_read_concatenates_timeout_safe_chunks():
    """BladeRF.read склеює шматки і не викликає sync_rx довшим за таймаут."""
    src = object.__new__(BladeRF)
    src._fs = 40e6
    src.timeout_ms = 3500
    src.clip_frac = 0.0
    calls: list[int] = []

    def _sync_rx(n):
        calls.append(int(n))
        src.clip_frac = 0.02
        return np.full(int(n), 0.5 + 0.25j, dtype=np.complex64)

    src._sync_rx = _sync_rx
    src._need_dev = lambda: None

    n = int(40e6 * 4.0)
    out = BladeRF.read(src, n)
    assert out.dtype == np.complex64
    assert out.shape == (n,)
    assert np.allclose(out[0], 0.5 + 0.25j)
    assert np.allclose(out[-1], 0.5 + 0.25j)
    assert len(calls) >= 2
    assert sum(calls) == n
    assert all(c / src._fs * 1000 < src.timeout_ms for c in calls)
    assert src.clip_frac == 0.02


def test_short_read_is_a_single_sync_rx():
    src = object.__new__(BladeRF)
    src._fs = 35e6
    src.timeout_ms = 3500
    calls: list[int] = []
    src._sync_rx = lambda n: calls.append(int(n)) or np.zeros(int(n), np.complex64)
    src._need_dev = lambda: None
    n = int(35e6 * 0.08)
    BladeRF.read(src, n)
    assert calls == [n]


if __name__ == "__main__":
    test_documented_4s_capture_exceeds_single_sync_rx()
    test_default_3s_capture_is_also_split()
    test_lock_and_inspect_reads_stay_one_usb_call()
    test_read_concatenates_timeout_safe_chunks()
    test_short_read_is_a_single_sync_rx()
    print("OK")
