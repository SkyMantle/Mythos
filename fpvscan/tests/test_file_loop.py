#!/usr/bin/env python3
"""File replay must splice at EOF. Restarting at 0 skips the tail."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan.sdr.file import FileSource, write_capture


def _open(data: np.ndarray, loop: bool = True) -> FileSource:
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "cap.cf32"
    write_capture(path, data, center_hz=5800e6, sample_rate=40e6)
    src = FileSource(path, loop=loop)
    src.open()
    src._tmpdir = tmp  # keep the directory alive
    return src


def test_loop_splices_tail_and_head():
    """LOCK reader chunks (5 ms) straddle EOF on the shipped 60 ms capture.

    Old read() reset _pos to 0, dropping the remaining samples. DecodeState
    tracking then saw a multi-millisecond jump and the picture rolled.
    """
    n = 10
    data = (np.arange(n, dtype=np.float32)
            + 1j * np.arange(n, dtype=np.float32)).astype(np.complex64)
    src = _open(data, loop=True)
    first = src.read(6)
    assert np.array_equal(first, data[:6])
    second = src.read(6)
    # samples 6,7,8,9 then wrap onto 0,1 — not a jump back to 0,1,2,3,4,5
    want = np.concatenate((data[6:], data[:2]))
    assert np.array_equal(second, want), (
        f"loop straddling EOF skipped the tail: got {second} want {want}"
    )
    third = src.read(6)
    # cursor after the wrap is 2; the next 6 samples are 2..7
    assert np.array_equal(third, data[2:8])
    src.close()


def test_loop_request_longer_than_file_keeps_cursor():
    n = 4
    data = (np.arange(n, dtype=np.float32)
            + 1j * np.zeros(n, dtype=np.float32)).astype(np.complex64)
    src = _open(data, loop=True)
    src.read(2)  # cursor at 2
    got = src.read(9)  # 2,3,0,1,2,3,0,1,2
    want = np.array([2, 3, 0, 1, 2, 3, 0, 1, 2], dtype=np.float32)
    assert np.array_equal(got.real, want), (
        f"long looping read jumped to start: got {got.real} want {want}"
    )
    src.close()


def test_shipped_capture_workflow_does_not_skip():
    """caps/vtx.cf32 is 2.4e6 samples @ 40 Msps (60 ms). Default inspect is
    25 ms, LOCK primes 10 ms, then the reader asks for 5 ms. Those sizes
    leave a remainder, so the old reset-to-0 path skipped ~3 ms every loop.
    """
    n = 2400  # 1000× smaller stand-in for 2_400_000
    data = (np.arange(n, dtype=np.float32)
            + 1j * np.zeros(n, dtype=np.float32)).astype(np.complex64)
    src = _open(data, loop=True)
    src.read(66)     # one sweep FFT grab (scaled)
    src.read(1000)   # inspect_ms=25
    src.read(400)    # _start_reader 10 ms prime
    # 5 ms reader chunks until past EOF
    chunks = [src.read(200) for _ in range(6)]
    stream = np.concatenate(chunks).real
    start = 66 + 1000 + 400  # 1466
    expected = (start + np.arange(len(stream))) % n
    assert np.array_equal(stream, expected.astype(np.float32)), (
        "workflow wrap inserted a gap: "
        f"first mismatch at {np.argmax(stream != expected)}"
    )
    src.close()


def test_read_returns_an_independent_copy():
    data = (np.arange(8, dtype=np.float32)
            + 1j * np.zeros(8, dtype=np.float32)).astype(np.complex64)
    src = _open(data, loop=True)
    got = src.read(4)
    got[:] = 99
    src._pos = 0
    again = src.read(4)
    assert np.array_equal(again.real, np.arange(4, dtype=np.float32))
    src.close()


def test_oneshot_returns_tail_then_eof():
    """loop=false must not raise before delivering the remaining samples."""
    data = (np.arange(8, dtype=np.float32)
            + 1j * np.zeros(8, dtype=np.float32)).astype(np.complex64)
    src = _open(data, loop=False)
    src.read(5)
    tail = src.read(10)
    assert np.array_equal(tail.real, np.arange(5, 8, dtype=np.float32)), (
        f"one-shot dropped the tail: got {tail.real}"
    )
    try:
        src.read(1)
    except EOFError:
        pass
    else:
        raise AssertionError("expected EOFError after the tail was consumed")
    src.close()


if __name__ == "__main__":
    test_loop_splices_tail_and_head()
    test_loop_request_longer_than_file_keeps_cursor()
    test_shipped_capture_workflow_does_not_skip()
    test_read_returns_an_independent_copy()
    test_oneshot_returns_tail_then_eof()
    print("OK")
