"""INSPECT classify gates that previous suites only source-inspected."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import demod
from fpvscan.dsp.demod import LINE_PAL


def _sync_train(fs: float, line_hz: float, seconds: float = 0.05) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    return ((t % (1.0 / line_hz)) < 4.7e-6).astype(np.float32)


def test_unknown_standard_halves_confidence():
    """16 kHz is labeled '?' and then conf *= 0.5. Without the penalty
    cheap VTx off-grid line rates confirm as video and auto-peek parks
    on a spur. The label-only test in the older suite does not catch this.
    """
    fs = 1e6
    pal = demod.classify_video(_sync_train(fs, LINE_PAL), fs, tol_hz=150.0)
    unk = demod.classify_video(_sync_train(fs, 16000.0), fs, tol_hz=150.0)
    assert pal.standard == "PAL"
    assert unk.standard == "?"
    assert abs(unk.line_rate - 16000.0) < 40
    assert unk.confidence <= pal.confidence * 0.55 + 0.02
    assert unk.confidence < pal.confidence
    # Default min_conf=0.45: a mid-prominence '?' must not sneak through
    # after the 0.5 cut when the same peak would pass as PAL.
    if pal.is_video and pal.confidence < 0.90:
        assert not unk.is_video or unk.confidence <= pal.confidence * 0.5 + 0.03


def test_nfft_is_capped_at_65536(monkeypatch):
    """A long INSPECT capture must not allocate a multi-megabin rfft on Pi."""
    fs = 400e3
    n = 200_000
    t = np.arange(n, dtype=np.float64) / fs
    x = (0.2 * np.sin(2 * np.pi * LINE_PAL * t)).astype(np.float32)
    seen: list[int] = []
    real = np.fft.rfft

    def spy(a, *args, **kwargs):
        seen.append(len(a))
        return real(a, *args, **kwargs)

    monkeypatch.setattr(np.fft, "rfft", spy)
    demod.classify_video(x, fs)
    assert seen
    assert seen[0] == 65536


def test_high_rate_short_inspect_is_rejected_after_decimation():
    """12 ms at 40 Msps is 480 k raw samples — looks long — but decimation
    to ~400 kHz leaves < 8192 and classify returns immediately. A mis-set
    inspect_ms then looks like 'no analog on the stand'.
    """
    fs = 40e6
    t = np.arange(int(fs * 0.012)) / fs
    x = ((t % (1.0 / LINE_PAL)) < 4.7e-6).astype(np.float32)
    score = demod.classify_video(x, fs)
    assert score.is_video is False
    assert score.line_rate == 0.0
    assert score.standard == "?"
