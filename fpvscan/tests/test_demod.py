from __future__ import annotations

import numpy as np
import pytest

from fpvscan.dsp.cvbs import STD_GEOM
from fpvscan.dsp.demod import (
    LINE_MAX_ERR_HZ,
    LINE_NTSC,
    LINE_PAL,
    _HAVE_NUMBA,
    _fm_demod_numpy,
    classify_video,
    fm_demod,
    standard_from_line_rate,
)


def _tone(hz: float, fs: float = 400e3, n: int = 16384) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / fs
    return np.sin(2 * np.pi * hz * t).astype(np.float32)


def _score(hz: float):
    return classify_video(
        _tone(hz), 400e3, min_harmonics=0, min_conf=0.0, min_prominence_db=0.0,
    )


def test_line_rate_15734_is_ntsc_not_pal_first() -> None:
    assert standard_from_line_rate(15734.0) == "NTSC"
    assert standard_from_line_rate(LINE_NTSC) == "NTSC"
    assert _score(LINE_NTSC).standard == "NTSC"


def test_line_rate_15625_is_pal() -> None:
    assert standard_from_line_rate(15625.0) == "PAL"
    assert standard_from_line_rate(LINE_PAL) == "PAL"
    assert _score(LINE_PAL).standard == "PAL"


def test_closer_to_ntsc_than_pal_is_not_pal() -> None:
    # PAL-first with 150 Hz labeled NTSC 15734 as PAL (Δ = 109 Hz).
    mid_ntsc = 15700.0
    assert abs(mid_ntsc - LINE_NTSC) < abs(mid_ntsc - LINE_PAL)
    assert standard_from_line_rate(mid_ntsc) == "NTSC"
    assert _score(mid_ntsc).standard == "NTSC"


def test_far_from_both_standards_is_unknown() -> None:
    far = LINE_PAL - (LINE_MAX_ERR_HZ + 20.0)
    assert standard_from_line_rate(far) == "?"
    assert "?" in STD_GEOM


def test_decode_geometry_uses_nearest_when_label_is_unknown() -> None:
    """Cheap analog often sits ~125 Hz off PAL; '?' field timing cannot lock."""
    far = LINE_PAL - (LINE_MAX_ERR_HZ + 20.0)
    assert abs(far - LINE_PAL) < abs(far - LINE_NTSC)
    assert standard_from_line_rate(far) == "?"
    assert standard_from_line_rate(far, max_err_hz=None) == "PAL"
    assert STD_GEOM["PAL"] != STD_GEOM["?"]


def _fm_tone(n: int = 4096, fs: float = 1e6, f0: float = 50e3) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / fs
    ph = 2 * np.pi * f0 * t
    return (np.cos(ph) + 1j * np.sin(ph)).astype(np.complex64)


def test_fm_demod_numpy_fallback_runs() -> None:
    iq = _fm_tone()
    out = _fm_demod_numpy(iq, 1e6, 4e6)
    assert out.shape == (len(iq) - 1,)
    assert abs(float(np.median(out)) - (50e3 / 4e6)) < 0.02
    public = fm_demod(iq, 1e6, 4e6)
    assert public.shape == (len(iq) - 1,)
    empty = fm_demod(np.zeros(1, dtype=np.complex64), 1e6)
    assert empty.size == 0


@pytest.mark.skipif(not _HAVE_NUMBA, reason="numba not installed")
def test_fm_demod_numba_matches_numpy() -> None:
    iq = _fm_tone()
    np.testing.assert_allclose(
        fm_demod(iq, 1e6, 4e6),
        _fm_demod_numpy(iq, 1e6, 4e6),
        rtol=1e-5,
        atol=1e-5,
    )
