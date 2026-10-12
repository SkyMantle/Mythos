"""Omitted sweep keys must keep a real PSD grab and analog occupancy floors."""
from __future__ import annotations

from unittest.mock import patch

from tests.helpers import MockSource, make_engine


def test_sweep_omitted_fft_averages_and_occupancy_floors():
    """#31 forwarded explicit YAML 5 dB / 2 MHz / 200 kHz. #18 pinned
    shipped fft_size=8192. The code fallbacks are fft=4096, averages=8,
    threshold_db=8, min_bw_hz=4e6, dc_notch_hz=200e3. A 0-average grab
    is empty; a 0 dB floor lists Wi-Fi; a 0 Hz min_bw lists CW as analog.
    helpers.py already omits these scan keys — that is the partial-config
    path.
    """
    src = MockSource()
    grabbed: list[int] = []
    real = src.retune_and_read

    def wrap(hz, n):
        grabbed.append(int(n))
        return real(hz, n)

    src.retune_and_read = wrap
    eng = make_engine(src=src)
    assert "fft_size" not in eng.cfg["scan"]
    assert "averages" not in eng.cfg["scan"]
    assert "threshold_db" not in eng.cfg["scan"]
    assert "min_bw_hz" not in eng.cfg["scan"]
    eng.cfg["scan"]["start_hz"] = 5700e6
    eng.cfg["scan"]["stop_hz"] = 5900e6
    eng.cfg["scan"]["step_hz"] = 200e6

    occ: dict = {}

    def capture(_psd, _center, _fs, **kw):
        occ.update(kw)
        return []

    with patch("fpvscan.engine.spectrum.find_occupied", side_effect=capture):
        eng._do_sweep()

    assert grabbed == [4096 * 8]
    assert occ["threshold_db"] == 8.0
    assert occ["min_bw_hz"] == 4e6
    assert occ["dc_notch_hz"] == 200e3
