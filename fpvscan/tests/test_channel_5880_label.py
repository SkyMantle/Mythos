"""R7 and F8 share 5880 MHz — nearest_channel must stay stable."""
from __future__ import annotations

from fpvscan.bands import ALL_5G8, RACEBAND, BAND_F, nearest_channel
from fpvscan.engine import Engine


def test_5880_is_both_r7_and_f8_but_labels_as_r7():
    """Raceband R7 and Band F F8 are the same frequency. ALL_5G8 keeps
    both keys (they do not overwrite). nearest_channel walks insertion
    order and keeps the first exact match — RACEBAND is merged first,
    so the console shows R7. Switching the loop to `d <= best_d`
    flips the label to F8 and operators click the wrong name on a
    Raceband map even though the Hz are identical.
    """
    assert RACEBAND["R7"] == 5880
    assert BAND_F["F8"] == 5880
    assert ALL_5G8["R7"] == 5880
    assert ALL_5G8["F8"] == 5880
    assert nearest_channel(5880e6) == "R7"


def test_engine_inspect_uses_a_fresh_retune_not_the_sweep_iq():
    """_inspect's first argument is `_iq` — deliberately unused. The
    sweep buffer is fft×avg long and fails classify after decimation
    (see test_inspect_longer_than_sweep). Reusing it to 'save a retune'
    would confirm nothing.
    """
    import inspect
    src = inspect.getsource(Engine._inspect)
    assert "def _inspect(self, _iq, center_hz, fs, occ)" in src
    assert "self.src.retune_and_read(occ.center_hz" in src
