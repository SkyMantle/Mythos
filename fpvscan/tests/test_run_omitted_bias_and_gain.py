"""Omitted startup keys must not apply 4.5 V at an unknown gain."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_omitted_bias_tee_stays_off_and_gain_is_30():
    """#40 factory still omits `bias_tee` from BladeRF(). #50 checked
    `_run` order with shipped-style True. #57 forwarded explicit
    factory gain_db=30. Engine `_run` fallbacks are bias_tee=False
    and gain_db=30. Defaulting True on a partial config applies
    4.5 V plus the LNA before any offset, which is the ADC-clip
    window. Defaulting gain to 0 deafens LOCK; a leftover 60 clips
    even with the tee off. helpers.py already omits bias_tee.
    """
    src = MockSource()
    eng = make_engine(src=src)
    assert "bias_tee" not in eng.cfg["sdr"]
    eng.cfg["sdr"].pop("gain_db")
    eng._stop.set()
    eng._run()
    assert src.bias is False
    assert src.gain == 30.0
