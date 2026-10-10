"""A config that omits the LNA offset must still cut receiver gain."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_bias_tee_offset_defaults_to_18_when_key_is_omitted():
    """Shipped YAML is 15 dB (#50 checked 30−15). The code fallback
    is 18 — the middle of the 15–20 dB LNA range in the comment.
    Dropping the key and then defaulting to 0 leaves the LNA on top
    of gain_db and the ADC clips; CVBS then paints bars instead of
    a picture. #5 / #50 always had the YAML key in the fixture.
    """
    src = MockSource()
    eng = make_engine(src=src)
    eng.cfg["sdr"].pop("bias_tee_gain_offset_db")
    eng._apply_bias_tee_gain(True)
    assert src.gain == 30.0 - 18.0
    eng._apply_bias_tee_gain(False)
    assert src.gain == 30.0
