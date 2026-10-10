"""factory must keep the 400 µs PLL dump when YAML omits settle_us."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_settle_defaults_to_400_us_when_omitted(monkeypatch):
    """#21 forwarded an explicit YAML 500. BladeRF / factory both
    default to 400 µs when the key is missing — `max(2048, fs·τ)`
    at 35 Msps is 14 000 samples. A 0 µs default (or forgetting to
    pass settle_us) makes inspect/LOCK read PLL garbage and look
    like 'no video' on a real hop.
    """
    seen: dict = {}

    class Fake:
        name = "bladerf"

        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.bladerf.BladeRF", Fake)
    src = make_source({"driver": "bladerf"})
    assert src.name == "bladerf"
    assert seen["settle_us"] == 400.0
    assert seen["use_meta"] is False
