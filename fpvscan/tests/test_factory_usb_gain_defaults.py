"""factory must keep USB stream and MGC defaults when YAML omits the keys."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_usb_gain_and_agc_defaults_when_omitted(monkeypatch):
    """#21 forwarded an explicit YAML block. #56 pinned omitted
    settle_us=400. The other constructor fallbacks are the ones that
    brick a partial config: gain_db=0 clips nothing but AGC-on
    (`True` fallback) walks the VGA during LOCK; buffer_size=0
    fails sync_config; channel=1 opens RX2 on a one-antenna Pi.
    """
    seen: dict = {}

    class Fake:
        name = "bladerf"

        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.bladerf.BladeRF", Fake)
    src = make_source({"driver": "bladerf"})
    assert src.name == "bladerf"
    assert seen["gain_db"] == 30.0
    assert seen["agc"] is False
    assert seen["num_buffers"] == 32
    assert seen["buffer_size"] == 32768
    assert seen["num_transfers"] == 16
    assert seen["channel"] == 0
    assert seen["device"] == ""
    assert seen["lib_path"] is None
