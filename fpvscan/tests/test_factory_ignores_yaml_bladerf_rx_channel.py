"""factory must map YAML rx_channel onto BladeRF.channel, not the YAML name."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_bladerf_rx_channel_kwarg(monkeypatch):
    """#21 maps `rx_channel: 1` → `channel=1` and `quick_tune` →
    `use_meta`. #40 omits `bias_tee`; #81 ignores file/sim keys;
    #89 ignores `bias_tee_gain_offset_db`; #90 ignores
    `lo_offset_hz`. The complementary leftover is still passing
    the YAML name: `BladeRF(..., rx_channel=cfg.get("rx_channel"),
    channel=...)`. BladeRF() takes `channel`, not `rx_channel`.
    Shipped YAML has `rx_channel: 0` (RX1); RX2 is `1`. A leftover
    extra `rx_channel=` TypeErrors on start and the board never
    opens, so `confirm_hits=2` never lights the bird. Constructor
    must still see `channel` (USB identity) and must not see
    `rx_channel`. Engine never reads that kwarg — CHANNEL_RX is
    applied inside BladeRF from `channel`.
    """
    seen: dict = {}

    class Fake:
        name = "bladerf"

        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.bladerf.BladeRF", Fake)
    src = make_source({
        "driver": "bladerf",
        "gain_db": 35,
        "rx_channel": 1,
        "quick_tune": False,
    })
    assert src.name == "bladerf"
    assert seen["gain_db"] == 35.0
    assert seen["channel"] == 1
    assert seen["use_meta"] is False
    assert "rx_channel" not in seen
    assert seen.get("rx_channel") != 1
    assert seen.get("channel") != "rx_channel"
