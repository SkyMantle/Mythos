"""factory must not forward the YAML driver selector into BladeRF."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_bladerf_driver_kwarg(monkeypatch):
    """#47 unknown `driver` raises `ValueError` before any
    constructor. #21 maps `rx_channel` → `channel` and
    `quick_tune` → `use_meta`. #91 still passes `channel=`
    (not the YAML name). #90 ignores `lo_offset_hz`; #89
    ignores `bias_tee_gain_offset_db`; #81 ignores file /
    sim keys. The complementary leftover is forwarding the
    selector itself: `BladeRF(driver=cfg.get("driver"),
    ...)`. BladeRF() has no `driver` kwarg. Shipped YAML
    always has `driver: bladerf`. A leftover extra
    `driver=` TypeErrors on start and the board never
    opens, so `confirm_hits=2` never lights the bird.
    Constructor must still see USB identity (`gain_db` /
    `channel`) and must not see `driver`. Engine never
    reads that kwarg — `make_source` already branched.
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
        "rx_channel": 0,
        "quick_tune": False,
    })
    assert src.name == "bladerf"
    assert seen["gain_db"] == 35.0
    assert seen["channel"] == 0
    assert seen["use_meta"] is False
    assert "driver" not in seen
    assert seen.get("driver") != "bladerf"
    assert seen.get("channel") != "bladerf"
