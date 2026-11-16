"""factory must default an omitted rx_channel to RX1 (channel 0)."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_omitted_rx_channel_still_constructs_bladerf(monkeypatch):
    """#91 maps explicit YAML `rx_channel: 1` → `channel=1`. #57
    bundled a fully omitted sdr block (channel=0 with USB/gain).
    Operators delete the `rx_channel:` line on a one-antenna Pi
    (RX1 is BladeRF() default 0) while leaving `device` /
    `gain_db`. factory is `int(cfg.get("rx_channel", 0))`. A
    leftover `cfg["rx_channel"]` (no default) KeyErrors on start
    and the board never opens, so `confirm_hits=2` never lights
    the bird. A leftover always-1 (copy-paste from #91) opens
    RX2 — empty spectrum on a Pi that only wired RX1. A leftover
    `int(cfg.get("rx_channel"))` TypeErrors on int(None).
    Constructor must see 0, not 1, and must not see YAML-only
    names. File/sim constructors must not run. Explicit 1 stays
    #91.
    """
    seen: dict = {}
    file_calls = []
    sim_calls = []

    class FakeBlade:
        name = "bladerf"

        def __init__(self, **kw):
            seen.update(kw)

    class FakeFile:
        name = "file"

        def __init__(self, *a, **kw):
            file_calls.append((a, kw))

    class FakeSim:
        name = "sim"

        def __init__(self, *a, **kw):
            sim_calls.append((a, kw))

    monkeypatch.setattr("fpvscan.sdr.bladerf.BladeRF", FakeBlade)
    monkeypatch.setattr("fpvscan.sdr.file.FileSource", FakeFile)
    monkeypatch.setattr("fpvscan.sdr.sim.SimSource", FakeSim)
    src = make_source({
        "device": "*:serial=abcd1234",
        "gain_db": 35,
        "agc": False,
        "settle_us": 500,
        "quick_tune": False,
    })
    assert src.name == "bladerf"
    assert file_calls == []
    assert sim_calls == []
    assert seen["channel"] == 0
    assert seen["channel"] != 1
    assert seen["device"] == "*:serial=abcd1234"
    assert seen["gain_db"] == 35.0
    assert seen["settle_us"] == 500.0
    assert seen["use_meta"] is False
    assert "driver" not in seen
    assert "rx_channel" not in seen
    assert "quick_tune" not in seen
