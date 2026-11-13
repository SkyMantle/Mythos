"""factory must default an omitted driver selector to BladeRF."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_omitted_driver_still_constructs_bladerf(monkeypatch):
    """#47 unknown `driver` raises before any constructor. #92
    ignores explicit `driver: bladerf` as a kwarg (BladeRF() has
    no such name). #93/#96 ignore it for file/sim. Shipped YAML
    always names `driver: bladerf`; operators comment that line
    while leaving `device` / `gain_db` / USB keys. factory is
    `cfg.get("driver", "bladerf")`. A leftover `cfg["driver"]`
    (no default) KeyErrors on start, or a leftover empty-string
    branch (`if driver == "bladerf"` misses `""`) falls through
    to `ValueError: Невідомий драйвер` and the board never
    opens, so `confirm_hits=2` never lights the bird. Constructor
    must still see BladeRF kwargs (`gain_db` / `channel` /
    `use_meta`) and must not see `driver`. File/sim constructors
    must not run.
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
        "rx_channel": 1,
        "agc": False,
        "settle_us": 500,
        "quick_tune": False,
    })
    assert src.name == "bladerf"
    assert file_calls == []
    assert sim_calls == []
    assert seen["device"] == "*:serial=abcd1234"
    assert seen["gain_db"] == 35.0
    assert seen["channel"] == 1
    assert seen["agc"] is False
    assert seen["settle_us"] == 500.0
    assert seen["use_meta"] is False
    assert "driver" not in seen
    assert "rx_channel" not in seen
    assert "quick_tune" not in seen
    assert seen.get("driver") != "bladerf"
    assert seen.get("channel") != "bladerf"
