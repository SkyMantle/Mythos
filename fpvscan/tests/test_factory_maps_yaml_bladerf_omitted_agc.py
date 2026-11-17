"""factory must default an omitted agc key to manual gain (False)."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_omitted_agc_still_constructs_bladerf(monkeypatch):
    """#21 / #57 bundled omitted `agc=False` with USB/gain. #100
    maps explicit YAML `agc: true` (search). Operators delete
    the `agc:` line after bringup so LOCK stays on MGC, while
    leaving `gain_db` / `settle_us`. factory is
    `bool(cfg.get("agc", False))`. A leftover `cfg["agc"]` (no
    default) KeyErrors on start and the board never opens, so
    `confirm_hits=2` never lights the bird. A leftover always-
    True (copy-paste from #100) walks the VGA during LOCK —
    brightness pumps, CVBS sync walks, tracking never shrinks.
    A leftover `bool(cfg.get("agc"))` is False on None *and*
    on missing, but `bool("false")` is True if YAML is a
    string. Constructor must see False, not True, and must not
    see YAML-only names. File/sim constructors must not run.
    Explicit True stays #100.
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
        "rx_channel": 0,
        "settle_us": 500,
        "quick_tune": False,
    })
    assert src.name == "bladerf"
    assert file_calls == []
    assert sim_calls == []
    assert seen["agc"] is False
    assert seen["agc"] is not True
    assert seen["device"] == "*:serial=abcd1234"
    assert seen["gain_db"] == 35.0
    assert seen["channel"] == 0
    assert seen["settle_us"] == 500.0
    assert seen["use_meta"] is False
    assert "driver" not in seen
    assert "rx_channel" not in seen
    assert "quick_tune" not in seen
