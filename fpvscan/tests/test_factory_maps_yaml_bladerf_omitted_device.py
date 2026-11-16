"""factory must default an omitted device to empty string, not None."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_omitted_device_still_constructs_bladerf(monkeypatch):
    """#21 maps a non-empty serial. #98 maps shipped YAML
    `device: ''` as empty string, not None — copy-pasting
    `lib_path or None` onto `device` makes `None.encode()`
    TypeError on `bladerf_open`. The remaining half: operators
    delete the `device:` line to use the first board while
    leaving `gain_db` / `settle_us`. factory is
    `cfg.get("device", "")`. A leftover `cfg["device"]` (no
    default) KeyErrors on start and the board never opens, so
    `confirm_hits=2` never lights the bird. A leftover
    `cfg.get("device") or None` still constructs while `open()`
    TypeErrors. A leftover always-serial from #21 ignores the
    first-board path. Constructor must see "", not None, and
    must not see YAML-only names. File/sim constructors must
    not run. Explicit empty string stays #98; non-empty serial
    stays #21.
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
        "gain_db": 35,
        "rx_channel": 1,
        "agc": False,
        "settle_us": 500,
        "quick_tune": False,
    })
    assert src.name == "bladerf"
    assert file_calls == []
    assert sim_calls == []
    assert seen["device"] == ""
    assert seen["device"] is not None
    assert seen["device"] != "*:serial=abcd1234"
    assert seen["gain_db"] == 35.0
    assert seen["channel"] == 1
    assert seen["settle_us"] == 500.0
    assert seen["use_meta"] is False
    assert "driver" not in seen
    assert "rx_channel" not in seen
    assert "quick_tune" not in seen
