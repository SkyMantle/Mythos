"""factory must default an omitted lib_path to search-path None."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_omitted_lib_path_still_constructs_bladerf(monkeypatch):
    """#21 empty `lib_path: ''` becomes None (`cfg.get("lib_path")
    or None`). #100 omitted `driver` still constructs BladeRF.
    Operators delete the `lib_path:` line while leaving `device` /
    `gain_db` (Pi install; the DLL is on the search path). factory
    is `cfg.get("lib_path") or None`. A leftover `cfg["lib_path"]`
    (no default) KeyErrors on start and the board never opens, so
    `confirm_hits=2` never lights the bird. Constructor must still
    see BladeRF kwargs and must see lib_path as None, not a missing
    key that raises before BladeRF(). File/sim constructors must
    not run. Non-empty `--lib` is the other half of this pair.
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
    assert seen.get("lib_path") is None
    assert seen.get("lib_path") != ""
    assert seen["device"] == "*:serial=abcd1234"
    assert seen["gain_db"] == 35.0
    assert seen["channel"] == 1
    assert seen["use_meta"] is False
    assert seen["settle_us"] == 500.0
    assert "driver" not in seen
    assert "rx_channel" not in seen
    assert "quick_tune" not in seen
