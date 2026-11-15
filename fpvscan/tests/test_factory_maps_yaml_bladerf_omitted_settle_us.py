"""factory must default an omitted settle_us to the 400 µs PLL dump."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_omitted_settle_us_still_constructs_bladerf(monkeypatch):
    """#21 explicit YAML `settle_us: 500` (bench, not BladeRF() 400)
    is forwarded. #101 omitted `lib_path` still constructs. Operators
    delete the `settle_us:` line while leaving `device` / `gain_db`
    (Pi install; 400 µs is enough on a warmed board). factory is
    `float(cfg.get("settle_us", 400))`. A leftover `cfg["settle_us"]`
    (no default) KeyErrors on start and the board never opens, so
    `confirm_hits=2` never lights the bird. A leftover always-500
    (copy-paste from shipped YAML / #21) still constructs while
    every 8 MHz hop waits an extra 100 µs × ~700 steps — the
    400 MHz–6 GHz walk stalls on a box that deleted the knob to
    go *faster*. Constructor must see 400.0, not 500, and must
    not see YAML-only names. File/sim constructors must not run.
    Non-empty 500 µs stays #21.
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
        "quick_tune": False,
    })
    assert src.name == "bladerf"
    assert file_calls == []
    assert sim_calls == []
    assert seen["settle_us"] == 400.0
    assert seen["settle_us"] != 500.0
    assert seen["settle_us"] != 500
    assert seen["device"] == "*:serial=abcd1234"
    assert seen["gain_db"] == 35.0
    assert seen["channel"] == 1
    assert seen["use_meta"] is False
    assert "driver" not in seen
    assert "rx_channel" not in seen
    assert "quick_tune" not in seen
