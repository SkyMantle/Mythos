"""factory must default an omitted gain_db to the constructor 30 dB."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_omitted_gain_db_still_constructs_bladerf(monkeypatch):
    """#21 explicit YAML `gain_db: 35` (bench, not BladeRF() 30) is
    forwarded. #101 omitted `lib_path` / this pair's `settle_us`
    still construct. Operators delete the `gain_db:` line while
    leaving `device` / `settle_us` (Pi install; 30 dB is the
    BladeRF MGC default and avoids clipping a close VTx). factory
    is `float(cfg.get("gain_db", 30))`. A leftover `cfg["gain_db"]`
    (no default) KeyErrors on start and the board never opens, so
    `confirm_hits=2` never lights the bird. A leftover always-35
    (copy-paste from shipped YAML / #21) still constructs while
    the ADC clips on a close bird the operator *lowered* gain to
    avoid — CVBS then emits stripes instead of a picture.
    Constructor must see 30.0, not 35, and must not see YAML-only
    names. File/sim constructors must not run. Explicit 35 stays #21.
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
        "rx_channel": 1,
        "agc": False,
        "settle_us": 500,
        "quick_tune": False,
    })
    assert src.name == "bladerf"
    assert file_calls == []
    assert sim_calls == []
    assert seen["gain_db"] == 30.0
    assert seen["gain_db"] != 35.0
    assert seen["gain_db"] != 35
    assert seen["device"] == "*:serial=abcd1234"
    assert seen["settle_us"] == 500.0
    assert seen["channel"] == 1
    assert seen["use_meta"] is False
    assert "driver" not in seen
    assert "rx_channel" not in seen
    assert "quick_tune" not in seen
