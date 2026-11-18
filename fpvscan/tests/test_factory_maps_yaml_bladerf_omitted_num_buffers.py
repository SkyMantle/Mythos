"""factory must default an omitted num_buffers key to 32."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_omitted_num_buffers_still_constructs_bladerf(monkeypatch):
    """#57 bundled omitted `num_buffers=32` with empty
    `{"driver": "bladerf"}`. #100 maps explicit YAML
    `num_buffers: 8` (small-RAM Pi). Operators delete the
    `num_buffers:` line after shrinking the USB ring for a
    bench, while leaving `device` / `gain_db` / `buffer_size`.
    factory is `int(cfg.get("num_buffers", 32))`. A leftover
    `cfg["num_buffers"]` (no default) KeyErrors on start and
    the board never opens, so `confirm_hits=2` never lights
    the bird. A leftover always-8 (copy-paste from #100)
    under-buffers a full-rate 35 Msps walk — USB overruns,
    occupancy never crosses `threshold_db`. A leftover 0
    fails `sync_config`. Constructor must see 32, not 8 or
    0, and must not see YAML-only names. File/sim
    constructors must not run. Explicit 8 stays #100.
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
        "agc": False,
        "settle_us": 500,
        "quick_tune": False,
        "buffer_size": 32768,
        "num_transfers": 16,
    })
    assert src.name == "bladerf"
    assert file_calls == []
    assert sim_calls == []
    assert seen["num_buffers"] == 32
    assert seen["num_buffers"] != 8
    assert seen["num_buffers"] != 0
    assert seen["buffer_size"] == 32768
    assert seen["num_transfers"] == 16
    assert seen["device"] == "*:serial=abcd1234"
    assert seen["gain_db"] == 35.0
    assert seen["channel"] == 0
    assert seen["agc"] is False
    assert seen["settle_us"] == 500.0
    assert seen["use_meta"] is False
    assert "driver" not in seen
    assert "rx_channel" not in seen
    assert "quick_tune" not in seen
