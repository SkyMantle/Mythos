"""factory must default omitted quick_tune to use_meta=False."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_omitted_quick_tune_still_constructs_bladerf(monkeypatch):
    """#97 maps explicit YAML `quick_tune: true` → `use_meta=True`
    and does not pass the YAML name. #21 / #100 map explicit
    `false`. #57 bundled omitted USB/gain but never asserted
    `use_meta`. Operators comment `# quick_tune: false` after
    `bench_retune.py` reports 0 profiles, while leaving
    `gain_db` / `device`. factory is
    `bool(cfg.get("quick_tune", False))` as `use_meta=`. A
    leftover `cfg["quick_tune"]` (no default) KeyErrors on
    start and the board never opens, so `confirm_hits=2` never
    lights the bird. A leftover always-True (copy-paste from
    #97) enables SC16_Q11_META on a board/library that has no
    profiles — `sync_rx` flags overruns, LOCK looks like a dead
    VTx. A leftover extra `quick_tune=` TypeErrors (#97).
    Constructor must see `use_meta is False`, not True, and
    must not see the YAML name. File/sim constructors must not
    run. Explicit True stays #97.
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
    })
    assert src.name == "bladerf"
    assert file_calls == []
    assert sim_calls == []
    assert seen["use_meta"] is False
    assert seen["use_meta"] is not True
    assert "quick_tune" not in seen
    assert seen.get("quick_tune") is not True
    assert seen.get("use_meta") != "quick_tune"
    assert seen["device"] == "*:serial=abcd1234"
    assert seen["gain_db"] == 35.0
    assert seen["channel"] == 0
    assert seen["agc"] is False
    assert seen["settle_us"] == 500.0
    assert "driver" not in seen
    assert "rx_channel" not in seen
