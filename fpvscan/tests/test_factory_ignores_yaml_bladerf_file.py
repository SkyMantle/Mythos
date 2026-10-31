"""factory must not forward file / sim YAML keys into BladeRF."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_bladerf_file_yaml(monkeypatch):
    """#80 factory ignores sim YAML rate / path / emitters
    (`SimSource()` takes no kwargs). #79 ignores file
    `sample_rate` / `center_hz` / `gain_db` (sidecar stays).
    #73 ignores bladerf `timeout_ms` / `bandwidth_ratio`.
    The complementary leftover is a `driver: bladerf` block
    that *names* `sample_rate: 35e6` next to `path` / `loop` /
    `realtime` / `emitters` / `noise_db` / `seed` — the keys
    operators copy from a file or sim stanza. BladeRF's
    constructor has no such kwargs. A leftover
    `BladeRF(path=cfg["path"], sample_rate=cfg.get(...))`
    TypeErrors on start and the board never opens, or, if
    honored as `device=`, opens the wrong USB handle. Honoring
    `loop` / `emitters` / `seed` is the same TypeError.
    Constructor must not see file / sim rate / LO / path /
    loop / realtime / emitters / noise / seed.
    """
    seen: dict = {}

    class Fake:
        name = "bladerf"

        def __init__(self, device="", lib_path=None, channel=0,
                     gain_db=30.0, agc=False, num_buffers=32,
                     buffer_size=32768, num_transfers=16,
                     settle_us=400.0, use_meta=False, **kw):
            seen["device"] = device
            seen["lib_path"] = lib_path
            seen["channel"] = channel
            seen["gain_db"] = gain_db
            seen["agc"] = agc
            seen["num_buffers"] = num_buffers
            seen["buffer_size"] = buffer_size
            seen["num_transfers"] = num_transfers
            seen["settle_us"] = settle_us
            seen["use_meta"] = use_meta
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.bladerf.BladeRF", Fake)
    src = make_source({
        "driver": "bladerf",
        "sample_rate": 35e6,
        "center_hz": 1280e6,
        "path": "caps/vtx.cf32",
        "loop": False,
        "realtime": True,
        "emitters": [],
        "noise_db": -20,
        "seed": 99,
    })
    assert src.name == "bladerf"
    assert "sample_rate" not in seen
    assert "center_hz" not in seen
    assert "path" not in seen
    assert "loop" not in seen
    assert "realtime" not in seen
    assert "emitters" not in seen
    assert "noise_db" not in seen
    assert "seed" not in seen
    assert seen.get("sample_rate") != 35e6
    assert seen.get("path") != "caps/vtx.cf32"
    assert seen.get("loop") is not False
    assert seen.get("emitters") != []
    assert seen.get("seed") != 99
