"""factory must not forward bladeRF USB YAML keys into FileSource."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_file_usb_yaml(monkeypatch):
    """#81 factory ignores file / sim keys for bladerf.
    #79 ignores file `sample_rate` / `center_hz` / `gain_db`.
    #77 ignores `realtime: true`; #74 forwards `loop: false`;
    #73 ignores bladerf `timeout_ms` / `bandwidth_ratio`.
    The complementary leftover is a `driver: file` block
    that *names* USB stream keys operators copy from a
    bladeRF stanza — `device` / `lib_path` / `rx_channel` /
    `agc` / `num_buffers` / `buffer_size` / `num_transfers` /
    `settle_us` / `quick_tune` / `timeout_ms` /
    `bandwidth_ratio`. FileSource's constructor is
    `path` / `loop` / `realtime` only. A leftover
    `FileSource(..., device=cfg.get("device"),
    num_buffers=cfg.get("num_buffers"))` TypeErrors on
    start and replay never opens, so `confirm_hits=2`
    never lights the bird. Constructor must not see USB
    handle / stream / settle / timeout / bandwidth keys.
    """
    seen: dict = {}

    class Fake:
        name = "file"

        def __init__(self, path, loop=True, realtime=False, **kw):
            seen["path"] = path
            seen["loop"] = loop
            seen["realtime"] = realtime
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.file.FileSource", Fake)
    src = make_source({
        "driver": "file",
        "path": "caps/vtx.cf32",
        "loop": True,
        "device": "*:serial=xyz",
        "lib_path": "/opt/bladeRF.dll",
        "rx_channel": 1,
        "agc": True,
        "num_buffers": 8,
        "buffer_size": 4096,
        "num_transfers": 4,
        "settle_us": 0,
        "quick_tune": True,
        "timeout_ms": 0,
        "bandwidth_ratio": 0,
    })
    assert src.name == "file"
    assert seen["path"] == "caps/vtx.cf32"
    assert seen["loop"] is True
    assert "device" not in seen
    assert "lib_path" not in seen
    assert "rx_channel" not in seen
    assert "channel" not in seen
    assert "agc" not in seen
    assert "num_buffers" not in seen
    assert "buffer_size" not in seen
    assert "num_transfers" not in seen
    assert "settle_us" not in seen
    assert "quick_tune" not in seen
    assert "use_meta" not in seen
    assert "timeout_ms" not in seen
    assert "bandwidth_ratio" not in seen
    assert seen.get("device") != "*:serial=xyz"
    assert seen.get("num_buffers") != 8
    assert seen.get("timeout_ms") != 0
    assert seen.get("agc") is not True
