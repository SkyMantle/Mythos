"""factory must keep shipped empty device as '', not None."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_maps_empty_bladerf_device_not_none(monkeypatch):
    """#21 maps a non-empty `device` serial onto BladeRF.
    #91 maps `rx_channel` → `channel` without the YAML
    name; #92 ignores `driver`; #97 maps `quick_tune` →
    `use_meta`. Shipped YAML is `device: ''` (first found
    board) and `lib_path: ''` (search). factory *does*
    `lib_path=cfg.get("lib_path") or None` so ctypes
    searches. Copy-pasting that onto
    `device=cfg.get("device") or None` turns the empty
    string into None. BladeRF.open() does
    `self.device.encode() or None` — `None.encode()`
    TypeErrors and the board never opens, so
    `confirm_hits=2` never lights the bird. Constructor
    must still see `device=""` (falsy; encode-or-None
    becomes the libbladeRF first-device pointer) and
    must not see `device=None`. `lib_path` empty still
    becomes None (the contrast). Must not pass YAML-only
    names (`driver` / `rx_channel` / `quick_tune`).
    """
    seen: dict = {}

    class Fake:
        name = "bladerf"

        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.bladerf.BladeRF", Fake)
    src = make_source({
        "driver": "bladerf",
        "device": "",
        "lib_path": "",
        "gain_db": 35,
        "rx_channel": 0,
        "quick_tune": False,
        "agc": False,
    })
    assert src.name == "bladerf"
    assert "device" in seen
    assert seen["device"] == ""
    assert seen["device"] is not None
    assert seen.get("device") != "bladerf"
    assert seen["lib_path"] is None
    assert seen["lib_path"] != ""
    assert seen["gain_db"] == 35.0
    assert seen["channel"] == 0
    assert seen["use_meta"] is False
    assert seen["agc"] is False
    assert "driver" not in seen
    assert "rx_channel" not in seen
    assert "quick_tune" not in seen
    assert seen.get("quick_tune") is not False
    assert seen.get("device") is not None
