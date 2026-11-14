"""factory must forward a real --lib path into BladeRF, not search-path None."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_maps_nonempty_bladerf_lib_path(monkeypatch):
    """#21 empty shipped YAML `lib_path: ''` becomes None so
    `load_lib` searches the default dirs. #98 empty `device: ''`
    stays "" — copy-pasting `or None` onto device is a TypeError
    on `None.encode()`. The missing half: run.py `--lib` /
    `sdr.lib_path` / `BLADERF_LIB_DIR` operators pass a real
    filename (`C:\\bladeRF\\x64\\bladeRF.dll`). factory is
    `lib_path=cfg.get("lib_path") or None`. A leftover always-None
    (`lib_path=None` / dropping the kwarg / treating any string
    like the empty-YAML branch) ignores the path the operator
    named; `load_lib` searches Program Files, misses a nonstandard
    install, and the board never opens, so `confirm_hits=2` never
    lights the bird. Constructor must see the YAML path, not None,
    and must not see YAML-only names (`driver` / `rx_channel` /
    `quick_tune`). Empty-string → None stays #21.
    """
    seen: dict = {}

    class Fake:
        name = "bladerf"

        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.bladerf.BladeRF", Fake)
    lib = r"C:\Program Files\bladeRF\x64\bladeRF.dll"
    src = make_source({
        "driver": "bladerf",
        "device": "*:serial=abcd1234",
        "lib_path": lib,
        "rx_channel": 0,
        "gain_db": 35,
        "agc": False,
        "settle_us": 500,
        "quick_tune": False,
    })
    assert src.name == "bladerf"
    assert seen["lib_path"] == lib
    assert seen["lib_path"] is not None
    assert seen["lib_path"] != ""
    assert seen["device"] == "*:serial=abcd1234"
    assert seen["gain_db"] == 35.0
    assert seen["channel"] == 0
    assert seen["use_meta"] is False
    assert seen["settle_us"] == 500.0
    assert "driver" not in seen
    assert "rx_channel" not in seen
    assert "quick_tune" not in seen
