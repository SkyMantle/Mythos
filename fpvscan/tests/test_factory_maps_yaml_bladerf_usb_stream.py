"""factory must forward USB stream sizes that are not constructor defaults."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_maps_nondefault_bladerf_usb_stream_yaml(monkeypatch):
    """#21 maps shipped-matching USB sizes (32 / 32768 / 16) and
    `agc: false` — those are also BladeRF() defaults, so dropping
    the kwargs still looks like a live open. #73 omitted
    `timeout_ms` / `bandwidth_ratio` stay constructor 3500 / 0.9
    (explicit 0 ERR_TIMEOUT-loops or aliases the 16.2 kHz line
    into DC). #91 maps `rx_channel` → `channel`; #97 maps
    `quick_tune` → `use_meta`; #98 empty `device: ''` stays "".
    The missing half: explicit YAML that *differs* from the
    constructor — `num_buffers: 8` / `buffer_size: 4096` /
    `num_transfers: 4` / `agc: true` — must actually reach
    BladeRF(). Operators shrink the USB ring on a Pi that
    cannot keep 32×32 KiB, or turn AGC on for search. A leftover
    omitted-kwarg still constructs (`sync_config` sees 32/32768/16
    and MGC) while the YAML the operator edited is ignored:
    USB overruns on the small-RAM box, or AGC-off search never
    lifts a weak VTx above `threshold_db`, so `confirm_hits=2`
    never lights the bird. Constructor must see the YAML values,
    not 32/32768/16/False, and must not see YAML-only names
    (`driver` / `rx_channel` / `quick_tune`). `timeout_ms` /
    `bandwidth_ratio` still stay constructor defaults (#73).
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
        "gain_db": 35,
        "rx_channel": 0,
        "agc": True,
        "num_buffers": 8,
        "buffer_size": 4096,
        "num_transfers": 4,
        "settle_us": 500,
        "quick_tune": False,
        "timeout_ms": 0,
        "bandwidth_ratio": 0,
    })
    assert src.name == "bladerf"
    assert seen["num_buffers"] == 8
    assert seen["buffer_size"] == 4096
    assert seen["num_transfers"] == 4
    assert seen["agc"] is True
    assert seen["num_buffers"] != 32
    assert seen["buffer_size"] != 32768
    assert seen["num_transfers"] != 16
    assert seen["agc"] is not False
    assert seen["gain_db"] == 35.0
    assert seen["settle_us"] == 500.0
    assert seen["channel"] == 0
    assert seen["use_meta"] is False
    assert seen["device"] == ""
    assert "timeout_ms" not in seen
    assert "bandwidth_ratio" not in seen
    assert seen.get("timeout_ms") != 0
    assert seen.get("bandwidth_ratio") != 0
    assert "driver" not in seen
    assert "rx_channel" not in seen
    assert "quick_tune" not in seen
