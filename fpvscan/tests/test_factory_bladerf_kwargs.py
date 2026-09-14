"""factory.make_source must map YAML onto BladeRF / FileSource correctly."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_maps_yaml_onto_bladerf(monkeypatch):
    """Empty `lib_path: ''` must become None (search path), not a
    filename. `quick_tune` is `use_meta` — swapping those starts a
    metadata stream without profiles and every read is the wrong size.
    """
    seen: dict = {}

    class Fake:
        name = "bladerf"

        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.bladerf.BladeRF", Fake)
    src = make_source({
        "driver": "bladerf",
        "device": "*:serial=abcd1234",
        "lib_path": "",
        "rx_channel": 1,
        "gain_db": 35,
        "agc": False,
        "num_buffers": 32,
        "buffer_size": 32768,
        "num_transfers": 16,
        "settle_us": 500,
        "quick_tune": True,
    })
    assert src.name == "bladerf"
    assert seen["lib_path"] is None
    assert seen["use_meta"] is True
    assert seen["settle_us"] == 500.0
    assert seen["agc"] is False
    assert seen["buffer_size"] == 32768
    assert seen["gain_db"] == 35.0
    assert seen["channel"] == 1
    assert seen["device"] == "*:serial=abcd1234"
    assert seen["num_buffers"] == 32
    assert seen["num_transfers"] == 16


def test_factory_file_loops_by_default(monkeypatch):
    """A 2–4 s capture is shorter than a sweep+LOCK session. loop=False
    raises EOFError mid-inspect and the replay looks like 'no signal'.
    """
    seen: dict = {}

    class Fake:
        name = "file"

        def __init__(self, path, loop=True):
            seen["path"] = path
            seen["loop"] = loop

    monkeypatch.setattr("fpvscan.sdr.file.FileSource", Fake)
    make_source({"driver": "file", "path": "caps/vtx.cf32"})
    assert seen["path"] == "caps/vtx.cf32"
    assert seen["loop"] is True
