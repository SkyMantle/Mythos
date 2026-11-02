"""factory must not forward bladeRF USB YAML keys into SimSource."""
from __future__ import annotations

from fpvscan.sdr.factory import make_source


def test_factory_ignores_explicit_sim_usb_yaml(monkeypatch):
    """#83 factory ignores USB stream keys for file.
    #81 ignores file / sim keys for bladerf. #80 ignores
    sim `sample_rate` / `center_hz` / `gain_db` / `path` /
    `loop` / `realtime` / `emitters` / `noise_db` / `seed`.
    #73 ignores bladerf `timeout_ms` / `bandwidth_ratio`.
    The complementary leftover is a `driver: sim` block
    that *names* USB stream keys operators copy from a
    bladeRF stanza — `device` / `lib_path` / `rx_channel` /
    `agc` / `num_buffers` / `buffer_size` / `num_transfers` /
    `settle_us` / `quick_tune` / `timeout_ms` /
    `bandwidth_ratio`. Factory is `SimSource()` with no
    kwargs. A leftover `SimSource(device=cfg.get("device"),
    num_buffers=cfg.get("num_buffers"))` TypeErrors on
    start (constructor has no such kwarg) and the default
    scene never opens, so `confirm_hits=2` never lights
    the bird. Constructor must not see USB handle / stream
    / settle / timeout / bandwidth keys.
    """
    seen: dict = {}

    class Fake:
        name = "sim"

        def __init__(self, emitters=None, noise_db=-70.0, seed=0, **kw):
            seen["emitters"] = emitters
            seen["noise_db"] = noise_db
            seen["seed"] = seed
            seen.update(kw)

    monkeypatch.setattr("fpvscan.sdr.sim.SimSource", Fake)
    src = make_source({
        "driver": "sim",
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
    assert src.name == "sim"
    assert seen["emitters"] is None
    assert seen["noise_db"] == -70.0
    assert seen["seed"] == 0
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
