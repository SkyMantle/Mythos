"""bladeRF USB stream contracts that hang the NIOS or restart LOCK."""
from __future__ import annotations

import inspect
from pathlib import Path

from fpvscan.sdr import bladerf as B


SRC = (Path(__file__).resolve().parents[1]
       / "fpvscan" / "sdr" / "bladerf.py").read_text(encoding="utf-8")


def test_timeout_ms_outlasts_a_usb_hiccup():
    """A 100 ms timeout turns every stalled sync_rx into ERR_TIMEOUT plus
    _config_stream. 3500 ms is longer than a USB3 re-enumeration glitch
    and still fails closed if the cable is actually gone.
    """
    assert inspect.signature(B.BladeRF.__init__).parameters["timeout_ms"].default == 3500
    assert "self.timeout_ms), \"bladerf_sync_config\"" in SRC
    assert "self.timeout_ms)" in SRC


def test_sync_config_uses_rx_x1_layout_not_the_channel_index():
    """bladerf_sync_config's second argument is a *layout* enum (RX_X1=0).
    CHANNEL_RX(n) happens to equal 0 for RX1, but CHANNEL_RX(1) is 2, which
    is RX_X2 — a stereo stream on a board opened with one RX channel.
    """
    assert B.RX_X1 == 0
    assert B.RX_X2 == 2
    assert B.CHANNEL_RX(0) == 0
    assert B.CHANNEL_RX(1) == 2
    assert "self._dev, RX_X1, fmt" in SRC
    assert "self.ch = CHANNEL_RX(channel)" in SRC


def test_stream_reconfig_disables_the_module_first():
    """Calling sync_config on a live stream reallocates USB buffers while
    transfers are still in flight and hangs the NIOS control interface.
    """
    assert "if self._streaming:" in SRC
    assert "bladerf_enable_module(self._dev, self.ch, False)" in SRC
    cfg = SRC[SRC.index("def _config_stream"):SRC.index("def close")]
    assert cfg.index("bladerf_enable_module") < cfg.index("bladerf_sync_config")


def test_set_sample_rate_skips_resync_when_fs_is_within_one_hz():
    """libbladeRF often returns 34999872 for 35000000. Re-running
    _config_stream every LOCK frame is the driver-side twin of the
    engine fs-mismatch restart (merged #2).
    """
    body = SRC[SRC.index("def set_sample_rate"):SRC.index("def set_bias_tee")]
    assert "if self._streaming and abs(self._fs - float(hz)) < 1.0:" in body
    assert "return" in body.split("if self._streaming and abs", 1)[1].split("\n", 2)[1]


def test_lib_calls_demand_an_open_device():
    """A null bladerf handle is an access violation, not a BladeRFError.
    USB recover leaves _dev empty on purpose; _need_dev must fire first.
    """
    assert "def _need_dev(self):" in SRC
    assert "пристрій не відкритий" in SRC
    assert SRC.count("self._need_dev()") >= 5


def test_open_refuses_an_unconfigured_fpga():
    """An xA4 with no bitstream enumerates and then every sync_rx fails.
    install_pi.sh must ship bladerf-fpga-hostedxa4 for this check to pass
    on a fresh Pi image.
    """
    body = SRC[SRC.index("def open"):SRC.index("def _config_stream")]
    assert "bladerf_is_fpga_configured(self._dev) <= 0" in body
    assert "bladerf-fpga-hostedxa4" in body
