"""BladeRF.close must drop the RX module before releasing the handle."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "fpvscan" / "sdr" / "bladerf.py").read_text(encoding="utf-8")


def test_close_disables_rx_then_closes_and_clears_dev():
    """`bladerf_close` on a still-enabled stream leaves USB transfers
    in flight. The next process's `open()` then hangs the NIOS
    control interface — the same failure `_config_stream` already
    guards. close() must disable the module, close the handle, and
    zero `_dev` so `_need_dev` is what the next caller hits, not a
    use-after-free.
    """
    body = SRC[SRC.index("def close(self)"):SRC.index("def _need_dev")]
    assert "bladerf_enable_module(self._dev, self.ch, False)" in body
    assert "bladerf_close(self._dev)" in body
    assert body.index("bladerf_enable_module") < body.index("bladerf_close")
    assert "self._streaming = False" in body
    assert "self._dev = C.c_void_p()" in body
