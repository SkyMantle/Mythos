"""Вибір джерела за конфігом."""
from __future__ import annotations

from .base import SdrSource


def make_source(cfg: dict) -> SdrSource:
    driver = cfg.get("driver", "bladerf")

    if driver == "bladerf":
        from .bladerf import BladeRF
        return BladeRF(
            device=cfg.get("device", ""),
            lib_path=cfg.get("lib_path") or None,
            channel=int(cfg.get("rx_channel", 0)),
            gain_db=float(cfg.get("gain_db", 30)),
            agc=bool(cfg.get("agc", False)),
            num_buffers=int(cfg.get("num_buffers", 32)),
            buffer_size=int(cfg.get("buffer_size", 32768)),
            num_transfers=int(cfg.get("num_transfers", 16)),
            settle_us=float(cfg.get("settle_us", 400)),
            use_meta=bool(cfg.get("quick_tune", False)),
        )

    if driver == "file":
        from .file import FileSource
        return FileSource(cfg["path"], loop=bool(cfg.get("loop", True)))

    if driver == "sim":
        from .sim import SimSource
        return SimSource()

    raise ValueError(f"Невідомий драйвер: {driver}")
