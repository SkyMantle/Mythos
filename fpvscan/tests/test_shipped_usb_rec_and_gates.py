"""Shipped YAML values that silently hang USB or hide a weak VTx."""
from __future__ import annotations

import inspect
from pathlib import Path

from fpvscan.config import load
from fpvscan.engine import Engine


CFG = load(Path(__file__).resolve().parents[1] / "config.yaml")


def test_usb_stream_knobs_keep_libbladerf_alive():
    """buffer_size must be a multiple of 1024. num_transfers >= num_buffers
    hangs sync_rx on the xA4. These are the values factory() forwards.
    """
    sdr = CFG["sdr"]
    assert int(sdr["num_buffers"]) == 32
    assert int(sdr["buffer_size"]) == 32768
    assert int(sdr["buffer_size"]) % 1024 == 0
    assert int(sdr["num_transfers"]) == 16
    assert int(sdr["num_transfers"]) < int(sdr["num_buffers"])
    assert int(sdr["rx_channel"]) == 0


def test_bias_tee_offset_is_the_bench_15db_not_code_18():
    """Engine fallback is 18 dB. The LNA on the stand was measured at
    ~15 dB; YAML 0 clips the ADC, YAML 18 under-gains a distant VTx.
    """
    assert float(CFG["sdr"]["bias_tee_gain_offset_db"]) == 15


def test_shipped_min_bw_beats_the_engine_4mhz_fallback():
    """_do_sweep does scan.get('min_bw_hz', 4e6). Dropping the YAML key
    restores the 4 MHz floor that rejected the 4.3 MHz bench bird.
    """
    assert float(CFG["scan"]["min_bw_hz"]) == 2e6
    src = inspect.getsource(Engine._do_sweep)
    assert 'scan.get("min_bw_hz", 4e6)' in src


def test_classifier_and_peek_floors_stay_at_the_tuned_values():
    sc = CFG["scan"]
    assert float(sc["min_confidence"]) == 0.45
    assert float(sc["line_prominence_db"]) == 8.0
    assert float(sc["auto_peek_min_conf"]) == 0.6


def test_recorder_crf_and_preset_are_pi_safe():
    """rec_crf 18 and preset=slow fill the SD card and stall LOCK on Pi 5."""
    assert int(CFG["video"]["rec_crf"]) == 24
    assert CFG["video"]["rec_preset"] == "veryfast"
