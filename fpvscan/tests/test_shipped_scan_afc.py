"""Shipped YAML knobs that are not the engine/DSP fallbacks."""
from pathlib import Path

from fpvscan.config import load


CFG = load(Path(__file__).resolve().parents[1] / "config.yaml")


def test_afc_deadband_is_tighter_than_the_engine_fallback():
    """Engine default is 150 kHz. YAML 100 kHz still ignores picture-mean
    wobble but a revert to 150 kHz lets a free-running VTx sit off-center
    for more LOCK frames before the mixer moves.
    """
    assert float(CFG["video"]["afc_deadband_hz"]) == 100e3
    assert float(CFG["video"]["afc_deadband_hz"]) < 150e3


def test_scan_averages_are_eight():
    """One FFT of noise makes occupancy flicker and INSPECT a CW spur.
    Eight averages is the dwell the 5 dB threshold was tuned against.
    """
    assert int(CFG["scan"]["averages"]) == 8


def test_stream_quality_is_60_not_encode_default():
    """cvbs.encode default quality is 75. On Wi-Fi to a phone that extra
    15 points stalls the WebSocket behind one fat WebP and the console
    looks frozen even though LOCK is still decoding.
    """
    assert int(CFG["video"]["stream_quality"]) == 60


def test_auto_peek_cooldown_is_a_full_minute():
    """Cooldown 0 re-locks the same bird every sweep pass and never
    finishes 400 MHz–6 GHz. 60 s is one peek per VTx per minute.
    """
    assert float(CFG["scan"]["auto_peek_cooldown_s"]) == 60
