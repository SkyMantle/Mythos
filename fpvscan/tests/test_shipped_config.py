"""Shipped config.yaml flags that silently kill NTSC LOCK or walk CVBS sync."""
from pathlib import Path

from fpvscan.config import load


CFG = load(Path(__file__).resolve().parents[1] / "config.yaml")


def test_agc_is_off_in_shipped_config():
    """AGC is useful while hunting; on LOCK it pumps the discriminator."""
    assert CFG["sdr"]["agc"] is False


def test_min_lines_is_absent_or_low_enough_for_one_field_ntsc():
    """A 232-line NTSC field is valid. min_lines 250 (old default) drops it."""
    min_lines = CFG["video"].get("min_lines", 200)
    assert min_lines <= 200


def test_confirm_hits_rejects_a_single_sweep_flash():
    assert int(CFG["scan"]["confirm_hits"]) >= 2


def test_bias_tee_gain_offset_is_positive():
    """Zero offset with bias-tee on clips the ADC on a typical LNA."""
    assert float(CFG["sdr"]["bias_tee_gain_offset_db"]) > 0
