"""Omitted auto_peek_min_conf must peek at exactly 0.6, not only above it."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_omitted_min_conf_peeks_at_exactly_06():
    """#36 / #61 used 0.50 (list, no LOCK) vs 0.70 (peek). The gate is
    `confidence < 0.6`, so 0.60 must still peek. A leftover `<=`
    parks a shipped-floor analog hit in the list and never shows a
    picture; helpers always pass 0.6 and shipped YAML is 0.6, so the
    inclusive edge is the operator-partial-config path.
    """
    eng = make_engine()
    eng.cfg["scan"]["confirm_hits"] = 1
    eng.cfg["scan"].pop("auto_peek_min_conf")
    eng._merge(Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=14.0,
        confidence=0.60, standard="PAL",
    ))
    assert eng.state.mode == "LOCK"
    assert eng.state.auto is True
    assert eng.state.lock_target == 5800e6
