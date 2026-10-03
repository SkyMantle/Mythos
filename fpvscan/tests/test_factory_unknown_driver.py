"""A typo in sdr.driver must fail at factory, not hang on the radio."""
from __future__ import annotations

import pytest

from fpvscan.sdr.factory import make_source


def test_unknown_driver_raises_before_open():
    """`driver: uhd` (or a leftover `rtlsdr`) must not fall through
    to BladeRF() and block `run.py` on a missing board while the
    operator thinks they asked for `--file`.
    """
    with pytest.raises(ValueError, match="uhd"):
        make_source({"driver": "uhd"})
